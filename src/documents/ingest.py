"""Register a document into S3 and drive its ingest job.

S3 is the only source of truth, so this is the one write in the pipeline that is not
reconstructible. Everything downstream (graph, vectors, chunks) can be dropped and
rebuilt from the object this module puts down.

Two consequences shape the code:

- Keys are content-addressed and prefixed by tenant. Re-uploading identical bytes
  resolves to the same key, so a double-submit is idempotent rather than a duplicate
  document with a second set of assertions.
- Nothing is overwritten. `register` refuses to replace an existing object with
  different bytes, which cannot happen under a content-addressed key but is worth
  failing loudly on rather than assuming.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Protocol

from src.documents.keys import document_key as _document_key
from src.documents.models import (
    DocumentMeta,
    IngestJob,
    JobState,
    sha256_hex,
)
from src.graph.scope import AuthContext

logger = logging.getLogger(__name__)

DEFAULT_MEDIA_TYPE = "application/octet-stream"

_MEDIA_TYPES = {
    "pdf": "application/pdf",
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "tif": "image/tiff",
    "tiff": "image/tiff",
    "txt": "text/plain",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}


class S3Like(Protocol):
    """The slice of the boto3 S3 client this module uses."""

    def put_object(self, **kwargs: Any) -> dict[str, Any]: ...
    def head_object(self, **kwargs: Any) -> dict[str, Any]: ...
    def get_object(self, **kwargs: Any) -> dict[str, Any]: ...


def guess_media_type(filename: str) -> str:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return _MEDIA_TYPES.get(ext, DEFAULT_MEDIA_TYPE)


def document_key(ctx: AuthContext, content_sha256: str, filename: str) -> str:
    """Tenant-prefixed content-addressed key for `ctx`.

    A thin adapter over `keys.document_key` so callers holding an `AuthContext` do not
    reach past it for `tenant_id`.
    """
    return _document_key(ctx.tenant_id, content_sha256, filename)


class DocumentAlreadyExists(ValueError):
    """Raised when a key holds different bytes than the caller is uploading."""


class IngestStore:
    """In-memory registry of documents and jobs.

    A seam, not a design. The real implementation is DynamoDB; keeping the interface
    this small means swapping it does not touch the pipeline. Jobs and documents are
    both derived from S3, so losing this store costs a re-scan, not data.
    """

    def __init__(self) -> None:
        self._documents: dict[str, DocumentMeta] = {}
        self._jobs: dict[str, IngestJob] = {}

    def put_document(self, doc: DocumentMeta) -> None:
        self._documents[doc.document_id] = doc

    def get_document(self, document_id: str) -> DocumentMeta | None:
        return self._documents.get(document_id)

    def put_job(self, job: IngestJob) -> None:
        self._jobs[job.job_id] = job

    def get_job(self, job_id: str) -> IngestJob | None:
        return self._jobs.get(job_id)

    def jobs_for_document(self, document_id: str) -> list[IngestJob]:
        return [j for j in self._jobs.values() if j.document_id == document_id]

    def jobs_in_state(self, tenant_id: str, state: JobState) -> list[IngestJob]:
        return [
            j for j in self._jobs.values() if j.tenant_id == tenant_id and j.state is state
        ]


class Ingestor:
    def __init__(
        self,
        bucket: str,
        *,
        s3: S3Like | None = None,
        store: IngestStore | None = None,
        s3_factory: Callable[[], S3Like] | None = None,
    ) -> None:
        self.bucket = bucket
        self.store = store or IngestStore()
        self._s3 = s3
        self._s3_factory = s3_factory

    @property
    def s3(self) -> S3Like:
        if self._s3 is None:
            factory = self._s3_factory
            if factory is None:
                import boto3

                factory = lambda: boto3.client("s3")  # noqa: E731
            self._s3 = factory()
        return self._s3

    def register(
        self,
        ctx: AuthContext,
        *,
        filename: str,
        body: bytes,
        matter_id: str | None = None,
        media_type: str | None = None,
    ) -> tuple[DocumentMeta, IngestJob]:
        """Put the bytes in S3 and open an ingest job.

        Returns the existing document if these exact bytes are already registered for
        this tenant; a fresh job is still opened, because re-ingesting with improved
        extractors is a legitimate reason to run the pipeline again.
        """
        if matter_id is not None:
            ctx.assert_can_read_matter(matter_id)

        content_sha256 = sha256_hex(body)
        key = document_key(ctx, content_sha256, filename)

        existing = self._head(key)
        if existing is not None and existing.get("Metadata", {}).get(
            "content-sha256", content_sha256
        ) != content_sha256:
            raise DocumentAlreadyExists(f"{key} holds different bytes")

        version_id = None
        if existing is None:
            put = self.s3.put_object(
                Bucket=self.bucket,
                Key=key,
                Body=body,
                ContentType=media_type or guess_media_type(filename),
                Metadata={
                    "content-sha256": content_sha256,
                    "tenant-id": ctx.tenant_id,
                    "uploaded-by": ctx.user_id,
                    "filename": filename,
                },
            )
            version_id = put.get("VersionId")
        else:
            version_id = existing.get("VersionId")

        doc = DocumentMeta(
            tenant_id=ctx.tenant_id,
            bucket=self.bucket,
            key=key,
            filename=filename,
            media_type=media_type or guess_media_type(filename),
            content_sha256=content_sha256,
            byte_size=len(body),
            uploaded_by=ctx.user_id,
            matter_id=matter_id,
            s3_version_id=version_id,
        )
        prior = self.store.get_document(doc.document_id)
        if prior is not None:
            doc = prior
        else:
            self.store.put_document(doc)

        job = IngestJob.for_document(doc)
        self.store.put_job(job)
        logger.info("registered %s as %s (job %s)", filename, doc.document_id, job.job_id)
        return doc, job

    def fetch(self, doc: DocumentMeta) -> bytes:
        """Read the bytes back, verifying they are the ones we recorded.

        The hash check is not paranoia: an assertion's `source_locator` offsets are
        only meaningful against the exact bytes that were parsed, so a mismatch means
        every offset for this document is suspect.
        """
        obj = self.s3.get_object(Bucket=doc.bucket, Key=doc.key)
        body = obj["Body"].read()
        actual = sha256_hex(body)
        if actual != doc.content_sha256:
            raise DocumentAlreadyExists(
                f"{doc.s3_uri} hash {actual[:12]} != recorded {doc.content_sha256[:12]}"
            )
        return body

    def advance(self, job: IngestJob, state: JobState, *, reason: str | None = None) -> IngestJob:
        job.advance(state, reason=reason)
        self.store.put_job(job)
        return job

    def fail(self, job: IngestJob, state: JobState, reason: str) -> IngestJob:
        logger.warning("job %s failed at %s: %s", job.job_id, state.value, reason)
        return self.advance(job, state, reason=reason)

    def _head(self, key: str) -> dict[str, Any] | None:
        try:
            return self.s3.head_object(Bucket=self.bucket, Key=key)
        except Exception as e:  # botocore raises ClientError; tests raise anything
            if "404" in str(e) or "NoSuchKey" in str(e) or "Not Found" in str(e):
                return None
            if type(e).__name__ == "ClientError":
                code = getattr(e, "response", {}).get("Error", {}).get("Code")
                if code in {"404", "NoSuchKey", "NotFound"}:
                    return None
            raise
