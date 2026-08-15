"""S3 storage for source documents, and the only place a download URL is minted.

S3 is the source of truth, so this module owns the one write in the pipeline that is
not reconstructible. Two rules shape everything here.

**Keys are content-addressed under a tenant prefix**, and the layout lives in
`keys.py` because the S3 notification filter and the IAM prefix boundary both depend
on it. The prefix is not decoration: a bucket policy can pin a role to
`processed/{tenant_id}/*`, so a bug that computes the wrong key fails closed on the
IAM boundary instead of reading another firm's file. Content addressing makes a
double-submit idempotent.

**Presigned URLs are minted per request and never persisted.** A URL expires in
minutes; an audit trail holding one is provenance that stops resolving, and a stored
URL is a bearer credential sitting in a database. `SourceLocator` therefore stores the
document id, and this module turns that into a link at the moment someone clicks it.

Authorization does not live here, with one exception. A presigned URL bypasses every
application check once it exists, so the caller must be authorised *before* the URL is
created — that is the route's job. What this module adds is the belt-and-braces half:
pass a `ctx` and it refuses to sign anything outside that caller's tenant.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol

from src.config import LexGraphConfig
from src.documents.ingest import guess_media_type
from src.documents.keys import (
    PROCESSED_PREFIX,
    document_key,
    parse_document_key,
    parse_raw_key,
    raw_key,
    safe_filename,
)
from src.documents.models import DocumentMeta, sha256_hex
from src.graph.scope import AuthContext, ScopeViolation

logger = logging.getLogger(__name__)

#: A provenance link is for checking a citation now, not for sharing. Short enough
#: that a URL pasted into an email is dead before it is read.
DEFAULT_EXPIRY_SECONDS = 900

#: Hard ceiling. Above an hour a presigned URL stops being a click-through and starts
#: being an unrevocable, unauditable copy of the document.
MAX_EXPIRY_SECONDS = 3600

#: An upload ticket has to outlive a slow connection uploading a large bundle, so it is
#: longer-lived than a download link. It only grants a write to one key under one
#: tenant prefix, which is a much narrower grant than read access to a document.
DEFAULT_UPLOAD_EXPIRY_SECONDS = 1800
MAX_UPLOAD_EXPIRY_SECONDS = 3600


class S3Like(Protocol):
    """The slice of the boto3 S3 client this module uses."""

    def put_object(self, **kwargs: Any) -> dict[str, Any]: ...
    def head_object(self, **kwargs: Any) -> dict[str, Any]: ...
    def get_object(self, **kwargs: Any) -> dict[str, Any]: ...
    def copy_object(self, **kwargs: Any) -> dict[str, Any]: ...
    def delete_object(self, **kwargs: Any) -> dict[str, Any]: ...
    def generate_presigned_url(self, ClientMethod: str, **kwargs: Any) -> str: ...
    def generate_presigned_post(self, Bucket: str, Key: str, **kwargs: Any) -> dict[str, Any]: ...
    def list_objects_v2(self, **kwargs: Any) -> dict[str, Any]: ...


class DocumentIndex(Protocol):
    """Resolves a document id to its S3 location.

    A cache, not a system of record. `document_id` is a one-way hash of tenant and content
    so it cannot be reversed, but the key contains both halves, which means a miss is
    recoverable by listing the tenant's prefix rather than fatal. `describe` does that.

    The default implementation is per-process, which is why the fallback matters: without it
    every deploy made previously uploaded documents report as "no longer in storage".
    """

    def put_document(self, doc: DocumentMeta) -> None: ...
    def get_document(self, document_id: str) -> DocumentMeta | None: ...


class InMemoryDocumentIndex:
    def __init__(self) -> None:
        self._documents: dict[str, DocumentMeta] = {}

    def put_document(self, doc: DocumentMeta) -> None:
        self._documents[doc.document_id] = doc

    def get_document(self, document_id: str) -> DocumentMeta | None:
        return self._documents.get(document_id)

    def drop_document(self, document_id: str) -> None:
        self._documents.pop(document_id, None)


class DocumentNotFound(LookupError):
    """No such document, or one this tenant cannot see.

    Deliberately does not distinguish the two, matching `scope.py`: confirming that a
    document exists elsewhere is itself a leak.
    """


@dataclass(frozen=True)
class DownloadLink:
    """A short-lived link, plus when it dies. Never stored on an assertion."""

    url: str
    expires_at: str
    expires_in: int
    filename: str
    page: int | None = None


@dataclass(frozen=True)
class UploadTicket:
    """A presigned POST the browser submits the file to directly.

    POST rather than PUT because only POST carries policy conditions, so the size cap
    is enforced by S3 instead of being a promise the client makes.
    """

    url: str
    fields: dict[str, str]
    key: str
    upload_id: str
    expires_in: int
    max_bytes: int


@dataclass(frozen=True)
class RawUpload:
    """An object that landed under `raw/`, with the tenant read back out of its key."""

    tenant_id: str
    upload_id: str
    filename: str
    key: str
    body: bytes
    media_type: str
    uploaded_by: str
    matter_id: str | None


class DocumentStorage:
    def __init__(
        self,
        bucket: str,
        *,
        kms_key_id: str = "",
        s3: S3Like | None = None,
        s3_factory: Callable[[], S3Like] | None = None,
        index: DocumentIndex | None = None,
    ) -> None:
        self.bucket = bucket
        self.kms_key_id = kms_key_id
        self.index = index or InMemoryDocumentIndex()
        self._s3 = s3
        self._s3_factory = s3_factory

    @property
    def s3(self) -> S3Like:
        if self._s3 is None:
            factory = self._s3_factory
            if factory is None:
                import boto3
                from botocore.config import Config

                # signature_version must be set explicitly. The client resolves to s3v4 on
                # its own, but `generate_presigned_post` still builds a v2 policy from the
                # default config and emits AWSAccessKeyId/signature, and S3 refuses a v2
                # POST to a KMS-encrypted bucket: "Requests specifying Server Side
                # Encryption with AWS KMS managed keys require AWS Signature Version 4".
                def factory() -> S3Like:
                    return boto3.client("s3", config=Config(signature_version="s3v4"))

            self._s3 = factory()
        return self._s3

    def put_document(
        self,
        ctx: AuthContext,
        *,
        filename: str,
        body: bytes,
        matter_id: str | None = None,
        media_type: str | None = None,
    ) -> DocumentMeta:
        """Upload the bytes and record where they went.

        An existing object is not re-put. The key is content-addressed, so a hit means
        the bytes are already there and a second version would differ only in its
        timestamp — noise in a bucket whose versions are cited by provenance.
        """
        if matter_id is not None:
            ctx.assert_can_read_matter(matter_id)

        content_sha256 = sha256_hex(body)
        key = document_key(ctx.tenant_id, content_sha256, filename)
        resolved_type = media_type or guess_media_type(filename)

        existing = self._head(key)
        if existing is None:
            args: dict[str, Any] = {
                "Bucket": self.bucket,
                "Key": key,
                "Body": body,
                "ContentType": resolved_type,
                "Metadata": {
                    "content-sha256": content_sha256,
                    "tenant-id": ctx.tenant_id,
                    "uploaded-by": ctx.user_id,
                    "filename": safe_filename(filename),
                },
            }
            if self.kms_key_id:
                # Omitted when unset so the bucket's default encryption applies;
                # naming AES256 here would silently downgrade a KMS bucket.
                args["ServerSideEncryption"] = "aws:kms"
                args["SSEKMSKeyId"] = self.kms_key_id
            version_id = self.s3.put_object(**args).get("VersionId")
        else:
            version_id = existing.get("VersionId")

        doc = DocumentMeta(
            tenant_id=ctx.tenant_id,
            bucket=self.bucket,
            key=key,
            filename=filename,
            media_type=resolved_type,
            content_sha256=content_sha256,
            byte_size=len(body),
            uploaded_by=ctx.user_id,
            matter_id=matter_id,
            s3_version_id=version_id,
        )
        prior = self.index.get_document(doc.document_id)
        if prior is not None:
            return prior
        self.index.put_document(doc)
        logger.info("stored %s as %s", doc.key, doc.document_id)
        return doc

    def describe(self, document_id: str, *, tenant_id: str = "") -> DocumentMeta | None:
        """Where a document lives. Callers use this to authorise *before* presigning.

        Falls back to S3 on a miss, because the default index is per-process: a document
        uploaded before the last deploy was reported as "no longer in storage" while its
        bytes sat in the bucket untouched. S3 is the source of truth, so the index is a
        cache, and a cache miss must not read as absence.

        The lookup works without any stored mapping because `processed/{tenant}/{sha}/{name}`
        already contains everything: `document_id` is `sha256(tenant:content_sha)`, so listing
        a tenant's prefix and recomputing the id finds the object. Scoped to one tenant's
        prefix, so a miss cannot walk into another firm's documents.
        """
        found = self.index.get_document(document_id)
        if found is not None or not tenant_id:
            return found
        return self._find_in_bucket(document_id, tenant_id)

    def _find_in_bucket(self, document_id: str, tenant_id: str) -> DocumentMeta | None:
        prefix = f"{PROCESSED_PREFIX}{tenant_id}/"
        token: str | None = None
        while True:
            kwargs: dict[str, Any] = {"Bucket": self.bucket, "Prefix": prefix}
            if token:
                kwargs["ContinuationToken"] = token
            try:
                page = self.s3.list_objects_v2(**kwargs)
            except Exception as e:
                logger.warning("could not list %s while resolving %s: %s", prefix, document_id, e)
                return None

            for obj in page.get("Contents", []):
                key = str(obj.get("Key", ""))
                parsed = parse_document_key(key)
                if parsed is None:
                    continue
                key_tenant, content_sha256, filename = parsed
                if f"doc-{sha256_hex(f'{key_tenant}:{content_sha256}')[:24]}" != document_id:
                    continue
                meta = DocumentMeta(
                    tenant_id=key_tenant,
                    bucket=self.bucket,
                    key=key,
                    filename=filename,
                    media_type=guess_media_type(filename),
                    content_sha256=content_sha256,
                    byte_size=int(obj.get("Size", 0) or 0),
                    # Not recoverable from the key. Left empty rather than guessed: an
                    # invented uploader on an audit surface is worse than a blank one.
                    uploaded_by="",
                )
                # Repopulate, so a burst of page requests costs one listing.
                self.index.put_document(meta)
                logger.info("resolved %s from the bucket after an index miss", document_id)
                return meta

            token = page.get("NextContinuationToken")
            if not token:
                return None

    def presign_upload(
        self,
        ctx: AuthContext,
        *,
        filename: str,
        media_type: str | None = None,
        matter_id: str | None = None,
        max_bytes: int,
        expires_in: int = DEFAULT_UPLOAD_EXPIRY_SECONDS,
        upload_id: str | None = None,
    ) -> UploadTicket:
        """Mint a presigned POST for a browser upload into `raw/`.

        The file never crosses the API, so a 400-page bundle is not bounded by the
        60s CloudFront origin timeout. What the API keeps is the part that must not be
        client-controlled: the key, and therefore the tenant prefix.

        `content-length-range` is a signed condition, so S3 rejects an oversized body
        itself. Without it the cap would be advice — the whole point of POST over PUT.

        The tenant is authorised here, but note that the *matter* travels as metadata
        on the object and is re-checked when the notification is processed. A presigned
        POST cannot be trusted to preserve it: conditions constrain what S3 accepts,
        they do not stop a caller replaying the ticket with different metadata.
        """
        if not 0 < expires_in <= MAX_UPLOAD_EXPIRY_SECONDS:
            raise ValueError(
                f"expires_in must be in 1..{MAX_UPLOAD_EXPIRY_SECONDS}s, got {expires_in}"
            )
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        if matter_id is not None:
            ctx.assert_can_read_matter(matter_id)

        upload = upload_id or uuid.uuid4().hex
        key = raw_key(ctx.tenant_id, upload, filename)
        resolved_type = media_type or guess_media_type(filename)

        # Echoed back by S3 on the notification, so the worker knows who uploaded and
        # which matter was intended without a second lookup.
        fields: dict[str, str] = {
            "Content-Type": resolved_type,
            "x-amz-meta-tenant-id": ctx.tenant_id,
            "x-amz-meta-uploaded-by": ctx.user_id,
            "x-amz-meta-filename": safe_filename(filename),
        }
        if matter_id:
            fields["x-amz-meta-matter-id"] = matter_id
        if self.kms_key_id:
            fields["x-amz-server-side-encryption"] = "aws:kms"
            fields["x-amz-server-side-encryption-aws-kms-key-id"] = self.kms_key_id

        conditions: list[Any] = [
            ["content-length-range", 1, max_bytes],
            {"Content-Type": resolved_type},
        ]
        conditions.extend({k: v} for k, v in fields.items() if k != "Content-Type")

        post = self.s3.generate_presigned_post(
            Bucket=self.bucket,
            Key=key,
            Fields=dict(fields),
            Conditions=conditions,
            ExpiresIn=expires_in,
        )
        logger.info("presigned upload %s for %s", key, ctx.user_id)
        return UploadTicket(
            url=post["url"],
            fields=post["fields"],
            key=key,
            upload_id=upload,
            expires_in=expires_in,
            max_bytes=max_bytes,
        )

    def read_raw(self, key: str, *, max_bytes: int) -> RawUpload:
        """Read an object that landed under `raw/`, refusing anything that is not one.

        The notification supplies only a key, so the tenant comes from the key and the
        uploader from object metadata. Both are server-written — the key by
        `presign_upload`, the metadata by a signed condition — so neither is a claim the
        client can vary after the fact.
        """
        parsed = parse_raw_key(key)
        if parsed is None:
            raise DocumentNotFound(f"{key!r} is not a raw upload key")
        tenant_id, upload_id, filename = parsed

        obj = self.s3.get_object(Bucket=self.bucket, Key=key)
        body = obj["Body"].read() if hasattr(obj["Body"], "read") else obj["Body"]
        if len(body) > max_bytes:
            # S3's own condition should have refused this, so reaching here means the
            # object was written by something other than a minted ticket.
            raise DocumentNotFound(f"{key!r} exceeds {max_bytes} bytes")

        meta = {k.lower(): v for k, v in (obj.get("Metadata") or {}).items()}
        # The key is authoritative for tenant: metadata is a convenience, the prefix is
        # the boundary. A mismatch means the object was not written by our ticket.
        if meta.get("tenant-id", tenant_id) != tenant_id:
            raise DocumentNotFound(f"{key!r} tenant does not match its prefix")

        return RawUpload(
            tenant_id=tenant_id,
            upload_id=upload_id,
            filename=meta.get("filename") or filename,
            key=key,
            body=body,
            media_type=obj.get("ContentType") or guess_media_type(filename),
            uploaded_by=meta.get("uploaded-by", "unknown"),
            matter_id=meta.get("matter-id") or None,
        )

    def discard_raw(self, key: str) -> None:
        """Drop a consumed or rejected upload. Best-effort: the lifecycle rule is the
        real guarantee, so a failure here is logged rather than raised."""
        try:
            self.s3.delete_object(Bucket=self.bucket, Key=key)
        except Exception as e:
            logger.warning("could not discard raw upload %s: %s", key, e)

    def presign_download(
        self,
        document_id: str,
        *,
        expires_in: int = DEFAULT_EXPIRY_SECONDS,
        ctx: AuthContext | None = None,
        page: int | None = None,
    ) -> DownloadLink:
        """Mint a short-lived GET URL for a document.

        The caller must already have checked that this principal may read the
        document's matter — once signed, the URL answers to nobody. Passing `ctx` adds
        a tenant check here as well, which is cheap and catches a wrong `document_id`
        before it becomes a credential.
        """
        if not 0 < expires_in <= MAX_EXPIRY_SECONDS:
            raise ValueError(
                f"expires_in must be in 1..{MAX_EXPIRY_SECONDS}s, got {expires_in}, a "
                "longer-lived URL is an unrevocable copy of the document"
            )

        doc = self.index.get_document(document_id)
        if doc is None:
            raise DocumentNotFound(f"no document {document_id!r}")
        if ctx is not None and doc.tenant_id != ctx.tenant_id:
            raise ScopeViolation(f"no document {document_id!r}")

        display = safe_filename(doc.filename)
        url = self.s3.generate_presigned_url(
            "get_object",
            Params={
                "Bucket": doc.bucket,
                "Key": doc.key,
                # inline so the browser's PDF viewer opens it in place — the reviewer is
                # checking a citation, not collecting a file.
                "ResponseContentDisposition": f'inline; filename="{display}"',
                "ResponseContentType": doc.media_type,
            },
            ExpiresIn=expires_in,
        )
        if page:
            # Fragments are never sent to S3, so this does not disturb the signature.
            # PDF viewers honour it and land the reviewer on the cited page.
            url = f"{url}#page={page}"

        expires_at = (datetime.now(timezone.utc) + timedelta(seconds=expires_in)).isoformat()
        logger.info("presigned %s for %ss", document_id, expires_in)
        return DownloadLink(
            url=url,
            expires_at=expires_at,
            expires_in=expires_in,
            filename=doc.filename,
            page=page,
        )

    def document_exists(self, document_id: str) -> bool:
        doc = self.index.get_document(document_id)
        if doc is None:
            return False
        return self._head(doc.key) is not None

    def delete_document(self, document_id: str) -> bool:
        """Delete the current version and forget the document.

        Under versioning this writes a delete marker rather than destroying bytes, so a
        routine delete stays recoverable. Erasing the bytes means deleting every
        version, which is deliberately a separate, privileged act.
        """
        doc = self.index.get_document(document_id)
        if doc is None:
            return False
        self.s3.delete_object(Bucket=doc.bucket, Key=doc.key)
        drop = getattr(self.index, "drop_document", None)
        if drop is not None:
            drop(document_id)
        logger.info("deleted %s (%s)", document_id, doc.key)
        return True

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


#: Storage is process-wide rather than per-request because the document index it wraps
#: is the process's memory of where documents live. `set_document_storage` is the
#: injection seam for tests and for wiring that does not go through config.
_override: DocumentStorage | None = None
_by_bucket: dict[str, DocumentStorage] = {}


def set_document_storage(storage: DocumentStorage | None) -> None:
    global _override
    _override = storage


def storage_from_config(config: LexGraphConfig) -> DocumentStorage | None:
    """None when no bucket is configured.

    Unconfigured storage degrades provenance to file/page/quote with no link, which is
    still checkable by hand. It is not an error, so it does not raise.
    """
    if _override is not None:
        return _override
    bucket = config.documents.bucket
    if not bucket:
        return None
    if bucket not in _by_bucket:
        _by_bucket[bucket] = DocumentStorage(bucket, kms_key_id=config.documents.kms_key_id)
    return _by_bucket[bucket]
