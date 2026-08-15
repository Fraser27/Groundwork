"""Runs the ingest pipeline off the request path.

The upload endpoint used to do all of this inside one POST: store, transcribe every
page, chunk, embed, extract, stage, promote. CloudFront's origin read timeout is 60s and
the ALB's idle timeout matches, so a large bundle returned a 504 *after* S3 and the graph
had already been written — the browser was told the upload failed while the work had
partly succeeded.

Now the browser uploads straight to S3 and this runs afterwards, triggered by the object
landing. Three properties make that safe rather than merely asynchronous:

**Job state is written before the phase it describes is attempted.** A job recorded as
PARSING with no live worker is retryable from `retry_target`; one still recorded as
REGISTERED is indistinguishable from an upload nobody picked up. This is what makes
container death survivable — see `job_store.JobTracker`.

**Ingestion is capped.** A bulk upload of fifty documents would otherwise be fifty
concurrent ingests times the per-document page concurrency, which is a burst of Bedrock
throttling and a surprise on the invoice.

**The raw object is promoted, not re-uploaded.** The bytes land under `raw/` where the
key cannot be content-addressed (nobody knows the digest yet); this hashes them and
writes the document to its content-addressed `processed/` key, so a double submit still
collapses to one document.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from typing import Any, Protocol

from src.documents.job_store import InMemoryJobStore, JobTracker
from src.documents.models import DocumentMeta, IngestJob, JobState
from src.documents.storage import DocumentStorage, RawUpload
from src.graph.scope import AuthContext

logger = logging.getLogger(__name__)


class Pipeline(Protocol):
    """The chunk/embed/extract/stage/promote half, injected to keep this module
    testable without a graph, a vector store or Bedrock."""

    def __call__(
        self,
        ctx: AuthContext,
        parsed: Any,
        *,
        matter_id: str | None,
        run_model_extraction: bool,
        job_id: str,
    ) -> dict[str, Any]: ...


class IngestLimiter:
    """Bounds concurrent ingests, refusing rather than queueing when full.

    Refusing is deliberate. The trigger is retried, and the object is still in S3 with a
    seven-day lifecycle, so a refusal costs a retry. An unbounded queue instead converts
    a bulk upload into memory pressure and a thundering herd at Bedrock, and the thing
    that eventually fails is a request that looked accepted.
    """

    def __init__(self, limit: int) -> None:
        self._sem = threading.BoundedSemaphore(max(1, limit))
        self.limit = max(1, limit)

    def try_acquire(self) -> bool:
        return self._sem.acquire(blocking=False)

    def release(self) -> None:
        self._sem.release()


class IngestBusy(RuntimeError):
    """Raised when the concurrency cap is already met. The caller should retry."""


class IngestRunner:
    def __init__(
        self,
        storage: DocumentStorage,
        *,
        pipeline: Pipeline,
        parser: Any | None = None,
        store: Any | None = None,
        limiter: IngestLimiter | None = None,
        max_upload_bytes: int,
        plain_text_types: frozenset[str],
        parse_plain_text: Callable[..., Any],
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self.storage = storage
        self.pipeline = pipeline
        self.parser = parser
        self.tracker = JobTracker(store or InMemoryJobStore())
        self.limiter = limiter or IngestLimiter(4)
        self.max_upload_bytes = max_upload_bytes
        self.plain_text_types = plain_text_types
        self.parse_plain_text = parse_plain_text
        self.on_event = on_event

    @property
    def store(self):
        return self.tracker.store

    def _emit(self, job: IngestJob, extra: dict[str, Any] | None = None) -> None:
        """Publish progress. Never allowed to break an ingest: the poller is the
        correctness guarantee and this is the enhancement on top of it."""
        if self.on_event is None:
            return
        payload = {
            "job_id": job.job_id,
            "document_id": job.document_id,
            "state": job.state.value,
            "reason": job.reason,
            **(extra or {}),
        }
        try:
            self.on_event(job.tenant_id, payload)
        except Exception as e:
            logger.warning("could not publish progress for %s: %s", job.job_id, e)

    def ingest_raw_key(self, key: str, *, run_model_extraction: bool = True) -> IngestJob:
        """Ingest an object that has landed under `raw/`.

        The tenant comes from the key and the uploader from object metadata, both
        server-written, so nothing here trusts a value the client could vary after the
        ticket was minted.
        """
        if not self.limiter.try_acquire():
            raise IngestBusy(f"at the {self.limiter.limit}-document ingest limit")
        try:
            raw = self.storage.read_raw(key, max_bytes=self.max_upload_bytes)
            return self._ingest(raw, run_model_extraction=run_model_extraction)
        finally:
            self.limiter.release()

    def _ingest(self, raw: RawUpload, *, run_model_extraction: bool) -> IngestJob:
        # A synthetic context: there is no request and no JWT on this path. It is scoped
        # to exactly the matter recorded on the object at ticket-minting time, where the
        # uploader's access *was* checked against their real grants. Narrow rather than
        # convenient: this context can reach one matter in one tenant and nothing else.
        ctx = AuthContext(
            tenant_id=raw.tenant_id,
            user_id=raw.uploaded_by,
            matter_allowlist=frozenset({raw.matter_id}) if raw.matter_id else frozenset(),
        )

        doc = self.storage.put_document(
            ctx,
            filename=raw.filename,
            body=raw.body,
            matter_id=raw.matter_id,
            media_type=raw.media_type,
        )
        job = self.tracker.open(IngestJob.for_document(doc))
        self._emit(job)

        # Consumed: the processed copy is the record now. Leaving it would re-trigger on
        # a redelivered notification and pay for the whole document twice.
        self.storage.discard_raw(raw.key)

        self.tracker.advance(job, JobState.FETCHING)
        self._emit(job)
        self.tracker.advance(job, JobState.PARSING)
        self._emit(job)

        parsed = self._parse(doc, raw.body, job)
        if parsed is None:
            return job

        self.tracker.advance(job, JobState.CHUNKING)
        self._emit(job)
        try:
            result = self.pipeline(
                ctx,
                parsed,
                matter_id=doc.matter_id,
                run_model_extraction=run_model_extraction,
                job_id=doc.document_id,
            )
        except Exception as e:
            logger.exception("pipeline failed for %s", doc.document_id)
            self.tracker.fail(job, JobState.CHUNK_FAILED, str(e))
            self._emit(job)
            return job

        job.chunk_count = int(result.get("chunks") or 0)
        for state in (JobState.EXTRACTING, JobState.EMBEDDING, JobState.GRAPH_STAGED):
            self.tracker.advance(job, state)
        self._emit(job, {"result": result})

        # This job's own staged-but-unpromoted count, not the tenant's. `pending_review` is
        # tenant-wide, so using it parked every later document at PENDING_REVIEW as soon as
        # one assertion was pending anywhere -- including documents that produced nothing,
        # which then showed "awaiting review" against an empty queue.
        staged = int(result.get("assertions_staged") or 0)
        promoted = int(result.get("assertions_live") or 0)
        if staged > promoted:
            self.tracker.advance(job, JobState.PENDING_REVIEW)
        else:
            # Nothing needs a human, so the job is done. APPROVED is passed through
            # rather than skipped: LIVE is only reachable from it, and the history should
            # show that no reviewer was involved.
            self.tracker.advance(job, JobState.PENDING_REVIEW)
            self.tracker.advance(job, JobState.APPROVED)
            self.tracker.advance(job, JobState.LIVE)
        self._emit(job, {"result": result})
        return job

    def _parse(self, doc: DocumentMeta, body: bytes, job: IngestJob) -> Any | None:
        """Transcribe, or record why not and stop.

        PARSE_FAILED rather than a success state with no text: the bytes are stored and
        the digest recorded, so this is re-runnable the moment Bedrock is reachable, and
        `retry_target` points back at PARSING.
        """
        if doc.media_type in self.plain_text_types:
            text = body.decode("utf-8", errors="replace")
            return self.parse_plain_text(doc.document_id, text, filename=doc.filename)

        if self.parser is None:
            self.tracker.fail(
                job, JobState.PARSE_FAILED, "no vision model client available (Bedrock unreachable)"
            )
            self._emit(job)
            return None

        try:
            return self.parser.parse(doc.document_id, body, filename=doc.filename)
        except Exception as e:
            logger.warning("transcription failed for %s: %s", doc.document_id, e)
            self.tracker.fail(job, JobState.PARSE_FAILED, str(e))
            self._emit(job)
            return None
