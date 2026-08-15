"""The async ingest path: presigned upload, then a notification-driven pipeline.

What is worth asserting here is not that the plumbing connects but that the properties
which make it *safe* to run off the request path hold:

- the size cap is enforced by S3, not promised by the client
- the key — and therefore the tenant prefix — is server-chosen
- a raw key from another tenant's prefix cannot be coaxed into ingesting
- job state is persisted before each phase, so container death is recoverable
- the concurrency cap actually refuses
- the raw object is consumed, so a redelivered notification is not a second charge

boto3 is stubbed throughout. No AWS credentials, no network.
"""

from __future__ import annotations

import io
from typing import Any

import pytest

from src.documents.job_store import InMemoryJobStore
from src.documents.keys import PROCESSED_PREFIX, RAW_PREFIX, raw_key
from src.documents.models import JobState
from src.documents.parse import parse_plain_text
from src.documents.runner import IngestBusy, IngestLimiter, IngestRunner
from src.documents.storage import DocumentNotFound, DocumentStorage
from src.graph.scope import AuthContext, ScopeViolation

TENANT = "firm-acme"
BUCKET = "lexgraph-docs"
TEXT = b"Held for the plaintiff. Costs reserved."
PLAIN_TEXT_TYPES = frozenset({"text/plain", "text/markdown", "text/csv"})


class FakeS3:
    def __init__(self) -> None:
        self.objects: dict[str, dict[str, Any]] = {}
        self.puts: list[dict[str, Any]] = []
        self.deletes: list[str] = []
        self.posts: list[dict[str, Any]] = []

    def put_object(self, **kw: Any) -> dict[str, Any]:
        self.puts.append(kw)
        self.objects[kw["Key"]] = kw
        return {"VersionId": f"v{len(self.puts)}"}

    def head_object(self, **kw: Any) -> dict[str, Any]:
        if kw["Key"] not in self.objects:
            raise RuntimeError("404 Not Found")
        return {"VersionId": "v1"}

    def get_object(self, **kw: Any) -> dict[str, Any]:
        obj = self.objects.get(kw["Key"])
        if obj is None:
            raise RuntimeError("404 Not Found")
        return {
            "Body": io.BytesIO(obj["Body"]),
            "ContentType": obj.get("ContentType", "application/pdf"),
            "Metadata": obj.get("Metadata", {}),
        }

    def copy_object(self, **kw: Any) -> dict[str, Any]:
        return {}

    def delete_object(self, **kw: Any) -> dict[str, Any]:
        self.deletes.append(kw["Key"])
        self.objects.pop(kw["Key"], None)
        return {}

    def generate_presigned_url(self, ClientMethod: str, **kw: Any) -> str:
        return "https://example.invalid/signed"

    def generate_presigned_post(self, Bucket: str, Key: str, **kw: Any) -> dict[str, Any]:
        self.posts.append({"Bucket": Bucket, "Key": Key, **kw})
        return {
            "url": f"https://{Bucket}.s3.amazonaws.com/",
            "fields": {"key": Key, **kw.get("Fields", {})},
        }

    def land(self, key: str, body: bytes, *, metadata: dict[str, str], content_type: str) -> None:
        """Simulate a browser completing a presigned POST."""
        self.objects[key] = {
            "Key": key,
            "Body": body,
            "Metadata": metadata,
            "ContentType": content_type,
        }


@pytest.fixture
def s3() -> FakeS3:
    return FakeS3()


@pytest.fixture
def storage(s3: FakeS3) -> DocumentStorage:
    return DocumentStorage(BUCKET, kms_key_id="alias/lexgraph-docs", s3=s3)


@pytest.fixture
def ctx() -> AuthContext:
    return AuthContext(tenant_id=TENANT, user_id="alice")


def make_runner(storage: DocumentStorage, **kw: Any) -> IngestRunner:
    calls: list[dict[str, Any]] = []

    def pipeline(ctx, parsed, *, matter_id, run_model_extraction, job_id):
        calls.append({"tenant": ctx.tenant_id, "job_id": job_id, "matter_id": matter_id})
        return {"chunks": 3, "pending_review": 0}

    runner = IngestRunner(
        storage,
        pipeline=kw.pop("pipeline", pipeline),
        store=kw.pop("store", InMemoryJobStore()),
        max_upload_bytes=kw.pop("max_upload_bytes", 1024),
        plain_text_types=PLAIN_TEXT_TYPES,
        parse_plain_text=parse_plain_text,
        **kw,
    )
    runner.pipeline_calls = calls  # type: ignore[attr-defined]
    return runner


class TestPresignUpload:
    def test_the_size_cap_is_a_signed_condition(self, storage: DocumentStorage, ctx, s3):
        """A PUT URL could only ask nicely. The whole reason for POST is that S3 itself
        rejects an oversized body."""
        storage.presign_upload(ctx, filename="motion.pdf", max_bytes=5_000)
        conditions = s3.posts[0]["Conditions"]
        assert ["content-length-range", 1, 5_000] in conditions

    def test_the_key_is_server_chosen_and_tenant_prefixed(self, storage, ctx, s3):
        """The client never supplies the key: the prefix is the IAM boundary."""
        ticket = storage.presign_upload(ctx, filename="motion.pdf", max_bytes=1024)
        assert ticket.key.startswith(f"{RAW_PREFIX}{TENANT}/")
        assert ticket.key.endswith("/motion.pdf")

    def test_a_traversing_filename_cannot_escape_the_prefix(self, storage, ctx):
        ticket = storage.presign_upload(ctx, filename="../../other-firm/secret.pdf", max_bytes=1024)
        assert ".." not in ticket.key
        assert ticket.key.startswith(f"{RAW_PREFIX}{TENANT}/")

    def test_two_uploads_of_one_filename_do_not_collide(self, storage, ctx):
        a = storage.presign_upload(ctx, filename="motion.pdf", max_bytes=1024)
        b = storage.presign_upload(ctx, filename="motion.pdf", max_bytes=1024)
        assert a.key != b.key

    def test_uploader_and_tenant_are_pinned_as_conditions(self, storage, ctx, s3):
        """Metadata the worker later trusts must be signed, or a replayed ticket could
        change who uploaded and which matter it was for."""
        storage.presign_upload(ctx, filename="motion.pdf", matter_id=None, max_bytes=1024)
        conditions = s3.posts[0]["Conditions"]
        assert {"x-amz-meta-tenant-id": TENANT} in conditions
        assert {"x-amz-meta-uploaded-by": "alice"} in conditions

    def test_a_matter_the_caller_cannot_read_is_refused(self, storage):
        """Authorisation happens before the ticket exists — once signed it answers to
        nobody."""
        narrow = AuthContext(tenant_id=TENANT, user_id="bob", matter_allowlist=frozenset({"m-1"}))
        with pytest.raises(ScopeViolation):
            storage.presign_upload(narrow, filename="x.pdf", matter_id="m-99", max_bytes=1024)

    def test_an_absurd_expiry_is_refused(self, storage, ctx):
        with pytest.raises(ValueError, match="expires_in"):
            storage.presign_upload(ctx, filename="x.pdf", max_bytes=1024, expires_in=99_999)


class TestReadRaw:
    def test_reads_back_tenant_and_uploader(self, storage: DocumentStorage, ctx, s3):
        key = raw_key(TENANT, "u1", "brief.txt")
        s3.land(
            key,
            TEXT,
            metadata={"tenant-id": TENANT, "uploaded-by": "alice", "filename": "brief.txt"},
            content_type="text/plain",
        )
        raw = storage.read_raw(key, max_bytes=1024)
        assert (raw.tenant_id, raw.uploaded_by, raw.filename) == (TENANT, "alice", "brief.txt")
        assert raw.body == TEXT

    def test_a_processed_key_is_refused(self, storage, s3):
        """Otherwise the pipeline could be pointed at its own output."""
        key = f"{PROCESSED_PREFIX}{TENANT}/{'a' * 64}/x.pdf"
        s3.land(key, TEXT, metadata={}, content_type="application/pdf")
        with pytest.raises(DocumentNotFound):
            storage.read_raw(key, max_bytes=1024)

    def test_metadata_cannot_override_the_prefix_tenant(self, storage, s3):
        """The key is the boundary. An object whose metadata disagrees was not written by
        one of our tickets, so it is refused rather than believed."""
        key = raw_key(TENANT, "u1", "x.txt")
        s3.land(key, TEXT, metadata={"tenant-id": "firm-other"}, content_type="text/plain")
        with pytest.raises(DocumentNotFound):
            storage.read_raw(key, max_bytes=1024)

    def test_an_oversized_object_is_refused(self, storage, s3):
        """S3's condition should have caught this, so reaching here means the object was
        not written by a minted ticket."""
        key = raw_key(TENANT, "u1", "x.txt")
        s3.land(key, b"x" * 100, metadata={}, content_type="text/plain")
        with pytest.raises(DocumentNotFound):
            storage.read_raw(key, max_bytes=10)


class TestIngestFromNotification:
    def _land(self, s3: FakeS3, *, matter_id: str | None = None) -> str:
        key = raw_key(TENANT, "u1", "brief.txt")
        meta = {"tenant-id": TENANT, "uploaded-by": "alice", "filename": "brief.txt"}
        if matter_id:
            meta["matter-id"] = matter_id
        s3.land(key, TEXT, metadata=meta, content_type="text/plain")
        return key

    def test_a_landed_object_runs_the_whole_pipeline(self, storage, s3):
        runner = make_runner(storage)
        job = runner.ingest_raw_key(self._land(s3))
        assert job.state is JobState.LIVE
        assert runner.pipeline_calls[0]["tenant"] == TENANT

    def test_the_document_is_written_to_its_content_addressed_key(self, storage, s3):
        """The raw key cannot be content-addressed because nobody knows the digest yet.
        Promoting it is what restores idempotency."""
        runner = make_runner(storage)
        runner.ingest_raw_key(self._land(s3))
        written = [p["Key"] for p in s3.puts]
        assert len(written) == 1
        assert written[0].startswith(f"{PROCESSED_PREFIX}{TENANT}/")

    def test_the_raw_object_is_consumed(self, storage, s3):
        """A redelivered notification would otherwise pay to transcribe the whole
        document a second time."""
        key = self._land(s3)
        make_runner(storage).ingest_raw_key(key)
        assert key in s3.deletes

    def test_the_matter_travels_from_the_object(self, storage, s3):
        runner = make_runner(storage)
        runner.ingest_raw_key(self._land(s3, matter_id="m-1"))
        assert runner.pipeline_calls[0]["matter_id"] == "m-1"

    def test_every_phase_is_recorded_in_order(self, storage, s3):
        """The history is what makes a stalled ingest diagnosable."""
        store = InMemoryJobStore()
        runner = make_runner(storage, store=store)
        job = runner.ingest_raw_key(self._land(s3))

        states = [h.state for h in store.get_job(TENANT, job.job_id).history]
        assert states[:5] == [
            JobState.REGISTERED,
            JobState.FETCHING,
            JobState.PARSING,
            JobState.CHUNKING,
            JobState.EXTRACTING,
        ]
        assert states[-1] is JobState.LIVE

    def test_state_is_persisted_before_the_pipeline_runs(self, storage, s3):
        """The property that makes container death survivable: if the process dies inside
        the pipeline, the stored state must already say so."""
        store = InMemoryJobStore()
        observed: list[JobState] = []

        def pipeline(ctx, parsed, *, matter_id, run_model_extraction, job_id):
            jobs = store.jobs_for_document(TENANT, parsed.document_id)
            observed.append(jobs[0].state)
            return {"chunks": 1, "pending_review": 0}

        runner = make_runner(storage, store=store, pipeline=pipeline)
        runner.ingest_raw_key(self._land(s3))
        assert observed == [JobState.CHUNKING]

    def test_a_pipeline_failure_is_recorded_with_a_retry_target(self, storage, s3):
        def boom(ctx, parsed, *, matter_id, run_model_extraction, job_id):
            raise RuntimeError("vector store unreachable")

        store = InMemoryJobStore()
        runner = make_runner(storage, store=store, pipeline=boom)
        job = runner.ingest_raw_key(self._land(s3))

        stored = store.get_job(TENANT, job.job_id)
        assert stored.state is JobState.CHUNK_FAILED
        assert stored.reason == "vector store unreachable"
        assert stored.state.retry_target is JobState.CHUNKING

    def test_a_missing_vision_model_fails_the_parse_not_the_upload(self, storage, s3):
        """The bytes are stored and hashed, so this is re-runnable once Bedrock exists."""
        key = raw_key(TENANT, "u1", "scan.pdf")
        s3.land(
            key,
            b"%PDF-1.7",
            metadata={"tenant-id": TENANT, "uploaded-by": "alice", "filename": "scan.pdf"},
            content_type="application/pdf",
        )
        runner = make_runner(storage, parser=None)
        job = runner.ingest_raw_key(key)
        assert job.state is JobState.PARSE_FAILED
        assert job.state.retry_target is JobState.PARSING
        # Stored anyway: the record is the bytes.
        assert any(p["Key"].startswith(PROCESSED_PREFIX) for p in s3.puts)

    def test_review_needed_stops_at_pending_review(self, storage, s3):
        """Staged but not promoted is what "needs review" means. This test used to pass a bare
        `pending_review`, which is a tenant-wide count, and so asserted the bug where one
        pending assertion anywhere held every later document open."""

        def needs_review(ctx, parsed, *, matter_id, run_model_extraction, job_id):
            return {"chunks": 2, "assertions_staged": 4, "assertions_live": 0, "pending_review": 4}

        runner = make_runner(storage, pipeline=needs_review)
        job = runner.ingest_raw_key(self._land(s3))
        assert job.state is JobState.PENDING_REVIEW
        assert not job.state.is_terminal

    def test_a_key_outside_raw_is_refused(self, storage, s3):
        runner = make_runner(storage)
        with pytest.raises(DocumentNotFound):
            runner.ingest_raw_key(f"{PROCESSED_PREFIX}{TENANT}/abc/x.pdf")


class TestProgressEvents:
    def test_progress_is_published_per_phase(self, storage, s3):
        seen: list[tuple[str, str]] = []
        key = raw_key(TENANT, "u1", "brief.txt")
        s3.land(
            key,
            TEXT,
            metadata={"tenant-id": TENANT, "uploaded-by": "alice", "filename": "brief.txt"},
            content_type="text/plain",
        )
        runner = make_runner(
            storage, on_event=lambda tenant, ev: seen.append((tenant, ev["state"]))
        )
        runner.ingest_raw_key(key)

        assert {t for t, _ in seen} == {TENANT}
        assert "LIVE" in [s for _, s in seen]

    def test_a_broken_subscriber_does_not_fail_the_ingest(self, storage, s3):
        """Progress reporting is the enhancement; the poll is the guarantee. A websocket
        must never be able to lose a document."""
        key = raw_key(TENANT, "u1", "brief.txt")
        s3.land(
            key,
            TEXT,
            metadata={"tenant-id": TENANT, "uploaded-by": "alice", "filename": "brief.txt"},
            content_type="text/plain",
        )

        def explode(tenant: str, event: dict) -> None:
            raise RuntimeError("socket gone")

        job = make_runner(storage, on_event=explode).ingest_raw_key(key)
        assert job.state is JobState.LIVE


class TestConcurrencyCap:
    def test_the_limiter_refuses_beyond_its_limit(self):
        limiter = IngestLimiter(2)
        assert limiter.try_acquire()
        assert limiter.try_acquire()
        assert not limiter.try_acquire()
        limiter.release()
        assert limiter.try_acquire()

    def test_an_ingest_beyond_the_cap_is_refused_not_queued(self, storage, s3):
        """Refusing costs a retry, and the object is still in S3. Queueing would turn a
        bulk upload into memory pressure and a thundering herd at Bedrock."""
        key = raw_key(TENANT, "u1", "brief.txt")
        s3.land(
            key,
            TEXT,
            metadata={"tenant-id": TENANT, "uploaded-by": "alice", "filename": "brief.txt"},
            content_type="text/plain",
        )
        full = IngestLimiter(1)
        assert full.try_acquire()

        runner = make_runner(storage, limiter=full)
        with pytest.raises(IngestBusy):
            runner.ingest_raw_key(key)

    def test_the_slot_is_released_after_a_failure(self, storage, s3):
        """A leaked slot would permanently shrink capacity, one failure at a time."""

        def boom(ctx, parsed, *, matter_id, run_model_extraction, job_id):
            raise RuntimeError("nope")

        key = raw_key(TENANT, "u1", "brief.txt")
        s3.land(
            key,
            TEXT,
            metadata={"tenant-id": TENANT, "uploaded-by": "alice", "filename": "brief.txt"},
            content_type="text/plain",
        )
        limiter = IngestLimiter(1)
        make_runner(storage, limiter=limiter, pipeline=boom).ingest_raw_key(key)
        assert limiter.try_acquire()


class TestTerminalStateReflectsThisDocument:
    """A job's final state must describe the document it ingested, not the tenant.

    `pending_review` in the pipeline result is a tenant-wide count, so keying the terminal
    state off it parked every subsequent document at PENDING_REVIEW as soon as one assertion
    was pending anywhere -- including a document that produced no assertions at all, which
    then read as "awaiting review" while nothing was queued for it.
    """

    def _land(self, s3: FakeS3) -> str:
        key = raw_key(TENANT, "u1", "brief.txt")
        s3.land(
            key,
            TEXT,
            metadata={"tenant-id": TENANT, "uploaded-by": "alice", "filename": "brief.txt"},
            content_type="text/plain",
        )
        return key

    def test_a_document_that_staged_nothing_reaches_live(self, storage, s3):
        """Seven assertions pending elsewhere in the tenant must not hold this one open."""

        def pipeline(ctx, parsed, *, matter_id, run_model_extraction, job_id):
            return {"chunks": 1, "assertions_staged": 0, "assertions_live": 0, "pending_review": 7}

        runner = make_runner(storage, pipeline=pipeline)
        assert runner.ingest_raw_key(self._land(s3)).state is JobState.LIVE

    def test_a_document_awaiting_a_person_stops_at_pending_review(self, storage, s3):
        def pipeline(ctx, parsed, *, matter_id, run_model_extraction, job_id):
            return {"chunks": 1, "assertions_staged": 3, "assertions_live": 1, "pending_review": 0}

        runner = make_runner(storage, pipeline=pipeline)
        assert runner.ingest_raw_key(self._land(s3)).state is JobState.PENDING_REVIEW

    def test_everything_auto_asserted_reaches_live(self, storage, s3):
        """All staged claims promoted immediately, so no reviewer is involved."""

        def pipeline(ctx, parsed, *, matter_id, run_model_extraction, job_id):
            return {"chunks": 1, "assertions_staged": 4, "assertions_live": 4, "pending_review": 2}

        runner = make_runner(storage, pipeline=pipeline)
        assert runner.ingest_raw_key(self._land(s3)).state is JobState.LIVE
