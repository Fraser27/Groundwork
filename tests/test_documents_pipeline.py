"""Tests for ingest and embed.

All AWS clients are injected fakes — the pipeline is testable without credentials by
design, because a pipeline you cannot test offline is a pipeline nobody tests.

Page transcription is covered in `test_vision_parse.py`, and extraction in
`test_model_extraction.py`, which is where the substantive question lives: what an
untrusted model is allowed to put in the graph.
"""

from __future__ import annotations

import io
import json

import pytest

from src.documents.embed import (
    Embedder,
    InMemoryVectorStore,
    index_name,
)
from src.documents.ingest import Ingestor, document_key, guess_media_type
from src.documents.keys import PROCESSED_PREFIX
from src.documents.keys import document_key as storage_document_key
from src.documents.models import (
    Chunk,
    DocumentMeta,
    IllegalTransition,
    IngestJob,
    JobState,
    sha256_hex,
)
from src.graph.scope import AuthContext

TENANT = "firm-acme"


def ctx(**kw) -> AuthContext:
    return AuthContext(user_id="alice", tenant_id=TENANT, **kw)


class FakeS3:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}
        self.metadata: dict[str, dict] = {}

    def put_object(self, **kw):
        self.objects[kw["Key"]] = kw["Body"]
        self.metadata[kw["Key"]] = kw.get("Metadata", {})
        return {"VersionId": "v1"}

    def head_object(self, **kw):
        key = kw["Key"]
        if key not in self.objects:
            raise RuntimeError("404 Not Found")
        return {"Metadata": self.metadata[key], "VersionId": "v1"}

    def get_object(self, **kw):
        return {"Body": io.BytesIO(self.objects[kw["Key"]])}


class TestIngest:
    def test_register_writes_to_s3_and_opens_a_job(self):
        s3 = FakeS3()
        ing = Ingestor("lex-docs", s3=s3)
        doc, job = ing.register(ctx(), filename="motion.pdf", body=b"%PDF-1.7 body")

        assert doc.bucket == "lex-docs"
        assert s3.objects[doc.key] == b"%PDF-1.7 body"
        assert job.state is JobState.REGISTERED
        assert doc.media_type == "application/pdf"

    def test_key_is_content_addressed_and_tenant_prefixed(self):
        digest = sha256_hex(b"x")
        key = document_key(ctx(), digest, "brief.pdf")
        assert key == f"{PROCESSED_PREFIX}firm-acme/{digest}/brief.pdf"

    def test_ingest_and_storage_agree_on_the_key(self):
        """These were separate implementations that had drifted apart. One key, or a
        document is written to one place and read from another."""
        digest = sha256_hex(b"x")
        assert document_key(ctx(), digest, "brief.pdf") == storage_document_key(
            ctx().tenant_id, digest, "brief.pdf"
        )

    def test_identical_bytes_resolve_to_one_document(self):
        """A double-submit must not become two documents with two sets of assertions."""
        ing = Ingestor("lex-docs", s3=FakeS3())
        first, _ = ing.register(ctx(), filename="a.pdf", body=b"same")
        second, _ = ing.register(ctx(), filename="a.pdf", body=b"same")
        assert first.document_id == second.document_id

    def test_resubmitting_still_opens_a_fresh_job(self):
        """Re-ingesting with an improved extractor is a legitimate re-run."""
        ing = Ingestor("lex-docs", s3=FakeS3())
        _, first = ing.register(ctx(), filename="a.pdf", body=b"same")
        _, second = ing.register(ctx(), filename="a.pdf", body=b"same")
        assert first.job_id != second.job_id

    def test_same_bytes_different_tenants_are_different_documents(self):
        ing = Ingestor("lex-docs", s3=FakeS3())
        mine, _ = ing.register(ctx(), filename="a.pdf", body=b"same")
        theirs, _ = ing.register(
            AuthContext(user_id="bob", tenant_id="firm-other"), filename="a.pdf", body=b"same"
        )
        assert mine.document_id != theirs.document_id
        assert mine.key != theirs.key

    def test_fetch_verifies_the_hash(self):
        """Offsets are only meaningful against the exact bytes that were parsed."""
        s3 = FakeS3()
        ing = Ingestor("lex-docs", s3=s3)
        doc, _ = ing.register(ctx(), filename="a.pdf", body=b"original")
        s3.objects[doc.key] = b"tampered"
        with pytest.raises(ValueError, match="hash"):
            ing.fetch(doc)

    def test_fetch_returns_verified_bytes(self):
        ing = Ingestor("lex-docs", s3=FakeS3())
        doc, _ = ing.register(ctx(), filename="a.pdf", body=b"original")
        assert ing.fetch(doc) == b"original"

    def test_matter_scope_enforced_at_registration(self):
        from src.graph.scope import ScopeViolation

        ing = Ingestor("lex-docs", s3=FakeS3())
        walled = ctx(matter_denylist=frozenset({"matter-9"}))
        with pytest.raises(ScopeViolation):
            ing.register(walled, filename="a.pdf", body=b"x", matter_id="matter-9")

    @pytest.mark.parametrize(
        "filename,media",
        [
            ("a.pdf", "application/pdf"),
            ("a.PNG", "image/png"),
            ("a.zzz", "application/octet-stream"),
        ],
    )
    def test_media_type_guessing(self, filename, media):
        assert guess_media_type(filename) == media


class TestJobStateMachine:
    def _doc(self) -> DocumentMeta:
        return DocumentMeta(
            tenant_id=TENANT,
            bucket="b",
            key="k",
            filename="a.pdf",
            media_type="application/pdf",
            content_sha256=sha256_hex(b"x"),
            byte_size=1,
            uploaded_by="alice",
        )

    def test_happy_path(self):
        job = IngestJob.for_document(self._doc())
        for state in (
            JobState.FETCHING,
            JobState.PARSING,
            JobState.CHUNKING,
            JobState.EXTRACTING,
            JobState.EMBEDDING,
            JobState.GRAPH_STAGED,
            JobState.PENDING_REVIEW,
            JobState.APPROVED,
            JobState.LIVE,
        ):
            job.advance(state)
        assert job.state is JobState.LIVE
        assert job.state.is_terminal

    def test_illegal_transition_refused(self):
        job = IngestJob.for_document(self._doc())
        with pytest.raises(IllegalTransition):
            job.advance(JobState.LIVE)

    def test_failure_requires_a_reason(self):
        job = IngestJob.for_document(self._doc())
        with pytest.raises(IllegalTransition, match="requires a reason"):
            job.advance(JobState.FETCH_FAILED)

    def test_failure_is_recoverable_to_the_right_phase(self):
        job = IngestJob.for_document(self._doc())
        job.advance(JobState.FETCHING).advance(JobState.PARSING)
        job.advance(JobState.PARSE_FAILED, reason="transcription throttled on page 12")
        assert job.state.is_failed
        assert job.retry().state is JobState.PARSING

    def test_stage_failure_restarts_at_extraction(self):
        """Extractions are not persisted between attempts; re-deriving is cheap."""
        assert JobState.STAGE_FAILED.retry_target is JobState.EXTRACTING

    def test_live_cannot_be_retried(self):
        job = IngestJob.for_document(self._doc())
        for state in (
            JobState.FETCHING,
            JobState.PARSING,
            JobState.CHUNKING,
            JobState.EXTRACTING,
            JobState.EMBEDDING,
            JobState.GRAPH_STAGED,
            JobState.PENDING_REVIEW,
            JobState.APPROVED,
            JobState.LIVE,
        ):
            job.advance(state)
        with pytest.raises(IllegalTransition, match="not a recoverable failure"):
            job.retry()

    def test_history_is_append_only(self):
        """A job that succeeded on its third attempt should show the first two."""
        job = IngestJob.for_document(self._doc())
        job.advance(JobState.FETCHING)
        job.advance(JobState.FETCH_FAILED, reason="throttled")
        job.retry()
        job.advance(JobState.PARSING)
        assert [h.state for h in job.history] == [
            JobState.REGISTERED,
            JobState.FETCHING,
            JobState.FETCH_FAILED,
            JobState.FETCHING,
            JobState.PARSING,
        ]
        assert job.attempts[JobState.FETCHING.value] == 2

    def test_reason_cleared_on_recovery(self):
        job = IngestJob.for_document(self._doc())
        job.advance(JobState.FETCHING)
        job.advance(JobState.FETCH_FAILED, reason="throttled")
        assert job.reason == "throttled"
        assert job.retry().reason is None


class FakeBedrock:
    def __init__(self, payload) -> None:
        self.payload = payload
        self.requests: list[dict] = []

    def invoke_model(self, **kw):
        self.requests.append({"modelId": kw["modelId"], "body": json.loads(kw["body"])})
        body = (
            self.payload
            if isinstance(self.payload, dict)
            else {"content": [{"text": self.payload}]}
        )
        return {"body": io.BytesIO(json.dumps(body).encode())}


def _chunk(text: str, *, page: int = 1, start: int = 500) -> Chunk:
    return Chunk(
        document_id="doc-1",
        tenant_id=TENANT,
        ordinal=0,
        page=page,
        char_start=start,
        char_end=start + len(text),
        text=text,
    )


class TestEmbedding:
    def _embedder(self, store=None, dims=4):
        payload = {"embedding": [0.1] * dims}
        return Embedder(store or InMemoryVectorStore(), bedrock=FakeBedrock(payload)), payload

    def test_index_is_per_tenant(self):
        assert index_name(ctx()) == "tenant-firm-acme-chunks"

    def test_records_carry_page_and_offsets(self):
        embedder, _ = self._embedder()
        [record] = embedder.embed_chunks(ctx(), [_chunk("some text", page=3, start=900)])
        assert (record.page, record.char_start, record.char_end) == (3, 900, 909)
        assert record.to_metadata()["document_id"] == "doc-1"

    def test_vector_id_is_the_chunk_id_so_reembedding_converges(self):
        store = InMemoryVectorStore()
        embedder, _ = self._embedder(store)
        chunks = [_chunk("some text")]
        embedder.embed_and_store(ctx(), chunks)
        embedder.embed_and_store(ctx(), chunks)
        assert store.count(index_name(ctx())) == 1

    def test_cross_tenant_chunk_is_refused(self):
        embedder, _ = self._embedder()
        alien = _chunk("x").model_copy(update={"tenant_id": "firm-other"})
        with pytest.raises(Exception, match="firm-other"):
            embedder.embed_chunks(ctx(), [alien])

    def test_search_is_scoped_by_matter_denylist(self):
        store = InMemoryVectorStore()
        embedder, _ = self._embedder(store)
        walled = _chunk("walled").model_copy(update={"matter_id": "matter-9"})
        open_ = _chunk("open", start=800).model_copy(update={"matter_id": "matter-1"})
        embedder.embed_and_store(ctx(), [walled, open_])

        hits = embedder.search(ctx(matter_denylist=frozenset({"matter-9"})), "anything")
        assert [h.record.matter_id for h in hits] == ["matter-1"]

    def test_search_is_scoped_by_matter_allowlist(self):
        store = InMemoryVectorStore()
        embedder, _ = self._embedder(store)
        a = _chunk("a").model_copy(update={"matter_id": "matter-1"})
        b = _chunk("b", start=800).model_copy(update={"matter_id": "matter-2"})
        embedder.embed_and_store(ctx(), [a, b])

        hits = embedder.search(ctx(matter_allowlist=frozenset({"matter-1"})), "anything")
        assert [h.record.matter_id for h in hits] == ["matter-1"]

    def test_another_tenants_index_is_empty(self):
        store = InMemoryVectorStore()
        embedder, _ = self._embedder(store)
        embedder.embed_and_store(ctx(), [_chunk("mine")])
        assert embedder.search(AuthContext(user_id="bob", tenant_id="firm-other"), "x") == []

    def test_batching_writes_everything(self):
        store = InMemoryVectorStore()
        embedder, _ = self._embedder(store)
        embedder.batch_size = 2
        chunks = [_chunk(f"text {i}", start=i * 100) for i in range(5)]
        assert embedder.embed_and_store(ctx(), chunks) == 5
        assert store.count(index_name(ctx())) == 5

    def test_delete_document_clears_its_vectors(self):
        store = InMemoryVectorStore()
        embedder, _ = self._embedder(store)
        embedder.embed_and_store(ctx(), [_chunk("a"), _chunk("b", start=800)])
        assert store.delete_document(index_name(ctx()), "doc-1") == 2
        assert store.count(index_name(ctx())) == 0

    def test_cohere_response_shape_supported(self):
        from src.documents.embed import COHERE_V3

        embedder = Embedder(
            InMemoryVectorStore(),
            bedrock=FakeBedrock({"embeddings": [[0.5, 0.5]]}),
            model_id=COHERE_V3,
        )
        assert embedder.embed_text("x") == [0.5, 0.5]

    def test_unknown_response_shape_raises(self):
        from src.documents.embed import EmbeddingFailed

        embedder = Embedder(InMemoryVectorStore(), bedrock=FakeBedrock({"nope": 1}))
        with pytest.raises(EmbeddingFailed, match="no embedding"):
            embedder.embed_text("x")

    def test_search_ranks_by_similarity(self):
        store = InMemoryVectorStore()
        embedder = Embedder(store, bedrock=FakeBedrock({"embedding": [1.0, 0.0]}))
        store.upsert(
            index_name(ctx()),
            embedder.embed_chunks(ctx(), [_chunk("aligned")]),
        )
        near = Embedder(store, bedrock=FakeBedrock({"embedding": [0.0, 1.0]}))
        store.upsert(index_name(ctx()), near.embed_chunks(ctx(), [_chunk("orthogonal", start=800)]))

        hits = embedder.search(ctx(), "query")
        assert hits[0].record.text == "aligned"
        assert hits[0].score > hits[1].score
