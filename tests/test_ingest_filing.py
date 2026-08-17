"""Which matter a re-ingest files a document under.

The production failure this pins: a document filed under a matter by `link_documents`, then
re-uploaded twice with no matter named, went back to reading unassigned. Relinking is
point-in-time and nothing re-applied it, so every re-ingest silently un-filed the document —
and matter access is allowlist-primary, so an unfiled fact is readable by everyone in the
tenant, including people screened from the matter it belongs to.

What is asserted is therefore not "the field is populated" but the two consequences: the
facts land under the right matter, and the pipeline runs inside a context scoped to it.
"""

from __future__ import annotations

import io
from typing import Any

import pytest

from src.documents.filing import Filing, resolve_filing
from src.documents.job_store import InMemoryJobStore
from src.documents.keys import raw_key
from src.documents.models import DocumentMeta, IngestJob, document_id_for, sha256_hex
from src.documents.parse import parse_plain_text
from src.documents.runner import IngestRunner
from src.documents.storage import DocumentStorage
from src.graph_audit import LINK_DOCUMENTS, InMemoryGraphAudit

TENANT = "firm-acme"
BUCKET = "lexgraph-docs"
TEXT = b"Held for the plaintiff. Costs reserved."
HALVESTON = "HAL-2026-0001"
OTHER = "NTL-2026-0114"
PLAIN_TEXT_TYPES = frozenset({"text/plain", "text/markdown", "text/csv"})


class FakeS3:
    def __init__(self) -> None:
        self.objects: dict[str, dict[str, Any]] = {}
        self.puts: list[dict[str, Any]] = []

    def put_object(self, **kw: Any) -> dict[str, Any]:
        self.puts.append(kw)
        self.objects[kw["Key"]] = kw
        return {"VersionId": f"v{len(self.puts)}"}

    def head_object(self, **kw: Any) -> dict[str, Any]:
        if kw["Key"] not in self.objects:
            raise RuntimeError("404 Not Found")
        return {"VersionId": "v1"}

    def get_object(self, **kw: Any) -> dict[str, Any]:
        obj = self.objects[kw["Key"]]
        return {
            "Body": io.BytesIO(obj["Body"]),
            "ContentType": obj.get("ContentType", "text/plain"),
            "Metadata": obj.get("Metadata", {}),
        }

    def delete_object(self, **kw: Any) -> dict[str, Any]:
        self.objects.pop(kw["Key"], None)
        return {}

    def land(self, key: str, body: bytes, *, matter_id: str | None) -> str:
        meta = {"tenant-id": TENANT, "uploaded-by": "alice", "filename": "brief.txt"}
        if matter_id:
            meta["matter-id"] = matter_id
        self.objects[key] = {
            "Key": key,
            "Body": body,
            "Metadata": meta,
            "ContentType": "text/plain",
        }
        return key


@pytest.fixture
def s3() -> FakeS3:
    return FakeS3()


@pytest.fixture
def storage(s3: FakeS3) -> DocumentStorage:
    return DocumentStorage(BUCKET, s3=s3)


@pytest.fixture
def store() -> InMemoryJobStore:
    return InMemoryJobStore()


def make_runner(storage: DocumentStorage, store: InMemoryJobStore, **kw: Any) -> IngestRunner:
    """A runner that records the context and matter every pipeline run saw."""
    seen: list[dict[str, Any]] = []

    def pipeline(ctx, parsed, *, matter_id, run_model_extraction, job_id):
        seen.append({"ctx": ctx, "matter_id": matter_id})
        return {"chunks": 1, "assertions_staged": 0, "assertions_live": 0}

    runner = IngestRunner(
        storage,
        pipeline=kw.pop("pipeline", pipeline),
        store=store,
        max_upload_bytes=4096,
        plain_text_types=PLAIN_TEXT_TYPES,
        parse_plain_text=parse_plain_text,
        **kw,
    )
    runner.runs = seen  # type: ignore[attr-defined]
    return runner


def upload(s3: FakeS3, runner: IngestRunner, *, matter_id: str | None, upload_id: str) -> IngestJob:
    """One full trip through the notification path, as a distinct upload of the same bytes."""
    return runner.ingest_raw_key(
        s3.land(raw_key(TENANT, upload_id, "brief.txt"), TEXT, matter_id=matter_id)
    )


def refile(store: InMemoryJobStore, document_id: str, matter_id: str) -> None:
    """Stand in for `link_documents`, which is the only thing that files a document after
    the fact. It moves `job.matter_id`; nothing re-applies it on a later ingest."""
    for job in store.jobs_for_document(TENANT, document_id):
        job.matter_id = matter_id
        store.put_job(job)


class TestAdoption:
    def test_a_re_ingest_naming_no_matter_keeps_the_prior_one(self, storage, s3, store):
        """The production bug, end to end: filed under Halveston, then re-uploaded twice with
        nothing named. Before the fix the second and third uploads filed under None."""
        runner = make_runner(storage, store)
        first = upload(s3, runner, matter_id=HALVESTON, upload_id="u1")
        assert first.matter_id == HALVESTON

        second = upload(s3, runner, matter_id=None, upload_id="u2")
        third = upload(s3, runner, matter_id=None, upload_id="u3")

        assert second.matter_id == HALVESTON
        assert third.matter_id == HALVESTON

    def test_a_matter_established_only_by_a_link_is_adopted(self, storage, s3, store):
        """The exact production sequence: the first upload named nothing, a bulk link filed it,
        and the re-ingest has to honour the link rather than the original upload."""
        runner = make_runner(storage, store)
        first = upload(s3, runner, matter_id=None, upload_id="u1")
        assert first.matter_id is None

        refile(store, first.document_id, HALVESTON)
        again = upload(s3, runner, matter_id=None, upload_id="u2")

        assert again.matter_id == HALVESTON

    def test_the_adopted_matter_reaches_the_facts(self, storage, s3, store):
        """`job.matter_id` is what the Documents page renders; `matter_id` on the pipeline call
        is what the chunks and assertions carry. A fix that moved only the first would leave
        the page right and the facts wrong, which is the worse half."""
        runner = make_runner(storage, store)
        upload(s3, runner, matter_id=HALVESTON, upload_id="u1")
        upload(s3, runner, matter_id=None, upload_id="u2")

        assert runner.runs[1]["matter_id"] == HALVESTON

    def test_the_pipeline_runs_inside_the_adopted_matter(self, storage, s3, store):
        """The synthetic context is scoped to one matter, and the facts are staged through it.
        Scoped to the upload's (absent) matter while filing under the adopted one would have the
        pipeline operating outside the wall it thinks it is inside."""
        runner = make_runner(storage, store)
        upload(s3, runner, matter_id=HALVESTON, upload_id="u1")
        upload(s3, runner, matter_id=None, upload_id="u2")

        ctx = runner.runs[1]["ctx"]
        assert ctx.matter_allowlist == frozenset({HALVESTON})
        assert ctx.can_read_matter(HALVESTON)
        assert not ctx.can_read_matter(OTHER)

    def test_a_first_ingest_with_nothing_to_adopt_still_works(self, storage, s3, store):
        """No prior job and no matter named. Unfiled is wrong, but refusing the upload would
        lose the document, and the bytes are the record."""
        runner = make_runner(storage, store)
        job = upload(s3, runner, matter_id=None, upload_id="u1")

        assert job.matter_id is None
        assert runner.runs[0]["matter_id"] is None
        assert runner.runs[0]["ctx"].matter_allowlist == frozenset()

    def test_another_document_is_not_adopted_from(self, storage, s3, store):
        """Adoption is per document. A matter established on one document must not leak onto
        the next unfiled upload, which would file facts under a matter nobody chose."""
        runner = make_runner(storage, store)
        upload(s3, runner, matter_id=HALVESTON, upload_id="u1")

        other = runner.ingest_raw_key(
            s3.land(
                raw_key(TENANT, "u2", "other.txt"), b"Different bytes entirely.", matter_id=None
            )
        )
        assert other.matter_id is None


class TestExplicitWins:
    def test_a_named_matter_overrides_the_stored_one(self, storage, s3, store):
        """A user naming a matter during upload is expressing intent, and it is the only way to
        re-file through the surface that files documents at all."""
        runner = make_runner(storage, store)
        upload(s3, runner, matter_id=HALVESTON, upload_id="u1")
        moved = upload(s3, runner, matter_id=OTHER, upload_id="u2")

        assert moved.matter_id == OTHER
        assert runner.runs[1]["matter_id"] == OTHER
        assert runner.runs[1]["ctx"].matter_allowlist == frozenset({OTHER})

    def test_an_override_is_audited(self, storage, s3, store):
        """It is an access change effected through an upload: allowlist-primary means the old
        matter's team loses the facts and the new matter's team gains them. Unrecorded, it is
        indistinguishable from the facts having always been there."""
        audit = InMemoryGraphAudit()
        runner = make_runner(storage, store, graph_audit=audit)
        upload(s3, runner, matter_id=HALVESTON, upload_id="u1")
        upload(s3, runner, matter_id=OTHER, upload_id="u2")

        events = audit.events(TENANT)
        assert len(events) == 1
        assert events[0].action == LINK_DOCUMENTS
        assert events[0].matter_id == OTHER
        # The previous matter, because afterwards it is unrecoverable from the data.
        assert events[0].detail["previous_matters"] == {events[0].document_id: HALVESTON}

    def test_adoption_is_not_audited(self, storage, s3, store):
        """Adoption restores the filing already in force. Recording it as a change would fill
        the audit log with events where nothing moved, which is how a real one gets missed."""
        audit = InMemoryGraphAudit()
        runner = make_runner(storage, store, graph_audit=audit)
        upload(s3, runner, matter_id=HALVESTON, upload_id="u1")
        upload(s3, runner, matter_id=None, upload_id="u2")

        assert audit.events(TENANT) == []

    def test_re_naming_the_same_matter_is_not_an_access_change(self, storage, s3, store):
        audit = InMemoryGraphAudit()
        runner = make_runner(storage, store, graph_audit=audit)
        upload(s3, runner, matter_id=HALVESTON, upload_id="u1")
        upload(s3, runner, matter_id=HALVESTON, upload_id="u2")

        assert audit.events(TENANT) == []

    def test_a_failed_audit_does_not_lose_the_document(self, storage, s3, store):
        """The bytes are the record. An audit store that is down must not turn a good upload
        into a lost one — it is logged instead, loudly."""

        class Broken(InMemoryGraphAudit):
            def append(self, event):
                raise RuntimeError("dynamo unreachable")

        runner = make_runner(storage, store, graph_audit=Broken())
        upload(s3, runner, matter_id=HALVESTON, upload_id="u1")
        moved = upload(s3, runner, matter_id=OTHER, upload_id="u2")

        assert moved.matter_id == OTHER


class TestSynchronousRoutes:
    """The two request-path entry points, which resolve through the same module.

    Both *require* a matter, so adoption cannot fire from the request itself — but a document
    can already be filed by an earlier async ingest or a link, and re-uploading it here naming
    a different matter moves it. That is an access change whichever route it arrives through, so
    it is audited on all three rather than only on the notification path.
    """

    TENANT_ID = "dev-tenant"

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient

        from src.api.app import create_app
        from src.api.deps import get_services
        from src.config import AuthConfig, DocumentConfig, GraphConfig, LexGraphConfig
        from src.documents.storage import set_document_storage

        cfg = LexGraphConfig(
            environment="local",
            auth=AuthConfig(dev_bypass_tenant=self.TENANT_ID),
            graph=GraphConfig(uri="bolt://127.0.0.1:1", user="none", password="none"),
            documents=DocumentConfig(bucket=BUCKET),
        )
        cfg.validate()
        set_document_storage(DocumentStorage(BUCKET, s3=FakeS3()))
        with TestClient(create_app(cfg)) as client:
            get_services().graph_audit = InMemoryGraphAudit()
            yield client
        set_document_storage(None)

    def _services(self):
        from src.api.deps import get_services

        return get_services()

    def _already_filed(self, document_id: str, matter_id: str) -> None:
        """A prior async ingest of these bytes, filed under `matter_id`."""
        doc = DocumentMeta(
            document_id=document_id,
            tenant_id=self.TENANT_ID,
            bucket=BUCKET,
            key="processed/x",
            filename="brief.txt",
            media_type="text/plain",
            content_sha256=sha256_hex(TEXT),
            byte_size=len(TEXT),
            uploaded_by="alice",
            matter_id=matter_id,
        )
        self._services().job_store.put_job(IngestJob.for_document(doc))

    def test_the_upload_route_moves_and_audits(self, client):
        self._already_filed(document_id_for(self.TENANT_ID, sha256_hex(TEXT)), HALVESTON)

        res = client.post(
            f"/api/tenants/{self.TENANT_ID}/documents",
            files={"file": ("brief.txt", TEXT, "text/plain")},
            data={"run_model_extraction": "false", "matter_id": OTHER},
        )
        assert res.status_code == 201, res.text
        assert res.json()["matter_id"] == OTHER

        events = self._services().graph_audit.events(self.TENANT_ID)
        assert [e.action for e in events] == [LINK_DOCUMENTS]
        assert events[0].detail["previous_matters"] == {events[0].document_id: HALVESTON}

    def test_the_upload_route_does_not_audit_an_unchanged_filing(self, client):
        """A re-upload to the same matter moves nothing. Recording it would bury the events
        where something did move."""
        self._already_filed(document_id_for(self.TENANT_ID, sha256_hex(TEXT)), HALVESTON)

        client.post(
            f"/api/tenants/{self.TENANT_ID}/documents",
            files={"file": ("brief.txt", TEXT, "text/plain")},
            data={"run_model_extraction": "false", "matter_id": HALVESTON},
        )
        assert self._services().graph_audit.events(self.TENANT_ID) == []

    def test_the_text_route_moves_and_audits(self, client):
        import hashlib

        text = "Held for the plaintiff. Costs reserved."
        # The text route's own id scheme, which differs from the byte-hash one.
        self._already_filed(hashlib.sha256(text.encode()).hexdigest()[:16], HALVESTON)

        res = client.post(
            f"/api/tenants/{self.TENANT_ID}/documents/text",
            json={
                "filename": "brief.txt",
                "text": text,
                "matter_id": OTHER,
                "run_model_extraction": False,
            },
        )
        assert res.status_code == 200, res.text
        assert res.json()["matter_id"] == OTHER
        assert [e.action for e in self._services().graph_audit.events(self.TENANT_ID)] == [
            LINK_DOCUMENTS
        ]


class TestResolveFiling:
    """The rule itself, away from the pipeline."""

    def _job(self, matter_id: str | None, *, created_at: str) -> IngestJob:
        doc = DocumentMeta(
            tenant_id=TENANT,
            bucket=BUCKET,
            key="processed/x",
            filename="brief.txt",
            media_type="text/plain",
            content_sha256=sha256_hex(TEXT),
            byte_size=len(TEXT),
            uploaded_by="alice",
            matter_id=matter_id,
        )
        job = IngestJob.for_document(doc)
        job.created_at = created_at
        return job

    def _store(self, *jobs: IngestJob) -> InMemoryJobStore:
        store = InMemoryJobStore()
        for job in jobs:
            store.put_job(job)
        return store

    def test_the_newest_job_that_names_a_matter_wins(self):
        """Not simply the newest. Nothing files a document to no matter deliberately, so a null
        on a later job is the bug being undone rather than a decision to respect — which is also
        what recovers the documents already un-filed in production."""
        store = self._store(
            self._job(HALVESTON, created_at="2026-08-16T11:21:00+00:00"),
            self._job(None, created_at="2026-08-16T14:31:00+00:00"),
            self._job(None, created_at="2026-08-17T07:10:00+00:00"),
        )
        document_id = document_id_for(TENANT, sha256_hex(TEXT))

        filing = resolve_filing(store, TENANT, document_id, None)

        assert filing.matter_id == HALVESTON
        assert filing.adopted

    def test_the_most_recent_filing_wins_over_an_older_one(self):
        store = self._store(
            self._job(HALVESTON, created_at="2026-08-16T11:21:00+00:00"),
            self._job(OTHER, created_at="2026-08-16T13:16:00+00:00"),
        )
        document_id = document_id_for(TENANT, sha256_hex(TEXT))

        assert resolve_filing(store, TENANT, document_id, None).matter_id == OTHER

    def test_an_unreachable_job_store_degrades_rather_than_failing(self):
        """A document is not worth losing over a store that cannot be read. This degrades to the
        old behaviour, which is wrong but recoverable by a link."""

        class Broken:
            def jobs_for_document(self, tenant_id, document_id):
                raise RuntimeError("dynamo unreachable")

        filing = resolve_filing(Broken(), TENANT, "doc-x", None)

        assert filing.matter_id is None
        assert not filing.adopted

    def test_an_empty_string_is_not_a_matter(self):
        """Form-encoded uploads can send one, and treating it as named would defeat adoption
        while looking like an explicit choice."""
        store = self._store(self._job(HALVESTON, created_at="2026-08-16T11:21:00+00:00"))
        document_id = document_id_for(TENANT, sha256_hex(TEXT))

        assert resolve_filing(store, TENANT, document_id, "").matter_id == HALVESTON

    def test_only_an_override_reads_as_an_access_change(self):
        assert Filing(matter_id=OTHER, requested=OTHER, previous=HALVESTON).overrode
        assert not Filing(matter_id=HALVESTON, requested=None, previous=HALVESTON).overrode
        assert not Filing(matter_id=OTHER, requested=OTHER, previous=None).overrode
        assert not Filing(matter_id=OTHER, requested=OTHER, previous=OTHER).overrode
