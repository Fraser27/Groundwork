"""Matters as records, requiring one on upload, and bulk linking.

The bug behind all of this: a matter was not stored, only derived by grouping assertions on
`matter_id`. Two consequences that look small and are not.

**An empty matter could not exist**, so a matter could not be created before a document was filed
under it. That is backwards -- a team is staffed and an ethical screen raised *before* the first
document arrives, and you cannot screen a lawyer from a matter the system has never heard of.

**A typo became a matter.** `NTL-2026-114` and `NTL-2026-0114` were two matters if two uploads
disagreed, and nothing noticed, because the list was whatever the data happened to contain. A
conflict check split across both returns half its rows and looks clean.

The graph tests run against a live Neo4j and skip without one, because the thing most likely to be
wrong is the Cypher and a hand-written double accepts a syntax error happily.
"""

from __future__ import annotations

import os

import pytest

from src.graph.client import GraphClient
from src.graph.scope import AuthContext, ScopeViolation
from src.graph_audit import LINK_DOCUMENTS, InMemoryGraphAudit
from src.matters import MatterError, MatterStore, link_documents

TENANT = "t-matters-test"
NTL = "NTL-2026-0114"
MBC = "MBC-2024-0431"

GRAPH_URI = os.getenv("TEST_GRAPH_URI", "bolt://127.0.0.1:7687")
GRAPH_USER = os.getenv("TEST_GRAPH_USER", "neo4j")
GRAPH_PASSWORD = os.getenv("TEST_GRAPH_PASSWORD", "groundwork-dev")


def _live_graph() -> GraphClient | None:
    try:
        client = GraphClient(uri=GRAPH_URI, user=GRAPH_USER, password=GRAPH_PASSWORD)
        if client.verify_connectivity():
            return client
    except Exception:
        return None
    return None


@pytest.fixture(scope="module")
def graph() -> GraphClient:
    client = _live_graph()
    if client is None:
        pytest.skip(f"no graph at {GRAPH_URI} (docker compose up -d neo4j)")
    return client


@pytest.fixture
def ctx() -> AuthContext:
    return AuthContext(user_id="partner@firm.example", tenant_id=TENANT)


@pytest.fixture
def store(graph: GraphClient, ctx: AuthContext) -> MatterStore:
    s = MatterStore(graph)
    for m in s.list(ctx):
        s.delete(ctx, m.matter_id)
    yield s
    for m in s.list(ctx):
        s.delete(ctx, m.matter_id)


class TestAMatterExistsBeforeAnyDocument:
    def test_a_created_matter_is_listed_with_no_documents(self, store, ctx):
        """The whole reason for storing them. Grouping assertions could never express this, so a
        matter could not be created before something was filed under it."""
        store.create(ctx, NTL, "Northwind Trading Ltd v Calder Shipping AG")

        listed = store.list(ctx)
        assert [m.matter_id for m in listed] == [NTL]
        assert listed[0].name == "Northwind Trading Ltd v Calder Shipping AG"

    def test_it_records_who_created_it_and_when(self, store, ctx):
        m = store.create(ctx, NTL, "Northwind v Calder")
        assert m.created_by == "partner@firm.example"
        assert m.created_at

    def test_recreating_renames_rather_than_duplicating(self, store, ctx):
        """Two people setting up the same matter should converge on one record, not have one of
        them refused."""
        first = store.create(ctx, NTL, "Northwind")
        again = store.create(ctx, NTL, "Northwind Trading Ltd v Calder Shipping AG")

        assert len(store.list(ctx)) == 1
        assert again.name == "Northwind Trading Ltd v Calder Shipping AG"
        # History is not rewritten by an update.
        assert again.created_at == first.created_at

    def test_another_tenants_matters_are_not_listed(self, store, ctx):
        store.create(ctx, NTL, "Mine")
        other = AuthContext(user_id="them@other.example", tenant_id="t-other-firm")
        store.create(other, "OTHER-1", "Theirs")
        try:
            assert [m.matter_id for m in store.list(ctx)] == [NTL]
        finally:
            store.delete(other, "OTHER-1")


class TestATypoIsNotAMatter:
    def test_exists_distinguishes_a_real_reference_from_a_near_miss(self, store, ctx):
        """The check an upload depends on. Without it `NTL-2026-114` becomes a second matter and
        a conflict check split across both returns half its rows while looking clean."""
        store.create(ctx, NTL, "Northwind v Calder")

        assert store.exists(ctx, NTL) is True
        assert store.exists(ctx, "NTL-2026-114") is False

    def test_a_reference_is_required(self, store, ctx):
        with pytest.raises(MatterError, match="reference"):
            store.create(ctx, "", "A name")

    def test_a_name_is_required(self, store, ctx):
        """A list of bare references is not readable, and the name is the only field a matter has
        beyond its id."""
        with pytest.raises(MatterError, match="name"):
            store.create(ctx, NTL, "")

    def test_whitespace_is_not_a_reference(self, store, ctx):
        with pytest.raises(MatterError):
            store.create(ctx, "   ", "A name")


class TestDeletingTheRecord:
    def test_it_removes_the_name_and_nothing_else(self, store, ctx):
        """Withdrawing the facts is `wipe_matter`, a separate and louder act. Deleting the record
        must not quietly take facts with it."""
        store.create(ctx, NTL, "Northwind v Calder")
        store.delete(ctx, NTL)
        assert store.list(ctx) == []


class TestBulkLinking:
    """Linking is a property update, not a rewrite.

    `matter_id` is deliberately absent from the hash that identifies a fact, so re-filing a
    document keeps every assertion id and therefore every citation.
    """

    def _services(self, graph, queue, audit=None):
        class Services:
            review_queue = queue
            graph_audit = audit if audit is not None else InMemoryGraphAudit()

        s = Services()
        s.graph = graph
        return s

    def _queue_with(self, ctx, *facts, store=None):
        from src.documents.review import ReviewQueue

        queue = ReviewQueue(store=store)
        if facts:
            queue.stage(ctx, list(facts))
        return queue

    def _fact(self, document_id: str, matter: str | None, subject: str = "counsel:us"):
        from src.graph.assertions import EpistemicClass, SourceLocator, build_assertion

        return build_assertion(
            tenant_id=TENANT,
            subject_id=f"{subject}-{document_id}",
            predicate="REPRESENTS",
            object_id="party:calder",
            epistemic_class=EpistemicClass.EXTRACTED_MODEL,
            method="llm:test@v1",
            confidence=0.8,
            matter_id=matter,
            source_locator=SourceLocator(
                document_id=document_id, filename="f.pdf", page=1, quote="a quote"
            ),
        )

    def test_an_unknown_matter_is_refused(self, store, ctx, graph):
        """Refused rather than created, because a link that invents a matter is how a typo becomes
        a permanent second matter."""
        services = self._services(graph, self._queue_with(ctx))

        with pytest.raises(MatterError, match="no matter"):
            link_documents(services, ctx, "NEVER-CREATED", ["doc-1"])

    def test_no_documents_is_refused(self, store, ctx, graph):
        store.create(ctx, NTL, "Northwind")
        services = self._services(graph, self._queue_with(ctx))

        with pytest.raises(MatterError, match="no documents"):
            link_documents(services, ctx, NTL, [])

    def test_several_documents_move_in_one_call(self, store, ctx, graph):
        store.create(ctx, NTL, "Northwind")
        facts = [self._fact("doc-a", None), self._fact("doc-b", None)]
        services = self._services(graph, self._queue_with(ctx, *facts))

        report = link_documents(services, ctx, NTL, ["doc-a", "doc-b"])
        assert set(report.documents) == {"doc-a", "doc-b"}
        assert report.matter_id == NTL

    def test_the_previous_matter_is_recorded(self, store, ctx, graph):
        """Recorded before the move, because afterwards it is unrecoverable from the data -- and
        "where did this document come from" is exactly what somebody asks later."""
        store.create(ctx, NTL, "Northwind")
        store.create(ctx, MBC, "Meridian")
        services = self._services(graph, self._queue_with(ctx, self._fact("doc-a", MBC)))

        report = link_documents(services, ctx, NTL, ["doc-a"])
        assert report.previous_matters["doc-a"] == MBC

    def test_a_link_is_audited(self, store, ctx, graph):
        """It is an access change effected through a data operation: matter access is
        allowlist-primary, so moving a document changes who can read its facts."""
        store.create(ctx, NTL, "Northwind")
        audit = InMemoryGraphAudit()
        services = self._services(graph, self._queue_with(ctx, self._fact("doc-a", None)), audit)

        link_documents(services, ctx, NTL, ["doc-a"], reason="filed under the right matter")

        events = audit.events(TENANT)
        assert len(events) == 1
        assert events[0].action == LINK_DOCUMENTS
        assert events[0].matter_id == NTL
        assert events[0].actor == "partner@firm.example"
        assert events[0].reason == "filed under the right matter"
        assert events[0].detail["documents"] == ["doc-a"]

    def test_an_unrecorded_link_says_so(self, store, ctx, graph):
        store.create(ctx, NTL, "Northwind")

        class NoAudit:
            review_queue = self._queue_with(ctx)
            graph_audit = None

        services = NoAudit()
        services.graph = graph

        report = link_documents(services, ctx, NTL, ["doc-a"])
        assert any("not recorded" in e for e in report.errors)

    def test_the_relinked_ids_come_back_from_the_graph(self, store, ctx, graph):
        """The ids, not a count. `matter_id` is absent from the hash that identifies a fact, so the
        ids survive the move and the audit can say *which* facts changed hands.

        Against a live graph because the returning shape is the Cypher's, and a double would
        accept a query that still returned only `count`."""
        from src.graph.assertion_store import GraphAssertionStore

        store.create(ctx, NTL, "Northwind")
        graph_store = GraphAssertionStore(graph=graph)
        graph_store.drop_tenant(TENANT)
        try:
            facts = [self._fact("doc-a", None), self._fact("doc-a", None, subject="counsel:uk")]
            queue = self._queue_with(ctx, *facts, store=graph_store)
            report = link_documents(self._services(graph, queue), ctx, NTL, ["doc-a"])

            assert set(report.assertion_ids) == {f.assertion_id for f in facts}
            assert report.assertions_relinked == 2
        finally:
            graph_store.drop_tenant(TENANT)

    def test_a_screened_target_matter_is_refused(self, store, ctx, graph):
        """A wall that holds for reads but not for filing is not a wall: moving a document into a
        screened matter would be a way to hide it from its own team."""
        store.create(ctx, NTL, "Northwind")
        screened = AuthContext(
            user_id="associate@firm.example",
            tenant_id=TENANT,
            matter_denylist=frozenset({NTL}),
        )
        services = self._services(graph, self._queue_with(screened))

        with pytest.raises(ScopeViolation):
            link_documents(services, screened, NTL, ["doc-a"])


class TestALinkIsAuditedAsAffectingTheFactsItMoved:
    """The regression that shipped: a link that moved 28 assertions was logged as affecting 0.

    `_audit_link` put a count in `detail` and left `assertion_ids` empty, and `affected` is
    derived from `assertion_ids` -- so the Audit page showed an access change that touched
    nothing. Moving a document between matters changes who can read its facts, which is the only
    reason this operation is audited, so under-reporting it to zero is a misleading audit trail.

    No live graph here, deliberately: this is about what reaches the audit log, and it has to fail
    in CI. The Cypher's returning shape is covered against a real Neo4j in `TestBulkLinking`.
    """

    def _services(self, ids: list[str], audit):
        from src.graph import matter_queries as q

        class FakeGraph:
            def query(self, cypher, params=None):
                if cypher is q.GET_MATTER:
                    return [{"m": {"matter_id": NTL, "name": "Northwind"}}]
                if cypher is q.RELINK_DOCUMENT_ASSERTIONS:
                    return [{"assertion_ids": list(ids)}]
                raise AssertionError(f"unexpected query: {cypher}")

        class Queue:
            def visible(self, ctx):
                return []

        class Services:
            review_queue = Queue()
            graph_audit = audit

        s = Services()
        s.graph = FakeGraph()
        return s

    def test_moving_n_assertions_is_audited_as_affecting_n(self, ctx):
        ids = [f"a-{i}" for i in range(28)]
        audit = InMemoryGraphAudit()

        report = link_documents(self._services(ids, audit), ctx, NTL, ["doc-a"])

        event = audit.events(TENANT)[0]
        assert event.action == LINK_DOCUMENTS
        assert event.affected == 28
        assert report.assertions_relinked == 28

    def test_the_ids_are_recorded_so_the_facts_that_moved_are_named(self, ctx):
        """A count says something moved; the ids answer "did this move the fact I am asking
        about", which is the question a conflict check turns into."""
        audit = InMemoryGraphAudit()

        link_documents(self._services(["a-1", "a-2"], audit), ctx, NTL, ["doc-a"])

        assert audit.events(TENANT)[0].assertion_ids == ("a-1", "a-2")

    def test_a_link_that_moved_nothing_is_still_audited_as_zero(self, ctx):
        """The honest zero, which the bug made indistinguishable from a real move: filing a
        document whose facts have not been extracted yet moves no assertions."""
        audit = InMemoryGraphAudit()

        link_documents(self._services([], audit), ctx, NTL, ["doc-a"])

        assert audit.events(TENANT)[0].affected == 0


class TestTheUploadRequiresARealMatter:
    """An upload with no matter produced facts nobody could attribute, silently."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient

        from src.api.app import create_app
        from src.config import AuthConfig, DocumentConfig, GraphConfig, GroundworkConfig
        from src.documents.storage import DocumentStorage, set_document_storage

        class FakeS3:
            def generate_presigned_post(self, Bucket, Key, **kw):
                return {"url": "https://s3.example/", "fields": {"key": Key}}

            def head_object(self, **kw):
                raise RuntimeError("404")

        cfg = GroundworkConfig(
            environment="local",
            auth=AuthConfig(dev_bypass_tenant="demo-firm"),
            graph=GraphConfig(uri="bolt://127.0.0.1:1", user="none", password="none"),
            documents=DocumentConfig(bucket="b"),
        )
        cfg.validate()
        set_document_storage(DocumentStorage("b", s3=FakeS3()))
        yield TestClient(create_app(cfg))
        set_document_storage(None)

    def test_an_upload_without_a_matter_is_refused(self, client):
        res = client.post("/api/tenants/demo-firm/documents/presign", json={"filename": "a.pdf"})
        assert res.status_code == 422

    def test_an_upload_with_a_matter_is_allowed(self, client):
        """No graph is reachable here, so verification degrades to allowing it. Refusing every
        upload because the graph is down would be worse than accepting one whose matter cannot be
        checked -- and ingest re-checks matter access regardless."""
        res = client.post(
            "/api/tenants/demo-firm/documents/presign",
            json={"filename": "a.pdf", "matter_id": NTL},
        )
        assert res.status_code == 200


class TestListingGlueDatabasesBeforeScanning:
    """Choosing what to govern, rather than reading whatever AWS permits.

    Scanning used to take every database the task role could see. A firm's catalog holds other
    teams' data, so that made the Tables page unusable and the graph misleading: a governed layer
    is supposed to hold what somebody chose to govern, and "everything I have permission to read"
    is not a choice.
    """

    def test_it_lists_names_with_table_counts(self):
        """Names alone cannot answer "which of these do I want", and an empty database is almost
        always the wrong pick."""
        from src.discovery.glue_scanner import list_databases

        class FakePaginator:
            def __init__(self, pages):
                self.pages = pages

            def paginate(self, **kw):
                if "DatabaseName" in kw:
                    return self.pages.get(kw["DatabaseName"], [{"TableList": []}])
                return self.pages["__dbs__"]

        class FakeGlue:
            def get_paginator(self, op):
                return FakePaginator(
                    {
                        "__dbs__": [{"DatabaseList": [{"Name": "aemo"}, {"Name": "other_team"}]}],
                        "aemo": [{"TableList": [{"Name": "t1"}, {"Name": "t2"}]}],
                        "other_team": [{"TableList": []}],
                    }
                )

        out = list_databases(FakeGlue())
        assert [d["name"] for d in out["databases"]] == ["aemo", "other_team"]
        assert out["databases"][0]["table_count"] == 2
        assert out["databases"][1]["table_count"] == 0

    def test_one_unreadable_database_does_not_hide_the_rest(self):
        """A permission gap on one database is a normal state in a shared catalog, so a partial
        list somebody can act on beats an error page."""
        from src.discovery.glue_scanner import list_databases

        class Paginator:
            def __init__(self, name):
                self.name = name

            def paginate(self, **kw):
                if kw.get("DatabaseName") == "forbidden":
                    raise RuntimeError("AccessDeniedException")
                if "DatabaseName" in kw:
                    return [{"TableList": [{"Name": "t1"}]}]
                return [{"DatabaseList": [{"Name": "aemo"}, {"Name": "forbidden"}]}]

        class FakeGlue:
            def get_paginator(self, op):
                return Paginator(op)

        out = list_databases(FakeGlue())
        assert len(out["databases"]) == 2
        forbidden = next(d for d in out["databases"] if d["name"] == "forbidden")
        assert forbidden["table_count"] is None
        assert "AccessDenied" in forbidden["error"]

    def test_a_catalog_that_cannot_be_listed_reports_why(self):
        from src.discovery.glue_scanner import list_databases

        class Broken:
            def get_paginator(self, op):
                class P:
                    def paginate(self, **kw):
                        raise RuntimeError("no such region")

                return P()

        out = list_databases(Broken())
        assert out["databases"] == []
        assert out["errors"]


class TestRelinkingMovesTheEmbeddingsToo:
    """The wall held for the graph and leaked through vector search.

    A chunk with no `matter_id` is deliberately tenant-wide -- `search()` admits it under any
    allowlist, mirroring `edge_scope`'s `matter_id IS NULL OR matter_id IN allowlist`. That is
    correct on its own, and it was `link_documents` that made it a leak: relinking updated the
    graph assertions and left the chunks carrying their old label, so a document moved into a
    matter somebody is screened from stayed retrievable by them. Verified in production, where a
    chunk of the Halveston note sat at `matter_id: None` after its 28 facts had moved.

    The graph half being scoped is what makes this the dangerous shape of the bug: a conflict check
    reads clean while retrieval still returns the text.
    """

    def _services(self, graph_ids, vector_store):
        from src.documents.embed import Embedder
        from src.graph import matter_queries as q

        class FakeGraph:
            def query(self, cypher, params=None):
                if cypher is q.GET_MATTER:
                    return [{"m": {"matter_id": NTL, "name": "Northwind"}}]
                if cypher is q.RELINK_DOCUMENT_ASSERTIONS:
                    return [{"assertion_ids": list(graph_ids)}]
                raise AssertionError(f"unexpected query: {cypher}")

        class Queue:
            def visible(self, ctx):
                return []

        class Services:
            review_queue = Queue()
            graph_audit = InMemoryGraphAudit()

        s = Services()
        s.graph = FakeGraph()
        s.embedder = Embedder(vector_store, model_id="test-model")
        return s

    def _chunk(self, document_id: str, matter_id: str | None, vector_id: str = "c1"):
        from src.documents.embed import VectorRecord

        return VectorRecord(
            vector_id=vector_id,
            tenant_id=TENANT,
            document_id=document_id,
            page=1,
            char_start=0,
            char_end=10,
            text="Calder Shipping AG has appeared as a counterparty",
            embedding=(0.1, 0.2),
            model_id="test-model",
            matter_id=matter_id,
        )

    def test_an_unfiled_chunk_is_filed_by_the_link(self, ctx):
        """The production state exactly: facts moved, chunk left at None."""
        from src.documents.embed import InMemoryVectorStore, index_name

        store = InMemoryVectorStore()
        index = index_name(ctx)
        store.upsert(index, [self._chunk("doc-a", None)])

        report = link_documents(self._services(["a-1"], store), ctx, NTL, ["doc-a"])

        assert report.chunks_refiled == 1
        assert store._indexes[index]["c1"].matter_id == NTL

    def test_a_screened_reader_stops_seeing_the_moved_chunk(self, ctx):
        """The consequence, stated as a read rather than a field. Before the fix this search
        returned the chunk, because a null matter clears every allowlist."""
        from src.documents.embed import InMemoryVectorStore, index_name

        store = InMemoryVectorStore()
        index = index_name(ctx)
        store.upsert(index, [self._chunk("doc-a", None)])

        # Somebody staffed on another matter only. The chunk is unfiled, so they can read it.
        before = store.search(index, [0.1, 0.2], top_k=5, matter_allowlist=frozenset({"OTHER-1"}))
        assert len(before) == 1, "an unfiled chunk is tenant-wide, which is the leak"

        link_documents(self._services(["a-1"], store), ctx, NTL, ["doc-a"])

        after = store.search(index, [0.1, 0.2], top_k=5, matter_allowlist=frozenset({"OTHER-1"}))
        assert after == [], "once filed under NTL it is out of reach of an OTHER-1 allowlist"

    def test_chunks_of_other_documents_are_left_alone(self, ctx):
        from src.documents.embed import InMemoryVectorStore, index_name

        store = InMemoryVectorStore()
        index = index_name(ctx)
        store.upsert(index, [self._chunk("doc-a", None), self._chunk("doc-b", MBC, "c2")])

        link_documents(self._services(["a-1"], store), ctx, NTL, ["doc-a"])

        assert store._indexes[index]["c2"].matter_id == MBC

    def test_a_vector_failure_is_reported_not_raised(self, ctx):
        """The facts have already moved by then. Failing the call would leave the two halves
        inconsistent with no record of why, so it is reported loudly instead."""
        from src.documents.embed import InMemoryVectorStore

        class Broken(InMemoryVectorStore):
            def relabel_matter(self, index, document_id, matter_id):
                raise RuntimeError("collection unreachable")

        report = link_documents(self._services(["a-1"], Broken()), ctx, NTL, ["doc-a"])

        assert report.assertions_relinked == 1
        assert any("vector chunks did not" in e for e in report.errors)


class TestLinkingMovesTheDocumentRecordToo:
    """Three stores hold a document's matter, and the Documents page reads the third.

    `_document_summary` renders `job.matter_id`. Linking updated the graph assertions and the
    vector chunks and not the job, so a user filed a document under a matter, watched the fact
    counts move on the Matters page, and saw the document row still say unassigned. The one store
    nothing wrote was the one on screen.

    Every job for the document, not just the newest: a re-ingest leaves earlier rows behind and the
    page picks among them, so refiling one can leave it reading a stale row.
    """

    def _services(self, jobs, graph_ids=("a-1",)):
        from src.graph import matter_queries as q

        class FakeGraph:
            def query(self, cypher, params=None):
                if cypher is q.GET_MATTER:
                    return [{"m": {"matter_id": NTL, "name": "Northwind"}}]
                if cypher is q.RELINK_DOCUMENT_ASSERTIONS:
                    return [{"assertion_ids": list(graph_ids)}]
                raise AssertionError(f"unexpected query: {cypher}")

        class Jobs:
            def __init__(self, rows):
                self.rows = rows
                self.written: list = []

            def jobs_for_document(self, tenant_id, document_id):
                return [j for j in self.rows if j.document_id == document_id]

            def put_job(self, job):
                self.written.append(job)

        class Queue:
            def visible(self, ctx):
                return []

        store = Jobs(jobs)

        class Services:
            review_queue = Queue()
            graph_audit = InMemoryGraphAudit()
            job_store = store

        s = Services()
        s.graph = FakeGraph()
        return s, store

    def _job(self, document_id="doc-a", matter_id=None):
        from src.documents.models import DocumentMeta, IngestJob

        doc = DocumentMeta(
            document_id=document_id,
            tenant_id=TENANT,
            bucket="b",
            key=f"processed/{document_id}",
            filename="note.pdf",
            media_type="application/pdf",
            content_sha256="a" * 64,
            byte_size=10,
            uploaded_by="u",
            matter_id=matter_id,
        )
        return IngestJob.for_document(doc)

    def test_the_job_is_refiled(self, store, ctx):
        store.create(ctx, NTL, "Northwind")
        services, jobs = self._services([self._job()])

        report = link_documents(services, ctx, NTL, ["doc-a"])

        assert report.jobs_refiled == 1
        assert jobs.written[0].matter_id == NTL

    def test_every_job_for_the_document_is_refiled(self, store, ctx):
        """A re-ingest leaves older rows, and the page picks among them."""
        store.create(ctx, NTL, "Northwind")
        services, _ = self._services([self._job(), self._job()])

        report = link_documents(services, ctx, NTL, ["doc-a"])

        assert report.jobs_refiled == 2

    def test_a_job_already_on_the_matter_is_left_alone(self, store, ctx):
        """Idempotent, so a re-link is not a write per job per attempt."""
        store.create(ctx, NTL, "Northwind")
        services, jobs = self._services([self._job(matter_id=NTL)])

        report = link_documents(services, ctx, NTL, ["doc-a"])

        assert report.jobs_refiled == 0
        assert jobs.written == []

    def test_a_job_store_failure_is_reported_not_raised(self, store, ctx):
        """The facts have already moved by then. Failing here would leave three stores
        disagreeing with no record of why."""
        store.create(ctx, NTL, "Northwind")

        class Broken:
            def jobs_for_document(self, tenant_id, document_id):
                raise RuntimeError("dynamo unreachable")

        services, _ = self._services([])
        services.job_store = Broken()

        report = link_documents(services, ctx, NTL, ["doc-a"])

        assert report.assertions_relinked == 1
        assert any("document record did not" in e for e in report.errors)
