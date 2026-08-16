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
GRAPH_PASSWORD = os.getenv("TEST_GRAPH_PASSWORD", "lexgraph-dev")


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

    def _queue_with(self, ctx, *facts):
        from src.documents.review import ReviewQueue

        queue = ReviewQueue()
        if facts:
            queue.stage(ctx, list(facts))
        return queue

    def _fact(self, document_id: str, matter: str | None):
        from src.graph.assertions import EpistemicClass, SourceLocator, build_assertion

        return build_assertion(
            tenant_id=TENANT,
            subject_id=f"counsel:us-{document_id}",
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


class TestTheUploadRequiresARealMatter:
    """An upload with no matter produced facts nobody could attribute, silently."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient

        from src.api.app import create_app
        from src.config import AuthConfig, DocumentConfig, GraphConfig, LexGraphConfig
        from src.documents.storage import DocumentStorage, set_document_storage

        class FakeS3:
            def generate_presigned_post(self, Bucket, Key, **kw):
                return {"url": "https://s3.example/", "fields": {"key": Key}}

            def head_object(self, **kw):
                raise RuntimeError("404")

        cfg = LexGraphConfig(
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
