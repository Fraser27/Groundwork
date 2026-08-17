"""Assertions persisted to a real graph.

Run against a live Neo4j rather than a fake, because the thing most likely to be wrong here is
the Cypher itself, and a hand-written double would accept a syntax error happily. Every
statement in `assertion_queries` is either parameterised or (for the relationship type)
interpolated, so only a real parser proves it.

Skipped when no graph is reachable, so the suite stays green without one:

    docker compose up -d neo4j
    .venv/bin/python -m pytest tests/test_graph_assertion_store.py -q

Two properties matter more than the round-trip. **A review decision must land on the edge as
well as the node**, because `edge_scope` filters traversals on the edge copy — an approval that
updated only the node would leave the fact approved and invisible to every query. And **the
PREMISE edge must be traversable backwards**, because that is what the retraction cascade walks
to withdraw conclusions whose reasons have gone.
"""

from __future__ import annotations

import os

import pytest

from src.documents.review import AssertionRecord, Lifecycle
from src.graph.assertion_queries import UnsafeRelationshipType, safe_type
from src.graph.assertion_store import GraphAssertionStore
from src.graph.assertions import (
    EpistemicClass,
    ReviewState,
    SourceLocator,
    answerable_confidence,
    build_assertion,
)
from src.graph.client import GraphClient
from src.graph.scope import AuthContext

TENANT = "t-assertion-store"
OTHER = "t-other-firm"

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
def store(graph: GraphClient) -> GraphAssertionStore:
    s = GraphAssertionStore(graph=graph)
    s.drop_tenant(TENANT)
    s.drop_tenant(OTHER)
    yield s
    s.drop_tenant(TENANT)
    s.drop_tenant(OTHER)


def _assertion(
    subject: str = "counsel:us",
    predicate: str = "REPRESENTS",
    obj: str = "party:acme",
    *,
    tenant: str = TENANT,
    epistemic_class: EpistemicClass = EpistemicClass.EXTRACTED_MODEL,
    premises: tuple[str, ...] = (),
    premise_confidences: tuple[float, ...] = (),
    confidence: float = 0.7,
):
    inferred = epistemic_class is EpistemicClass.INFERRED
    return build_assertion(
        tenant_id=tenant,
        subject_id=subject,
        predicate=predicate,
        object_id=obj,
        epistemic_class=epistemic_class,
        method="rule:conflict_check@v1" if inferred else "llm:test@v1",
        confidence=confidence,
        matter_id="M-1",
        source_locator=SourceLocator(
            document_id="doc-1", filename="engagement.pdf", page=2, quote="the Adverse Party"
        ),
        premises=premises,
        premise_confidences=premise_confidences,
        rule_id="conflict_check" if inferred else None,
        rule_version="v1" if inferred else None,
    )


class TestRoundTrip:
    def test_an_assertion_survives_a_write_and_read(self, store):
        a = _assertion()
        store.put(AssertionRecord(assertion=a))

        got = store.get(TENANT, a.assertion_id)
        assert got is not None
        assert got.assertion.predicate == "REPRESENTS"
        assert got.assertion.epistemic_class is EpistemicClass.EXTRACTED_MODEL
        assert got.assertion.confidence == pytest.approx(0.7)

    def test_the_citation_survives(self, store):
        """The page and the quote are the product. A store that loses them keeps facts
        nobody can check."""
        a = _assertion()
        store.put(AssertionRecord(assertion=a))

        loc = store.get(TENANT, a.assertion_id).assertion.source_locator
        assert loc.filename == "engagement.pdf"
        assert loc.page == 2
        assert loc.quote == "the Adverse Party"

    def test_rewriting_the_same_assertion_does_not_duplicate(self, store):
        """Ids are content-addressed, so re-ingesting a document must converge."""
        a = _assertion()
        store.put(AssertionRecord(assertion=a))
        store.put(AssertionRecord(assertion=a))
        assert len(store.all_for_tenant(TENANT)) == 1


class TestTenantIsolation:
    def test_another_tenants_assertions_are_not_returned(self, store):
        store.put(AssertionRecord(assertion=_assertion()))
        store.put(AssertionRecord(assertion=_assertion(tenant=OTHER)))

        mine = store.all_for_tenant(TENANT)
        assert len(mine) == 1
        assert all(r.assertion.tenant_id == TENANT for r in mine)

    def test_another_tenants_assertion_is_not_fetchable_by_id(self, store):
        theirs = _assertion(tenant=OTHER)
        store.put(AssertionRecord(assertion=theirs))
        assert store.get(TENANT, theirs.assertion_id) is None


class TestPremisesAreTraversable:
    def test_an_inference_records_what_it_rests_on(self, store):
        p1 = _assertion("counsel:us", "REPRESENTS", "party:acme")
        p2 = _assertion("matter:M-1", "ADVERSE_TO", "party:acme")
        store.put(AssertionRecord(assertion=p1))
        store.put(AssertionRecord(assertion=p2))

        inferred = _assertion(
            "matter:M-1",
            "MENTIONS",
            "party:acme",
            epistemic_class=EpistemicClass.INFERRED,
            premises=(p1.assertion_id, p2.assertion_id),
            premise_confidences=(0.7, 0.7),
        )
        store.put(AssertionRecord(assertion=inferred))

        got = store.get(TENANT, inferred.assertion_id)
        assert set(got.assertion.premises) == {p1.assertion_id, p2.assertion_id}
        assert got.assertion.rule_id == "conflict_check"

    def test_dependents_walks_the_premise_edge_backwards(self, store):
        """What the retraction cascade needs: withdraw a premise and find every conclusion
        resting on it, or a conclusion outlives its reason."""
        premise = _assertion("counsel:us", "REPRESENTS", "party:acme")
        store.put(AssertionRecord(assertion=premise))

        inferred = _assertion(
            "matter:M-1",
            "MENTIONS",
            "party:acme",
            epistemic_class=EpistemicClass.INFERRED,
            premises=(premise.assertion_id,),
            premise_confidences=(0.7,),
        )
        store.put(AssertionRecord(assertion=inferred))

        dependents = store.dependents_of(TENANT, premise.assertion_id)
        assert [d.assertion_id for d in dependents] == [inferred.assertion_id]

    def test_a_fact_nothing_rests_on_has_no_dependents(self, store):
        a = _assertion()
        store.put(AssertionRecord(assertion=a))
        assert store.dependents_of(TENANT, a.assertion_id) == []


class TestReviewDecisionsReachBothCopies:
    """The property that makes an approval mean anything.

    Trust fields live on the edge *and* the node: the edge copy is what `edge_scope` filters a
    traversal on, the node copy is what the audit read returns. Updating one and not the other
    produces a fact that is approved and unreachable, which is worse than one that is visibly
    pending.
    """

    def test_an_approval_updates_the_edge(self, store, graph):
        a = _assertion()
        record = AssertionRecord(assertion=a)
        store.put(record)

        a.review_state = ReviewState.APPROVED
        a.reviewed_by = "reviewer@firm.example"
        a.reviewed_at = "2026-01-01T00:00:00Z"
        record.lifecycle = Lifecycle.LIVE
        store.set_review_state(record)

        rows = graph.query(
            "MATCH ()-[r {tenant_id: $t, assertion_id: $a}]->() RETURN r.review_state AS state",
            {"t": TENANT, "a": a.assertion_id},
        )
        assert rows[0]["state"] == "APPROVED"

    def test_an_approval_updates_the_node(self, store):
        a = _assertion()
        record = AssertionRecord(assertion=a)
        store.put(record)

        a.review_state = ReviewState.APPROVED
        record.lifecycle = Lifecycle.LIVE
        store.set_review_state(record)

        got = store.get(TENANT, a.assertion_id)
        assert got.assertion.review_state is ReviewState.APPROVED
        assert got.lifecycle is Lifecycle.LIVE

    def test_the_rescaled_confidence_reaches_the_edge(self, store, graph):
        """`ReviewQueue.approve` rescales confidence and writes through `put`, and it is the
        *edge* copy `edge_scope` filters on. A rescale that landed only on the node would
        leave an approved fact above the floor in the audit read and below it in every
        traversal -- the same class of bug as an approval that skipped the edge."""
        a = _assertion(confidence=0.55)
        store.put(AssertionRecord(assertion=a))

        a.raw_confidence = a.confidence
        a.confidence = answerable_confidence(a.confidence)
        a.review_state = ReviewState.APPROVED
        store.put(AssertionRecord(assertion=a, lifecycle=Lifecycle.LIVE))

        rows = graph.query(
            "MATCH ()-[r {tenant_id: $t, assertion_id: $a}]->() RETURN r.confidence AS c",
            {"t": TENANT, "a": a.assertion_id},
        )
        assert rows[0]["c"] == pytest.approx(0.91)

    def test_the_raw_score_survives_the_round_trip(self, store):
        """Provenance is the product, so the number the model actually claimed has to be
        readable back out -- otherwise the bump is a confidence nobody can explain."""
        a = _assertion(confidence=0.79)
        a.raw_confidence = 0.93
        store.put(AssertionRecord(assertion=a))
        assert store.get(TENANT, a.assertion_id).assertion.raw_confidence == pytest.approx(0.93)

    def test_a_fact_with_no_self_report_reads_back_as_none(self, store):
        """A catalog scan asserts its confidence rather than estimating it, so None must not
        come back as 0.0 -- that would read as "the model was certain it was wrong"."""
        a = _assertion()
        store.put(AssertionRecord(assertion=a))
        assert store.get(TENANT, a.assertion_id).assertion.raw_confidence is None


class TestSupersedeRatherThanDelete:
    def test_a_superseded_assertion_leaves_the_default_read(self, store):
        """Facts are closed, never edited away. It stops being current, and an `as_of` read
        can still reconstruct what the file showed at the time."""
        a = _assertion()
        store.put(AssertionRecord(assertion=a))
        store.supersede(TENANT, a.assertion_id, at="2026-01-01T00:00:00Z")

        assert store.get(TENANT, a.assertion_id) is None
        assert store.all_for_tenant(TENANT) == []


class TestDropTenant:
    def test_a_reset_removes_this_tenant_only(self, store):
        store.put(AssertionRecord(assertion=_assertion()))
        store.put(AssertionRecord(assertion=_assertion(tenant=OTHER)))

        dropped = store.drop_tenant(TENANT)
        assert dropped == 1
        assert store.all_for_tenant(TENANT) == []
        assert len(store.all_for_tenant(OTHER)) == 1


class TestRelationshipTypeSafety:
    """A relationship type cannot be parameterised in Cypher, so it is interpolated.

    `build_assertion` validates against the pack's closed vocabulary before anything reaches
    the store, which makes this unreachable in practice. It is checked anyway because
    "unreachable" and "unchecked" must not become the same thing on the one path that builds a
    query string.
    """

    @pytest.mark.parametrize(
        "bad",
        [
            "REPRESENTS]->() DETACH DELETE (n",
            "HAS-DASH",
            "1STARTS_WITH_DIGIT",
            "",
            "with space",
        ],
    )
    def test_a_type_that_is_not_an_identifier_is_refused(self, bad):
        with pytest.raises(UnsafeRelationshipType):
            safe_type(bad)

    def test_a_normal_predicate_passes(self):
        assert safe_type("REPRESENTS") == "REPRESENTS"


class TestRecordFieldsSurviveTheRoundTrip:
    """`job_id` is load-bearing, not informational.

    `promote(job_id=...)` filters on it, so a store that dropped it made every scoped promotion a
    silent no-op. Seven auto-asserted facts sat STAGED in production because of this: nothing else
    promotes, and nothing approves what needs no approval, so they were unreachable forever.

    The rest are the record's own fields rather than the assertion's -- the review note, the
    retraction reason and who withdrew it, and the id a correction replaces. All of them are what
    the Audit page reads, so losing them loses the explanation rather than the fact.
    """

    def test_the_job_id_survives(self, store):
        a = _assertion()
        store.put(AssertionRecord(assertion=a, job_id="doc-7"))
        assert store.get(TENANT, a.assertion_id).job_id == "doc-7"

    def test_a_scoped_promotion_finds_the_record(self, store, graph):
        """The end-to-end shape of the bug: stage under a job, promote that job, expect it live."""
        from src.documents.review import ReviewQueue

        queue = ReviewQueue(store)
        ctx = AuthContext(user_id="sys", tenant_id=TENANT)
        a = build_assertion(
            tenant_id=TENANT,
            subject_id="document:d1",
            predicate="MENTIONS",
            object_id="party:calder",
            # EXTRACTED_DET on a presence predicate is AUTO_ASSERTED: a check confirmed the words
            # are there, so no person is needed and nothing will ever approve it.
            epistemic_class=EpistemicClass.EXTRACTED_DET,
            method="llm:test+verify:quote@v1",
            confidence=0.95,
            source_locator=SourceLocator(
                document_id="d1", filename="f.pdf", page=1, quote="Calder"
            ),
        )
        queue.stage(ctx, [a], job_id="doc-d1")

        assert queue.promote(ctx, job_id="doc-d1") == [a.assertion_id]
        assert [r.assertion_id for r in queue.live_assertions(ctx)] == [a.assertion_id]

    def test_the_review_note_survives(self, store):
        a = _assertion()
        record = AssertionRecord(assertion=a)
        record.review_note = "the letter names Calder as the adverse party"
        store.put(record)
        assert "adverse party" in (store.get(TENANT, a.assertion_id).review_note or "")

    def test_the_withdrawal_reason_and_actor_survive(self, store):
        """What the Audit page shows beside a withdrawn fact. Losing these leaves a fact that is
        gone with no recorded reason, which is the shape of an unexplained deletion."""
        a = _assertion()
        record = AssertionRecord(assertion=a)
        record.retracted_reason = "re-extracting with a corrected model"
        record.retracted_by = "partner@firm.example"
        store.put(record)

        got = store.get(TENANT, a.assertion_id)
        assert got.retracted_by == "partner@firm.example"
        assert "corrected model" in (got.retracted_reason or "")

    def test_the_correction_link_survives(self, store):
        a = _assertion()
        record = AssertionRecord(assertion=a)
        record.corrects = "some-earlier-assertion-id"
        store.put(record)
        assert store.get(TENANT, a.assertion_id).corrects == "some-earlier-assertion-id"
