"""Correcting a claim, and withdrawing a document or a matter.

Both are destructive-looking operations on a record that must stay defensible, so the properties
tested here are mostly about what *survives*.

**A correction is never an edit.** An assertion says a named method at a named version, reading a
named span, produced this claim, and its id is a hash over exactly that. Editing the predicate in
place would leave the record asserting the model extracted something it did not -- a provenance
lie that still looks authoritative. So the reviewer's version is a new DECLARED assertion and the
original is closed.

**A wipe is soft, and does not cascade.** An inference resting on a withdrawn premise is left
standing, because it *was* drawn from evidence present at the time and its premises are still
resolvable. Retracting it would assert the firm never held a belief it did hold, which is the
kind of tidying a compliance record must not do.
"""

from __future__ import annotations

import pytest

from src.documents.review import Lifecycle, ReviewError, ReviewQueue
from src.graph.assertions import EpistemicClass, ReviewState, SourceLocator, build_assertion
from src.graph.scope import AuthContext, ScopeViolation
from src.graph_audit import (
    ACTIONS,
    MAX_STORED_IDS,
    SUPERSEDE,
    WIPE_DOCUMENT,
    WIPE_MATTER,
    GraphAudit,
    GraphEvent,
    InMemoryGraphAudit,
    graph_pk,
)
from src.ontology.loader import load_ontology

TENANT = "demo-firm"
NTL = "NTL-2026-0114"
MBC = "MBC-2024-0431"


@pytest.fixture
def ctx() -> AuthContext:
    return AuthContext(user_id="partner@firm.example", tenant_id=TENANT)


@pytest.fixture
def onto():
    return load_ontology("legal")


def fact(
    subject: str = "counsel:us",
    predicate: str = "REPRESENTS",
    obj: str = "party:calder",
    *,
    document_id: str = "doc-1",
    matter: str | None = NTL,
    epistemic_class: EpistemicClass = EpistemicClass.EXTRACTED_MODEL,
    confidence: float = 0.72,
):
    return build_assertion(
        tenant_id=TENANT,
        subject_id=subject,
        predicate=predicate,
        object_id=obj,
        epistemic_class=epistemic_class,
        method="llm:sonnet-5@v1",
        confidence=confidence,
        matter_id=matter,
        source_locator=SourceLocator(
            document_id=document_id,
            filename="engagement.pdf",
            page=2,
            quote="adverse to Calder Shipping AG",
        ),
    )


class TestCorrectingAClaim:
    def test_the_reviewers_version_is_declared_not_extracted(self, ctx, onto):
        """The class is the axis. A lawyer saying so and a model reading it must not become the
        same kind of object, or the graph loses the distinction it exists to keep."""
        queue = ReviewQueue()
        a = fact()
        queue.stage(ctx, [a])

        corrected, _ = queue.supersede(
            ctx,
            a.assertion_id,
            predicate="ADVERSE_TO",
            subject_id="matter:" + NTL,
            reason="the letter names Calder as the adverse party",
            allowed_predicates=onto.allowed_for("ADVERSE_TO"),
        )

        assert corrected.assertion.epistemic_class is EpistemicClass.DECLARED
        assert corrected.assertion.method == "reviewer:partner@firm.example"

    def test_it_needs_no_second_review(self, ctx, onto):
        """DECLARED auto-asserts: asking a person to approve their own correction would be
        theatre."""
        queue = ReviewQueue()
        a = fact()
        queue.stage(ctx, [a])

        corrected, _ = queue.supersede(
            ctx,
            a.assertion_id,
            predicate="ADVERSE_TO",
            reason="wrong side",
            allowed_predicates=onto.allowed_for("ADVERSE_TO"),
        )

        assert corrected.assertion.review_state is ReviewState.AUTO_ASSERTED
        assert corrected.lifecycle is Lifecycle.LIVE

    def test_the_original_is_closed_not_deleted(self, ctx, onto):
        """An as-of read before the correction has to still show what the model proposed."""
        queue = ReviewQueue()
        a = fact()
        queue.stage(ctx, [a])

        _, original = queue.supersede(
            ctx,
            a.assertion_id,
            predicate="ADVERSE_TO",
            reason="wrong side",
            allowed_predicates=onto.allowed_for("ADVERSE_TO"),
        )

        assert original.assertion.superseded_at is not None
        assert original.is_current is False
        # Still fetchable: closed, not gone.
        assert queue.store.get(TENANT, a.assertion_id) is not None

    def test_the_original_records_who_overrode_it_and_why(self, ctx, onto):
        queue = ReviewQueue()
        a = fact()
        queue.stage(ctx, [a])

        _, original = queue.supersede(
            ctx,
            a.assertion_id,
            predicate="ADVERSE_TO",
            reason="the letter names Calder as the adverse party",
            allowed_predicates=onto.allowed_for("ADVERSE_TO"),
        )

        assert original.retracted_by == "partner@firm.example"
        assert "adverse party" in (original.retracted_reason or "")

    def test_the_citation_is_kept(self, ctx, onto):
        """A correction with no citation is an opinion. The reviewer re-read the same span."""
        queue = ReviewQueue()
        a = fact()
        queue.stage(ctx, [a])

        corrected, _ = queue.supersede(
            ctx,
            a.assertion_id,
            predicate="ADVERSE_TO",
            reason="wrong side",
            allowed_predicates=onto.allowed_for("ADVERSE_TO"),
        )

        loc = corrected.assertion.source_locator
        assert loc.filename == "engagement.pdf"
        assert loc.page == 2
        assert loc.quote == "adverse to Calder Shipping AG"

    def test_it_records_what_it_corrects_without_calling_it_a_premise(self, ctx, onto):
        """`premises` means "derived from", and the contract refuses them on a DECLARED assertion
        -- rightly, because the reviewer read the document rather than deriving anything from the
        model's mistake. The link is recorded separately."""
        queue = ReviewQueue()
        a = fact()
        queue.stage(ctx, [a])

        corrected, _ = queue.supersede(
            ctx,
            a.assertion_id,
            predicate="ADVERSE_TO",
            reason="wrong side",
            allowed_predicates=onto.allowed_for("ADVERSE_TO"),
        )

        assert corrected.assertion.premises == ()
        assert corrected.corrects == a.assertion_id

    def test_only_the_corrected_version_is_live(self, ctx, onto):
        queue = ReviewQueue()
        a = fact()
        queue.stage(ctx, [a])
        queue.supersede(
            ctx,
            a.assertion_id,
            predicate="ADVERSE_TO",
            reason="wrong side",
            allowed_predicates=onto.allowed_for("ADVERSE_TO"),
        )

        live = [r.assertion.predicate for r in queue.live_assertions(ctx)]
        assert live == ["ADVERSE_TO"]

    def test_a_correction_that_changes_nothing_is_refused(self, ctx, onto):
        """Accepting a claim as it stands is what approve is for. A no-op correction would write
        a DECLARED duplicate and close the original for no reason."""
        queue = ReviewQueue()
        a = fact()
        queue.stage(ctx, [a])

        with pytest.raises(ReviewError, match="must change"):
            queue.supersede(ctx, a.assertion_id, reason="looks right")

    def test_a_reason_is_mandatory(self, ctx):
        queue = ReviewQueue()
        a = fact()
        queue.stage(ctx, [a])

        with pytest.raises(ReviewError, match="reason"):
            queue.supersede(ctx, a.assertion_id, predicate="ADVERSE_TO", reason="")

    def test_a_withdrawn_claim_cannot_be_corrected(self, ctx, onto):
        """Correcting it would revive a retracted claim under a new id."""
        queue = ReviewQueue()
        a = fact()
        queue.stage(ctx, [a])
        record = queue.store.get(TENANT, a.assertion_id)
        record.assertion.superseded_at = "2026-01-01T00:00:00Z"
        queue.store.put(record)

        with pytest.raises(ReviewError, match="withdrawn"):
            queue.supersede(
                ctx,
                a.assertion_id,
                predicate="ADVERSE_TO",
                reason="wrong side",
                allowed_predicates=onto.allowed_for("ADVERSE_TO"),
            )

    def test_an_unknown_governing_predicate_is_refused(self, ctx, onto):
        """The closed vocabulary holds for a reviewer too. A conflict check that misses
        `is_counsel_to` looks exactly like a clean conflict check."""
        from src.graph.assertions import AssertionError_

        queue = ReviewQueue()
        a = fact()
        queue.stage(ctx, [a])

        with pytest.raises(AssertionError_):
            queue.supersede(
                ctx,
                a.assertion_id,
                predicate="IS_COUNSEL_TO",
                reason="synonym",
                allowed_predicates=onto.governing_predicates,
            )


class TestWipingADocument:
    def _services(self, queue, audit=None):
        class Services:
            review_queue = queue
            embedder = None
            job_store = None
            graph_audit = audit if audit is not None else InMemoryGraphAudit()

        return Services()

    def test_facts_from_the_document_leave_the_current_graph(self, ctx):
        from src.documents.wipe import wipe_document

        queue = ReviewQueue()
        a = fact(document_id="doc-1")
        b = fact("counsel:us", "REPRESENTS", "party:other", document_id="doc-2")
        queue.stage(ctx, [a, b])
        for x in (a, b):
            queue.approve(ctx, x.assertion_id)

        report = wipe_document(
            self._services(queue),
            ctx,
            "doc-1",
            reason="re-extracting with a better model",
            drop_vectors=False,
            drop_jobs=False,
        )

        assert report.assertions_superseded == 1
        live = [r.assertion.assertion_id for r in queue.live_assertions(ctx)]
        assert live == [b.assertion_id]

    def test_the_facts_are_closed_not_deleted(self, ctx):
        """The whole point of a soft delete: an as-of read before now still reconstructs them."""
        from src.documents.wipe import wipe_document

        queue = ReviewQueue()
        a = fact(document_id="doc-1")
        queue.stage(ctx, [a])
        queue.approve(ctx, a.assertion_id)

        wipe_document(
            self._services(queue),
            ctx,
            "doc-1",
            reason="re-extracting",
            drop_vectors=False,
            drop_jobs=False,
        )

        stored = queue.store.get(TENANT, a.assertion_id)
        assert stored is not None
        assert stored.assertion.superseded_at is not None
        assert stored.retracted_by == "partner@firm.example"

    def test_a_live_fact_keeps_its_lifecycle(self, ctx):
        """LIVE is retained because it records that the fact *was* believed and acted upon.
        Rewriting it to DISCARDED would erase that."""
        from src.documents.wipe import wipe_document

        queue = ReviewQueue()
        a = fact(document_id="doc-1")
        queue.stage(ctx, [a])
        queue.approve(ctx, a.assertion_id)

        wipe_document(
            self._services(queue),
            ctx,
            "doc-1",
            reason="re-extracting",
            drop_vectors=False,
            drop_jobs=False,
        )

        assert queue.store.get(TENANT, a.assertion_id).lifecycle is Lifecycle.LIVE

    def test_an_inference_is_left_standing(self, ctx):
        """No cascade, and this is the decision worth defending.

        The conclusion was drawn from evidence present at the time, and its premises are closed
        rather than gone, so the proof tree still resolves. Retracting it would assert the firm
        never held a belief it did hold.
        """
        from src.documents.wipe import wipe_document

        queue = ReviewQueue()
        premise = fact(document_id="doc-1", confidence=0.9)
        queue.stage(ctx, [premise])
        queue.approve(ctx, premise.assertion_id)

        inferred = build_assertion(
            tenant_id=TENANT,
            subject_id="matter:" + NTL,
            predicate="MENTIONS",
            object_id="party:calder",
            epistemic_class=EpistemicClass.INFERRED,
            method="rule:conflict_check@v1",
            confidence=0.8,
            source_locator=SourceLocator(source_id="reasoner", table="conflict_check"),
            premises=(premise.assertion_id,),
            premise_confidences=(0.9,),
            rule_id="conflict_check",
            rule_version="v1",
        )
        queue.stage(ctx, [inferred])
        queue.approve(ctx, inferred.assertion_id)

        wipe_document(
            self._services(queue),
            ctx,
            "doc-1",
            reason="re-extracting",
            drop_vectors=False,
            drop_jobs=False,
        )

        still_live = [r.assertion.assertion_id for r in queue.live_assertions(ctx)]
        assert inferred.assertion_id in still_live
        # And its premise is still readable, so "why did you believe that" still answers.
        assert queue.store.get(TENANT, premise.assertion_id) is not None

    def test_a_reason_is_mandatory(self, ctx):
        from src.documents.wipe import wipe_document

        with pytest.raises(ValueError, match="reason"):
            wipe_document(self._services(ReviewQueue()), ctx, "doc-1", reason="")

    def test_the_wipe_is_audited(self, ctx):
        from src.documents.wipe import wipe_document

        queue = ReviewQueue()
        a = fact(document_id="doc-1")
        queue.stage(ctx, [a])
        audit = InMemoryGraphAudit()

        wipe_document(
            self._services(queue, audit),
            ctx,
            "doc-1",
            reason="loaded by mistake",
            drop_vectors=False,
            drop_jobs=False,
        )

        events = audit.events(TENANT)
        assert len(events) == 1
        assert events[0].action == WIPE_DOCUMENT
        assert events[0].actor == "partner@firm.example"
        assert events[0].reason == "loaded by mistake"
        assert a.assertion_id in events[0].assertion_ids

    def test_an_unrecorded_wipe_says_so(self, ctx):
        """The one outcome this must not produce silently: facts gone from the current graph and
        nothing saying who removed them."""
        from src.documents.wipe import wipe_document

        class NoAudit:
            review_queue = ReviewQueue()
            embedder = None
            job_store = None
            graph_audit = None

        report = wipe_document(
            NoAudit(), ctx, "doc-1", reason="x", drop_vectors=False, drop_jobs=False
        )
        assert any("not recorded" in e for e in report.errors)


class TestWipingAMatter:
    def _services(self, queue, audit=None):
        class Services:
            review_queue = queue
            embedder = None
            job_store = None
            graph_audit = audit if audit is not None else InMemoryGraphAudit()

        return Services()

    def test_every_document_on_the_matter_is_withdrawn(self, ctx):
        from src.documents.wipe import wipe_matter

        queue = ReviewQueue()
        on_matter = [
            fact(document_id="doc-1", matter=NTL),
            fact("counsel:us", "REPRESENTS", "party:x", document_id="doc-2", matter=NTL),
        ]
        elsewhere = fact("counsel:us", "REPRESENTS", "party:y", document_id="doc-3", matter=MBC)
        queue.stage(ctx, [*on_matter, elsewhere])
        for x in (*on_matter, elsewhere):
            queue.approve(ctx, x.assertion_id)

        report = wipe_matter(
            self._services(queue),
            ctx,
            NTL,
            reason="matter closed",
            drop_vectors=False,
            drop_jobs=False,
        )

        assert report.assertions_superseded == 2
        assert set(report.documents) == {"doc-1", "doc-2"}
        live = [r.assertion.assertion_id for r in queue.live_assertions(ctx)]
        assert live == [elsewhere.assertion_id]

    def test_a_screened_matter_cannot_be_wiped(self, ctx):
        """An ethical wall that holds for reads and not for deletions is not a wall."""
        from src.documents.wipe import wipe_matter

        screened = AuthContext(
            user_id="associate@firm.example",
            tenant_id=TENANT,
            matter_denylist=frozenset({NTL}),
        )
        with pytest.raises(ScopeViolation):
            wipe_matter(self._services(ReviewQueue()), screened, NTL, reason="x")

    def test_the_matter_wipe_is_audited(self, ctx):
        from src.documents.wipe import wipe_matter

        queue = ReviewQueue()
        a = fact(document_id="doc-1", matter=NTL)
        queue.stage(ctx, [a])
        audit = InMemoryGraphAudit()

        wipe_matter(
            self._services(queue, audit),
            ctx,
            NTL,
            reason="matter closed",
            drop_vectors=False,
            drop_jobs=False,
        )

        events = audit.events(TENANT)
        assert events[0].action == WIPE_MATTER
        assert events[0].matter_id == NTL


class TestTheAuditLog:
    def test_it_refuses_an_unknown_action(self):
        """A log whose vocabulary drifts cannot be filtered reliably."""
        with pytest.raises(ValueError, match="unknown"):
            InMemoryGraphAudit().append(GraphEvent(tenant_id=TENANT, actor="me", action="TIDY_UP"))

    def test_the_known_actions_are_closed(self):
        from src.graph_audit import LINK_DOCUMENTS

        assert ACTIONS == {SUPERSEDE, WIPE_DOCUMENT, WIPE_MATTER, LINK_DOCUMENTS}

    def test_newest_first(self):
        audit = InMemoryGraphAudit()
        for i in range(3):
            audit.append(
                GraphEvent(
                    tenant_id=TENANT,
                    actor="me",
                    action=WIPE_DOCUMENT,
                    at=f"2026-01-0{i + 1}T00:00:00Z",
                    document_id=f"doc-{i}",
                )
            )
        assert [e.document_id for e in audit.events(TENANT)] == ["doc-2", "doc-1", "doc-0"]

    def test_another_tenants_events_are_not_returned(self):
        audit = InMemoryGraphAudit()
        audit.append(GraphEvent(tenant_id="other", actor="them", action=WIPE_DOCUMENT))
        assert audit.events(TENANT) == []


class TestTheAuditLogIsAppendOnly:
    def test_a_duplicate_key_is_refused_rather_than_overwritten(self):
        """What makes it append-only rather than merely append-shaped: no later write can rewrite
        the record of who deleted what."""
        conditions: list[str] = []

        class FakeTable:
            def put_item(self, **kw):
                conditions.append(kw.get("ConditionExpression", ""))
                return {}

            def query(self, **kw):
                return {"Items": []}

        GraphAudit(table=FakeTable()).append(
            GraphEvent(tenant_id=TENANT, actor="me", action=WIPE_DOCUMENT, document_id="d")
        )
        assert "attribute_not_exists" in conditions[0]

    def test_the_key_is_scoped_to_the_tenant(self):
        items: list[dict] = []

        class FakeTable:
            def put_item(self, **kw):
                items.append(kw["Item"])
                return {}

            def query(self, **kw):
                return {"Items": []}

        GraphAudit(table=FakeTable()).append(
            GraphEvent(tenant_id=TENANT, actor="me", action=WIPE_MATTER, matter_id=NTL)
        )
        assert items[0]["PK"] == graph_pk(TENANT)

    def test_a_huge_id_list_is_capped_but_the_count_is_exact(self):
        """A DynamoDB item is capped at 400KB and a large wipe would exceed it. The count has to
        stay right even when the list is clipped, or the audit under-reports."""
        items: list[dict] = []

        class FakeTable:
            def put_item(self, **kw):
                items.append(kw["Item"])
                return {}

            def query(self, **kw):
                return {"Items": []}

        many = tuple(f"a-{i}" for i in range(MAX_STORED_IDS + 50))
        GraphAudit(table=FakeTable()).append(
            GraphEvent(tenant_id=TENANT, actor="me", action=WIPE_MATTER, assertion_ids=many)
        )
        assert len(items[0]["assertion_ids"]) == MAX_STORED_IDS
        assert items[0]["affected"] == MAX_STORED_IDS + 50
        assert items[0]["ids_truncated"] is True


class TestOverHttp:
    """The wiring, which is where these go wrong."""

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient

        from src.api.app import create_app
        from src.api.deps import get_services
        from src.config import AuthConfig, GraphConfig, LexGraphConfig

        cfg = LexGraphConfig(
            environment="local",
            auth=AuthConfig(dev_bypass_tenant=TENANT),
            graph=GraphConfig(uri="bolt://127.0.0.1:1", user="none", password="none"),
        )
        cfg.validate()
        return TestClient(create_app(cfg)), get_services()

    def _staged(self, services, ctx, **kw):
        a = fact(**kw)
        services.review_queue.stage(ctx, [a])
        return a

    def test_correcting_over_http(self, client, ctx):
        c, services = client
        a = self._staged(services, ctx)

        r = c.post(
            f"/api/tenants/{TENANT}/assertions/{a.assertion_id}/correct",
            json={"predicate": "ADVERSE_TO", "subject_id": "matter:" + NTL, "reason": "wrong side"},
        )
        assert r.status_code == 200
        body = r.json()
        assert body["corrected"]["predicate"] == "ADVERSE_TO"
        assert body["corrected"]["epistemic_class"] == "DECLARED"
        assert body["superseded"]["assertion_id"] == a.assertion_id

    def test_a_correction_is_audited(self, client, ctx):
        c, services = client
        a = self._staged(services, ctx)
        c.post(
            f"/api/tenants/{TENANT}/assertions/{a.assertion_id}/correct",
            json={"predicate": "ADVERSE_TO", "reason": "wrong side"},
        )

        log = c.get(f"/api/tenants/{TENANT}/audit/graph").json()
        assert log["count"] >= 1
        assert log["events"][0]["action"] == SUPERSEDE
        assert log["events"][0]["reason"] == "wrong side"

    def test_a_reviewer_cannot_assert_a_rule_conclusion(self, client, ctx):
        """Only a rule may conclude POTENTIAL_CONFLICT. A reviewer asserting one directly would
        create a conflict flag with no premises -- exactly what the extractor is barred from."""
        c, services = client
        a = self._staged(services, ctx)

        r = c.post(
            f"/api/tenants/{TENANT}/assertions/{a.assertion_id}/correct",
            json={"predicate": "POTENTIAL_CONFLICT", "reason": "it is a conflict"},
        )
        assert r.status_code == 422
        assert "rule" in r.json()["detail"]

    def test_a_correction_without_a_reason_is_refused(self, client, ctx):
        c, services = client
        a = self._staged(services, ctx)
        r = c.post(
            f"/api/tenants/{TENANT}/assertions/{a.assertion_id}/correct",
            json={"predicate": "ADVERSE_TO", "reason": ""},
        )
        assert r.status_code == 422

    def test_wiping_a_document_over_http(self, client, ctx):
        c, services = client
        a = self._staged(services, ctx, document_id="doc-9")
        services.review_queue.approve(ctx, a.assertion_id)

        r = c.post(
            f"/api/tenants/{TENANT}/documents/doc-9/wipe",
            json={"reason": "re-extracting", "drop_vectors": False, "drop_jobs": False},
        )
        assert r.status_code == 200
        assert r.json()["assertions_superseded"] == 1
        assert services.review_queue.live_assertions(ctx) == []

    def test_wiping_a_matter_over_http(self, client, ctx):
        c, services = client
        a = self._staged(services, ctx, document_id="doc-8", matter=NTL)
        services.review_queue.approve(ctx, a.assertion_id)

        r = c.post(
            f"/api/tenants/{TENANT}/matters/{NTL}/wipe",
            json={"reason": "matter closed", "drop_vectors": False, "drop_jobs": False},
        )
        assert r.status_code == 200
        assert r.json()["assertions_superseded"] == 1

    def test_a_wipe_without_a_reason_is_refused(self, client):
        c, _ = client
        r = c.post(f"/api/tenants/{TENANT}/documents/doc-1/wipe", json={"reason": ""})
        assert r.status_code == 422

    def test_the_audit_log_records_a_wipe_with_who_and_when(self, client, ctx):
        """The trace back the whole soft delete exists for."""
        c, services = client
        a = self._staged(services, ctx, document_id="doc-7")
        c.post(
            f"/api/tenants/{TENANT}/documents/doc-7/wipe",
            json={"reason": "loaded by mistake", "drop_vectors": False, "drop_jobs": False},
        )

        event = c.get(f"/api/tenants/{TENANT}/audit/graph").json()["events"][0]
        assert event["action"] == WIPE_DOCUMENT
        assert event["document_id"] == "doc-7"
        assert event["actor"]
        assert event["at"]
        assert a.assertion_id in event["assertion_ids"]
