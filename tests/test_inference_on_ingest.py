"""Inference runs when a document lands, not only when someone remembers to ask.

The bug this closes was invisible from outside. A conflict was derivable from facts already in the
graph, no `POTENTIAL_CONFLICT` existed because nothing had called `/reason`, and the ethical wall
reported "nothing refused" -- honestly, because there was no conclusion in the graph to refuse on.
A model narrating the conflict in prose was the only thing telling anyone about it.

Ingest is the right trigger because a conflict is almost never visible in one document. It exists
in the join between this filing and one already on file, so the moment a new document goes live is
exactly the moment the join becomes possible.
"""

from __future__ import annotations

import pytest

from src.documents.review import Lifecycle
from src.graph.assertions import ReviewState
from src.graph.scope import AuthContext
from src.ontology.loader import load_ontology
from src.reasoning.engine import infer_and_stage

TENANT = "demo-firm"
NTL = "NTL-2026-0114"
MBC = "MBC-2024-0431"


@pytest.fixture
def ctx() -> AuthContext:
    return AuthContext(user_id="partner@firm.example", tenant_id=TENANT)


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    from src.api.app import create_app
    from src.api.deps import get_services
    from src.config import AuthConfig, GraphConfig, GroundworkConfig

    cfg = GroundworkConfig(
        # Pinned rather than defaulted: this file asserts the legal pack's rules and
        # vocabulary, so it must not follow a change of default pack.
        ontology_pack="legal",
        environment="local",
        auth=AuthConfig(dev_bypass_tenant=TENANT),
        graph=GraphConfig(uri="bolt://127.0.0.1:1", user="none", password="none"),
    )
    cfg.validate()
    return TestClient(create_app(cfg)), get_services()


def _ingest(c, text: str, *, matter_id: str, filename: str = "note.txt") -> dict:
    return c.post(
        f"/api/tenants/{TENANT}/documents/text",
        json={
            "text": text,
            "filename": filename,
            "matter_id": matter_id,
            "run_model_extraction": False,
        },
    ).json()


class TestIngestReportsWhatItInferred:
    def test_the_response_says_inference_ran(self, client):
        """ "Ran and found nothing" and "never ran" are the pair this codebase keeps confusing,
        and only one of them is reassuring."""
        c, _ = client
        body = _ingest(c, "A short note naming nobody in particular.", matter_id=NTL)

        assert body["inferred"]["ran"] is True
        assert body["inferred"]["rules_evaluated"] == len(load_ontology("legal").rules)

    def test_a_document_with_nothing_to_join_infers_nothing(self, client):
        c, _ = client
        body = _ingest(c, "A short note naming nobody in particular.", matter_id=NTL)
        assert body["inferred"]["staged"] == 0


class TestTheCrossDocumentConflictAppearsByItself:
    """The failure from production, as a test: two documents, neither wrong alone."""

    def _approve_all_pending(self, c, services, ctx):
        pending = c.get(f"/api/tenants/{TENANT}/assertions?review_state=PENDING").json()
        for row in pending["assertions"]:
            if row["predicate"] != "POTENTIAL_CONFLICT":
                c.post(f"/api/tenants/{TENANT}/assertions/{row['assertion_id']}/approve")
        services.review_queue.promote(ctx, job_id=None)

    def test_the_second_document_produces_the_conflict(self, client, ctx):
        """The firm acts for Calder on one matter; a filing on another opposes it. Nobody asks
        for a conflict check -- ingesting the second document is what surfaces it."""
        c, services = client
        onto = load_ontology("legal")

        from src.graph.assertions import (
            EpistemicClass,
            ReviewState,
            SourceLocator,
            build_assertion,
        )

        def live(subject, predicate, obj, matter):
            a = build_assertion(
                tenant_id=TENANT,
                subject_id=subject,
                predicate=predicate,
                object_id=obj,
                epistemic_class=EpistemicClass.EXTRACTED_MODEL,
                method="llm:test@v1",
                confidence=0.9,
                source_locator=SourceLocator(
                    document_id="d-prior", filename="engagement.pdf", page=1, quote="a quote"
                ),
                matter_id=matter,
                allowed_predicates=onto.extractable_predicates,
            )
            a.review_state = ReviewState.APPROVED
            return a

        # The prior file: the firm acts for Calder.
        services.review_queue.stage(
            ctx,
            [live("counsel:sian-aldridge", "REPRESENTS", "party:calder-shipping-ag", MBC)],
            job_id="prior",
        )
        services.review_queue.promote(ctx, job_id="prior")

        before = c.get(f"/api/tenants/{TENANT}/assertions?review_state=PENDING").json()
        assert not [a for a in before["assertions"] if a["predicate"] == "POTENTIAL_CONFLICT"], (
            "no conflict should exist before the second fact arrives"
        )

        # The new filing opposes the same company. Staged live the same way an approved
        # extraction would be, then ingest anything at all to trigger the pass.
        services.review_queue.stage(
            ctx,
            [live("matter:" + NTL, "ADVERSE_TO", "party:calder-shipping-ag", NTL)],
            job_id="new",
        )
        services.review_queue.promote(ctx, job_id="new")

        body = _ingest(c, "A filing in the Northwind matter.", matter_id=NTL)
        assert body["inferred"]["staged"] >= 1

        after = c.get(f"/api/tenants/{TENANT}/assertions?review_state=PENDING").json()
        conflicts = [a for a in after["assertions"] if a["predicate"] == "POTENTIAL_CONFLICT"]
        assert len(conflicts) == 1
        assert conflicts[0]["object_id"] == "party:calder-shipping-ag"

    def test_the_conclusion_is_pending_not_live(self, client, ctx):
        """The review gate is unchanged. Auto-running inference must not auto-publish a conflict
        flag -- a lawyer decides whether it is real."""
        c, services = client
        onto = load_ontology("legal")

        from src.graph.assertions import (
            EpistemicClass,
            ReviewState,
            SourceLocator,
            build_assertion,
        )

        def live(subject, predicate, obj, matter):
            a = build_assertion(
                tenant_id=TENANT,
                subject_id=subject,
                predicate=predicate,
                object_id=obj,
                epistemic_class=EpistemicClass.EXTRACTED_MODEL,
                method="llm:test@v1",
                confidence=0.9,
                source_locator=SourceLocator(
                    document_id="d-prior", filename="x.pdf", page=1, quote="a quote"
                ),
                matter_id=matter,
                allowed_predicates=onto.extractable_predicates,
            )
            a.review_state = ReviewState.APPROVED
            return a

        services.review_queue.stage(
            ctx,
            [
                live("counsel:sian-aldridge", "REPRESENTS", "party:calder-shipping-ag", MBC),
                live("matter:" + NTL, "ADVERSE_TO", "party:calder-shipping-ag", NTL),
            ],
            job_id="prior",
        )
        services.review_queue.promote(ctx, job_id="prior")

        _ingest(c, "A filing.", matter_id=NTL)

        conflicts = [
            r
            for r in services.review_queue.visible(ctx)
            if r.assertion.predicate == "POTENTIAL_CONFLICT"
        ]
        assert len(conflicts) == 1
        assert conflicts[0].assertion.review_state is ReviewState.PENDING


class TestItIsIdempotent:
    def test_a_second_pass_does_not_duplicate(self, client, ctx):
        """`assertion_id` is content-addressed, so re-running converges. Otherwise every upload
        would add another copy of every standing conclusion."""
        c, services = client
        onto = load_ontology("legal")

        from src.graph.assertions import (
            EpistemicClass,
            ReviewState,
            SourceLocator,
            build_assertion,
        )

        def live(subject, predicate, obj, matter):
            a = build_assertion(
                tenant_id=TENANT,
                subject_id=subject,
                predicate=predicate,
                object_id=obj,
                epistemic_class=EpistemicClass.EXTRACTED_MODEL,
                method="llm:test@v1",
                confidence=0.9,
                source_locator=SourceLocator(
                    document_id="d1", filename="x.pdf", page=1, quote="a quote"
                ),
                matter_id=matter,
                allowed_predicates=onto.extractable_predicates,
            )
            a.review_state = ReviewState.APPROVED
            return a

        services.review_queue.stage(
            ctx,
            [
                live("counsel:sian-aldridge", "REPRESENTS", "party:calder-shipping-ag", MBC),
                live("matter:" + NTL, "ADVERSE_TO", "party:calder-shipping-ag", NTL),
            ],
            job_id="prior",
        )
        services.review_queue.promote(ctx, job_id="prior")

        _ingest(c, "First filing.", matter_id=NTL, filename="one.txt")
        _ingest(c, "Second filing.", matter_id=NTL, filename="two.txt")

        conflicts = [
            r
            for r in services.review_queue.visible(ctx)
            if r.assertion.predicate == "POTENTIAL_CONFLICT"
        ]
        assert len(conflicts) == 1


class TestApprovalDrawsWhatItMakesPossible:
    """The gap ingest alone left, found by a reviewer looking at a real queue.

    A conflict's premises are EXTRACTED_MODEL, so they are PENDING when the document lands and
    `live_assertions` rightly excludes them — the ingest pass correctly finds nothing. Approval is
    the moment they become usable, and nothing re-ran inference there. So both premises sat
    approved and live while the conflict they imply existed nowhere, and the wall reported "nothing
    refused" over a graph that could have refused.
    """

    def _pending_premises(self, services, ctx):
        """Two model claims that together imply a conflict, staged and awaiting review."""
        onto = load_ontology("legal")
        from src.graph.assertions import EpistemicClass, SourceLocator, build_assertion

        claims = [
            build_assertion(
                tenant_id=TENANT,
                subject_id=subject,
                predicate=predicate,
                object_id=obj,
                epistemic_class=EpistemicClass.EXTRACTED_MODEL,
                method="llm:test@v1",
                confidence=0.9,
                source_locator=SourceLocator(
                    document_id="d1", filename="advice.pdf", page=1, quote="a quote"
                ),
                matter_id=matter,
                allowed_predicates=onto.extractable_predicates,
            )
            for subject, predicate, obj, matter in (
                ("counsel:sian-aldridge", "REPRESENTS", "party:calder-shipping-ag", MBC),
                ("matter:" + NTL, "ADVERSE_TO", "party:calder-shipping-ag", NTL),
            )
        ]
        services.review_queue.stage(ctx, claims, job_id="j1")
        return claims

    def _conflicts(self, services, ctx):
        return [
            r
            for r in services.review_queue.visible(ctx)
            if r.assertion.predicate == "POTENTIAL_CONFLICT"
        ]

    def test_pending_premises_draw_nothing(self, client, ctx):
        """Not a bug: an unreviewed model claim must not be a premise. This is the state the
        ingest pass legitimately leaves behind."""
        _, services = client
        self._pending_premises(services, ctx)
        assert infer_and_stage(load_ontology("legal"), services.review_queue, ctx).count == 0

    def test_approving_the_last_premise_draws_the_conflict(self, client, ctx):
        c, services = client
        first, second = self._pending_premises(services, ctx)

        c.post(f"/api/tenants/{TENANT}/assertions/{first.assertion_id}/approve")
        assert self._conflicts(services, ctx) == [], "one premise is not a conflict"

        r = c.post(f"/api/tenants/{TENANT}/assertions/{second.assertion_id}/approve")
        assert r.status_code == 200
        assert r.json()["inferred"] == 1

        conflicts = self._conflicts(services, ctx)
        assert len(conflicts) == 1
        assert conflicts[0].assertion.object_id == "party:calder-shipping-ag"

    def test_the_drawn_conflict_is_pending(self, client, ctx):
        """Auto-drawing must not auto-publish. Whether a conflict is real stays a lawyer's call."""
        c, services = client
        for claim in self._pending_premises(services, ctx):
            c.post(f"/api/tenants/{TENANT}/assertions/{claim.assertion_id}/approve")

        assert self._conflicts(services, ctx)[0].assertion.review_state is ReviewState.PENDING

    def test_approving_an_ordinary_fact_reports_nothing_drawn(self, client, ctx):
        """Most facts are not the last premise of anything, and the field should say so rather
        than being noise on every approval."""
        c, services = client
        first, _ = self._pending_premises(services, ctx)
        body = c.post(f"/api/tenants/{TENANT}/assertions/{first.assertion_id}/approve").json()
        assert body["inferred"] == 0

    def test_a_rejection_draws_nothing(self, client, ctx):
        """Rejecting removes a premise rather than supplying one, so there is nothing to draw."""
        c, services = client
        first, second = self._pending_premises(services, ctx)
        c.post(f"/api/tenants/{TENANT}/assertions/{first.assertion_id}/approve")
        c.post(
            f"/api/tenants/{TENANT}/assertions/{second.assertion_id}/reject",
            json={"note": "the letter does not support this"},
        )
        assert self._conflicts(services, ctx) == []

    def test_a_failed_inference_does_not_lose_the_approval(self, client, ctx, monkeypatch):
        """The approval is the reviewer's decision and already succeeded. Failing the request
        afterwards would report a recorded decision as lost."""
        c, services = client
        first, _ = self._pending_premises(services, ctx)

        def boom(*_args, **_kwargs):
            raise RuntimeError("reasoner unavailable")

        monkeypatch.setattr("src.reasoning.engine.infer_and_stage", boom)
        r = c.post(f"/api/tenants/{TENANT}/assertions/{first.assertion_id}/approve")
        assert r.status_code == 200
        assert r.json()["review_state"] == "APPROVED"
        assert r.json()["inferred"] == 0


class TestApprovingAConclusionSurvivesTheNextPass:
    """The regression that made approving a conflict a no-op.

    `assertion_id` is content-addressed, so a reasoning pass over unchanged premises re-derives
    the *same* conclusion with the *same* id. Re-staging it wrote a fresh PENDING record over the
    approved one — so approving a conflict silently un-approved it, the veto never took effect,
    and the UI reported success. "Idempotent" was true of the id and false of the decision.

    Found by a reviewer who approved a conflict and watched it come back.
    """

    def _approved_conflict(self, c, services, ctx):
        onto = load_ontology("legal")
        from src.graph.assertions import EpistemicClass, SourceLocator, build_assertion

        premises = [
            build_assertion(
                tenant_id=TENANT,
                subject_id=subject,
                predicate=predicate,
                object_id=obj,
                epistemic_class=EpistemicClass.EXTRACTED_MODEL,
                method="llm:test@v1",
                confidence=0.9,
                source_locator=SourceLocator(
                    document_id="d1", filename="x.pdf", page=1, quote="a quote"
                ),
                matter_id=matter,
                allowed_predicates=onto.extractable_predicates,
            )
            for subject, predicate, obj, matter in (
                ("counsel:sian-aldridge", "REPRESENTS", "party:calder-shipping-ag", MBC),
                ("matter:" + NTL, "ADVERSE_TO", "party:calder-shipping-ag", NTL),
            )
        ]
        services.review_queue.stage(ctx, premises, job_id="j1")
        for p in premises:
            c.post(f"/api/tenants/{TENANT}/assertions/{p.assertion_id}/approve")

        conflict = next(
            r
            for r in services.review_queue.visible(ctx)
            if r.assertion.predicate == "POTENTIAL_CONFLICT"
        )
        c.post(f"/api/tenants/{TENANT}/assertions/{conflict.assertion_id}/approve")
        return conflict.assertion_id

    def test_the_approval_holds(self, client, ctx):
        c, services = client
        conflict_id = self._approved_conflict(c, services, ctx)

        record = services.review_queue.fetch(ctx, conflict_id)
        assert record.assertion.review_state is ReviewState.APPROVED
        assert record.assertion.reviewed_by is not None

    def test_it_holds_through_a_later_pass(self, client, ctx):
        """The pass that clobbered it ran inside the approval request. Running another one
        explicitly is the same hazard, and the ingest and /reason paths have always done it."""
        c, services = client
        conflict_id = self._approved_conflict(c, services, ctx)

        infer_and_stage(load_ontology("legal"), services.review_queue, ctx)

        record = services.review_queue.fetch(ctx, conflict_id)
        assert record.assertion.review_state is ReviewState.APPROVED
        assert record.lifecycle is Lifecycle.LIVE

    def test_the_approved_conflict_actually_reports(self, client, ctx):
        """What the whole thing is for. An approval that silently reset to PENDING produced a
        conflict nobody was told about, which is indistinguishable from a clean conflict check.

        Asserts *reporting* rather than refusing: the pack declares `effect: notify`, so a signed-off
        conflict names itself and suppresses nothing. `awaiting_review` must be empty because that
        channel is for the unreviewed -- a finding in both places would read as two findings."""
        c, services = client
        self._approved_conflict(c, services, ctx)

        from src.query.blocks import blocks_for
        from src.query.graph_reader import GraphReader

        reader = GraphReader(services.review_queue, ontology=load_ontology("legal"))
        screen = blocks_for(ctx, graph_reader=reader, seeds=["party:calder-shipping-ag"])
        assert screen.advisories, "the approved conflict must be reported"
        assert screen.awaiting_review == (), "it is signed off, so it is not awaiting review"

    def test_a_pending_claim_is_still_refreshed(self, client, ctx):
        """The restraint. Re-extraction must keep updating a claim nobody has looked at, so only
        a *decided* record is protected."""
        _, services = client
        onto = load_ontology("legal")
        from src.graph.assertions import EpistemicClass, SourceLocator, build_assertion

        def claim():
            return build_assertion(
                tenant_id=TENANT,
                subject_id="counsel:sian-aldridge",
                predicate="REPRESENTS",
                object_id="party:calder-shipping-ag",
                epistemic_class=EpistemicClass.EXTRACTED_MODEL,
                method="llm:test@v1",
                confidence=0.9,
                source_locator=SourceLocator(
                    document_id="d1", filename="x.pdf", page=1, quote="a quote"
                ),
                matter_id=MBC,
                allowed_predicates=onto.extractable_predicates,
            )

        first = claim()
        services.review_queue.stage(ctx, [first], job_id="j1")
        services.review_queue.stage(ctx, [claim()], job_id="j2")

        record = services.review_queue.fetch(ctx, first.assertion_id)
        assert record.job_id == "j2", "a pending claim should still be refreshed"

    def test_staging_still_reports_every_id_it_was_given(self, client, ctx):
        """The caller asked what is in the queue for this job. Omitting the settled ones would
        read as a partial failure."""
        c, services = client
        conflict_id = self._approved_conflict(c, services, ctx)
        record = services.review_queue.fetch(ctx, conflict_id)

        staged = services.review_queue.stage(ctx, [record.assertion], job_id="again")
        assert staged == [conflict_id]


class TestApprovingABatchRunsOnePass:
    """N approvals must not mean N passes over the tenant's graph, and a conflict approved as a
    batch should be drawn from all of its premises rather than from however many were live
    partway through a loop."""

    def _two_premises(self, services, ctx):
        onto = load_ontology("legal")
        from src.graph.assertions import EpistemicClass, SourceLocator, build_assertion

        claims = [
            build_assertion(
                tenant_id=TENANT,
                subject_id=subject,
                predicate=predicate,
                object_id=obj,
                epistemic_class=EpistemicClass.EXTRACTED_MODEL,
                method="llm:test@v1",
                confidence=0.9,
                source_locator=SourceLocator(
                    document_id="d1", filename="advice.pdf", page=1, quote="a quote"
                ),
                matter_id=matter,
                allowed_predicates=onto.extractable_predicates,
            )
            for subject, predicate, obj, matter in (
                ("counsel:sian-aldridge", "REPRESENTS", "party:calder-shipping-ag", MBC),
                ("matter:" + NTL, "ADVERSE_TO", "party:calder-shipping-ag", NTL),
            )
        ]
        services.review_queue.stage(ctx, claims, job_id="j1")
        return claims

    def test_one_call_approves_both_and_draws_the_conflict(self, client, ctx):
        c, services = client
        claims = self._two_premises(services, ctx)

        body = c.post(
            f"/api/tenants/{TENANT}/assertions/approve",
            json={"assertion_ids": [a.assertion_id for a in claims]},
        ).json()

        assert len(body["approved"]) == 2
        assert body["failed"] == []
        assert body["inferred"] == 1

    def test_the_pass_runs_once_not_per_assertion(self, client, ctx, monkeypatch):
        c, services = client
        claims = self._two_premises(services, ctx)

        calls = {"n": 0}
        real = infer_and_stage

        def counted(*args, **kwargs):
            calls["n"] += 1
            return real(*args, **kwargs)

        monkeypatch.setattr("src.reasoning.engine.infer_and_stage", counted)
        c.post(
            f"/api/tenants/{TENANT}/assertions/approve",
            json={"assertion_ids": [a.assertion_id for a in claims]},
        )
        assert calls["n"] == 1

    def test_one_bad_id_does_not_lose_the_others(self, client, ctx):
        """A reviewer clearing twenty claims should not lose nineteen decisions to a twentieth a
        cascade had already rejected -- and the failures are named, because "19 of 20" with no
        list is not something anyone can act on."""
        c, services = client
        claims = self._two_premises(services, ctx)

        body = c.post(
            f"/api/tenants/{TENANT}/assertions/approve",
            json={"assertion_ids": [claims[0].assertion_id, "does-not-exist"]},
        ).json()

        assert len(body["approved"]) == 1
        assert body["failed"][0]["assertion_id"] == "does-not-exist"
        assert "does-not-exist" in body["failed"][0]["reason"]

    def test_nothing_approved_means_no_pass(self, client, ctx):
        c, _ = client
        body = c.post(
            f"/api/tenants/{TENANT}/assertions/approve",
            json={"assertion_ids": ["nope"]},
        ).json()
        assert body["approved"] == []
        assert body["inferred"] == 0


class TestTheSharedDefinition:
    """The endpoint and the ingest path must not drift, because the drift is silent: a conflict
    would exist or not depending on how the document arrived."""

    def test_the_endpoint_uses_the_shared_pass(self):
        import inspect

        from src.api import routes_review

        assert "infer_and_stage" in inspect.getsource(routes_review.run_reasoner)

    def test_ingest_uses_the_shared_pass(self):
        import inspect

        from src.api import routes_documents

        assert "infer_and_stage" in inspect.getsource(routes_documents._infer)

    def test_it_stages_rather_than_publishing(self, ctx):
        """The one thing no code path may do is opt out of review."""
        from src.documents.review import Lifecycle, ReviewQueue
        from src.graph.assertions import (
            EpistemicClass,
            ReviewState,
            SourceLocator,
            build_assertion,
        )

        onto = load_ontology("legal")
        queue = ReviewQueue(governing_predicates=onto.governing_predicates)

        def live(subject, predicate, obj, matter):
            a = build_assertion(
                tenant_id=TENANT,
                subject_id=subject,
                predicate=predicate,
                object_id=obj,
                epistemic_class=EpistemicClass.EXTRACTED_MODEL,
                method="llm:test@v1",
                confidence=0.9,
                source_locator=SourceLocator(
                    document_id="d1", filename="x.pdf", page=1, quote="a quote"
                ),
                matter_id=matter,
                allowed_predicates=onto.extractable_predicates,
            )
            a.review_state = ReviewState.APPROVED
            return a

        queue.stage(
            ctx,
            [
                live("counsel:sian", "REPRESENTS", "party:calder-shipping-ag", MBC),
                live("matter:" + NTL, "ADVERSE_TO", "party:calder-shipping-ag", NTL),
            ],
            job_id="j1",
        )
        queue.promote(ctx, job_id="j1")

        report = infer_and_stage(onto, queue, ctx)
        assert report.count == 1

        conclusion = next(
            r for r in queue.visible(ctx) if r.assertion.predicate == "POTENTIAL_CONFLICT"
        )
        assert conclusion.lifecycle is Lifecycle.STAGED
        assert conclusion.assertion.review_state is ReviewState.PENDING
