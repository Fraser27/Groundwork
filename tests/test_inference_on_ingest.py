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
    from src.config import AuthConfig, GraphConfig, LexGraphConfig

    cfg = LexGraphConfig(
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
        """"Ran and found nothing" and "never ran" are the pair this codebase keeps confusing,
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
        assert not [
            a for a in before["assertions"] if a["predicate"] == "POTENTIAL_CONFLICT"
        ], "no conflict should exist before the second fact arrives"

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
