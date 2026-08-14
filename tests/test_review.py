"""Tests for the review queue.

The behaviours worth defending here are the ones a caller would happily route around
if the code let them: a model extraction reaching the live graph without a human, an
approval with no name attached, a rejection that quietly deletes the evidence.
"""

from __future__ import annotations

import pytest

from src.documents.review import (
    Lifecycle,
    ReviewError,
    ReviewQueue,
)
from src.graph.assertions import (
    EpistemicClass,
    ReviewState,
    SourceLocator,
    build_assertion,
)
from src.graph.scope import AuthContext, ScopeViolation

TENANT = "firm-acme"
LOC = SourceLocator(document_id="doc-1", filename="a.pdf", page=2, quote="the Adverse Party")


def ctx(user="reviewer-1", tenant=TENANT, **kw) -> AuthContext:
    return AuthContext(user_id=user, tenant_id=tenant, **kw)


def det(object_id="authority:410-u-s-113", **over):
    """A quote-verified presence claim — the only shape EXTRACTED_DET permits.

    A model proposed the text was there; a string search confirmed it. Presence is all
    a check can establish, hence MENTIONS rather than CITES.
    """
    kw = dict(
        tenant_id=TENANT,
        subject_id="document:doc-1",
        predicate="MENTIONS",
        object_id=object_id,
        epistemic_class=EpistemicClass.EXTRACTED_DET,
        method="llm:opus-5+verify:quote@v1",
        confidence=0.98,
        source_locator=LOC,
    )
    kw.update(over)
    return build_assertion(**kw)


def model(object_id="authority:550-u-s-544", confidence=0.7, **over):
    kw = dict(
        tenant_id=TENANT,
        subject_id="document:doc-1",
        predicate="DISTINGUISHES",
        object_id=object_id,
        epistemic_class=EpistemicClass.EXTRACTED_MODEL,
        method="llm:claude-sonnet-4-5",
        confidence=confidence,
        source_locator=LOC,
    )
    kw.update(over)
    return build_assertion(**kw)


class TestStaging:
    def test_staging_returns_ids_and_stores_records(self):
        q = ReviewQueue()
        ids = q.stage(ctx(), [det(), model()], job_id="job-1")
        assert len(ids) == 2
        assert len(q.list_staged(ctx(), job_id="job-1")) == 2

    def test_cross_tenant_assertion_is_refused(self):
        q = ReviewQueue()
        with pytest.raises(ScopeViolation, match="tenant"):
            q.stage(ctx(), [det(tenant_id="firm-other")])

    def test_a_bad_assertion_stages_nothing(self):
        """All-or-nothing: a half-ingested filing must not look complete."""
        q = ReviewQueue()
        with pytest.raises(ScopeViolation):
            q.stage(ctx(), [det(), det(tenant_id="firm-other", object_id="authority:x")])
        assert q.list_staged(ctx()) == []

    def test_matter_outside_scope_is_refused(self):
        q = ReviewQueue()
        walled = ctx(matter_allowlist=frozenset({"matter-1"}))
        with pytest.raises(ScopeViolation):
            q.stage(walled, [det(matter_id="matter-9")])

    def test_restaging_the_same_assertion_is_a_noop(self):
        q = ReviewQueue()
        q.stage(ctx(), [det()], job_id="job-1")
        q.stage(ctx(), [det()], job_id="job-2")
        assert len(q.list_staged(ctx())) == 1


class TestQueueContents:
    def test_only_model_extractions_are_pending(self):
        q = ReviewQueue()
        q.stage(ctx(), [det(), model()])
        pending = q.list_pending(ctx())
        assert [i.epistemic_class for i in pending] == [EpistemicClass.EXTRACTED_MODEL]

    def test_queue_item_carries_the_citation(self):
        """File, page and quote — what a reviewer opens the PDF and searches for.

        Previously asserted char offsets; those index the extracted text buffer rather
        than the PDF, so they are debug metadata now, not the citation.
        """
        q = ReviewQueue()
        q.stage(ctx(), [model()])
        item = q.list_pending(ctx())[0]
        assert (item.document_id, item.page) == ("doc-1", 2)

    def test_lowest_confidence_first(self):
        """A reviewer's attention is worth most where the model was least sure."""
        q = ReviewQueue()
        q.stage(
            ctx(),
            [
                model(object_id="authority:a", confidence=0.75),
                model(object_id="authority:b", confidence=0.4),
                model(object_id="authority:c", confidence=0.6),
            ],
        )
        assert [i.confidence for i in q.list_pending(ctx())] == [0.4, 0.6, 0.75]

    def test_filter_by_document(self):
        q = ReviewQueue()
        other = SourceLocator(document_id="doc-2", filename="b.pdf", page=1, quote="Beta Holdings")
        q.stage(ctx(), [model(), model(object_id="authority:z", source_locator=other)])
        assert len(q.list_pending(ctx(), document_id="doc-2")) == 1

    def test_filter_by_job(self):
        q = ReviewQueue()
        q.stage(ctx(), [model()], job_id="job-1")
        q.stage(ctx(), [model(object_id="authority:z")], job_id="job-2")
        assert len(q.list_pending(ctx(), job_id="job-1")) == 1

    def test_walled_matters_are_invisible_in_the_queue(self):
        q = ReviewQueue()
        q.stage(ctx(), [model(matter_id="matter-1"), model(object_id="a", matter_id="matter-2")])
        walled = ctx(matter_denylist=frozenset({"matter-2"}))
        assert [i.matter_id for i in q.list_pending(walled)] == ["matter-1"]

    def test_another_tenant_sees_nothing(self):
        q = ReviewQueue()
        q.stage(ctx(), [model()])
        assert q.list_pending(ctx(tenant="firm-other")) == []

    def test_limit_is_applied(self):
        q = ReviewQueue()
        q.stage(ctx(), [model(object_id=f"authority:{i}") for i in range(10)])
        assert len(q.list_pending(ctx(), limit=3)) == 3


class TestApproval:
    def test_approving_records_who_and_when(self):
        q = ReviewQueue()
        a = model()
        q.stage(ctx(), [a])
        record = q.approve(ctx(user="alice"), a.assertion_id, note="checked the passage")
        assert record.assertion.review_state is ReviewState.APPROVED
        assert record.assertion.reviewed_by == "alice"
        assert record.assertion.reviewed_at
        assert record.review_note == "checked the passage"

    def test_approving_is_idempotent(self):
        q = ReviewQueue()
        a = model()
        q.stage(ctx(), [a])
        first = q.approve(ctx(), a.assertion_id).assertion.reviewed_at
        assert q.approve(ctx(), a.assertion_id).assertion.reviewed_at == first

    def test_cannot_approve_an_auto_asserted_assertion(self):
        """It would imply a human checked something they did not."""
        q = ReviewQueue()
        a = det()
        q.stage(ctx(), [a])
        with pytest.raises(ReviewError, match="needs no approval"):
            q.approve(ctx(), a.assertion_id)

    def test_cannot_revive_a_rejected_assertion(self):
        q = ReviewQueue()
        a = model()
        q.stage(ctx(), [a])
        q.reject(ctx(), a.assertion_id, reason="hallucinated")
        with pytest.raises(ReviewError, match="re-extract"):
            q.approve(ctx(), a.assertion_id)

    def test_approving_across_a_wall_is_refused(self):
        q = ReviewQueue()
        a = model(matter_id="matter-9")
        q.stage(ctx(), [a])
        walled = ctx(matter_denylist=frozenset({"matter-9"}))
        with pytest.raises(ScopeViolation):
            q.approve(walled, a.assertion_id)

    def test_unknown_assertion_is_vague(self):
        """Distinguishing walled-off from nonexistent leaks matter existence."""
        q = ReviewQueue()
        with pytest.raises(ReviewError, match="no assertion"):
            q.approve(ctx(), "nope")

    def test_approve_many(self):
        q = ReviewQueue()
        assertions = [model(object_id=f"authority:{i}") for i in range(3)]
        q.stage(ctx(), assertions)
        approved = q.approve_many(ctx(), [a.assertion_id for a in assertions])
        assert len(approved) == 3
        assert q.pending_count(ctx()) == 0


class TestRejection:
    def test_rejection_requires_a_reason(self):
        q = ReviewQueue()
        a = model()
        q.stage(ctx(), [a])
        with pytest.raises(ReviewError, match="reason"):
            q.reject(ctx(), a.assertion_id, reason="")

    def test_rejected_assertions_are_kept_not_deleted(self):
        """A rejected extraction is evidence about the extractor."""
        q = ReviewQueue()
        a = model()
        q.stage(ctx(), [a])
        q.reject(ctx(), a.assertion_id, reason="the passage says the opposite")
        record = q.fetch(ctx(), a.assertion_id)
        assert record.assertion.review_state is ReviewState.REJECTED
        assert record.lifecycle is Lifecycle.DISCARDED
        assert record.review_note == "the passage says the opposite"

    def test_rejection_removes_it_from_the_queue(self):
        q = ReviewQueue()
        a = model()
        q.stage(ctx(), [a])
        q.reject(ctx(), a.assertion_id, reason="wrong")
        assert q.list_pending(ctx()) == []

    def test_rejecting_a_live_assertion_points_at_retract(self):
        """A live claim may have inferences resting on it; rejection does not cascade."""
        q = ReviewQueue()
        a = model()
        q.stage(ctx(), [a])
        q.approve(ctx(), a.assertion_id)
        with pytest.raises(ReviewError, match="retract"):
            q.reject(ctx(), a.assertion_id, reason="changed my mind")


class TestPromotion:
    def test_auto_asserted_promotes_without_review(self):
        q = ReviewQueue()
        a = det()
        q.stage(ctx(), [a], job_id="job-1")
        assert q.promote(ctx(), job_id="job-1") == [a.assertion_id]

    def test_pending_model_extraction_does_not_promote(self):
        """This is the gate working, not a partial failure."""
        q = ReviewQueue()
        a = model()
        q.stage(ctx(), [a], job_id="job-1")
        assert q.promote(ctx(), job_id="job-1") == []
        assert q.fetch(ctx(), a.assertion_id).lifecycle is Lifecycle.STAGED

    def test_approved_model_extraction_promotes(self):
        q = ReviewQueue()
        a = model()
        q.stage(ctx(), [a], job_id="job-1")
        q.approve(ctx(), a.assertion_id)
        assert q.promote(ctx(), job_id="job-1") == [a.assertion_id]
        assert q.fetch(ctx(), a.assertion_id).lifecycle is Lifecycle.LIVE

    def test_rejected_never_promotes(self):
        q = ReviewQueue()
        a = model()
        q.stage(ctx(), [a], job_id="job-1")
        q.reject(ctx(), a.assertion_id, reason="wrong")
        assert q.promote(ctx(), job_id="job-1") == []

    def test_promotion_is_idempotent(self):
        q = ReviewQueue()
        q.stage(ctx(), [det()], job_id="job-1")
        assert len(q.promote(ctx(), job_id="job-1")) == 1
        assert q.promote(ctx(), job_id="job-1") == []

    def test_promotion_is_scoped_to_the_job(self):
        q = ReviewQueue()
        q.stage(ctx(), [det()], job_id="job-1")
        q.stage(ctx(), [det(object_id="authority:z")], job_id="job-2")
        assert len(q.promote(ctx(), job_id="job-1")) == 1
        assert len(q.live_assertions(ctx())) == 1

    def test_mixed_batch_promotes_only_the_signed_off(self):
        q = ReviewQueue()
        approved, pending, auto = model(object_id="a"), model(object_id="b"), det()
        q.stage(ctx(), [approved, pending, auto], job_id="job-1")
        q.approve(ctx(), approved.assertion_id)
        promoted = set(q.promote(ctx(), job_id="job-1"))
        assert promoted == {approved.assertion_id, auto.assertion_id}
        assert pending.assertion_id not in promoted

    def test_another_tenant_cannot_promote_your_staging(self):
        q = ReviewQueue()
        q.stage(ctx(), [det()], job_id="job-1")
        assert q.promote(ctx(tenant="firm-other"), job_id="job-1") == []
        assert len(q.live_assertions(ctx())) == 0


class TestPipelineShape:
    def test_full_path_from_extraction_to_live(self):
        q = ReviewQueue()
        citation, interpretation = det(), model()
        q.stage(ctx(), [citation, interpretation], job_id="job-1")

        # The deterministic citation is live immediately; the model claim waits.
        assert q.promote(ctx(), job_id="job-1") == [citation.assertion_id]
        assert q.pending_count(ctx()) == 1

        q.approve(ctx(user="alice"), interpretation.assertion_id)
        assert q.promote(ctx(), job_id="job-1") == [interpretation.assertion_id]
        assert {r.assertion_id for r in q.live_assertions(ctx())} == {
            citation.assertion_id,
            interpretation.assertion_id,
        }

    def test_auto_asserted_ids_lists_the_reproducible_ones(self):
        q = ReviewQueue()
        citation = det()
        q.stage(ctx(), [citation, model()])
        assert q.auto_asserted_ids(ctx()) == [citation.assertion_id]
