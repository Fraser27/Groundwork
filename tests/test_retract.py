"""Tests for cascading retraction — invariant 5.

The cascade is the point. The scenario these tests defend, in the words a regulator
would use:

    An extraction said Counsel represents Party. A rule inferred a conflict from it.
    The extraction turned out to be wrong and was retracted. Does the graph still
    assert the conflict?

If it does, every "why do you believe this?" answer for that flag cites a premise that
has been withdrawn — which is precisely the question the system exists to answer well.

The second theme is that retraction is **append-only**. A bitemporal read from before
the retraction must still show the fact, and still show what it supported. "What did
the file show when we advised?" has to survive a correction.
"""

from __future__ import annotations

import pytest

from src.documents.review import AssertionRecord, Lifecycle, ReviewError, ReviewQueue
from src.documents.retract import (
    RetractionError,
    dependency_closure,
    explain_retraction,
    retract,
    retract_document,
    supersede,
    would_cascade,
)
from src.graph.assertions import (
    EpistemicClass,
    SourceLocator,
    build_assertion,
)
from src.graph.scope import AuthContext, ScopeViolation

TENANT = "firm-acme"
LOC = SourceLocator(document_id="doc-1", filename="a.pdf", page=1, quote="clause 4 of the deed")


def ctx(user="reviewer-1", tenant=TENANT, **kw) -> AuthContext:
    return AuthContext(user_id=user, tenant_id=tenant, **kw)


def extracted(object_id, *, predicate="MENTIONS", confidence=0.95, **over):
    """A quote-verified presence claim, so it auto-asserts and can be a live premise."""
    kw = dict(
        tenant_id=TENANT,
        subject_id="counsel:smith",
        predicate=predicate,
        object_id=object_id,
        epistemic_class=EpistemicClass.EXTRACTED_DET,
        method="llm:opus-5+verify:quote@v1",
        confidence=confidence,
        source_locator=LOC,
    )
    kw.update(over)
    return build_assertion(**kw)


def inferred(object_id, premises, *, confidence=0.9, rule="conflict_check", **over):
    kw = dict(
        tenant_id=TENANT,
        subject_id="matter-4471",
        predicate="ADVERSE_TO",
        object_id=object_id,
        epistemic_class=EpistemicClass.INFERRED,
        method=f"rule:{rule}@v1",
        confidence=confidence,
        source_locator=LOC,
        premises=tuple(premises),
        rule_id=rule,
        rule_version="v1",
    )
    kw.update(over)
    return build_assertion(**kw)


def _chain(q: ReviewQueue, depth: int, *, live: bool = True):
    """A premise chain: root -> level1 -> level2 -> ... of length `depth`.

    INFERRED is not in AUTO_ASSERT_CLASSES, so each inference is approved explicitly
    to reach the live graph. That is the contract's choice — a rule's conclusion is
    only as reviewable as the rule.
    """
    root = extracted("party:acme")
    q.stage(ctx(), [root])
    chain = [root]
    for level in range(depth):
        node = inferred(
            f"party:derived-{level}",
            [chain[-1].assertion_id],
            confidence=0.9 - level * 0.01,
        )
        q.stage(ctx(), [node])
        if live:
            q.approve(ctx(), node.assertion_id)
        chain.append(node)
    if live:
        q.promote(ctx())
    return chain


class TestSingleRetraction:
    def test_retraction_supersedes_rather_than_deletes(self):
        q = ReviewQueue()
        a = extracted("party:acme")
        q.stage(ctx(), [a])
        q.promote(ctx())
        retract(q, ctx(), a.assertion_id, reason="wrong party")

        record = q.fetch(ctx(), a.assertion_id)
        assert record.assertion.superseded_at is not None
        assert not record.is_current
        # Still in the store — a retracted fact remains part of the audit record.
        assert q.store.get(TENANT, a.assertion_id) is not None

    def test_retraction_records_who_and_why(self):
        q = ReviewQueue()
        a = extracted("party:acme")
        q.stage(ctx(), [a])
        result = retract(q, ctx(user="alice"), a.assertion_id, reason="OCR misread the caption")
        assert result.retracted_by == "alice"
        assert result.reason == "OCR misread the caption"
        record = q.fetch(ctx(), a.assertion_id)
        assert record.retracted_by == "alice"
        assert record.retracted_reason == "OCR misread the caption"

    def test_reason_is_mandatory(self):
        q = ReviewQueue()
        a = extracted("party:acme")
        q.stage(ctx(), [a])
        with pytest.raises(RetractionError, match="must carry a reason"):
            retract(q, ctx(), a.assertion_id, reason="")

    def test_double_retraction_is_refused(self):
        """Re-stamping would rewrite when we actually stopped believing it."""
        q = ReviewQueue()
        a = extracted("party:acme")
        q.stage(ctx(), [a])
        retract(q, ctx(), a.assertion_id, reason="first")
        with pytest.raises(RetractionError, match="already retracted"):
            retract(q, ctx(), a.assertion_id, reason="second")

    def test_retracting_across_a_wall_is_refused(self):
        q = ReviewQueue()
        a = extracted("party:acme", matter_id="matter-9")
        q.stage(ctx(), [a])
        walled = ctx(matter_denylist=frozenset({"matter-9"}))
        with pytest.raises(ScopeViolation):
            retract(q, walled, a.assertion_id, reason="nope")

    def test_live_assertion_keeps_its_lifecycle(self):
        """Rewriting lifecycle would lose that this was once believed and acted upon."""
        q = ReviewQueue()
        a = extracted("party:acme")
        q.stage(ctx(), [a])
        q.promote(ctx())
        retract(q, ctx(), a.assertion_id, reason="wrong")
        assert q.fetch(ctx(), a.assertion_id).lifecycle is Lifecycle.LIVE

    def test_staged_assertion_is_discarded(self):
        q = ReviewQueue()
        a = extracted("party:acme")
        q.stage(ctx(), [a])
        retract(q, ctx(), a.assertion_id, reason="wrong")
        assert q.fetch(ctx(), a.assertion_id).lifecycle is Lifecycle.DISCARDED


class TestCascade:
    def test_one_level_cascade(self):
        """The headline case: retract the premise, the conflict flag falls with it."""
        q = ReviewQueue()
        premise = extracted("party:acme")
        q.stage(ctx(), [premise])
        conflict = inferred("party:acme", [premise.assertion_id])
        q.stage(ctx(), [conflict])
        q.promote(ctx())

        result = retract(q, ctx(), premise.assertion_id, reason="misidentified party")

        assert result.cascaded == (conflict.assertion_id,)
        assert not q.fetch(ctx(), conflict.assertion_id).is_current

    def test_cascade_is_transitive(self):
        q = ReviewQueue()
        chain = _chain(q, depth=4)
        result = retract(q, ctx(), chain[0].assertion_id, reason="root was wrong")
        assert len(result.cascaded) == 4
        for node in chain:
            assert not q.fetch(ctx(), node.assertion_id).is_current

    def test_cascade_reaches_a_diamond_once(self):
        """Two independent inferences, then one resting on both."""
        q = ReviewQueue()
        root = extracted("party:acme")
        q.stage(ctx(), [root])
        left = inferred("party:left", [root.assertion_id], confidence=0.9)
        right = inferred("party:right", [root.assertion_id], confidence=0.85)
        q.stage(ctx(), [left, right])
        joint = inferred(
            "party:joint",
            [left.assertion_id, right.assertion_id],
            confidence=0.8,
            rule="authority_stale",
        )
        q.stage(ctx(), [joint])
        q.promote(ctx())

        result = retract(q, ctx(), root.assertion_id, reason="root wrong")
        assert set(result.cascaded) == {
            left.assertion_id,
            right.assertion_id,
            joint.assertion_id,
        }
        assert len(result.cascaded) == 3, "an assertion must not be retracted twice"

    def test_cascade_does_not_touch_unrelated_assertions(self):
        q = ReviewQueue()
        a = extracted("party:acme")
        b = extracted(
            "party:beta",
            predicate="ADVERSE_TO",
            epistemic_class=EpistemicClass.EXTRACTED_MODEL,
        )
        q.stage(ctx(), [a, b])
        from_b = inferred("party:beta", [b.assertion_id])
        q.stage(ctx(), [from_b])
        q.promote(ctx())

        retract(q, ctx(), a.assertion_id, reason="wrong")
        assert q.fetch(ctx(), b.assertion_id).is_current
        assert q.fetch(ctx(), from_b.assertion_id).is_current

    def test_cascade_terminates_on_a_cycle(self):
        """A buggy rule must not hang the correction path."""
        q = ReviewQueue()
        root = extracted("party:acme")
        q.stage(ctx(), [root])
        first = inferred("party:x", [root.assertion_id])
        q.stage(ctx(), [first])
        # A rule inferring a fact from its own consequence. Should not exist; must not
        # loop if it does.
        loop = inferred("party:y", [first.assertion_id], confidence=0.8)
        q.stage(ctx(), [loop])
        q.store._dependents.setdefault((TENANT, loop.assertion_id), set()).add(
            first.assertion_id
        )

        result = retract(q, ctx(), root.assertion_id, reason="wrong")
        assert set(result.cascaded) == {first.assertion_id, loop.assertion_id}

    def test_cascade_reason_names_the_retracted_premise(self):
        q = ReviewQueue()
        premise = extracted("party:acme")
        q.stage(ctx(), [premise])
        conflict = inferred("party:acme", [premise.assertion_id])
        q.stage(ctx(), [conflict])

        retract(q, ctx(), premise.assertion_id, reason="bad OCR")
        note = q.fetch(ctx(), conflict.assertion_id).retracted_reason
        assert premise.assertion_id in note
        assert "bad OCR" in note

    def test_already_retracted_dependents_are_not_restamped(self):
        q = ReviewQueue()
        root = extracted("party:acme")
        q.stage(ctx(), [root])
        child = inferred("party:x", [root.assertion_id])
        q.stage(ctx(), [child])
        q.promote(ctx())

        retract(q, ctx(), child.assertion_id, reason="child wrong on its own terms")
        first_at = q.fetch(ctx(), child.assertion_id).assertion.superseded_at

        result = retract(q, ctx(), root.assertion_id, reason="root also wrong")
        assert child.assertion_id not in result.cascaded
        assert q.fetch(ctx(), child.assertion_id).assertion.superseded_at == first_at

    def test_orphaned_rules_are_reported(self):
        q = ReviewQueue()
        root = extracted("party:acme")
        q.stage(ctx(), [root])
        q.stage(
            ctx(),
            [
                inferred("party:x", [root.assertion_id], rule="conflict_check"),
                inferred("party:y", [root.assertion_id], rule="authority_stale"),
            ],
        )
        result = retract(q, ctx(), root.assertion_id, reason="wrong")
        assert result.orphaned_rules == ("authority_stale", "conflict_check")

    def test_total_counts_root_plus_cascade(self):
        q = ReviewQueue()
        chain = _chain(q, depth=3)
        assert retract(q, ctx(), chain[0].assertion_id, reason="wrong").total == 4


class TestDryRun:
    def test_dry_run_reports_the_closure_without_writing(self):
        """A reviewer needs "this withdraws 4 conclusions" before deciding, not after."""
        q = ReviewQueue()
        chain = _chain(q, depth=3)
        result = retract(q, ctx(), chain[0].assertion_id, reason="checking", dry_run=True)

        assert len(result.cascaded) == 3
        for node in chain:
            assert q.fetch(ctx(), node.assertion_id).is_current

    def test_dry_run_matches_the_real_cascade(self):
        q = ReviewQueue()
        chain = _chain(q, depth=3)
        preview = retract(q, ctx(), chain[0].assertion_id, reason="x", dry_run=True)
        actual = retract(q, ctx(), chain[0].assertion_id, reason="x")
        assert set(preview.cascaded) == set(actual.cascaded)

    def test_would_cascade_unions_multiple_roots(self):
        q = ReviewQueue()
        a = extracted("party:a")
        b = extracted(
            "party:b",
            predicate="ADVERSE_TO",
            epistemic_class=EpistemicClass.EXTRACTED_MODEL,
        )
        q.stage(ctx(), [a, b])
        from_a = inferred("party:x", [a.assertion_id])
        from_b = inferred("party:y", [b.assertion_id])
        q.stage(ctx(), [from_a, from_b])

        affected = would_cascade(q, ctx(), [a.assertion_id, b.assertion_id])
        assert affected == {
            a.assertion_id,
            b.assertion_id,
            from_a.assertion_id,
            from_b.assertion_id,
        }


class TestClosure:
    def test_closure_excludes_the_root(self):
        q = ReviewQueue()
        chain = _chain(q, depth=2)
        closure = dependency_closure(q, ctx(), chain[0].assertion_id)
        assert chain[0].assertion_id not in {r.assertion_id for r in closure}
        assert len(closure) == 2

    def test_closure_of_a_leaf_is_empty(self):
        q = ReviewQueue()
        chain = _chain(q, depth=1)
        assert dependency_closure(q, ctx(), chain[-1].assertion_id) == []


class TestBitemporalSurvival:
    def test_history_before_the_retraction_is_intact(self):
        """"What did the file show when we advised?" must survive a correction."""
        q = ReviewQueue()
        premise = extracted("party:acme")
        q.stage(ctx(), [premise])
        conflict = inferred("party:acme", [premise.assertion_id])
        q.stage(ctx(), [conflict])
        q.promote(ctx())
        recorded_at = q.fetch(ctx(), conflict.assertion_id).assertion.recorded_at

        retract(q, ctx(), premise.assertion_id, reason="wrong")

        record = q.fetch(ctx(), conflict.assertion_id)
        assert record.assertion.recorded_at == recorded_at
        assert record.assertion.superseded_at > recorded_at
        assert record.assertion.premises == (premise.assertion_id,)

    def test_root_and_cascade_share_one_timestamp(self):
        """One decision, one instant — not N separate events."""
        q = ReviewQueue()
        chain = _chain(q, depth=3)
        retract(q, ctx(), chain[0].assertion_id, reason="wrong")
        stamps = {q.fetch(ctx(), n.assertion_id).assertion.superseded_at for n in chain}
        assert len(stamps) == 1

    def test_retracted_assertions_leave_the_live_set(self):
        q = ReviewQueue()
        chain = _chain(q, depth=2)
        assert len(q.live_assertions(ctx())) == 3
        retract(q, ctx(), chain[0].assertion_id, reason="wrong")
        assert q.live_assertions(ctx()) == []

    def test_inferences_need_review_before_going_live(self):
        """INFERRED is not auto-asserted: a conclusion is as reviewable as its rule."""
        q = ReviewQueue()
        chain = _chain(q, depth=1, live=False)
        assert q.promote(ctx()) == [chain[0].assertion_id]
        assert [i.assertion_id for i in q.list_pending(ctx())] == [chain[1].assertion_id]

    def test_cascade_reaches_pending_inferences_too(self):
        """A pending conclusion must not survive its premise being withdrawn."""
        q = ReviewQueue()
        chain = _chain(q, depth=2, live=False)
        result = retract(q, ctx(), chain[0].assertion_id, reason="wrong")
        assert len(result.cascaded) == 2
        for node in chain[1:]:
            assert not q.fetch(ctx(), node.assertion_id).is_current

    def test_cascaded_claims_leave_the_review_queue(self):
        """Adjudicating an already-withdrawn claim wastes the scarcest resource."""
        q = ReviewQueue()
        chain = _chain(q, depth=2, live=False)
        assert q.pending_count(ctx()) == 2
        retract(q, ctx(), chain[0].assertion_id, reason="wrong")
        assert q.list_pending(ctx()) == []
        assert q.pending_count(ctx()) == 0

    def test_a_retracted_claim_cannot_be_approved(self):
        q = ReviewQueue()
        chain = _chain(q, depth=1, live=False)
        retract(q, ctx(), chain[0].assertion_id, reason="premise was wrong")
        with pytest.raises(ReviewError, match="was retracted"):
            q.approve(ctx(), chain[1].assertion_id)


class TestRetractDocument:
    def test_every_assertion_from_a_document_goes(self):
        q = ReviewQueue()
        a = extracted("party:acme")
        b = extracted("authority:410-u-s-113", predicate="MENTIONS")
        q.stage(ctx(), [a, b])
        q.promote(ctx())

        results = retract_document(q, ctx(), "doc-1", reason="re-parsed with a fixed extractor")
        assert len(results) == 2
        assert not q.fetch(ctx(), a.assertion_id).is_current
        assert not q.fetch(ctx(), b.assertion_id).is_current

    def test_inferences_fall_via_the_cascade_not_as_roots(self):
        q = ReviewQueue()
        premise = extracted("party:acme")
        q.stage(ctx(), [premise])
        conflict = inferred("party:acme", [premise.assertion_id])
        q.stage(ctx(), [conflict])
        q.promote(ctx())

        results = retract_document(q, ctx(), "doc-1", reason="withdrawn")
        assert len(results) == 1
        assert results[0].cascaded == (conflict.assertion_id,)
        assert not q.fetch(ctx(), conflict.assertion_id).is_current

    def test_other_documents_untouched(self):
        q = ReviewQueue()
        mine = extracted("party:acme")
        other = extracted(
            "party:beta",
            source_locator=SourceLocator(document_id="doc-2", filename="b.pdf", page=1, quote="the Trustees"),
        )
        q.stage(ctx(), [mine, other])
        retract_document(q, ctx(), "doc-1", reason="withdrawn")
        assert q.fetch(ctx(), other.assertion_id).is_current


class TestSupersede:
    def test_new_generation_replaces_the_old_and_cascades(self):
        q = ReviewQueue()
        old = extracted("party:acme", method="regex:party_caption@v1")
        q.stage(ctx(), [old])
        derived = inferred("party:acme", [old.assertion_id])
        q.stage(ctx(), [derived])
        q.promote(ctx())

        new = extracted("party:acme-corporation", method="regex:party_caption@v2")
        supersede(
            q,
            ctx(),
            old_assertion_id=old.assertion_id,
            new_record=AssertionRecord(assertion=new),
            reason="v2 handles corporate suffixes",
        )

        assert not q.fetch(ctx(), old.assertion_id).is_current
        assert not q.fetch(ctx(), derived.assertion_id).is_current
        assert q.fetch(ctx(), new.assertion_id).is_current

    def test_cross_tenant_replacement_refused(self):
        q = ReviewQueue()
        old = extracted("party:acme")
        q.stage(ctx(), [old])
        alien = build_assertion(
            tenant_id="firm-other",
            subject_id="counsel:smith",
            predicate="REPRESENTS",
            object_id="party:acme",
            epistemic_class=EpistemicClass.EXTRACTED_MODEL,
            method="regex:party_caption@v2",
            confidence=0.9,
            source_locator=LOC,
        )
        with pytest.raises(RetractionError, match="another tenant"):
            supersede(
                q,
                ctx(),
                old_assertion_id=old.assertion_id,
                new_record=AssertionRecord(assertion=alien),
                reason="x",
            )


class TestExplain:
    def test_explains_a_cascaded_retraction(self):
        q = ReviewQueue()
        premise = extracted("party:acme")
        q.stage(ctx(), [premise])
        conflict = inferred("party:acme", [premise.assertion_id])
        q.stage(ctx(), [conflict])
        retract(q, ctx(user="alice"), premise.assertion_id, reason="bad OCR")

        explained = explain_retraction(q, ctx(), conflict.assertion_id)
        assert explained["is_current"] is False
        assert explained["retracted_by"] == "alice"
        assert explained["rule_id"] == "conflict_check"
        assert explained["premises"] == [premise.assertion_id]

    def test_explains_a_current_assertion(self):
        q = ReviewQueue()
        a = extracted("party:acme")
        q.stage(ctx(), [a])
        explained = explain_retraction(q, ctx(), a.assertion_id)
        assert explained["is_current"] is True
        assert explained["superseded_at"] is None
