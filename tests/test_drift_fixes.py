"""Tests for six pieces of drift between the code and its own documentation.

These are worth keeping because each one was invisible: the code did something other
than what the docs, the UI or a config field claimed. A wrong explanation is a bug —
an administrator who trusts a toggle that does nothing has been misled by the system.
"""

from __future__ import annotations

import pytest

from src.governance import GovernanceSettings
from src.graph.assertions import (
    DEFAULT_REVIEW_POLICY,
    EpistemicClass,
    ReviewPolicy,
    ReviewState,
    SourceLocator,
    build_assertion,
)
from src.ontology.loader import load_ontology

DOC = SourceLocator(document_id="doc-1", filename="memo.pdf", page=2, quote="Acme Corporation")
TBL = SourceLocator(source_id="cms-1", table="matters")


def _verified(**over):
    kw = dict(
        tenant_id="firm-acme",
        subject_id="document:doc-1",
        predicate="MENTIONS",
        object_id="party:acme",
        epistemic_class=EpistemicClass.EXTRACTED_DET,
        method="llm:opus-5+verify:quote@v1",
        confidence=0.95,
        source_locator=DOC,
    )
    kw.update(over)
    return build_assertion(**kw)


class TestReviewPolicyIsActuallyRead:
    """These settings were stored, validated, warned about in the UI — and read by
    nothing. An administrator could switch one and be wrong about the consequence."""

    def test_default_auto_asserts_verified_presence(self):
        assert _verified().review_state is ReviewState.AUTO_ASSERTED

    def test_turning_off_auto_assert_sends_verified_claims_to_review(self):
        policy = ReviewPolicy(auto_assert_verified=False)
        assert _verified(policy=policy).review_state is ReviewState.PENDING

    def test_governing_predicate_review_catches_declared_facts(self):
        """The case the presence rule alone misses: a case-management export asserting
        adversity would otherwise drive a conflict check with nobody having read it."""
        policy = ReviewPolicy(governing_predicates=frozenset({"ADVERSE_TO"}))
        a = build_assertion(
            tenant_id="firm-acme",
            subject_id="party:a",
            predicate="ADVERSE_TO",
            object_id="party:b",
            epistemic_class=EpistemicClass.DECLARED,
            method="cms:matter_export",
            confidence=1.0,
            source_locator=TBL,
            policy=policy,
        )
        assert a.review_state is ReviewState.PENDING

    def test_governing_review_can_be_relaxed(self):
        policy = ReviewPolicy(
            require_review_for_governing=False,
            governing_predicates=frozenset({"ADVERSE_TO"}),
        )
        a = build_assertion(
            tenant_id="firm-acme",
            subject_id="party:a",
            predicate="ADVERSE_TO",
            object_id="party:b",
            epistemic_class=EpistemicClass.DECLARED,
            method="cms:matter_export",
            confidence=1.0,
            source_locator=TBL,
            policy=policy,
        )
        assert a.review_state is ReviewState.AUTO_ASSERTED

    def test_policy_can_never_make_a_model_claim_live(self):
        """One-directional by design: a setting may add review, never remove it."""
        permissive = ReviewPolicy(auto_assert_verified=True, require_review_for_governing=False)
        a = _verified(
            epistemic_class=EpistemicClass.EXTRACTED_MODEL,
            method="llm:opus-5",
            confidence=0.7,
            policy=permissive,
        )
        assert a.review_state is ReviewState.PENDING

    def test_absent_policy_keeps_the_safe_default(self):
        assert DEFAULT_REVIEW_POLICY.auto_assert_verified is True
        assert DEFAULT_REVIEW_POLICY.require_review_for_governing is True


class TestModelConfidenceCapIsApplied:
    def test_cap_clamps_an_overconfident_model(self):
        assert GovernanceSettings().effective_model_confidence(0.99) == 0.79

    def test_lowered_cap_takes_effect(self):
        s = GovernanceSettings(min_confidence_floor=0.6, model_confidence_cap=0.3)
        assert s.effective_model_confidence(0.9) == 0.3


class TestSymmetricPredicatesCollapse:
    """`symmetric: true` was parsed and exposed through the API but enforced nowhere, so
    one relationship could occupy two edges with two different content hashes."""

    def test_adverse_to_is_order_independent(self):
        o = load_ontology("legal")
        assert o.canonical_pair("ADVERSE_TO", "party:zeta", "party:alpha") == o.canonical_pair(
            "ADVERSE_TO", "party:alpha", "party:zeta"
        )

    def test_asymmetric_predicate_keeps_its_direction(self):
        """REPRESENTS is not symmetric — counsel represents a party, not the reverse."""
        o = load_ontology("legal")
        assert o.canonical_pair("REPRESENTS", "counsel:x", "party:y") == (
            "counsel:x",
            "party:y",
        )

    def test_unknown_predicate_is_left_alone(self):
        o = load_ontology("legal")
        assert o.canonical_pair("NOT_A_PREDICATE", "b", "a") == ("b", "a")

    def test_collapsing_yields_one_assertion_id(self):
        """The consequence that matters: a conflict check reads one edge, not two."""
        o = load_ontology("legal")

        def mk(subj, obj):
            s, t = o.canonical_pair("ADVERSE_TO", subj, obj)
            return build_assertion(
                tenant_id="firm-acme",
                subject_id=s,
                predicate="ADVERSE_TO",
                object_id=t,
                epistemic_class=EpistemicClass.EXTRACTED_MODEL,
                method="llm:opus-5",
                confidence=0.7,
                source_locator=DOC,
            ).assertion_id

        assert mk("party:alpha", "party:zeta") == mk("party:zeta", "party:alpha")


class TestRulePremiseFloorIsReadable:
    @pytest.mark.parametrize("rule_id", ["conflict_check", "authority_stale"])
    def test_every_rule_declares_a_floor(self, rule_id):
        assert load_ontology("legal").rule_premise_floor(rule_id) is not None

    @pytest.mark.parametrize("rule_id", ["conflict_check", "authority_stale"])
    def test_the_declared_floor_is_satisfiable(self, rule_id):
        """A floor no real fact can meet is a rule that silently never fires.

        Both rules declared EXTRACTED_DET, which no governing predicate may ever carry:
        `build_assertion` restricts that class to presence predicates, because a quote match
        proves text is present and never that it is significant. So the floor was unsatisfiable
        for anything read from a document, and neither rule could have fired on the unstructured
        half of the product. What actually keeps a conflict flag honest is the reasoner's review
        gate, not the epistemic class.
        """
        from src.graph.assertions import EpistemicClass
        from src.reasoning.engine import _STRENGTH

        floor = load_ontology("legal").rule_premise_floor(rule_id)
        assert _STRENGTH[EpistemicClass.EXTRACTED_MODEL] <= _STRENGTH[EpistemicClass(floor)]

    def test_unknown_rule_has_no_floor(self):
        assert load_ontology("legal").rule_premise_floor("nope") is None


class TestModelIdsAgree:
    def test_governance_matches_constants(self):
        """These diverged: the pipeline read config while the UI showed governance, so an
        administrator saw a model id the system was not using."""
        from src.constants import DEFAULT_EXTRACTION_MODEL

        assert GovernanceSettings().extraction_model == DEFAULT_EXTRACTION_MODEL


class TestOntologyHelpTextMatchesBehaviour:
    def test_cites_help_does_not_promise_auto_assertion(self):
        """`build_assertion` refuses CITES as EXTRACTED_DET, so help text claiming the
        parser produces it deterministically described the opposite of the code."""
        help_text = load_ontology("legal").predicates["CITES"].help or ""
        assert "parser" not in help_text.lower()
        assert "review" in help_text.lower()

    def test_cites_is_refused_as_verified_presence(self):
        from src.graph.assertions import AssertionError_

        with pytest.raises(AssertionError_, match="only that text appears"):
            _verified(predicate="CITES")
