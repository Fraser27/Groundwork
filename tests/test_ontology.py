"""Tests for ontology packs and the predicate gate.

The healthcare cases exist to keep domain-agnosticism honest. Legal is the focus,
but if a second pack stops loading we want a red test rather than a surprised
customer.
"""

from __future__ import annotations

import pytest

from src.graph.assertions import (
    AssertionError_,
    EpistemicClass,
    SourceLocator,
    build_assertion,
)
from src.ontology.loader import load_ontology

LOC = SourceLocator(document_id="doc-1", filename="a.pdf", page=1, quote="Acme Corporation")


class TestPacksLoad:
    @pytest.mark.parametrize("domain", ["legal", "healthcare"])
    def test_pack_loads(self, domain):
        o = load_ontology(domain)
        assert o.domain == domain
        assert o.entities and o.governing_predicates and o.rules

    def test_unknown_domain_lists_alternatives(self):
        with pytest.raises(FileNotFoundError, match="available:"):
            load_ontology("aerospace")

    def test_rule_method_is_versioned(self):
        """INFERRED assertions must name the exact rule version that produced them."""
        rule = load_ontology("legal").rules[0]
        assert rule.method == f"rule:{rule.id}@{rule.version}"


class TestTwoTierVocabulary:
    def test_governing_and_descriptive_are_disjoint(self):
        o = load_ontology("legal")
        assert not (o.governing_predicates & o.descriptive_predicates)

    @pytest.mark.parametrize(
        "pid", ["REPRESENTS", "ADVERSE_TO", "CITES", "SUBJECT_TO_PRIVILEGE", "DEADLINE_FOR"]
    )
    def test_consequential_predicates_are_governing(self, pid):
        """Conflicts, privilege, deadlines, citation authority — all closed."""
        assert load_ontology("legal").is_governing(pid)

    @pytest.mark.parametrize("pid", ["CONCERNS_TOPIC", "IN_INDUSTRY", "MENTIONS"])
    def test_descriptive_predicates_are_open(self, pid):
        o = load_ontology("legal")
        assert not o.is_governing(pid)
        assert o.allowed_for(pid) is None

    def test_healthcare_applies_the_same_test(self):
        """Direct-harm predicates are governing in healthcare too."""
        o = load_ontology("healthcare")
        assert o.is_governing("CONTRAINDICATED_WITH")
        assert o.is_governing("ALLERGIC_TO")
        assert not o.is_governing("IN_SPECIALTY")


class TestGroundingIsDeterministic:
    """STANDARD mode only: exact match, no LLM adjudicating schema."""

    @pytest.mark.parametrize(
        "probe,expected",
        [
            ("REPRESENTS", "REPRESENTS"),
            ("represents", "REPRESENTS"),
            ("adverse to", "ADVERSE_TO"),
            ("adverse-to", "ADVERSE_TO"),
        ],
    )
    def test_exact_and_label_matches_ground(self, probe, expected):
        assert load_ontology("legal").ground(probe) == expected

    @pytest.mark.parametrize("probe", ["is_counsel_to", "acts_on_behalf_of", "retained_by"])
    def test_semantic_synonyms_do_not_ground(self, probe):
        """These *mean* REPRESENTS, but only an LLM could say so — so we refuse."""
        assert load_ontology("legal").ground(probe) is None


class TestGateIntegration:
    """The ontology and the assertion contract must actually meet."""

    def test_governing_synonym_is_rejected_at_write(self):
        o = load_ontology("legal")
        with pytest.raises(AssertionError_, match="closed vocabulary"):
            build_assertion(
                tenant_id="firm-acme",
                subject_id="Counsel-1",
                predicate="is_counsel_to",
                object_id="Party-Acme",
                epistemic_class=EpistemicClass.EXTRACTED_MODEL,
                method="llm:claude-sonnet-5",
                confidence=0.9,
                source_locator=LOC,
                allowed_predicates=o.allowed_for("REPRESENTS"),
            )

    def test_approved_governing_predicate_writes(self):
        o = load_ontology("legal")
        a = build_assertion(
            tenant_id="firm-acme",
            subject_id="Counsel-1",
            predicate="REPRESENTS",
            object_id="Party-Acme",
            epistemic_class=EpistemicClass.DECLARED,
            method="cms:matter_export",
            confidence=1.0,
            source_locator=SourceLocator(source_id="src-1", table="matters"),
            allowed_predicates=o.allowed_for("REPRESENTS"),
        )
        assert a.predicate == "REPRESENTS"

    def test_descriptive_predicate_needs_no_approval(self):
        o = load_ontology("legal")
        a = build_assertion(
            tenant_id="firm-acme",
            subject_id="Doc-1",
            predicate="CONCERNS_TOPIC",
            object_id="Topic-Antitrust",
            epistemic_class=EpistemicClass.EXTRACTED_MODEL,
            method="llm:claude-sonnet-5",
            confidence=0.7,
            source_locator=LOC,
            allowed_predicates=o.allowed_for("CONCERNS_TOPIC"),
        )
        assert a.predicate == "CONCERNS_TOPIC"
