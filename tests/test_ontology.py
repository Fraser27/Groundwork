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
from src.ontology.loader import available_domains, load_ontology

LOC = SourceLocator(document_id="doc-1", filename="a.pdf", page=1, quote="Acme Corporation")

#: Every pack on disk, discovered rather than listed. A pack added to the directory and to no test
#: is a pack nobody checks, which is how the abstraction rots without anything going red.
ALL_PACKS = available_domains()


class TestPacksLoad:
    def test_more_than_one_pack_exists(self):
        """The parametrised cases below pass vacuously if the glob finds nothing, and a single pack
        cannot demonstrate that anything is domain-agnostic."""
        assert len(ALL_PACKS) >= 2, ALL_PACKS

    @pytest.mark.parametrize("domain", ALL_PACKS)
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


class TestTransitiveIsDeclaredNotAssumed:
    """Which predicates a rule may walk as a chain is a claim about the world, so the pack
    makes it and the engine does not guess."""

    def test_affiliation_chains(self):
        assert "AFFILIATE_OF" in load_ontology("legal").transitive_predicates

    def test_adversity_does_not_chain(self):
        """Opposing A, who opposes B, does not put the firm against B. Declaring this transitive
        would manufacture conflicts out of unrelated litigation. It is also Matter -> Party now, so
        the object of one hop is not a legal subject of the next and the chain has nowhere to go."""
        assert "ADVERSE_TO" not in load_ontology("legal").transitive_predicates

    def test_representation_does_not_chain(self):
        assert "REPRESENTS" not in load_ontology("legal").transitive_predicates

    def test_nothing_is_transitive_by_default(self):
        assert load_ontology("healthcare").transitive_predicates == frozenset()

    def test_a_descriptive_predicate_may_not_chain(self, tmp_path):
        """The descriptive half is open, so an extractor inventing a tag could mint the links
        of the chain. A path has to run over the closed vocabulary."""
        from src.ontology import loader

        pack = tmp_path / "bad.yaml"
        pack.write_text(
            "domain: bad\nversion: 1\nentity_types: []\ngoverning_predicates: []\n"
            "descriptive_predicates:\n  - id: TAG\n    transitive: true\nrules: []\n"
        )
        with pytest.raises(ValueError, match="descriptive"):
            loader._parse(__import__("yaml").safe_load(pack.read_text()))

    def test_a_chain_with_nothing_to_continue_into_is_refused(self, tmp_path):
        """Counsel->Party stops after one hop. Declared transitive it would be a silent no-op
        rather than an error, which is the failure mode this whole pack refuses."""
        from src.ontology import loader

        pack = tmp_path / "bad2.yaml"
        pack.write_text(
            "domain: bad\nversion: 1\nentity_types: []\n"
            "governing_predicates:\n  - id: ACTS_FOR\n    domain: [Counsel]\n"
            "    range: [Party]\n    transitive: true\n"
            "descriptive_predicates: []\nrules: []\n"
        )
        with pytest.raises(ValueError, match="do not overlap"):
            loader._parse(__import__("yaml").safe_load(pack.read_text()))

    def test_an_overlapping_domain_and_range_is_accepted(self, tmp_path):
        from src.ontology import loader

        pack = tmp_path / "ok.yaml"
        pack.write_text(
            "domain: ok\nversion: 1\nentity_types: []\n"
            "governing_predicates:\n  - id: OWNS\n    domain: [Party]\n"
            "    range: [Party]\n    transitive: true\n"
            "descriptive_predicates: []\nrules: []\n"
        )
        onto = loader._parse(__import__("yaml").safe_load(pack.read_text()))
        assert onto.transitive_predicates == frozenset({"OWNS"})


class TestSymmetricNeedsEqualEnds:
    """Interchangeable endpoints and unequal kinds cannot both be true.

    `canonical_pair` acts on `symmetric: true` by byte-sorting the endpoints before anything reads
    the edge, so with unequal kinds the sort *invents* an orientation the pack never declared.
    `ADVERSE_TO` was `[Party, Matter] -> [Party]` and symmetric: a matter subject survived only when
    its id happened to sort first, and `conflict_check` binding `(m:Matter)` bound it to a party and
    flagged the firm's own client. The pack is where that has to be caught, because by the time a
    rule matches, the orientation is already gone.
    """

    def test_the_shipped_packs_are_coherent(self):
        for domain in ALL_PACKS:
            onto = load_ontology(domain)
            for pdef in onto.predicates.values():
                if pdef.symmetric:
                    assert set(pdef.domain) == set(pdef.range), f"{domain}: {pdef.id}"

    def test_unequal_ends_are_refused(self, tmp_path):
        """The shape `ADVERSE_TO` had. Loud at load, so the class cannot come back through a
        different predicate."""
        from src.ontology import loader

        pack = tmp_path / "bad.yaml"
        pack.write_text(
            "domain: bad\nversion: 1\nentity_types: []\n"
            "governing_predicates:\n  - id: OPPOSES\n    domain: [Party, Matter]\n"
            "    range: [Party]\n    symmetric: true\n"
            "descriptive_predicates: []\nrules: []\n"
        )
        with pytest.raises(ValueError, match="symmetric"):
            loader._parse(__import__("yaml").safe_load(pack.read_text()))

    def test_a_partial_overlap_is_still_refused(self, tmp_path):
        """Equality, not overlap. Overlap admits one orientation the pack declared and one
        canonicalisation mints, which is the same defect with a smaller blast radius."""
        from src.ontology import loader

        pack = tmp_path / "bad2.yaml"
        pack.write_text(
            "domain: bad\nversion: 1\nentity_types: []\n"
            "governing_predicates:\n  - id: OPPOSES\n    domain: [Party]\n"
            "    range: [Party, Matter]\n    symmetric: true\n"
            "descriptive_predicates: []\nrules: []\n"
        )
        with pytest.raises(ValueError, match="symmetric"):
            loader._parse(__import__("yaml").safe_load(pack.read_text()))

    def test_equal_ends_are_accepted(self, tmp_path):
        from src.ontology import loader

        pack = tmp_path / "ok.yaml"
        pack.write_text(
            "domain: ok\nversion: 1\nentity_types: []\n"
            "governing_predicates:\n  - id: SIBLING_OF\n    domain: [Party]\n"
            "    range: [Party]\n    symmetric: true\n"
            "descriptive_predicates: []\nrules: []\n"
        )
        assert loader._parse(__import__("yaml").safe_load(pack.read_text())).predicates[
            "SIBLING_OF"
        ].symmetric

    def test_an_asymmetric_predicate_may_have_unequal_ends(self, tmp_path):
        """The validator must only fire on the combination. Most governing predicates relate two
        different kinds and that is ordinary."""
        from src.ontology import loader

        pack = tmp_path / "ok2.yaml"
        pack.write_text(
            "domain: ok\nversion: 1\nentity_types: []\n"
            "governing_predicates:\n  - id: ACTS_FOR\n    domain: [Counsel]\n"
            "    range: [Party]\n"
            "descriptive_predicates: []\nrules: []\n"
        )
        assert not loader._parse(
            __import__("yaml").safe_load(pack.read_text())
        ).predicates["ACTS_FOR"].symmetric

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
