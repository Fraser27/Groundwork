"""The reasoner, and the reasons it refuses to fire.

Two things are being tested, and the second matters more.

**That it fires.** The conflict check and the stale-authority check are the conclusions this
whole design exists to produce, and neither is visible in any single document — they exist only
in the join across matters.

**That it does not fire when it should not.** A conflict flag is acted on. One resting on an
unreviewed model guess, or on a fact since withdrawn, or on a cross product of unrelated
parties, is worse than no flag at all, because someone relies on it. Most of this file is that
half.

The healthcare cases are load-bearing rather than decorative: they run the same engine over a
different pack with no new Python, which is what "domain-agnostic by construction" has to mean
to be a real claim.
"""

from __future__ import annotations

import pytest

from src.graph.assertions import (
    EpistemicClass,
    ReviewState,
    SourceLocator,
    build_assertion,
)
from src.graph.scope import AuthContext
from src.ontology.loader import available_domains, load_ontology

ALL_PACKS = available_domains()
from src.ontology.patterns import MAX_PATH_HOPS, PatternError, parse_edge, parse_rule
from src.reasoning.engine import Reasoner, accepts

TENANT = "demo-firm"
NTL = "NTL-2026-0114"
MBC = "MBC-2024-0431"
HAL = "HAL-2025-0092"


@pytest.fixture
def ctx() -> AuthContext:
    return AuthContext(user_id="lawyer@firm.example", tenant_id=TENANT)


def fact(
    subject: str,
    predicate: str,
    obj: str,
    *,
    matter: str | None = NTL,
    epistemic_class: EpistemicClass = EpistemicClass.EXTRACTED_MODEL,
    confidence: float = 0.9,
    review_state: ReviewState = ReviewState.APPROVED,
    superseded_at: str | None = None,
):
    a = build_assertion(
        tenant_id=TENANT,
        subject_id=subject,
        predicate=predicate,
        object_id=obj,
        epistemic_class=epistemic_class,
        method="llm:test@v1",
        confidence=confidence,
        matter_id=matter,
        source_locator=SourceLocator(
            document_id="doc-1", filename="f.pdf", page=1, quote="a quoted span"
        ),
    )
    # Set after construction: review state is derived from the epistemic class by design, so
    # a caller cannot request it. Approval is a later event, which is what this simulates.
    a.review_state = review_state
    a.superseded_at = superseded_at
    return a


def _conflict_facts():
    """The demo scenario. The firm opposes Calder on one matter and acts for it on another."""
    return [
        fact("matter:" + NTL, "ADVERSE_TO", "party:calder", matter=NTL),
        fact("counsel:thorne-vaux", "REPRESENTS", "party:calder", matter=MBC),
    ]


def _stale_authority_facts():
    return [
        fact("document:advice", "CITES", "authority:aquitaine", matter=NTL),
        fact("authority:marisol", "OVERRULES", "authority:aquitaine", matter=HAL),
    ]


class TestTheConflictCheckFires:
    def test_a_cross_matter_conflict_is_found(self, ctx):
        """No document says this. It is true only once both matters are in one graph, which
        is why conflict checking is cross-matter by definition."""
        report = Reasoner(load_ontology("legal")).run(ctx, _conflict_facts())

        assert report.count == 1
        inferred = report.inferences[0].assertion
        assert inferred.predicate == "POTENTIAL_CONFLICT"
        assert inferred.object_id == "party:calder"
        assert inferred.epistemic_class is EpistemicClass.INFERRED

    def test_the_conclusion_carries_its_premises(self, ctx):
        """Without premises there is no proof tree, and "why does the system believe this"
        has no answer. build_assertion refuses an INFERRED assertion without them."""
        facts = _conflict_facts()
        report = Reasoner(load_ontology("legal")).run(ctx, facts)

        premises = set(report.inferences[0].assertion.premises)
        assert premises == {f.assertion_id for f in facts}

    def test_it_names_the_rule_and_version(self, ctx):
        report = Reasoner(load_ontology("legal")).run(ctx, _conflict_facts())
        a = report.inferences[0].assertion
        assert a.rule_id == "conflict_check"
        assert a.rule_version == "v1"
        assert a.method == "rule:conflict_check@v1"

    def test_an_inference_waits_for_review(self, ctx):
        """INFERRED is never auto-asserted. A potential conflict is a lawyer's judgement,
        so the system raises it rather than concluding it."""
        report = Reasoner(load_ontology("legal")).run(ctx, _conflict_facts())
        assert report.inferences[0].assertion.review_state is ReviewState.PENDING

    def test_a_direct_join_still_decays_exactly_once(self, ctx):
        """Per-hop decay must leave every existing rule where it was. Two premises, one edge
        each, so one factor -- and `assertion_id` is content-addressed over the confidence, so a
        change here would fork the ids of conflicts already sitting in a review queue."""
        report = Reasoner(load_ontology("legal")).run(ctx, _conflict_facts())
        assert report.inferences[0].assertion.confidence == pytest.approx(0.9 * 0.95)

    def test_no_conflict_when_the_parties_differ(self, ctx):
        """The join is the rule. Acting for one company and against another is normal work."""
        facts = [
            fact("matter:" + NTL, "ADVERSE_TO", "party:calder"),
            fact("counsel:thorne-vaux", "REPRESENTS", "party:northwind", matter=MBC),
        ]
        assert Reasoner(load_ontology("legal")).run(ctx, facts).count == 0


def _affiliate_conflict_facts():
    """The firm acts for Northwind, opposes Calder, and the two are in one corporate group.

    No two of these three facts amount to anything. `conflict_check` finds nothing here because
    the party it represents and the party it opposes are not the same node.
    """
    return [
        fact("counsel:thorne-vaux", "REPRESENTS", "party:northwind", matter=MBC),
        fact("party:northwind", "AFFILIATE_OF", "party:calder", matter=HAL),
        fact("matter:" + NTL, "ADVERSE_TO", "party:calder", matter=NTL),
    ]


class TestAConflictThroughAnAffiliateIsFound:
    """Where the conflicting parties are not adjacent. The affiliation is the fact in the
    middle; without a predicate to record it these three documents stay unrelated."""

    def test_the_conflict_is_found_through_the_group_company(self, ctx):
        report = Reasoner(load_ontology("legal")).run(ctx, _affiliate_conflict_facts())

        assert [i.rule_id for i in report.inferences] == ["conflict_via_affiliate"]
        a = report.inferences[0].assertion
        assert a.predicate == "POTENTIAL_CONFLICT"
        assert a.subject_id == "matter:" + NTL
        assert a.object_id == "party:northwind"

    def test_it_rests_on_all_three_premises(self, ctx):
        """A two-premise proof tree here would mean the affiliation was assumed rather than
        recorded, and the conclusion would survive the affiliation being withdrawn."""
        facts = _affiliate_conflict_facts()
        report = Reasoner(load_ontology("legal")).run(ctx, facts)

        assert set(report.inferences[0].assertion.premises) == {f.assertion_id for f in facts}

    def test_no_conflict_when_the_affiliation_is_absent(self, ctx):
        """Drop the middle fact and the parties are unrelated again. This is what makes the
        affiliation load-bearing rather than decorative."""
        facts = [f for f in _affiliate_conflict_facts() if f.predicate != "AFFILIATE_OF"]
        assert Reasoner(load_ontology("legal")).run(ctx, facts).count == 0

    def test_an_unreviewed_affiliation_does_not_produce_a_conflict(self, ctx):
        """The middle fact is held to the same gate as the ends. A model's guess that two
        companies are related must not be enough to flag a conflict on its own."""
        facts = _affiliate_conflict_facts()
        for f in facts:
            if f.predicate == "AFFILIATE_OF":
                f.review_state = ReviewState.PENDING
        assert Reasoner(load_ontology("legal")).run(ctx, facts).count == 0

    def test_the_same_shape_works_in_the_healthcare_pack(self, ctx):
        """An allergy against one brand and a prescription for another, sharing an ingredient.
        Same three-premise shape, same engine, no new Python — which is the claim."""
        facts = [
            fact("clinician:okafor", "PRESCRIBED", "medication:brand-a"),
            fact("medication:brand-a", "SAME_INGREDIENT_AS", "medication:brand-b"),
            fact("patient:h-mora", "ALLERGIC_TO", "medication:brand-b"),
        ]
        report = Reasoner(load_ontology("healthcare")).run(ctx, facts)

        assert [i.rule_id for i in report.inferences] == ["contraindication_via_ingredient"]
        a = report.inferences[0].assertion
        assert (a.subject_id, a.predicate) == ("patient:h-mora", "CONTRAINDICATION_ALERT")
        assert len(a.premises) == 3


def _two_hop_affiliate_facts():
    """The same conflict with one more holding company in the way.

    Northwind and Calder are still in one group, but the document trail records it in two steps:
    Northwind is an affiliate of Meridian, Meridian of Calder. Nothing about the conflict changed;
    only the number of links did.
    """
    return [
        fact("counsel:thorne-vaux", "REPRESENTS", "party:northwind", matter=MBC),
        fact("party:northwind", "AFFILIATE_OF", "party:meridian", matter=HAL),
        fact("party:meridian", "AFFILIATE_OF", "party:calder", matter=HAL),
        fact("matter:" + NTL, "ADVERSE_TO", "party:calder", matter=NTL),
    ]


class TestTheAffiliateChainIsFollowed:
    """`AFFILIATE_OF` is declared transitive, and that declaration was inert.

    The rule wrote its middle premise as a plain edge, so `max_hops` was 1, `is_path` was False,
    and the walk in `_match` never ran. A two-hop group structure produced zero inferences while
    every rule reported as evaluated — the shape of failure this codebase is organised against,
    and worse here because a structure deep enough to hide a conflict is often one built to.
    """

    def test_the_rule_asks_to_walk_the_chain(self):
        """Pinned on the pack, not the engine. The walk exists and is correct; the bug was a rule
        that never asked for it, which no engine test could catch."""
        rule = next(r for r in load_ontology("legal").rules if r.id == "conflict_via_affiliate")
        affiliate = next(p for p in parse_rule(rule).premises if p.predicate == "AFFILIATE_OF")
        assert affiliate.is_path
        assert (affiliate.min_hops, affiliate.max_hops) == (1, MAX_PATH_HOPS)

    def test_a_two_hop_group_structure_produces_the_conflict(self, ctx):
        """The regression that was silent. One intermediate company was enough to hide it."""
        report = Reasoner(load_ontology("legal")).run(ctx, _two_hop_affiliate_facts())

        assert [i.rule_id for i in report.inferences] == ["conflict_via_affiliate"]
        a = report.inferences[0].assertion
        assert a.predicate == "POTENTIAL_CONFLICT"
        assert a.subject_id == "matter:" + NTL
        assert a.object_id == "party:northwind"

    def test_the_adjacent_case_still_fires(self, ctx):
        """`*1..3` includes 1. A path premise that stopped finding the direct affiliation would
        trade one silent miss for another."""
        report = Reasoner(load_ontology("legal")).run(ctx, _affiliate_conflict_facts())

        assert [i.rule_id for i in report.inferences] == ["conflict_via_affiliate"]
        assert report.inferences[0].assertion.object_id == "party:northwind"

    def test_a_chain_past_the_hop_bound_is_not_followed(self, ctx):
        """Four links, a bound of three. Out of reach rather than the walk quietly running as far
        as the data goes, because a proof tree past three siblings is not defensible."""
        facts = [
            fact("counsel:thorne-vaux", "REPRESENTS", "party:l1", matter=MBC),
            fact("party:l1", "AFFILIATE_OF", "party:l2", matter=HAL),
            fact("party:l2", "AFFILIATE_OF", "party:l3", matter=HAL),
            fact("party:l3", "AFFILIATE_OF", "party:l4", matter=HAL),
            fact("party:l4", "AFFILIATE_OF", "party:l5", matter=HAL),
            fact("matter:" + NTL, "ADVERSE_TO", "party:l5", matter=NTL),
        ]
        assert Reasoner(load_ontology("legal")).run(ctx, facts).count == 0

    def test_every_link_of_the_chain_is_a_premise(self, ctx):
        """The whole reason a path is allowed where negation is not: each step is a signed-off
        assertion, so the proof tree stays whole and withdrawing any link withdraws the
        conclusion."""
        facts = _two_hop_affiliate_facts()
        report = Reasoner(load_ontology("legal")).run(ctx, facts)

        drawn = report.inferences[0].assertion
        assert set(drawn.premises) == {f.assertion_id for f in facts}
        assert len(drawn.premises) == 4

    def test_the_chained_conclusion_is_capped_by_its_weakest_link(self, ctx):
        """Invariant 3, over a walked chain. The weakest premise is a middle link here, which is
        the one an extra hop could have laundered away."""
        facts = _two_hop_affiliate_facts()
        for f in facts:
            if (f.subject_id, f.object_id) == ("party:meridian", "party:calder"):
                weakest = f
                f.confidence = 0.6
        report = Reasoner(load_ontology("legal")).run(ctx, facts)

        assert report.inferences[0].assertion.confidence <= weakest.confidence

    def test_a_chained_conclusion_is_less_confident_than_an_adjacent_one(self, ctx):
        """One step of doubt per edge crossed. A conflict reached through two holding companies
        may well be more urgent, but urgency is not confidence."""
        legal = load_ontology("legal")
        chained = Reasoner(legal).run(ctx, _two_hop_affiliate_facts()).inferences[0]
        adjacent = Reasoner(legal).run(ctx, _affiliate_conflict_facts()).inferences[0]

        assert chained.assertion.confidence < adjacent.assertion.confidence

    def test_an_unreviewed_link_breaks_the_chain(self, ctx):
        """Each step is held to the same gate as an ordinary premise. A model's guess in the
        middle of an ownership ladder must not carry a conflict across it."""
        facts = _two_hop_affiliate_facts()
        for f in facts:
            if (f.subject_id, f.object_id) == ("party:northwind", "party:meridian"):
                f.review_state = ReviewState.PENDING
        assert Reasoner(load_ontology("legal")).run(ctx, facts).count == 0

    def test_a_cycle_terminates_without_concluding_about_a_party_from_itself(self, ctx):
        """Northwind and Calder each recorded as an affiliate of the other. `_match` tracks
        visited nodes per path, so the walk cannot return to Northwind and hand the rule
        `q == p` — a conflict between the firm's own client and itself."""
        facts = [
            fact("counsel:thorne-vaux", "REPRESENTS", "party:northwind", matter=MBC),
            fact("party:northwind", "AFFILIATE_OF", "party:calder", matter=HAL),
            fact("party:calder", "AFFILIATE_OF", "party:northwind", matter=HAL),
            fact("matter:" + NTL, "ADVERSE_TO", "party:northwind", matter=NTL),
        ]
        report = Reasoner(load_ontology("legal")).run(ctx, facts)

        assert [i.rule_id for i in report.inferences] == ["conflict_check"]


class TestASpellingVariantNoLongerHidesAConflict:
    """The failure `CLAUDE.md` rule 5 describes, reproduced for entity names rather than
    predicates: two documents naming one company differently used to produce two nodes, and the
    conflict check joined on neither while reporting success.

    Goes through the real `ModelExtractor.validate` rather than hand-built ids, because the whole
    point is that normalisation happens at the extraction boundary. Asserting canonical ids
    directly would test nothing about the path a document actually takes.
    """

    def _extracted(self, claims):
        from src.documents.extractors.model import ModelExtractor, ProposedClaim
        from src.documents.models import Chunk

        text = (
            "Sian Aldridge acts for Halveston Chartering Limited. "
            "Calder Shipping AG appeared as a counterparty in two unrelated fixtures."
        )
        chunk = Chunk(
            document_id="doc-1",
            tenant_id=TENANT,
            filename="advice.pdf",
            ordinal=0,
            page=1,
            char_start=0,
            char_end=len(text),
            text=text,
        )
        extractor = ModelExtractor(load_ontology("legal"), bedrock=None)
        out = []
        for subject, predicate, obj in claims:
            proposed = ProposedClaim(
                subject_id=subject,
                predicate=predicate,
                object_id=obj,
                quote="Calder Shipping AG appeared as a counterparty",
                confidence=0.9,
            )
            out.extend(extractor.validate([proposed], chunk=chunk))
        for a in out:
            a.review_state = ReviewState.APPROVED
        return out

    def test_one_company_spelled_two_ways_still_produces_the_conflict(self, ctx):
        """`REPRESENTS` names the party in slug form and `ADVERSE_TO` names it as it appears in
        prose. Revert the normalisation and this returns zero conflicts while the reasoner reports
        that every rule ran -- which is the shape of the bug, not an error anyone would see."""
        facts = self._extracted(
            [
                ("counsel:sian-aldridge", "REPRESENTS", "party:calder-shipping-ag"),
                ("matter:" + NTL, "ADVERSE_TO", "Party:Calder Shipping AG"),
            ]
        )
        assert len({f.object_id for f in facts}) == 1, "the two spellings did not converge"

        report = Reasoner(load_ontology("legal")).run(ctx, facts)
        assert [i.rule_id for i in report.inferences] == ["conflict_check"]
        assert report.inferences[0].assertion.object_id == "party:calder-shipping-ag"

    def test_two_genuinely_different_companies_still_do_not_conflict(self, ctx):
        """The guard on the guard. If normalisation over-collapsed, this would produce a false
        conflict -- which is worse than the miss it replaced, because someone acts on it."""
        facts = self._extracted(
            [
                ("counsel:sian-aldridge", "REPRESENTS", "party:calder-shipping-ag"),
                ("matter:" + NTL, "ADVERSE_TO", "party:halveston-chartering-limited"),
            ]
        )
        assert Reasoner(load_ontology("legal")).run(ctx, facts).count == 0


class TestAForbiddenConclusionIsRefusedAndReported:
    """The incident, at the reasoner level.

    `conflict_check` declares `(m:Matter)-[ADVERSE_TO]->(p:Party)`, but pattern types are not
    enforced, so a party-subject `ADVERSE_TO` bound `m` to a Party and the rule drew
    `POTENTIAL_CONFLICT(party -> party)` against its own declared domain. With endpoint kinds
    enforced the write is refused — and the refusal has to be *reported*, or a rule that draws
    nothing is indistinguishable from a rule that found nothing.
    """

    def _party_subject_adversity(self):
        """What the extractor actually produced from the Halveston note. It was legal on the pack's
        own terms at the time -- `ADVERSE_TO` declared `[Party, Matter] -> [Party]` -- and the pack
        has since narrowed to `[Matter] -> [Party]`, so `build_assertion` now refuses this shape
        outright. The facts are built here without the endpoint check, deliberately: a graph written
        before the narrowing still holds them, and a starved rule must stay reportable over it."""
        return [
            fact("counsel:sian-aldridge", "REPRESENTS", "party:halveston-chartering-limited"),
            fact(
                "party:calder-shipping-ag",
                "ADVERSE_TO",
                "party:halveston-chartering-limited",
                matter=HAL,
            ),
        ]

    def test_the_party_to_party_conflict_is_not_written(self, ctx):
        report = Reasoner(load_ontology("legal")).run(ctx, self._party_subject_adversity())
        assert report.count == 0

    def test_the_premise_never_binds_so_there_is_nothing_to_refuse(self, ctx):
        """Two guards stop this conflict, and which one fires is the point.

        Premise type filtering rejects the fact before the rule binds `m`, so nothing is built and
        `conclusions_refused` stays *empty*. The endpoint check in `build_assertion` is the belt
        behind it, and it would populate that list instead. Asserting empty here is what proves the
        filter ran rather than the contract catching it afterwards -- two different states, so the
        assertion has something to bite on.
        """
        report = Reasoner(load_ontology("legal")).run(ctx, self._party_subject_adversity())
        assert report.conclusions_refused == []

    def test_the_contract_still_refuses_it_if_the_filter_is_bypassed(self, ctx):
        """Defence in depth, asserted directly rather than assumed. A rule with untyped premises
        binds the party-subject adversity, and then invariant 7 is the only thing standing
        between it and a conflict about the firm's own client."""

        class Untyped:
            id = "untyped_conflict"
            version = "v1"
            description = ""
            when = ("(c)-[:REPRESENTS]->(p)", "(m)-[:ADVERSE_TO]->(p)")
            then = "(m)-[:POTENTIAL_CONFLICT]->(p)"
            min_premise_class = "EXTRACTED_MODEL"

        legal = load_ontology("legal")

        class Patched:
            domain = legal.domain
            rules = (Untyped(),)
            governing_predicates = legal.governing_predicates
            descriptive_predicates = legal.descriptive_predicates
            endpoint_kinds = legal.endpoint_kinds
            entity_kind_of = legal.entity_kind_of

        report = Reasoner(Patched()).run(ctx, self._party_subject_adversity())
        assert report.count == 0
        assert len(report.conclusions_refused) == 1
        assert "party -> party" in report.conclusions_refused[0]

    def test_the_starved_rule_says_which_premise_emptied_the_join(self, ctx):
        """The state that was unreportable, and the reason this bug was found by hand.

        `rules_skipped` covers a rule that could never fire and `conclusions_refused` a conclusion
        the contract refused. A rule whose premises simply matched nothing produced no signal at
        all — so a conflict check starved by a wrongly-shaped fact looked exactly like a clean
        conflict check, which is the failure the whole design is organised against.
        """
        report = Reasoner(load_ontology("legal")).run(ctx, self._party_subject_adversity())

        assert report.count == 0
        starved = report.rules_starved["conflict_check"]
        assert "ADVERSE_TO" in starved, "naming the rule alone is not actionable"
        assert "Matter" in starved, "the reader needs the shape that was wanted"

    def test_it_distinguishes_no_candidates_from_none_joining(self, ctx):
        """Two different problems. No matching fact is a missing or wrongly-shaped claim; facts
        that matched but did not join is a genuine absence of the relationship."""
        onto = load_ontology("legal")
        matched_but_unjoined = [
            fact("counsel:sian-aldridge", "REPRESENTS", "party:halveston-chartering-limited"),
            fact("matter:" + NTL, "ADVERSE_TO", "party:someone-unrelated", matter=NTL),
        ]
        starved = Reasoner(onto).run(ctx, matched_but_unjoined).rules_starved["conflict_check"]
        assert "none joining" in starved

    def test_a_rule_that_fires_is_not_reported_as_starved(self, ctx):
        """The guard on the guard. A report that named every rule would be noise nobody reads."""
        facts = [
            fact("counsel:thorne-vaux", "REPRESENTS", "party:calder-shipping-ag", matter=MBC),
            fact("matter:" + NTL, "ADVERSE_TO", "party:calder-shipping-ag", matter=NTL),
        ]
        report = Reasoner(load_ontology("legal")).run(ctx, facts)

        assert report.count == 1
        assert "conflict_check" not in report.rules_starved

    def test_it_reaches_the_report_a_person_reads(self, ctx):
        report = Reasoner(load_ontology("legal")).run(ctx, self._party_subject_adversity())
        assert "conflict_check" in report.to_dict()["rules_starved"]

    def test_the_affiliate_rule_needs_the_matter_orientation_too(self, ctx):
        """Why this mattered beyond the one bad conflict. `conflict_via_affiliate` was reachable in
        production -- `AFFILIATE_OF` was extracted and approved -- and drew nothing, because every
        `ADVERSE_TO` named the counterparty as *subject*. The rule wants the firm's matter adverse
        to the affiliate, so the inverted orientation matched nothing and the real conflict, the
        one the fixture's risk memo is about, stayed invisible."""
        facts = [
            fact("counsel:thorne-vaux", "REPRESENTS", "party:meridian-bulk-carriers-sa"),
            fact("party:meridian-bulk-carriers-sa", "AFFILIATE_OF", "party:calder-shipping-ag"),
            fact("matter:" + NTL, "ADVERSE_TO", "party:calder-shipping-ag", matter=NTL),
        ]
        report = Reasoner(load_ontology("legal")).run(ctx, facts)

        assert [i.rule_id for i in report.inferences] == ["conflict_via_affiliate"]
        drawn = report.inferences[0].assertion
        assert drawn.subject_id == "matter:" + NTL
        assert drawn.object_id == "party:meridian-bulk-carriers-sa"

    def test_the_matter_subject_form_still_fires(self, ctx):
        """The guard on the guard. `matter -> party` is what the rule declares and what the
        conflict predicate accepts, so this must still draw the conflict."""
        facts = [
            fact("counsel:thorne-vaux", "REPRESENTS", "party:calder-shipping-ag", matter=MBC),
            fact("matter:" + NTL, "ADVERSE_TO", "party:calder-shipping-ag", matter=NTL),
        ]
        report = Reasoner(load_ontology("legal")).run(ctx, facts)

        assert [i.rule_id for i in report.inferences] == ["conflict_check"]
        assert report.inferences[0].assertion.subject_id == "matter:" + NTL
        assert report.conclusions_refused == []


class TestTheStaleAuthorityCheckFires:
    def test_reliance_on_an_overruled_case_is_flagged(self, ctx):
        report = Reasoner(load_ontology("legal")).run(ctx, _stale_authority_facts())

        assert report.count == 1
        a = report.inferences[0].assertion
        assert a.predicate == "RELIES_ON_STALE_AUTHORITY"
        assert a.subject_id == "document:advice"
        assert a.object_id == "authority:aquitaine"

    def test_a_different_case_being_overruled_changes_nothing(self, ctx):
        facts = [
            fact("document:advice", "CITES", "authority:aquitaine"),
            fact("authority:marisol", "OVERRULES", "authority:unrelated", matter=HAL),
        ]
        assert Reasoner(load_ontology("legal")).run(ctx, facts).count == 0


class TestItRefusesUnsafePremises:
    """The half that keeps a conflict flag worth acting on."""

    def test_an_unreviewed_claim_is_not_a_premise(self, ctx):
        """The gate that matters. A model proposing REPRESENTS is a proposal, and a conflict
        flag built on one would be relied upon by someone who never saw the proposal."""
        facts = [
            fact("matter:" + NTL, "ADVERSE_TO", "party:calder", review_state=ReviewState.PENDING),
            fact(
                "counsel:us",
                "REPRESENTS",
                "party:calder",
                matter=MBC,
                review_state=ReviewState.PENDING,
            ),
        ]
        assert Reasoner(load_ontology("legal")).run(ctx, facts).count == 0

    def test_one_unreviewed_premise_is_enough_to_stop_it(self, ctx):
        facts = [
            fact("matter:" + NTL, "ADVERSE_TO", "party:calder"),
            fact(
                "counsel:us",
                "REPRESENTS",
                "party:calder",
                matter=MBC,
                review_state=ReviewState.PENDING,
            ),
        ]
        assert Reasoner(load_ontology("legal")).run(ctx, facts).count == 0

    def test_a_rejected_claim_is_not_a_premise(self, ctx):
        facts = [
            fact("matter:" + NTL, "ADVERSE_TO", "party:calder"),
            fact(
                "counsel:us",
                "REPRESENTS",
                "party:calder",
                matter=MBC,
                review_state=ReviewState.REJECTED,
            ),
        ]
        assert Reasoner(load_ontology("legal")).run(ctx, facts).count == 0

    def test_a_withdrawn_claim_is_not_a_premise(self, ctx):
        """A conclusion must not outlive the reason it was drawn."""
        facts = [
            fact("matter:" + NTL, "ADVERSE_TO", "party:calder"),
            fact(
                "counsel:us",
                "REPRESENTS",
                "party:calder",
                matter=MBC,
                superseded_at="2026-01-01T00:00:00Z",
            ),
        ]
        assert Reasoner(load_ontology("legal")).run(ctx, facts).count == 0

    def test_an_auto_asserted_fact_is_a_valid_premise(self, ctx):
        """AUTO_ASSERTED means no person was needed, not that no person stands behind it: a
        system of record declared it, or a check confirmed it."""
        facts = [
            fact(
                "matter:" + NTL,
                "ADVERSE_TO",
                "party:calder",
                epistemic_class=EpistemicClass.DECLARED,
                review_state=ReviewState.AUTO_ASSERTED,
            ),
            fact(
                "counsel:us",
                "REPRESENTS",
                "party:calder",
                matter=MBC,
                epistemic_class=EpistemicClass.DECLARED,
                review_state=ReviewState.AUTO_ASSERTED,
            ),
        ]
        assert Reasoner(load_ontology("legal")).run(ctx, facts).count == 1


class TestConfidence:
    def test_an_inference_never_exceeds_its_weakest_premise(self, ctx):
        """Invariant 3. A chain of guesses must not launder itself into a certainty."""
        facts = [
            fact("matter:" + NTL, "ADVERSE_TO", "party:calder", confidence=0.95),
            fact("counsel:us", "REPRESENTS", "party:calder", matter=MBC, confidence=0.55),
        ]
        report = Reasoner(load_ontology("legal")).run(ctx, facts)
        assert report.inferences[0].assertion.confidence <= 0.55

    def test_an_inference_is_slightly_less_certain_than_its_premises(self, ctx):
        """An inference adds a step, and a step can be wrong even when every premise is
        right, so it is not presented as equal to them either."""
        facts = [
            fact("matter:" + NTL, "ADVERSE_TO", "party:calder", confidence=0.8),
            fact("counsel:us", "REPRESENTS", "party:calder", matter=MBC, confidence=0.8),
        ]
        report = Reasoner(load_ontology("legal")).run(ctx, facts)
        assert report.inferences[0].assertion.confidence < 0.8


class TestMatterScoping:
    def test_a_cross_matter_conclusion_belongs_to_no_single_matter(self, ctx):
        """Stamping it with one of its matters would hide it from the other, and a conflict
        invisible to one of the matters it concerns is the failure the design prevents."""
        report = Reasoner(load_ontology("legal")).run(ctx, _conflict_facts())
        assert report.inferences[0].assertion.matter_id is None

    def test_a_single_matter_conclusion_inherits_the_matter(self, ctx):
        facts = [
            fact("matter:" + NTL, "ADVERSE_TO", "party:calder", matter=NTL),
            fact("counsel:us", "REPRESENTS", "party:calder", matter=NTL),
        ]
        report = Reasoner(load_ontology("legal")).run(ctx, facts)
        assert report.inferences[0].assertion.matter_id == NTL


class TestNoDuplicatesOrSelfJoins:
    def test_the_same_conclusion_is_reached_once(self, ctx):
        """Two different documents can evidence the same conflict. One edge, not two."""
        facts = [
            fact("matter:" + NTL, "ADVERSE_TO", "party:calder"),
            fact("counsel:a", "REPRESENTS", "party:calder", matter=MBC),
            fact("counsel:b", "REPRESENTS", "party:calder", matter=MBC),
        ]
        report = Reasoner(load_ontology("legal")).run(ctx, facts)
        assert report.count == 1

    def test_one_fact_cannot_be_two_premises(self, ctx):
        """ADVERSE_TO is symmetric, so without a guard a single fact could satisfy both
        premises of a rule and "prove" a conclusion from itself."""
        single = [fact("authority:x", "OVERRULES", "authority:x")]
        assert Reasoner(load_ontology("legal")).run(ctx, single).count == 0


class TestTheSameEngineRunsAnotherDomain:
    """No legal knowledge in the engine. This is the test that proves it."""

    def test_the_healthcare_rule_fires(self):
        ctx = AuthContext(user_id="dr@clinic.example", tenant_id=TENANT)
        facts = [
            fact("clinician:jones", "PRESCRIBED", "medication:penicillin", matter="ENC-1"),
            fact("patient:smith", "ALLERGIC_TO", "medication:penicillin", matter="ENC-1"),
        ]
        report = Reasoner(load_ontology("healthcare")).run(ctx, facts)

        assert report.count == 1
        a = report.inferences[0].assertion
        assert a.predicate == "CONTRAINDICATION_ALERT"
        assert a.subject_id == "patient:smith"
        assert a.rule_id == "contraindication_alert"

    def test_a_different_medication_does_not_alert(self):
        ctx = AuthContext(user_id="dr@clinic.example", tenant_id=TENANT)
        facts = [
            fact("clinician:jones", "PRESCRIBED", "medication:amoxicillin", matter="ENC-1"),
            fact("patient:smith", "ALLERGIC_TO", "medication:penicillin", matter="ENC-1"),
        ]
        assert Reasoner(load_ontology("healthcare")).run(ctx, facts).count == 0


class TestReportsAreLegible:
    def test_the_report_says_what_it_did(self, ctx):
        legal = load_ontology("legal")
        report = Reasoner(legal).run(ctx, _conflict_facts())
        out = report.to_dict()
        assert out["count"] == 1
        assert out["rules_evaluated"] == len(legal.rules)
        assert out["facts_considered"] == 2
        assert out["inferences"][0]["rule_id"] == "conflict_check"

    def test_no_facts_is_not_an_error(self, ctx):
        legal = load_ontology("legal")
        report = Reasoner(legal).run(ctx, [])
        assert report.count == 0
        assert report.rules_evaluated == len(legal.rules)


class TestPatternParsing:
    def test_a_normal_pattern_reads(self):
        edge = parse_edge("(c:Counsel)-[:REPRESENTS]->(p:Party)")
        assert (edge.subject_var, edge.predicate, edge.object_var) == ("c", "REPRESENTS", "p")
        assert edge.subject_type == "Counsel"

    def test_types_are_optional(self):
        edge = parse_edge("(m)-[:POTENTIAL_CONFLICT]->(p)")
        assert edge.subject_type is None

    @pytest.mark.parametrize(
        "bad",
        [
            "not a pattern",
            "(c)-[REPRESENTS]->(p)",
            "(c)-[:REPRESENTS]-(p)",
            "(c:Counsel)-[:REPRESENTS]->(p:Party) EXTRA",
            "",
        ],
    )
    def test_a_malformed_pattern_is_refused(self, bad):
        with pytest.raises(PatternError):
            parse_edge(bad)

    def test_every_shipped_rule_parses(self):
        """A rule nobody can parse would silently never fire, and a conflict check that never
        fires looks exactly like a clean conflict check."""
        for domain in ALL_PACKS:
            for rule in load_ontology(domain).rules:
                parsed = parse_rule(rule)
                assert parsed.join_variables, f"{parsed.rule_id} has no join"

    def test_a_rule_whose_premises_do_not_join_is_refused(self):
        """The dangerous one: unjoined premises produce a cross product, flagging conflicts
        between parties that have nothing to do with each other."""

        class Unjoined:
            id = "bad"
            version = "v1"
            description = ""
            when = ("(a:Counsel)-[:REPRESENTS]->(b:Party)", "(c:Matter)-[:ADVERSE_TO]->(d:Party)")
            then = "(a)-[:POTENTIAL_CONFLICT]->(b)"
            min_premise_class = "EXTRACTED_MODEL"

        with pytest.raises(PatternError, match="sharing no variable"):
            parse_rule(Unjoined())

    def test_a_third_premise_that_joins_nothing_is_refused(self):
        """The two-premise case above cannot distinguish "some variable is shared" from "every
        premise is connected". At three premises it can, and the weaker test passed this: `q`
        is shared, so `join_variables` is non-empty, while the third premise still
        cross-products every ADVERSE_TO against every match."""

        class Stranded:
            id = "bad"
            version = "v1"
            description = ""
            when = (
                "(c:Counsel)-[:REPRESENTS]->(q:Party)",
                "(q:Party)-[:PARTY_TO]->(m:Matter)",
                "(x:Party)-[:ADVERSE_TO]->(y:Party)",
            )
            then = "(m)-[:POTENTIAL_CONFLICT]->(q)"
            min_premise_class = "EXTRACTED_MODEL"

        with pytest.raises(PatternError, match="sharing no variable"):
            parse_rule(Stranded())

    def test_a_premise_joining_only_through_a_middle_premise_is_kept(self):
        """Connectivity is transitive, not adjacency to the first premise. `conflict_via_affiliate`
        has this shape: ADVERSE_TO reaches REPRESENTS only through AFFILIATE_OF."""
        parsed = parse_rule(
            next(r for r in load_ontology("legal").rules if r.id == "conflict_via_affiliate")
        )
        assert parsed.disconnected_premises == ()
        assert len(parsed.premises) == 3

    def test_a_bounded_path_reads(self):
        edge = parse_edge("(a:Party)-[:AFFILIATE_OF*1..3]->(b:Party)")
        assert (edge.predicate, edge.min_hops, edge.max_hops) == ("AFFILIATE_OF", 1, 3)
        assert edge.is_path

    def test_an_ordinary_edge_is_a_path_of_one(self):
        """Both bounds default to 1, so the common case needs no separate code path."""
        edge = parse_edge("(c:Counsel)-[:REPRESENTS]->(p:Party)")
        assert (edge.min_hops, edge.max_hops) == (1, 1)
        assert not edge.is_path

    def test_a_path_longer_than_the_bound_is_refused(self):
        """A proof tree past three siblings stops being defensible in front of a regulator."""
        with pytest.raises(PatternError, match=f"at most {MAX_PATH_HOPS}"):
            parse_edge("(a:Party)-[:AFFILIATE_OF*1..4]->(b:Party)")

    @pytest.mark.parametrize("lower", [0, 2])
    def test_a_lower_bound_other_than_one_is_refused(self, lower):
        """`*2..3` requires a longer path, which asserts that no shorter one exists. That is
        negation in disguise: no assertion witnesses an absence, so the proof tree would have a
        hole where its reason belongs."""
        with pytest.raises(PatternError, match="lower bound must be 1"):
            parse_edge(f"(a:Party)-[:AFFILIATE_OF*{lower}..3]->(b:Party)")

    def test_a_conclusion_may_not_be_a_path(self):
        class PathConclusion:
            id = "bad"
            version = "v1"
            description = ""
            when = ("(a:Counsel)-[:REPRESENTS]->(b:Party)", "(c:Matter)-[:ADVERSE_TO]->(b:Party)")
            then = "(c)-[:POTENTIAL_CONFLICT*1..3]->(b)"
            min_premise_class = "EXTRACTED_MODEL"

        with pytest.raises(PatternError, match="concludes a path"):
            parse_rule(PathConclusion())

    def test_a_conclusion_about_an_unbound_variable_is_refused(self):
        class Unbound:
            id = "bad"
            version = "v1"
            description = ""
            when = ("(a:Counsel)-[:REPRESENTS]->(b:Party)", "(c:Matter)-[:ADVERSE_TO]->(b:Party)")
            then = "(zzz)-[:POTENTIAL_CONFLICT]->(b)"
            min_premise_class = "EXTRACTED_MODEL"

        with pytest.raises(PatternError, match="no premise binds"):
            parse_rule(Unbound())


class TestUnparseableRulesAreSkippedNotFatal:
    def test_one_bad_rule_does_not_stop_the_others(self, ctx):
        """Reported rather than raised: a pack with one typo should still fire its good rules,
        and the skip is recorded so it does not merely look like a rule that found nothing."""
        legal = load_ontology("legal")

        class BadRule:
            id = "broken"
            version = "v1"
            description = ""
            when = ("nonsense",)
            then = "(a)-[:POTENTIAL_CONFLICT]->(b)"
            min_premise_class = "EXTRACTED_MODEL"

        class Patched:
            domain = legal.domain
            rules = (*legal.rules, BadRule())
            governing_predicates = legal.governing_predicates
            descriptive_predicates = legal.descriptive_predicates
            endpoint_kinds = legal.endpoint_kinds
            entity_kind_of = legal.entity_kind_of

        reasoner = Reasoner(Patched())
        report = reasoner.run(ctx, _conflict_facts())
        assert report.count == 1
        assert "broken" in report.rules_skipped


class TestTheStrengthFloor:
    def test_a_weaker_class_than_the_floor_is_refused(self):
        weak = fact(
            "a",
            "MENTIONS",
            "b",
            epistemic_class=EpistemicClass.PREDICTED,
            review_state=ReviewState.AUTO_ASSERTED,
        )
        assert accepts(weak, "EXTRACTED_MODEL") is False

    def test_a_stronger_class_than_the_floor_is_accepted(self):
        strong = fact(
            "a",
            "MENTIONS",
            "b",
            epistemic_class=EpistemicClass.DECLARED,
            review_state=ReviewState.AUTO_ASSERTED,
        )
        assert accepts(strong, "EXTRACTED_MODEL") is True

    def test_an_unknown_floor_is_treated_as_strictest(self):
        """A typo in a pack must not quietly widen what a rule fires on."""
        model = fact(
            "a",
            "MENTIONS",
            "b",
            epistemic_class=EpistemicClass.EXTRACTED_MODEL,
            review_state=ReviewState.AUTO_ASSERTED,
        )
        assert accepts(model, "TYPO_CLASS") is False


class TestTheEndpoint:
    """Over HTTP, because the wiring is where this goes wrong.

    The route reads live assertions, runs the pack's rules, and stages the conclusions. Every
    one of those three is a place a mistake looks like "the reasoner found nothing".
    """

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
        c = TestClient(create_app(cfg))
        return c, get_services()

    def _make_live(self, services, ctx, facts):
        """Stage and promote, so the facts are live rather than merely present."""
        services.review_queue.stage(ctx, facts)
        for f in facts:
            record = services.review_queue.store.get(TENANT, f.assertion_id)
            record.assertion.review_state = ReviewState.APPROVED
            services.review_queue.store.put(record)
        services.review_queue.promote(ctx)

    def test_it_infers_the_conflict_over_http(self, client, ctx):
        c, services = client
        self._make_live(services, ctx, _conflict_facts())

        body = c.post(f"/api/tenants/{TENANT}/reason").json()
        assert body["count"] == 1
        assert body["staged"] == 1
        assert body["inferences"][0]["predicate"] == "POTENTIAL_CONFLICT"

    def test_the_inference_lands_in_the_review_queue(self, client, ctx):
        """Staged, not published. An inference goes through the same gate as an extraction."""
        c, services = client
        self._make_live(services, ctx, _conflict_facts())
        c.post(f"/api/tenants/{TENANT}/reason")

        pending = c.get(f"/api/tenants/{TENANT}/assertions?review_state=PENDING").json()
        predicates = [a["predicate"] for a in pending["assertions"]]
        assert "POTENTIAL_CONFLICT" in predicates

    def test_the_staged_inference_carries_its_premises(self, client, ctx):
        c, services = client
        facts = _conflict_facts()
        self._make_live(services, ctx, facts)
        c.post(f"/api/tenants/{TENANT}/reason")

        pending = c.get(f"/api/tenants/{TENANT}/assertions?review_state=PENDING").json()
        inferred = next(a for a in pending["assertions"] if a["predicate"] == "POTENTIAL_CONFLICT")
        assert set(inferred["premises"]) == {f.assertion_id for f in facts}

    def test_running_it_twice_does_not_duplicate(self, client, ctx):
        """Ids are content-addressed, so a second pass converges."""
        c, services = client
        self._make_live(services, ctx, _conflict_facts())

        c.post(f"/api/tenants/{TENANT}/reason")
        second = c.post(f"/api/tenants/{TENANT}/reason").json()
        pending = c.get(f"/api/tenants/{TENANT}/assertions?review_state=PENDING").json()
        conflicts = [a for a in pending["assertions"] if a["predicate"] == "POTENTIAL_CONFLICT"]
        assert second["count"] == 1
        assert len(conflicts) == 1

    def test_nothing_to_infer_is_not_an_error(self, client):
        c, _ = client
        body = c.post(f"/api/tenants/{TENANT}/reason").json()
        assert body["count"] == 0
        assert body["rules_evaluated"] == len(load_ontology("legal").rules)

    def test_it_does_not_fire_on_unpromoted_facts(self, client, ctx):
        """Staged but never promoted is not a fact anyone stands behind."""
        c, services = client
        services.review_queue.stage(ctx, _conflict_facts())

        assert c.post(f"/api/tenants/{TENANT}/reason").json()["count"] == 0


class TestPackValidationHappensAtLoad:
    """A rule that cannot fire must fail loudly when the pack is read.

    Both failure modes are otherwise silent, and silence is the danger: a rule that never fires
    and a rule that finds nothing are indistinguishable from outside, and "no conflicts found"
    is precisely the answer nobody re-checks.
    """

    def test_an_undeclared_conclusion_fails_the_pack(self, tmp_path, monkeypatch):
        pack = tmp_path / "broken.yaml"
        pack.write_text(
            "domain: broken\n"
            "version: 1\n"
            "entity_types:\n"
            "  - id: Thing\n"
            "    label: Thing\n"
            "    description: t\n"
            "governing_predicates:\n"
            "  - id: KNOWS\n"
            "    label: knows\n"
            "    description: d\n"
            "rules:\n"
            "  - id: bad\n"
            "    version: v1\n"
            "    description: concludes something undeclared\n"
            "    when:\n"
            '      - "(a:Thing)-[:KNOWS]->(b:Thing)"\n'
            '      - "(c:Thing)-[:KNOWS]->(b:Thing)"\n'
            '    then: "(a)-[:NEVER_DECLARED]->(b)"\n'
            "    min_premise_class: EXTRACTED_MODEL\n"
        )
        monkeypatch.setattr("src.ontology.loader.ONTOLOGY_DIR", tmp_path)
        load_ontology.cache_clear()

        with pytest.raises(PatternError, match="NEVER_DECLARED"):
            load_ontology("broken")
        load_ontology.cache_clear()

    def test_matching_on_an_undeclared_predicate_fails_the_pack(self, tmp_path, monkeypatch):
        pack = tmp_path / "broken2.yaml"
        pack.write_text(
            "domain: broken2\n"
            "version: 1\n"
            "entity_types:\n"
            "  - id: Thing\n"
            "    label: Thing\n"
            "    description: t\n"
            "governing_predicates:\n"
            "  - id: RESULT\n"
            "    label: result\n"
            "    description: d\n"
            "rules:\n"
            "  - id: bad\n"
            "    version: v1\n"
            "    description: matches on something undeclared\n"
            "    when:\n"
            '      - "(a:Thing)-[:NOT_A_PREDICATE]->(b:Thing)"\n'
            '      - "(c:Thing)-[:NOT_A_PREDICATE]->(b:Thing)"\n'
            '    then: "(a)-[:RESULT]->(b)"\n'
            "    min_premise_class: EXTRACTED_MODEL\n"
        )
        monkeypatch.setattr("src.ontology.loader.ONTOLOGY_DIR", tmp_path)
        load_ontology.cache_clear()

        with pytest.raises(PatternError, match="can never match"):
            load_ontology("broken2")
        load_ontology.cache_clear()

    def test_the_shipped_packs_still_load(self):
        for domain in ALL_PACKS:
            assert load_ontology(domain).rules

    def test_walking_a_predicate_the_pack_never_declared_transitive_fails(
        self, tmp_path, monkeypatch
    ):
        """The pack decides what a chain carries. Walking a predicate it never declared
        transitive would conclude something it never said follows from a chain."""
        pack = tmp_path / "broken3.yaml"
        pack.write_text(
            "domain: broken3\n"
            "version: 1\n"
            "entity_types:\n"
            "  - id: Thing\n"
            "    label: Thing\n"
            "    description: t\n"
            "governing_predicates:\n"
            "  - id: KNOWS\n"
            "    label: knows\n"
            "    domain: [Thing]\n"
            "    range: [Thing]\n"
            "    description: d\n"
            "  - id: RESULT\n"
            "    label: result\n"
            "    description: d\n"
            "rules:\n"
            "  - id: bad\n"
            "    version: v1\n"
            "    description: walks a chain the pack did not license\n"
            "    when:\n"
            '      - "(a:Thing)-[:KNOWS*1..3]->(b:Thing)"\n'
            '      - "(c:Thing)-[:KNOWS]->(b:Thing)"\n'
            '    then: "(a)-[:RESULT]->(b)"\n'
            "    min_premise_class: EXTRACTED_MODEL\n"
        )
        monkeypatch.setattr("src.ontology.loader.ONTOLOGY_DIR", tmp_path)
        load_ontology.cache_clear()

        with pytest.raises(PatternError, match="does not declare transitive"):
            load_ontology("broken3")
        load_ontology.cache_clear()


class TestARuleWalksAChain:
    """A chain of unknown length, which is the only thing a path premise buys over a third
    premise. An ownership ladder is the case: the pack author cannot know how many holding
    companies sit between the firm's client and the company it is opposing.
    """

    def _pack(self, tmp_path, monkeypatch):
        pack = tmp_path / "paths.yaml"
        pack.write_text(
            "domain: paths\n"
            "version: 1\n"
            "entity_types:\n"
            "  - id: Party\n"
            "    label: Party\n"
            "    description: p\n"
            "governing_predicates:\n"
            "  - id: OWNS\n"
            "    label: owns\n"
            "    domain: [Party]\n"
            "    range: [Party]\n"
            "    description: d\n"
            "    transitive: true\n"
            "  - id: ADVERSE_TO\n"
            "    label: adverse to\n"
            "    domain: [Party]\n"
            "    range: [Party]\n"
            "    description: d\n"
            "  - id: RESULT\n"
            "    label: result\n"
            "    domain: [Party]\n"
            "    range: [Party]\n"
            "    description: d\n"
            "rules:\n"
            "  - id: chain\n"
            "    version: v1\n"
            "    description: walks an ownership ladder\n"
            "    when:\n"
            '      - "(a:Party)-[:OWNS*1..3]->(b:Party)"\n'
            '      - "(c:Party)-[:ADVERSE_TO]->(b:Party)"\n'
            '    then: "(c)-[:RESULT]->(a)"\n'
            "    min_premise_class: EXTRACTED_MODEL\n"
        )
        monkeypatch.setattr("src.ontology.loader.ONTOLOGY_DIR", tmp_path)
        load_ontology.cache_clear()
        return load_ontology("paths")

    def test_the_pack_loads_because_the_predicate_is_transitive(self, tmp_path, monkeypatch):
        onto = self._pack(tmp_path, monkeypatch)
        assert onto.transitive_predicates == frozenset({"OWNS"})
        load_ontology.cache_clear()

    def test_a_one_hop_path_still_matches(self, tmp_path, monkeypatch):
        """`*1..3` includes 1. A path premise that stopped finding direct edges would be a
        regression dressed as a feature."""
        onto = self._pack(tmp_path, monkeypatch)
        ctx = AuthContext(user_id="a@b.example", tenant_id=TENANT)
        facts = [
            fact("party:top", "OWNS", "party:mid"),
            fact("party:rival", "ADVERSE_TO", "party:mid"),
        ]
        report = Reasoner(onto).run(ctx, facts)

        assert report.rules_skipped == {}
        assert [(i.assertion.subject_id, i.assertion.object_id) for i in report.inferences] == [
            ("party:rival", "party:top")
        ]
        load_ontology.cache_clear()

    def test_it_reaches_the_far_end_of_a_ladder(self, tmp_path, monkeypatch):
        """The conclusion nothing else finds: rival opposes low, top owns mid owns low, so the
        firm is against a company two steps below its own client."""
        onto = self._pack(tmp_path, monkeypatch)
        ctx = AuthContext(user_id="a@b.example", tenant_id=TENANT)
        facts = [
            fact("party:top", "OWNS", "party:mid"),
            fact("party:mid", "OWNS", "party:low"),
            fact("party:rival", "ADVERSE_TO", "party:low"),
        ]
        report = Reasoner(onto).run(ctx, facts)

        reached = {i.assertion.object_id for i in report.inferences}
        assert "party:top" in reached
        load_ontology.cache_clear()

    def test_every_step_of_the_chain_is_a_premise(self, tmp_path, monkeypatch):
        """The whole reason a path is allowed where negation is not. Two hops means three
        premises, so withdrawing any link withdraws the conclusion."""
        onto = self._pack(tmp_path, monkeypatch)
        ctx = AuthContext(user_id="a@b.example", tenant_id=TENANT)
        facts = [
            fact("party:top", "OWNS", "party:mid"),
            fact("party:mid", "OWNS", "party:low"),
            fact("party:rival", "ADVERSE_TO", "party:low"),
        ]
        report = Reasoner(onto).run(ctx, facts)

        far = next(i for i in report.inferences if i.assertion.object_id == "party:top")
        assert set(far.premise_ids) == {f.assertion_id for f in facts}
        load_ontology.cache_clear()

    def test_a_chain_longer_than_the_bound_is_not_followed(self, tmp_path, monkeypatch):
        """Four links, a bound of three. The far end must stay out of reach rather than the
        walk quietly running as far as the data goes."""
        onto = self._pack(tmp_path, monkeypatch)
        ctx = AuthContext(user_id="a@b.example", tenant_id=TENANT)
        facts = [
            fact("party:l1", "OWNS", "party:l2"),
            fact("party:l2", "OWNS", "party:l3"),
            fact("party:l3", "OWNS", "party:l4"),
            fact("party:l4", "OWNS", "party:l5"),
            fact("party:rival", "ADVERSE_TO", "party:l5"),
        ]
        reached = {i.assertion.object_id for i in Reasoner(onto).run(ctx, facts).inferences}
        assert "party:l2" in reached
        assert "party:l1" not in reached
        load_ontology.cache_clear()

    def test_a_cycle_terminates_and_concludes_nothing_about_itself(self, tmp_path, monkeypatch):
        """A OWNS B OWNS A. Without per-path visited nodes the walk runs to the hop cap and
        concludes something about A from A."""
        onto = self._pack(tmp_path, monkeypatch)
        ctx = AuthContext(user_id="a@b.example", tenant_id=TENANT)
        facts = [
            fact("party:a", "OWNS", "party:b"),
            fact("party:b", "OWNS", "party:a"),
            fact("party:r", "ADVERSE_TO", "party:b"),
        ]
        report = Reasoner(onto).run(ctx, facts)

        assert [(i.assertion.subject_id, i.assertion.object_id) for i in report.inferences] == [
            ("party:r", "party:a")
        ]
        load_ontology.cache_clear()

    def test_the_shortest_proof_is_the_one_cited(self, tmp_path, monkeypatch):
        """Two chains reach `top` from `low`: directly, and via `mid`. The conclusion is one
        edge either way, so the premises cited must be the tighter pair -- deterministically,
        or two runs of one conflict check disagree about why."""
        onto = self._pack(tmp_path, monkeypatch)
        ctx = AuthContext(user_id="a@b.example", tenant_id=TENANT)
        facts = [
            fact("party:top", "OWNS", "party:low"),
            fact("party:top", "OWNS", "party:mid"),
            fact("party:mid", "OWNS", "party:low"),
            fact("party:rival", "ADVERSE_TO", "party:low"),
        ]
        report = Reasoner(onto).run(ctx, facts)

        far = next(i for i in report.inferences if i.assertion.object_id == "party:top")
        assert len(far.premise_ids) == 2
        load_ontology.cache_clear()

    def test_one_fact_cannot_be_two_premises_of_a_chain(self, tmp_path, monkeypatch):
        """A single OWNS edge must not be walked twice to fake a two-hop ladder."""
        onto = self._pack(tmp_path, monkeypatch)
        ctx = AuthContext(user_id="a@b.example", tenant_id=TENANT)
        facts = [
            fact("party:x", "OWNS", "party:x"),
            fact("party:rival", "ADVERSE_TO", "party:x"),
        ]
        for i in Reasoner(onto).run(ctx, facts).inferences:
            assert len(set(i.premise_ids)) == len(i.premise_ids)
        load_ontology.cache_clear()

    def test_the_chain_is_cited_in_walk_order(self, tmp_path, monkeypatch):
        """`premises` is a flat tuple, so a 2-hop path contributes three ids with no structure
        saying which followed which. Order is preserved incidentally -- facts are appended as
        they match -- and that is worth pinning: it is the only thing letting a reader
        reconstruct `top -> mid -> low` from a proof tree that renders them as siblings.

        If a `premise_paths` structure ever lands, this test is what it replaces."""
        onto = self._pack(tmp_path, monkeypatch)
        ctx = AuthContext(user_id="a@b.example", tenant_id=TENANT)
        top = fact("party:top", "OWNS", "party:mid")
        mid = fact("party:mid", "OWNS", "party:low")
        adverse = fact("party:rival", "ADVERSE_TO", "party:low")
        report = Reasoner(onto).run(ctx, [top, mid, adverse])

        far = next(i for i in report.inferences if i.assertion.object_id == "party:top")
        assert list(far.premise_ids) == [top.assertion_id, mid.assertion_id, adverse.assertion_id]
        assert list(far.assertion.premises) == list(far.premise_ids)
        load_ontology.cache_clear()

    def test_a_longer_chain_is_less_confident(self, tmp_path, monkeypatch):
        """One step of doubt per edge crossed. Applied once per rule, a 3-hop conclusion was
        presented as firmly as a direct join, which is not an honest ordering: a longer chain has
        more places for a correct premise to lead to a wrong conclusion.

        Deliberately *not* the other direction. A conflict reached further out may well be more
        urgent, but urgency is not confidence, and `build_assertion` caps an inference at its
        weakest premise precisely so inference cannot manufacture certainty. Hop count in the
        proof tree is the honest record of how far this reached."""
        onto = self._pack(tmp_path, monkeypatch)
        ctx = AuthContext(user_id="a@b.example", tenant_id=TENANT)
        facts = [
            fact("party:l1", "OWNS", "party:l2", confidence=0.95),
            fact("party:l2", "OWNS", "party:l3", confidence=0.95),
            fact("party:l3", "OWNS", "party:l4", confidence=0.95),
            fact("party:rival", "ADVERSE_TO", "party:l4", confidence=0.95),
        ]
        report = Reasoner(onto).run(ctx, facts)
        by_target = {i.assertion.object_id: i.assertion.confidence for i in report.inferences}

        assert by_target["party:l3"] == pytest.approx(0.95 * 0.95)
        assert by_target["party:l2"] == pytest.approx(0.95 * 0.95**2)
        assert by_target["party:l1"] == pytest.approx(0.95 * 0.95**3)
        load_ontology.cache_clear()

    def test_a_decayed_conflict_would_fall_under_the_governance_floor(
        self, tmp_path, monkeypatch
    ):
        """Why vetoes had to stop being filtered by confidence first. A 3-hop conclusion off a
        0.95 premise lands under 0.8, so had the trust floor still applied to blocks this
        conclusion would be derived, approved, and then dropped by the veto path -- a conflict
        check returning nothing for a reason nobody could see."""
        onto = self._pack(tmp_path, monkeypatch)
        ctx = AuthContext(user_id="a@b.example", tenant_id=TENANT)
        facts = [
            fact("party:l1", "OWNS", "party:l2", confidence=0.9),
            fact("party:l2", "OWNS", "party:l3", confidence=0.9),
            fact("party:l3", "OWNS", "party:l4", confidence=0.9),
            fact("party:rival", "ADVERSE_TO", "party:l4", confidence=0.9),
        ]
        report = Reasoner(onto).run(ctx, facts)
        far = next(i for i in report.inferences if i.assertion.object_id == "party:l1")
        assert far.assertion.confidence < 0.8
        load_ontology.cache_clear()

    def test_an_unreviewed_link_breaks_the_chain(self, tmp_path, monkeypatch):
        """Every step is held to the same gate as an ordinary premise. A model's guess in the
        middle of a ladder must not carry a conclusion across it."""
        onto = self._pack(tmp_path, monkeypatch)
        ctx = AuthContext(user_id="a@b.example", tenant_id=TENANT)
        facts = [
            fact("party:top", "OWNS", "party:mid"),
            fact("party:mid", "OWNS", "party:low", review_state=ReviewState.PENDING),
            fact("party:rival", "ADVERSE_TO", "party:low"),
        ]
        reached = {i.assertion.object_id for i in Reasoner(onto).run(ctx, facts).inferences}
        assert reached == set()
        load_ontology.cache_clear()


class TestOnlyARuleMayConcludeAConclusion:
    """The conclusions have to be governing predicates, and that had a consequence I missed.

    Declaring them so that `build_assertion` accepts them, and so a typo in a rule fails at pack
    load, also put them in the list handed to the extractor's prompt. Production showed the
    result: five `POTENTIAL_CONFLICT` assertions with epistemic class EXTRACTED_MODEL -- a model
    asserting a conflict it thought it read on a page, with no premises and nothing to defend.

    That is precisely the collapse the epistemic axis exists to prevent. A conflict a rule
    derived from two signed-off facts and one a model guessed at would be the same kind of
    object, and only one of them can be followed back to a document.
    """

    def test_a_conclusion_is_not_offered_to_the_extractor(self):
        onto = load_ontology("legal")
        assert "POTENTIAL_CONFLICT" not in onto.extractable_predicates
        assert "RELIES_ON_STALE_AUTHORITY" not in onto.extractable_predicates

    def test_ordinary_predicates_are_still_offered(self):
        onto = load_ontology("legal")
        assert {"REPRESENTS", "ADVERSE_TO", "CITES", "MENTIONS"} <= onto.extractable_predicates

    def test_the_healthcare_conclusion_is_excluded_too(self):
        onto = load_ontology("healthcare")
        assert "CONTRAINDICATION_ALERT" not in onto.extractable_predicates
        assert "PRESCRIBED" in onto.extractable_predicates

    def test_a_proposed_conclusion_is_refused_at_the_boundary(self):
        """Not merely omitted from the prompt. A prompt is a request; this is the boundary, and a
        model that proposes one anyway must be refused rather than trusted to have read the
        instructions."""
        from src.documents.extractors.model import ModelExtractor, ProposedClaim
        from src.documents.models import Chunk

        onto = load_ontology("legal")
        extractor = ModelExtractor(onto, model_id="test", region="us-east-1")
        chunk = Chunk(
            chunk_id="c1",
            document_id="doc-1",
            tenant_id=TENANT,
            matter_id=NTL,
            page=1,
            ordinal=0,
            char_start=0,
            char_end=41,
            text="Meridian is adverse to Calder Shipping AG",
            filename="f.pdf",
        )
        claims = [
            ProposedClaim(
                subject_id="matter:NTL",
                predicate="POTENTIAL_CONFLICT",
                object_id="party:calder",
                quote="Meridian is adverse to Calder",
                confidence=0.9,
            )
        ]
        assert extractor.validate(claims, chunk=chunk) == []


class TestApprovalMakesAFactLive:
    """Approving is what makes a fact live. Separating the two hid every approval.

    `promote()` is called only during ingest, when nothing has been approved yet, so a reviewer's
    decision was never promoted by anything. Four approved facts sat at STAGED indefinitely: the
    UI reported success, the graph stored APPROVED, and no read path returned them, because
    `live_assertions` filters on lifecycle.
    """

    def test_an_approved_fact_becomes_live(self, ctx):
        from src.documents.review import Lifecycle, ReviewQueue

        queue = ReviewQueue()
        fact_a = fact("counsel:us", "REPRESENTS", "party:acme", review_state=ReviewState.PENDING)
        queue.stage(ctx, [fact_a])
        record = queue.approve(ctx, fact_a.assertion_id)

        assert record.lifecycle is Lifecycle.LIVE

    def test_an_approved_fact_is_returned_by_the_live_read(self, ctx):
        """The read the reasoner and the graph explorer both use."""
        from src.documents.review import ReviewQueue

        queue = ReviewQueue()
        fact_a = fact("counsel:us", "REPRESENTS", "party:acme", review_state=ReviewState.PENDING)
        queue.stage(ctx, [fact_a])
        queue.approve(ctx, fact_a.assertion_id)

        live = [r.assertion.assertion_id for r in queue.live_assertions(ctx)]
        assert fact_a.assertion_id in live

    def test_an_unapproved_fact_stays_out_of_the_live_read(self, ctx):
        """The gate still holds: approving is what promotes, so not approving does not."""
        from src.documents.review import ReviewQueue

        queue = ReviewQueue()
        fact_a = fact("counsel:us", "REPRESENTS", "party:acme", review_state=ReviewState.PENDING)
        queue.stage(ctx, [fact_a])

        assert queue.live_assertions(ctx) == []

    def test_approving_both_premises_lets_the_reasoner_fire(self, ctx):
        """End to end: this is the chain that was broken. Two facts approved through the queue,
        then the reasoner over what is live."""
        from src.documents.review import ReviewQueue

        queue = ReviewQueue()
        premises = [
            fact("matter:" + NTL, "ADVERSE_TO", "party:calder", review_state=ReviewState.PENDING),
            fact(
                "counsel:us",
                "REPRESENTS",
                "party:calder",
                matter=MBC,
                review_state=ReviewState.PENDING,
            ),
        ]
        queue.stage(ctx, premises)
        for p in premises:
            queue.approve(ctx, p.assertion_id)

        live = [r.assertion for r in queue.live_assertions(ctx)]
        report = Reasoner(load_ontology("legal")).run(ctx, live)
        assert report.count == 1
        assert report.inferences[0].assertion.predicate == "POTENTIAL_CONFLICT"
