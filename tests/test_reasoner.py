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
from src.ontology.loader import load_ontology
from src.ontology.patterns import PatternError, parse_edge, parse_rule
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

    def test_no_conflict_when_the_parties_differ(self, ctx):
        """The join is the rule. Acting for one company and against another is normal work."""
        facts = [
            fact("matter:" + NTL, "ADVERSE_TO", "party:calder"),
            fact("counsel:thorne-vaux", "REPRESENTS", "party:northwind", matter=MBC),
        ]
        assert Reasoner(load_ontology("legal")).run(ctx, facts).count == 0


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
        report = Reasoner(load_ontology("legal")).run(ctx, _conflict_facts())
        out = report.to_dict()
        assert out["count"] == 1
        assert out["rules_evaluated"] == 2
        assert out["facts_considered"] == 2
        assert out["inferences"][0]["rule_id"] == "conflict_check"

    def test_no_facts_is_not_an_error(self, ctx):
        report = Reasoner(load_ontology("legal")).run(ctx, [])
        assert report.count == 0
        assert report.rules_evaluated == 2


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
        for domain in ("legal", "healthcare"):
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

        with pytest.raises(PatternError, match="no variable shared"):
            parse_rule(Unjoined())

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
        assert body["rules_evaluated"] == 2

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
        for domain in ("legal", "healthcare"):
            assert load_ontology(domain).rules


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
