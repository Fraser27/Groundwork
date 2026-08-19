"""The rule-block half of the ethical wall, against the real `GraphReader`.

Every test here builds `src.query.graph_reader.GraphReader` and, where a conflict is needed,
derives it with the real `Reasoner` from the real pack. No fake reader appears in this file, and
that is the point of the file existing.

`blocking_facts` was called by `blocks_for` and defined only on the fakes in `test_planner.py`
and `test_resolver_tiers.py`. So the call raised `AttributeError` on every request, a handler
swallowed it at debug level, and step 4 of the trace reported "nothing refused" while being
structurally incapable of refusing. A suite that proves the design instead of the system.
"""

from __future__ import annotations

import pytest

from src.documents.review import InMemoryAssertionStore, ReviewQueue
from src.governance import GovernanceSettings
from src.graph.assertions import (
    EpistemicClass,
    ReviewState,
    SourceLocator,
    build_assertion,
)
from src.graph.scope import AuthContext
from src.ontology.loader import load_ontology
from src.query.blocks import BlockCheckUnavailable, blocks_for
from src.query.graph_reader import GraphReader
from src.query.planner import Planner
from src.query.resolver import Resolver, Tier
from src.reasoning.engine import Reasoner

TENANT = "firm-acme"


@pytest.fixture
def ctx() -> AuthContext:
    return AuthContext(user_id="alice@firm.com", tenant_id=TENANT)


def _live(ctx: AuthContext, queue: ReviewQueue, facts, job: str) -> None:
    """Stage, approve what needs it, and promote. What a reviewer would do."""
    onto = load_ontology("legal")
    assertions = [
        build_assertion(
            tenant_id=TENANT,
            subject_id=s,
            predicate=p,
            object_id=o,
            epistemic_class=EpistemicClass.EXTRACTED_MODEL,
            method="llm:opus-5",
            confidence=0.9,
            source_locator=SourceLocator(document_id="d1", filename="d1.pdf", page=1, quote="text"),
            matter_id=m,
            allowed_predicates=onto.extractable_predicates,
        )
        for s, p, o, m in facts
    ]
    queue.stage(ctx, assertions, job_id=job)
    for a in assertions:
        if a.review_state is ReviewState.PENDING:
            queue.approve(ctx, a.assertion_id)
    queue.promote(ctx, job_id=job)


def _reader_with_conflict(ctx: AuthContext) -> GraphReader:
    """A real reader over a real inferred POTENTIAL_CONFLICT.

    The conflict is derived by the real `Reasoner` rather than hand-written, so this also holds
    the join between what the reasoner concludes and what the wall reads. A veto on a predicate
    no rule produces would pass a hand-written test and refuse nothing in production.
    """
    onto = load_ontology("legal")
    queue = ReviewQueue(InMemoryAssertionStore(), governing_predicates=onto.governing_predicates)
    _live(
        ctx,
        queue,
        [
            ("counsel:dalgleish-rowe", "REPRESENTS", "party:calder-plc", "M-1"),
            ("matter:m-2", "ADVERSE_TO", "party:calder-plc", "M-2"),
        ],
        "j1",
    )

    live = [r.assertion for r in queue.visible(ctx) if r.is_current]
    inferences = [i.assertion for i in Reasoner(onto).run(ctx, live).inferences]
    assert any(a.predicate == "POTENTIAL_CONFLICT" for a in inferences), (
        "the reasoner produced no conflict, so this fixture proves nothing about the wall"
    )
    queue.stage(ctx, inferences, job_id="j2")
    for a in inferences:
        if a.review_state is ReviewState.PENDING:
            queue.approve(ctx, a.assertion_id)
    queue.promote(ctx, job_id="j2")
    return GraphReader(queue, ontology=onto)


def _reader_with_pending_conflict(ctx: AuthContext) -> GraphReader:
    """A conflict the reasoner derived and nobody has reviewed. Staged, never approved.

    This is the state the deployed system was in: the facts supported a conflict, inference had
    not run, and once it did the conclusion sat PENDING. The wall correctly refuses nothing on it,
    which is exactly why the reader has to be told it is there.
    """
    onto = load_ontology("legal")
    queue = ReviewQueue(InMemoryAssertionStore(), governing_predicates=onto.governing_predicates)
    _live(
        ctx,
        queue,
        [
            ("counsel:dalgleish-rowe", "REPRESENTS", "party:calder-plc", "M-1"),
            ("matter:m-2", "ADVERSE_TO", "party:calder-plc", "M-2"),
        ],
        "j1",
    )
    live = [r.assertion for r in queue.visible(ctx) if r.is_current]
    inferences = [i.assertion for i in Reasoner(onto).run(ctx, live).inferences]
    assert inferences, "the fixture proves nothing if the reasoner drew no conclusion"
    queue.stage(ctx, inferences, job_id="j2")  # staged and left PENDING on purpose
    return GraphReader(queue, ontology=onto)


class TestAConflictAwaitingReviewIsNotSilent:
    """An unreviewed conflict may not refuse anything, and may not be invisible either.

    Both halves matter. Firing on it would let a model's proposal withhold evidence, which is what
    `SIGNED_OFF` exists to prevent. But reporting "nothing refused" over a graph that holds a
    conflict about this very party is a true sentence which reads as a false one — and a model
    narrating the conflict in prose while the deterministic control said nothing is how this was
    found in production.
    """

    def test_it_still_refuses_nothing(self, ctx):
        reader = _reader_with_pending_conflict(ctx)
        assert reader.blocking_facts(ctx, ["party:calder-plc"]) == []

    def test_but_it_is_reported(self, ctx):
        reader = _reader_with_pending_conflict(ctx)
        rows = reader.unreviewed_blocks(ctx, ["party:calder-plc"])
        assert [r["predicate"] for r in rows] == ["POTENTIAL_CONFLICT"]
        assert rows[0]["rule"] == "conflict_check"

    def test_the_screen_names_the_party(self, ctx):
        screen = blocks_for(
            ctx, graph_reader=_reader_with_pending_conflict(ctx), seeds=["party:calder-plc"]
        )
        assert "party:calder-plc" in screen.awaiting_review

    def test_an_advisory_alone_does_not_make_the_screen_truthy(self, ctx):
        """`bool(screen)` drives whether rows get withheld. An unreviewed conflict must not start
        withholding evidence by the back door."""
        screen = blocks_for(
            ctx, graph_reader=_reader_with_pending_conflict(ctx), seeds=["party:calder-plc"]
        )
        assert not screen
        assert screen.blocks == []

    def test_it_reaches_the_trace(self, ctx):
        screen = blocks_for(
            ctx, graph_reader=_reader_with_pending_conflict(ctx), seeds=["party:calder-plc"]
        )
        # Both ends of the edge, because a conflict is about the matter as well as the party --
        # the same reason `POTENTIAL_CONFLICT` declares `blocks: both`.
        assert screen.trace(items_withheld=0)["awaiting_review"] == [
            "matter:m-2",
            "party:calder-plc",
        ]

    def test_an_approved_conflict_is_a_veto_not_an_advisory(self, ctx):
        """Once signed off it is reported by `blocking_facts`. Naming it in both places would
        double every refusal and make the advisory look like a second, weaker wall."""
        reader = _reader_with_conflict(ctx)
        assert reader.blocking_facts(ctx, ["party:calder-plc"])
        assert reader.unreviewed_blocks(ctx, ["party:calder-plc"]) == []

    def test_a_clean_graph_reports_nothing(self, ctx):
        reader = _reader_with_pending_conflict(ctx)
        assert reader.unreviewed_blocks(ctx, ["party:unrelated-gmbh"]) == []

    def test_another_tenant_sees_nothing(self, ctx):
        """Advisory or not, it is a graph read and scoped like every other one."""
        reader = _reader_with_pending_conflict(ctx)
        other = AuthContext(user_id="mallory@other.com", tenant_id="firm-beta")
        assert reader.unreviewed_blocks(other, ["party:calder-plc"]) == []

    def test_a_degraded_wall_reports_no_advisory(self, ctx):
        """"A conflict is awaiting review" would read as confirmation the check ran, which is the
        opposite of what `degraded` is saying."""
        screen = blocks_for(ctx, graph_reader=_BrokenReader(), seeds=["party:calder-plc"])
        assert screen.degraded
        assert screen.awaiting_review == ()


class TestTheRealReaderCanRefuse:
    """`GraphReader`, not a fake. These are the tests the fakes were standing in for."""

    def test_blocking_facts_exists_on_the_real_class(self):
        """The whole bug in one line. `blocks_for` has called this since it was written."""
        assert callable(getattr(GraphReader, "blocking_facts", None))

    def test_an_inferred_conflict_blocks_the_party(self, ctx):
        reader = _reader_with_conflict(ctx)
        found = reader.blocking_facts(ctx, ["party:calder-plc"])
        assert "party:calder-plc" in {row["subject_id"] for row in found}

    def test_a_block_names_the_rule_that_produced_it(self, ctx):
        """`rule` comes off the assertion, so a reader can find the proof tree."""
        found = _reader_with_conflict(ctx).blocking_facts(ctx, ["party:calder-plc"])
        assert {row["rule"] for row in found} == {"conflict_check"}

    def test_every_field_blocks_for_reads_is_populated(self, ctx):
        """`blocks_for` reads the first four. A missing `reason` renders as "blocked", which is
        a refusal with no explanation -- worse than no block, because it looks like a screen
        nobody can appeal. `confidence` rides along so a reviewer can see how firm the veto is;
        nothing filters on it."""
        for row in _reader_with_conflict(ctx).blocking_facts(ctx, ["party:calder-plc"]):
            assert set(row) == {
                "subject_id",
                "reason",
                "rule",
                "matter_id",
                "confidence",
                "premise_count",
                "effect",
            }
            assert row["subject_id"] and row["reason"] and row["rule"]

    def test_the_reason_names_the_other_end_by_its_real_id(self, ctx):
        """`matter:m-2`, not "m 2". A prettified id is not something anyone can look up."""
        found = _reader_with_conflict(ctx).blocking_facts(ctx, ["party:calder-plc"])
        reasons = " ".join(row["reason"] for row in found)
        assert "matter:m-2" in reasons

    def test_an_ordinary_governing_fact_does_not_block(self, ctx):
        """REPRESENTS is governing and central to a conflict check, and blocks nothing on its
        own. Blocking on it would withhold the firm's own client list."""
        onto = load_ontology("legal")
        queue = ReviewQueue(
            InMemoryAssertionStore(), governing_predicates=onto.governing_predicates
        )
        _live(ctx, queue, [("counsel:dr", "REPRESENTS", "party:calder-plc", "M-1")], "j1")
        assert GraphReader(queue, ontology=onto).blocking_facts(ctx, ["party:calder-plc"]) == []

    def test_a_seed_the_conflict_does_not_touch_is_not_blocked(self, ctx):
        reader = _reader_with_conflict(ctx)
        assert reader.blocking_facts(ctx, ["party:unrelated-gmbh"]) == []

    def test_no_seeds_means_no_blocks(self, ctx):
        assert _reader_with_conflict(ctx).blocking_facts(ctx, []) == []

    def test_repeated_reads_agree(self, ctx):
        """Two runs of one question returning different vetoes is indistinguishable from the
        graph having changed."""
        reader = _reader_with_conflict(ctx)
        first = reader.blocking_facts(ctx, ["party:calder-plc"])
        assert first == reader.blocking_facts(ctx, ["party:calder-plc"])


class TestThePackDecidesWhatAFindingDoes:
    """`blocks:` says which end a finding is about; `effect:` says what happens because of it.

    Two axes, because collapsing them would lose the endpoint. Three different things routed
    through one mechanism whose only behaviour was to drop the row, and only one of them is a wall.
    """

    def test_a_conflict_notifies(self):
        assert load_ontology("legal").effect_of("POTENTIAL_CONFLICT") == "notify"

    def test_a_conflict_is_still_a_finding(self):
        """Notify must not mean invisible. It stays in the set `blocking_facts` scans, or the
        reader is told nothing at all -- which is worse than withholding."""
        onto = load_ontology("legal")
        assert "POTENTIAL_CONFLICT" in onto.finding_predicates
        assert "POTENTIAL_CONFLICT" not in onto.blocking_predicates

    def test_the_endpoint_survives_the_effect(self):
        """Why `effect:` is a second key rather than a `blocks: notify` value. Spelling it as an
        endpoint would erase which end the finding is about, and the reason text is built from it."""
        assert load_ontology("legal").blocked_endpoints("POTENTIAL_CONFLICT") == ("object",)

    def test_stale_authority_notifies_too(self):
        """A quality finding, not a conduct one: nobody is harmed by seeing their own memo, and
        suppressing it hides the very advice that needs revising."""
        assert load_ontology("legal").effect_of("RELIES_ON_STALE_AUTHORITY") == "notify"

    def test_no_rule_finding_in_the_legal_pack_withholds(self):
        """The shape this arrives at, stated as an invariant. An ethical screen is the only wall
        left: a recorded instruction that a person must not see a matter. A rule finding is
        something the graph noticed and a lawyer decides about."""
        assert load_ontology("legal").blocking_predicates == frozenset()
        assert load_ontology("legal").finding_predicates

    def test_a_healthcare_contraindication_still_withholds(self):
        """The packs are allowed to disagree, and this one should: a clinician acting on a
        suppressed allergy is direct harm, which is not true of a conflict a lawyer must weigh."""
        onto = load_ontology("healthcare")
        assert onto.effect_of("CONTRAINDICATION_ALERT") == "withhold"
        assert "CONTRAINDICATION_ALERT" in onto.blocking_predicates

    def test_withhold_is_the_default(self):
        """Forgetting the flag must fail in the loud direction. A default of notify would silently
        un-veto every block in both packs."""
        assert load_ontology("legal").effect_of("REPRESENTS") == "withhold"
        assert load_ontology("legal").effect_of("NOT_A_PREDICATE") == "withhold"

    def test_an_unreadable_effect_fails_the_pack(self, tmp_path):
        from src.ontology import loader

        pack = tmp_path / "bad.yaml"
        pack.write_text(
            "domain: bad\nversion: 1\nentity_types: []\n"
            "governing_predicates:\n  - id: X\n    blocks: object\n    effect: maybe\n"
            "descriptive_predicates: []\nrules: []\n"
        )
        with pytest.raises(ValueError, match="effect"):
            loader._parse(__import__("yaml").safe_load(pack.read_text()))

    def test_an_effect_with_no_finding_fails_the_pack(self, tmp_path):
        """A consequence attached to nothing. `effect:` says what happens *because of* a finding,
        so without `blocks:` there is no finding for it to apply to."""
        from src.ontology import loader

        pack = tmp_path / "bad2.yaml"
        pack.write_text(
            "domain: bad\nversion: 1\nentity_types: []\n"
            "governing_predicates:\n  - id: X\n    effect: notify\n"
            "descriptive_predicates: []\nrules: []\n"
        )
        with pytest.raises(ValueError, match="no finding"):
            loader._parse(__import__("yaml").safe_load(pack.read_text()))


class TestThePackDecidesWhatBlocks:
    """Rule 5: the vocabulary is closed, so a veto cannot be a list in Python."""

    def test_both_packs_declare_a_finding(self):
        """A finding expressible in the legal pack and not the healthcare one would make this a
        legal feature the generic code happens to run.

        `finding_predicates` rather than `blocking_predicates`: the legal pack now declares both of
        its findings `effect: notify`, so nothing there withholds and an ethical screen is the only
        wall. That is the intended shape, and it is why the invariant is "a finding is declared and
        queryable" rather than "something vetoes"."""
        assert load_ontology("legal").finding_predicates
        assert load_ontology("healthcare").finding_predicates

    def test_every_blocking_predicate_is_governing(self):
        """A descriptive predicate is open, so a veto resting on one could be minted by any
        extractor inventing a tag."""
        for domain in ("legal", "healthcare"):
            onto = load_ontology(domain)
            assert onto.blocking_predicates <= onto.governing_predicates

    def test_stale_authority_blocks_the_document_not_the_authority(self):
        """The one place the endpoint choice is load-bearing. Blocking the authority would
        suppress "Brown was overruled" -- the single fact the reader most needs."""
        onto = load_ontology("legal")
        assert onto.blocked_endpoints("RELIES_ON_STALE_AUTHORITY") == ("subject",)

    def test_a_conflict_blocks_the_party_not_the_matter(self):
        """`both` blacked out the matter as well -- the file the disputes team is retained to run,
        on a memo whose own decision was "may be accepted subject to an information barrier".
        Withholding the party is the barrier's substance; withholding the matter withholds the
        work. A conflict about a party the firm also represents made that client's own file
        unanswerable."""
        assert load_ontology("legal").blocked_endpoints("POTENTIAL_CONFLICT") == ("object",)

    def test_a_conflict_does_not_withhold_the_matter_it_is_about(self, ctx):
        """The behaviour, not just the declaration. A matter-subject conflict is the shape the
        pack declares, and its matter has to stay readable."""
        onto = load_ontology("legal")
        queue = ReviewQueue(
            InMemoryAssertionStore(), governing_predicates=onto.governing_predicates
        )
        _live(
            ctx,
            queue,
            [
                ("counsel:dalgleish-rowe", "REPRESENTS", "party:calder-plc", "M-1"),
                ("matter:m-2", "ADVERSE_TO", "party:calder-plc", "M-2"),
            ],
            "j1",
        )
        live = [r.assertion for r in queue.visible(ctx) if r.is_current]
        inferences = [i.assertion for i in Reasoner(onto).run(ctx, live).inferences]
        assert inferences, "the fixture proves nothing if the rule did not fire"
        queue.stage(ctx, inferences, job_id="j2")
        for a in inferences:
            queue.approve(ctx, a.assertion_id)
        queue.promote(ctx, job_id="j2")

        reader = GraphReader(queue, ontology=onto)
        subjects = {row["subject_id"] for row in reader.blocking_facts(ctx, ["matter:m-2"])}
        assert subjects == {"party:calder-plc"}
        assert "matter:m-2" not in subjects

    def test_a_contraindication_blocks_the_drug_not_the_patient(self):
        """Suppressing evidence about the patient would withhold the record in order to
        protect the record."""
        onto = load_ontology("healthcare")
        assert onto.blocked_endpoints("CONTRAINDICATION_ALERT") == ("object",)

    def test_a_non_blocking_predicate_taints_nothing(self):
        assert load_ontology("legal").blocked_endpoints("REPRESENTS") == ()

    def test_an_unreadable_blocks_value_fails_the_pack(self, tmp_path):
        """Loud at load. Skipping it would leave a predicate the author wrote as a veto
        informing answers instead of forbidding them, which nothing downstream can detect."""
        from src.ontology import loader

        pack = tmp_path / "bad.yaml"
        pack.write_text(
            "domain: bad\nversion: 1\nentity_types: []\n"
            "governing_predicates:\n  - id: X\n    blocks: sideways\n"
            "descriptive_predicates: []\nrules: []\n"
        )
        with pytest.raises(ValueError, match="blocks"):
            loader._parse(__import__("yaml").safe_load(pack.read_text()))

    def test_a_descriptive_predicate_may_not_veto(self, tmp_path):
        from src.ontology import loader

        pack = tmp_path / "bad2.yaml"
        pack.write_text(
            "domain: bad\nversion: 1\nentity_types: []\ngoverning_predicates: []\n"
            "descriptive_predicates:\n  - id: TAG\n    blocks: subject\nrules: []\n"
        )
        with pytest.raises(ValueError, match="descriptive"):
            loader._parse(__import__("yaml").safe_load(pack.read_text()))


class TestAVetoIsNotWeighedAgainstAFloor:
    """The one gate a block does *not* inherit from retrieval.

    The trust floor decides what may inform an answer. A veto does not inform one -- it refuses
    it -- and filtering refusals by confidence means the least certain conflict is the one
    silently dropped. A check returning nothing because the veto fell under a floor is
    indistinguishable from a clean check, which is the single failure a conflict check may not
    have. It matters more now a conclusion decays per hop: a conflict reached four steps out is
    exactly the one nobody finds by hand.
    """

    def _with_weak_conflict(self, ctx: AuthContext, confidence: float) -> GraphReader:
        """A live conflict sitting below the 0.8 governance floor, as per-hop decay produces."""
        reader = _reader_with_conflict(ctx)
        for record in reader._queue.visible(ctx):
            if record.assertion.predicate == "POTENTIAL_CONFLICT":
                record.assertion.confidence = confidence
        return reader

    def test_a_conflict_below_the_floor_still_refuses(self, ctx):
        """0.77 is what a 4-hop chain off a 0.95 premise decays to."""
        reader = self._with_weak_conflict(ctx, 0.77)
        found = reader.blocking_facts(ctx, ["party:calder-plc"])
        assert "party:calder-plc" in {row["subject_id"] for row in found}

    def test_the_weak_confidence_travels_so_a_reviewer_can_judge_it(self, ctx):
        """Reported, not filtered on. How firm the veto is stays a reviewer's call."""
        reader = self._with_weak_conflict(ctx, 0.77)
        found = reader.blocking_facts(ctx, ["party:calder-plc"])
        assert {row["confidence"] for row in found} == {0.77}

    def test_a_very_weak_conflict_still_refuses(self, ctx):
        """There is no floor at all, not merely a lower one."""
        reader = self._with_weak_conflict(ctx, 0.05)
        assert reader.blocking_facts(ctx, ["party:calder-plc"])

    def test_the_floor_is_not_a_parameter_anyone_can_reinstate(self, ctx):
        """A `min_confidence` argument would let one caller quietly restore the bug."""
        import inspect

        assert "min_confidence" not in inspect.signature(GraphReader.blocking_facts).parameters


class TestTheFurthestReachingRefusalIsShownFirst:
    """Which refusal a reader sees first, and why it is not the most confident one.

    A direct conflict is one a partner has probably already spotted; one derived from several
    facts across separate matters is not visible in any single document by construction. Ordering
    these alphabetically -- what this did before -- buried the only finding nobody could have
    reached unaided. Reach and confidence move in opposite directions here, since each hop decays
    the conclusion, so ranking on confidence would put the obvious case on top.
    """

    def _reader_with_both(self, ctx: AuthContext) -> GraphReader:
        """A direct conflict on one party and an affiliate-derived one on another.

        Both are produced by the real `Reasoner`, so the premise counts are the ones production
        would see rather than numbers chosen to make the sort look right.
        """
        onto = load_ontology("legal")
        queue = ReviewQueue(
            InMemoryAssertionStore(), governing_predicates=onto.governing_predicates
        )
        _live(
            ctx,
            queue,
            [
                # Direct: represented and opposed, same party. Two premises.
                ("counsel:dalgleish-rowe", "REPRESENTS", "party:obvious-ltd", "M-1"),
                ("matter:m-2", "ADVERSE_TO", "party:obvious-ltd", "M-2"),
                # Indirect: reached only through the affiliation. Three premises.
                ("counsel:dalgleish-rowe", "REPRESENTS", "party:northwind", "M-4"),
                ("party:northwind", "AFFILIATE_OF", "party:calder-plc", "M-5"),
                ("matter:m-6", "ADVERSE_TO", "party:calder-plc", "M-6"),
            ],
            "j1",
        )
        live = [r.assertion for r in queue.visible(ctx) if r.is_current]
        inferences = [i.assertion for i in Reasoner(onto).run(ctx, live).inferences]
        queue.stage(ctx, inferences, job_id="j2")
        for a in inferences:
            if a.review_state is ReviewState.PENDING:
                queue.approve(ctx, a.assertion_id)
        queue.promote(ctx, job_id="j2")
        return GraphReader(queue, ontology=onto)

    def test_the_indirect_conflict_outranks_the_direct_one(self, ctx):
        reader = self._reader_with_both(ctx)
        found = reader.blocking_facts(ctx, ["party:northwind", "party:obvious-ltd"])
        counts = [row["premise_count"] for row in found]

        assert counts == sorted(counts, reverse=True)
        assert counts[0] > counts[-1], "the fixture must contain refusals of differing reach"

    def test_the_premise_count_is_the_real_one(self, ctx):
        """Three for the affiliate route, two for the direct one. Read off the assertion, so it
        cannot drift from what the proof tree actually holds."""
        reader = self._reader_with_both(ctx)
        found = reader.blocking_facts(ctx, ["party:northwind", "party:obvious-ltd"])
        by_subject = {row["subject_id"]: row["premise_count"] for row in found}

        assert by_subject["party:northwind"] == 3
        assert by_subject["party:obvious-ltd"] == 2

    def test_a_screen_sorts_below_a_derived_conflict(self, ctx):
        """A screen is an instruction about a matter the reader knows exists; a derived conflict
        is news. Listing the known thing first buries the discovered one."""
        screened = AuthContext(
            user_id="alice@firm.com",
            tenant_id=TENANT,
            matter_denylist=frozenset({"M-9"}),
            screen_reasons={"M-9": "acted for the opposing party"},
        )
        screen = blocks_for(
            screened,
            graph_reader=self._reader_with_both(screened),
            seeds=["party:northwind"],
        )
        rules = [b.rule for b in screen.blocks]

        assert "ethical_screen" in rules
        assert rules.index("conflict_via_affiliate") < rules.index("ethical_screen")

    def test_ordering_is_stable_across_runs(self, ctx):
        """Two runs of one conflict check listing refusals differently is indistinguishable from
        the graph having changed."""
        seeds = ["party:northwind", "party:obvious-ltd"]
        first = self._reader_with_both(ctx).blocking_facts(ctx, seeds)
        second = self._reader_with_both(ctx).blocking_facts(ctx, seeds)
        assert [r["subject_id"] for r in first] == [r["subject_id"] for r in second]


class TestTheRestOfTheTrustPolicyIsNotBypassed:
    """Dropping the floor drops *only* the floor. Every other gate a veto inherits stays."""

    def test_a_pending_conflict_does_not_block(self, ctx):
        """`_readable` admits only signed-off facts. A veto on an unreviewed model guess would
        withhold evidence on the strength of something nobody stands behind."""
        onto = load_ontology("legal")
        queue = ReviewQueue(
            InMemoryAssertionStore(), governing_predicates=onto.governing_predicates
        )
        pending = build_assertion(
            tenant_id=TENANT,
            subject_id="matter:m-2",
            predicate="POTENTIAL_CONFLICT",
            object_id="party:calder-plc",
            epistemic_class=EpistemicClass.EXTRACTED_MODEL,
            method="llm:opus-5",
            confidence=0.95,
            source_locator=SourceLocator(document_id="d1", page=1, quote="t"),
            matter_id="M-2",
            allowed_predicates=onto.governing_predicates,
        )
        assert pending.review_state is ReviewState.PENDING
        queue.stage(ctx, [pending], job_id="j1")
        reader = GraphReader(queue, ontology=onto)
        assert reader.blocking_facts(ctx, ["party:calder-plc"]) == []

    def test_another_tenant_sees_nothing(self, ctx):
        """The veto reads the graph, so it is scoped like every other read."""
        reader = _reader_with_conflict(ctx)
        other = AuthContext(user_id="mallory@other.com", tenant_id="firm-beta")
        assert reader.blocking_facts(other, ["party:calder-plc"]) == []


class TestFailsOpenButLoudly:
    """Refusing every answer on a graph error is its own outage. Failing open silently is how
    this bug survived. So it fails open and says so."""

    def test_a_reader_with_no_pack_refuses_to_claim_a_clean_wall(self, ctx):
        """Without a pack the blocking vocabulary is unknowable, so "no vetoes" is a claim the
        reader cannot make."""
        onto = load_ontology("legal")
        queue = ReviewQueue(
            InMemoryAssertionStore(), governing_predicates=onto.governing_predicates
        )
        _live(ctx, queue, [("counsel:dr", "REPRESENTS", "party:calder-plc", "M-1")], "j1")
        with pytest.raises(BlockCheckUnavailable):
            GraphReader(queue).blocking_facts(ctx, ["party:calder-plc"])

    def test_the_screen_records_why_it_could_not_check(self, ctx):
        screen = blocks_for(ctx, graph_reader=_BrokenReader(), seeds=["party:calder-plc"])
        assert screen.degraded

    def test_a_degraded_screen_reports_nothing_cleared(self, ctx):
        """Zero, not the seed count. Nothing was cleared; the check was skipped."""
        screen = blocks_for(ctx, graph_reader=_BrokenReader(), seeds=["party:calder-plc"])
        assert screen.cleared == 0
        assert screen.trace(items_withheld=0)["degraded"]

    def test_a_clean_wall_is_not_degraded_and_counts_what_cleared(self, ctx):
        screen = blocks_for(
            ctx, graph_reader=_reader_with_conflict(ctx), seeds=["party:unrelated-gmbh"]
        )
        assert screen.degraded == ""
        assert screen.trace(items_withheld=0) == {
            "seeds_considered": 1,
            "subjects_cleared": 1,
            "subjects_flagged": 0,
            "items_withheld": 0,
            "degraded": None,
            "blocks": [],
            # Empty because the conflict in this fixture is signed off, so it is a veto rather
            # than an advisory -- and it names a different party than the seed anyway.
            "awaiting_review": [],
        }

    def test_screens_still_apply_when_the_graph_check_fails(self, ctx):
        """The half that does not need the graph must survive the half that does."""
        screened = AuthContext(
            user_id="bob@firm.com",
            tenant_id=TENANT,
            matter_denylist=frozenset({"M-9"}),
            screen_reasons={"M-9": "Acting for the counterparty."},
        )
        screen = blocks_for(screened, graph_reader=_BrokenReader(), seeds=["party:calder-plc"])
        assert [b.rule for b in screen.blocks] == ["ethical_screen"]
        assert screen.degraded


class _BrokenReader:
    """A reader whose block check fails. Stands in for a graph outage, not for a missing method."""

    def blocking_facts(self, ctx, seeds, **kw):
        raise BlockCheckUnavailable("the graph is unreachable")


class TestTheApiReportsWhatTheWallDid:
    """The UI has declared `GateTrace` since before anything populated it. `tsc` passed and the
    page told a reader no count was recorded."""

    def test_query_sends_the_gate_counters(self, ctx):
        res = Resolver(graph_reader=_reader_with_conflict(ctx)).resolve(
            ctx, "calder", GovernanceSettings()
        )
        gate = res.to_dict()["gate"]
        assert gate is not None
        assert set(gate) == {
            "seeds_considered",
            "subjects_cleared",
            "subjects_flagged",
            "items_withheld",
            "degraded",
            "blocks",
            "awaiting_review",
        }

    def test_compose_sends_the_gate_counters(self, ctx):
        answer = Planner(graph_reader=_reader_with_conflict(ctx)).plan(
            ctx, "calder", GovernanceSettings()
        )
        assert answer.to_dict()["gate"] is not None

    def test_a_clean_wall_still_sends_a_gate(self, ctx):
        """A step visible only when it blocks reads as an exception rather than as a gate
        everything passed through."""
        res = Resolver(graph_reader=_reader_with_conflict(ctx)).resolve(
            ctx, "unrelated", GovernanceSettings()
        )
        assert res.to_dict()["gate"] is not None

    def test_a_degraded_wall_warns_in_the_response_body(self, ctx):
        """Not only in a log. The bug survived because the handler logged at debug."""
        onto = load_ontology("legal")
        queue = ReviewQueue(
            InMemoryAssertionStore(), governing_predicates=onto.governing_predicates
        )
        _live(ctx, queue, [("counsel:dr", "REPRESENTS", "party:calder-plc", "M-1")], "j1")
        # No ontology, so the real reader cannot say which predicates veto.
        res = Resolver(graph_reader=GraphReader(queue)).resolve(ctx, "calder", GovernanceSettings())
        assert res.to_dict()["gate"]["degraded"]
        assert any("could not be checked for conflicts" in w for w in res.warnings)

    def test_a_degraded_compose_warns_too(self, ctx):
        onto = load_ontology("legal")
        queue = ReviewQueue(
            InMemoryAssertionStore(), governing_predicates=onto.governing_predicates
        )
        _live(ctx, queue, [("counsel:dr", "REPRESENTS", "party:calder-plc", "M-1")], "j1")
        answer = Planner(graph_reader=GraphReader(queue)).plan(ctx, "calder", GovernanceSettings())
        assert answer.to_dict()["gate"]["degraded"]
        assert any("could not be checked for conflicts" in w for w in answer.warnings)

    def test_tier_one_stays_exempt(self, ctx):
        """A compiled metric carries no ids to veto, so there is no gate to report."""
        from src.metrics.loader import load_metrics
        from src.metrics.models import StaticCatalog
        from src.query.metric_matcher import MetricMatcher

        matcher = MetricMatcher(
            load_metrics("sample/metrics.yaml").metrics, StaticCatalog(tables={})
        )
        res = Resolver(metric_matcher=matcher).resolve(
            ctx, "fees billed by month", GovernanceSettings(), execute=False
        )
        assert res.tier is Tier.GOVERNED_METRIC
        assert res.gate is None


class TestProductionWiringCanActuallyRefuse:
    def test_the_wired_reader_has_a_pack(self):
        """Without one, `blocking_facts` raises and the wall degrades on every request for the
        life of the deployment -- which is the failure this whole change exists to end. Asserted
        against the real `build_services`, because the constructor argument is the single point
        where a refactor could switch the veto off without touching the veto.
        """
        from src.api.deps import build_services

        reader = build_services().graph_reader
        assert isinstance(reader, GraphReader)
        assert reader._ontology is not None
        assert reader._ontology.finding_predicates


def _reader_with_stale_authority(ctx: AuthContext) -> GraphReader:
    """A finding that still *withholds*, so the suppression path stays under test.

    `POTENTIAL_CONFLICT` notifies now, so it can no longer stand in for a veto. Stale authority
    is the pack's remaining `effect: withhold` finding in the legal pack.
    """
    onto = load_ontology("legal")
    queue = ReviewQueue(InMemoryAssertionStore(), governing_predicates=onto.governing_predicates)
    _live(
        ctx,
        queue,
        [
            ("document:advice", "CITES", "authority:aquitaine", "M-1"),
            ("authority:marisol", "OVERRULES", "authority:aquitaine", "M-1"),
        ],
        "j1",
    )
    live = [r.assertion for r in queue.visible(ctx) if r.is_current]
    inferences = [i.assertion for i in Reasoner(onto).run(ctx, live).inferences]
    assert any(a.predicate == "RELIES_ON_STALE_AUTHORITY" for a in inferences), (
        "the reasoner drew no stale-authority finding, so this fixture proves nothing"
    )
    queue.stage(ctx, inferences, job_id="j2")
    for a in inferences:
        if a.review_state is ReviewState.PENDING:
            queue.approve(ctx, a.assertion_id)
    queue.promote(ctx, job_id="j2")
    return GraphReader(queue, ontology=onto)


class TestAConflictNotifiesRatherThanWithholds:
    """The distinction the pack now draws, and the reason it draws it.

    Whether a potential conflict is a real one is a lawyer's judgement, and they cannot make it
    from evidence they were never shown. So a conflict names itself, sorts first, carries its
    premises — and suppresses nothing. An ethical screen remains the one true prohibition.
    """

    def test_the_conflict_is_still_reported(self, ctx):
        res = Resolver(graph_reader=_reader_with_conflict(ctx)).resolve(
            ctx, "calder", GovernanceSettings()
        )
        assert res.blocks, "a notify finding must still be named, or nobody learns of it"
        assert [b.rule for b in res.blocks] == ["conflict_check"]

    def test_but_the_evidence_survives(self, ctx):
        """The behaviour the user asked for: notified, not walled."""
        reader = _reader_with_conflict(ctx)
        res = Resolver(graph_reader=reader).resolve(ctx, "calder", GovernanceSettings())
        assert res.gate["items_withheld"] == 0
        assert any(
            row.get("subject_id") == "party:calder-plc" or row.get("object_id") == "party:calder-plc"
            for row in res.answer or []
        ), "the party the conflict is about must still appear in the evidence"

    def test_an_advisory_alone_does_not_make_the_screen_truthy(self, ctx):
        """`bool(screen)` is what drives filtering, so a notify finding must not trip it."""
        screen = blocks_for(
            ctx, graph_reader=_reader_with_conflict(ctx), seeds=["party:calder-plc"]
        )
        assert screen.advisories
        assert not screen.withholding
        assert not screen

    def test_a_flagged_party_is_not_counted_as_cleared(self, ctx):
        """The trap. Counting a flagged subject as cleared would render "3 of 3 cleared" beside a
        conflict notice -- a true sentence that reads as a false one."""
        screen = blocks_for(
            ctx, graph_reader=_reader_with_conflict(ctx), seeds=["party:calder-plc"]
        )
        trace = screen.trace(items_withheld=0)
        assert trace["subjects_cleared"] == 0
        assert trace["subjects_flagged"] == 1

    def test_the_synthesiser_does_see_a_notified_fact(self, ctx):
        """The inverse of the withholding guarantee, and the point of notifying: a model asked to
        summarise a conflict question needs the facts the conflict is about."""

        class Synth:
            def __init__(self) -> None:
                self.saw = ""

            def summarise(self, question, *, parts, blocks) -> str:
                self.saw = str(parts)
                return "summary"

        synth = Synth()
        Planner(graph_reader=_reader_with_conflict(ctx), synthesiser=synth).plan(
            ctx, "calder", GovernanceSettings()
        )
        assert "party:calder-plc" in synth.saw


class TestTheWallActuallyWithholdsEvidence:
    """A block that names a subject and does not remove its rows is a label, not a veto.

    Uses an **ethical screen**, which is now the only thing in the legal pack that withholds.
    Both rule findings declare `effect: notify`, so neither can stand in for the suppression path
    -- and a screen is the more honest subject anyway, since it is the one true prohibition: a
    recorded instruction that a person must not see a matter, whatever else they hold.
    """

    @staticmethod
    def _screened(ctx: AuthContext) -> AuthContext:
        return AuthContext(
            user_id=ctx.user_id,
            tenant_id=ctx.tenant_id,
            matter_denylist=frozenset({"M-1"}),
            screen_reasons={"M-1": "acted for the opposing party"},
        )

    @staticmethod
    def _unscoped():
        """A reader that returns screened rows anyway, so the `Screen` layer is what is tested.

        The real `GraphReader` applies the matter wall in `_readable`, so a screened row never
        reaches `Screen.keep` and `items_withheld` is always zero -- a test resting on the first
        line cannot tell whether the second exists. Same reasoning as `UnscopedReader` in
        `test_resolver_tiers.py`.
        """

        row = {
            "assertion_id": "a1",
            "subject_id": "document:advice",
            "predicate": "CITES",
            "object_id": "authority:aquitaine",
            "matter_id": "M-1",
            "confidence": 0.9,
            "epistemic_class": "EXTRACTED_MODEL",
            "matched_on": ["aquitaine"],
            "source": {},
        }

        class Unscoped:
            def search(self, ctx, question, **kw):
                return [dict(row)]

            def expand(self, ctx, seeds, **kw):
                return [dict(row)]

            def blocking_facts(self, ctx, seeds, **kw):
                return []

        return Unscoped()

    def test_a_blocked_subject_is_dropped_from_the_answer(self, ctx):
        res = Resolver(graph_reader=self._unscoped()).resolve(
            self._screened(ctx), "aquitaine", GovernanceSettings()
        )
        assert res.blocks, "the screen produced no block, so nothing was under test"
        assert res.answer == [], "a screened row must not survive the block check"

    def test_a_blocked_entity_is_dropped_when_it_is_the_object_of_an_edge(self, ctx):
        """`object_id` is in `SEED_KEYS`, so it seeds a block; `Screen.allows` did not test it.
        An edge "d1 MENTIONS party:calder" produced a conflict block and then survived it."""
        from src.query.blocks import Block, Screen

        screen = Screen(blocks=[Block(subject="party:calder-plc", reason="Conflict.", rule="r")])
        rows = [{"subject_id": "document:d1", "object_id": "party:calder-plc"}]
        assert screen.keep(rows) == []

    def test_the_withheld_count_reaches_the_trace(self, ctx):
        res = Resolver(graph_reader=self._unscoped()).resolve(
            self._screened(ctx), "aquitaine", GovernanceSettings()
        )
        assert res.gate["items_withheld"] >= 1

    def test_the_synthesiser_never_sees_a_rule_blocked_fact(self, ctx):
        """Same guarantee the screens have. A model that could reason about a blocked fact
        would reinstate it."""

        class Synth:
            def __init__(self) -> None:
                self.saw = ""

            def summarise(self, question, *, parts, blocks) -> str:
                self.saw = str(parts)
                return "summary"

        synth = Synth()
        Planner(graph_reader=self._unscoped(), synthesiser=synth).plan(
            self._screened(ctx), "aquitaine", GovernanceSettings()
        )
        assert "document:advice" not in synth.saw
