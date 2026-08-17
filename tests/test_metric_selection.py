"""How tier 1 chooses a metric, and how honestly it says so.

The gap this file covers, verified in production. "Whats the money charged for a matter ?" reached
no metric at all: `lg_fees_billed`'s vocabulary is fees/billed/invoiced/charges, the user said
*money* and *charged*, and `MetricMatcher` is pure keyword overlap. The tier router scored the
metric layer 0.554 on the same question -- comfortably the top layer -- so the router said "tier 1
looks right" and the keyword matcher declined anyway. The router widened tier selection and never
metric selection.

Three properties are worth more than the rest of the file:

**An exact word beats a cosine.** The keyword path keeps priority, so a metric the question names
is never displaced by one it merely resembles.

**A barely-ahead candidate declines.** The same reasoning as the keyword tie-break: an arbitrary
pick between two plausible metrics returns a number that looks authoritative.

**A similarity-selected metric is never labelled the way a keyword-selected one is.** Its SQL is
still compiled from an approved definition with no model in the path, but the *choice* of which
definition came from an embedding, so it cannot inherit "the same answer every time".

No AWS. Every dependency is injected.
"""

from __future__ import annotations

import pytest

from src.governance import GovernanceSettings
from src.graph.scope import AuthContext
from src.metrics.models import MetricDefinition, StaticCatalog
from src.query.metric_matcher import (
    MIN_CANDIDATE_MARGIN,
    MIN_CANDIDATE_SIMILARITY,
    SELECTED_BY_KEYWORD,
    SELECTED_BY_ROUTER,
    MetricCandidate,
    MetricMatcher,
)
from src.query.planner import Lane, Planner, Provenance
from src.query.resolver import Resolver, Tier

TENANT = "demo-firm"

#: The metric layer's score on the failing question, from the production container.
PRODUCTION_SIMILARITY = 0.554


@pytest.fixture
def ctx() -> AuthContext:
    return AuthContext(user_id="alice@firm.example", tenant_id=TENANT)


def metrics() -> list[MetricDefinition]:
    """The three approved in production, with their real vocabularies."""
    return [
        MetricDefinition(
            metric_id="lg_fees_billed",
            name="fees_billed",
            synonyms=["fees", "fees billed", "billed", "billings", "invoiced", "charges"],
            definition="Value of recorded time at each fee earner's charge-out rate.",
            expression="SUM(amount_gbp)",
            source_table="lexgraph_legal.time_entries",
            time_grain_column="entry_date",
            time_grains=["day", "month", "quarter", "year"],
        ),
        MetricDefinition(
            metric_id="lg_hours_recorded",
            name="hours_recorded",
            synonyms=["hours", "hours recorded", "time recorded", "time spent"],
            definition="Hours of recorded time, by matter and fee earner.",
            expression="SUM(hours)",
            source_table="lexgraph_legal.time_entries",
        ),
        MetricDefinition(
            metric_id="lg_matter_count",
            name="matter_count",
            synonyms=["matters", "open matters", "matter count", "caseload"],
            definition="Distinct matters, by practice area and status.",
            expression="COUNT(DISTINCT matter_id)",
            source_table="lexgraph_legal.matters",
        ),
    ]


class FakeCandidates:
    """A router stand-in. Records what it was asked so the scope arguments can be asserted."""

    def __init__(self, candidates: list[MetricCandidate] | None = None, fail: bool = False) -> None:
        self.candidates = candidates or []
        self.fail = fail
        self.calls: list[tuple[str, int]] = []

    def metric_candidates(self, ctx, question, *, top_k=5):
        self.calls.append((question, top_k))
        if self.fail:
            raise RuntimeError("opensearch unreachable")
        return self.candidates


def matcher(candidates=None, **kw) -> MetricMatcher:
    return MetricMatcher(
        metrics(), StaticCatalog(tables={}), candidate_source=candidates, **kw
    )


def settings(**over) -> GovernanceSettings:
    return GovernanceSettings(**over)


class TestTheProductionMiss:
    """The question that fell through to tiers 2 and 3 with a governed metric sitting right there."""

    QUESTION = "Whats the money charged for a matter ?"

    def test_keyword_matching_alone_still_misses_it(self):
        """Not a regression to fix in the matcher: *money* and *charged* are semantically the
        metric and lexically absent, and hand-maintaining a synonym per paraphrase is the
        maintenance burden the routing index exists to remove."""
        assert matcher().match(self.QUESTION, AuthContext(user_id="u", tenant_id=TENANT)) is None

    def test_a_routing_candidate_reaches_the_governed_metric(self, ctx):
        candidates = FakeCandidates(
            [
                MetricCandidate("lg_fees_billed", PRODUCTION_SIMILARITY),
                MetricCandidate("lg_matter_count", 0.41),
            ]
        )
        match = matcher(candidates).match(self.QUESTION, ctx, settings())
        assert match is not None
        assert match.metric.metric_id == "lg_fees_billed"

    def test_the_sql_is_still_compiled_from_the_approved_definition(self, ctx):
        """What the router changed is which metric; it did not write anything."""
        candidates = FakeCandidates([MetricCandidate("lg_fees_billed", PRODUCTION_SIMILARITY)])
        match = matcher(candidates).match(self.QUESTION, ctx, settings())
        assert match.compile() == matcher().match("fees billed by matter").compile()


class TestKeywordKeepsPriority:
    """An exact word is stronger evidence of intent than cosine proximity."""

    def test_a_keyword_match_is_not_displaced_by_a_candidate(self, ctx):
        candidates = FakeCandidates([MetricCandidate("lg_matter_count", 0.99)])
        match = matcher(candidates).match("fees billed by matter", ctx, settings())
        assert match.metric.metric_id == "lg_fees_billed"
        assert match.selected_by == SELECTED_BY_KEYWORD

    def test_a_keyword_match_never_asks_for_candidates(self, ctx):
        """Not merely ignored -- an embedding call and a vector search on the latency path of
        every question a keyword already answered."""
        candidates = FakeCandidates([MetricCandidate("lg_matter_count", 0.99)])
        matcher(candidates).match("fees billed by matter", ctx, settings())
        assert candidates.calls == []

    def test_a_keyword_match_reports_no_similarity(self, ctx):
        match = matcher(FakeCandidates()).match("fees billed by matter", ctx, settings())
        assert match.similarity is None
        assert match.selected_deterministically is True


class TestAmbiguityStillDeclines:
    """A wrong metric returns a number that looks authoritative, on either path."""

    def test_two_close_candidates_decline(self, ctx):
        near = MIN_CANDIDATE_MARGIN / 2
        candidates = FakeCandidates(
            [
                MetricCandidate("lg_fees_billed", 0.70),
                MetricCandidate("lg_hours_recorded", 0.70 - near),
            ]
        )
        assert matcher(candidates).match("money charged", ctx, settings()) is None

    def test_a_clear_leader_is_selected(self, ctx):
        """The margin is a threshold, not a refusal to ever choose."""
        candidates = FakeCandidates(
            [
                MetricCandidate("lg_fees_billed", 0.70),
                MetricCandidate("lg_hours_recorded", 0.70 - MIN_CANDIDATE_MARGIN * 2),
            ]
        )
        match = matcher(candidates).match("money charged", ctx, settings())
        assert match.metric.metric_id == "lg_fees_billed"

    def test_a_weak_best_candidate_declines(self, ctx):
        """The selection floor is well above `router_min_similarity`, which decides something
        much cheaper -- whether tier 1 is worth trying at all."""
        candidates = FakeCandidates(
            [MetricCandidate("lg_fees_billed", MIN_CANDIDATE_SIMILARITY - 0.01)]
        )
        assert matcher(candidates).match("something vague", ctx, settings()) is None

    def test_a_lexical_tie_is_not_broken_by_similarity(self, ctx):
        """A tie means two metrics are each lexically plausible. Letting a cosine break it would
        be exactly the arbitrary tie-break the keyword path refuses to make."""
        pack = [
            MetricDefinition(
                metric_id="m_a",
                name="alpha",
                synonyms=["shared word"],
                expression="SUM(x)",
                source_table="db.t",
            ),
            MetricDefinition(
                metric_id="m_b",
                name="beta",
                synonyms=["shared word"],
                expression="SUM(y)",
                source_table="db.t",
            ),
        ]
        candidates = FakeCandidates([MetricCandidate("m_a", 0.95)])
        tied = MetricMatcher(pack, StaticCatalog(tables={}), candidate_source=candidates)
        assert tied.match("shared word please", ctx, settings()) is None
        assert candidates.calls == []


class TestTheLabelDoesNotOverstate:
    """The SQL being reproducible does not make the choice of SQL reproducible."""

    def _router_match(self, ctx):
        candidates = FakeCandidates([MetricCandidate("lg_fees_billed", PRODUCTION_SIMILARITY)])
        return matcher(candidates).match("money charged for a matter", ctx, settings())

    def test_a_router_selected_match_is_not_marked_deterministic(self, ctx):
        match = self._router_match(ctx)
        assert match.selected_by == SELECTED_BY_ROUTER
        assert match.selected_deterministically is False

    def test_the_note_says_the_choice_was_by_similarity(self, ctx):
        note = self._router_match(ctx).selection_note
        assert "similarity" in note
        assert "0.55" in note

    def test_the_note_still_says_the_sql_had_no_ai_in_it(self, ctx):
        """Understating this would be the opposite error, and just as wrong: the figure is an
        exact aggregate over a definition a human approved."""
        assert "no AI" in self._router_match(ctx).selection_note

    def test_a_keyword_note_names_the_words_that_matched(self, ctx):
        match = matcher().match("fees billed by matter", ctx, settings())
        assert "billed" in match.selection_note
        assert "similarity" not in match.selection_note


class TestDegradingWithoutARouter:
    """A deployment with no vector store keeps working, keyword-only."""

    def test_no_candidate_source_is_keyword_only(self, ctx):
        assert matcher().match("money charged for a matter", ctx, settings()) is None
        assert matcher().match("fees billed", ctx, settings()) is not None

    def test_no_ctx_is_keyword_only(self):
        """Every pre-existing caller passes one argument. Without scope there is no tenant index
        to search, and guessing one would search another firm's descriptions."""
        candidates = FakeCandidates([MetricCandidate("lg_fees_billed", 0.99)])
        assert matcher(candidates).match("money charged for a matter") is None

    def test_a_failed_candidate_search_is_a_miss_not_an_error(self, ctx):
        """Tier 1 finding nothing is a fall-through. A broken similarity search must not be worse
        than not having one at all."""
        assert matcher(FakeCandidates(fail=True)).match("money charged", ctx, settings()) is None

    def test_routing_turned_off_disables_the_fallback(self, ctx):
        """An administrator who turned routing off asked for keyword matching, and a switch that
        still let similarity pick the metric would be decorative."""
        candidates = FakeCandidates([MetricCandidate("lg_fees_billed", 0.99)])
        off = settings(router_enabled=False)
        assert matcher(candidates).match("money charged for a matter", ctx, off) is None
        assert candidates.calls == []

    def test_a_candidate_for_an_unknown_metric_is_dropped(self, ctx):
        """A routing record outlives the metric it describes. A stale one winning would select a
        metric this matcher cannot compile, which is a 500 where a fall-through belongs."""
        candidates = FakeCandidates([MetricCandidate("lg_deleted_metric", 0.99)])
        assert matcher(candidates).match("money charged", ctx, settings()) is None

    def test_a_stale_candidate_does_not_crowd_out_a_live_one(self, ctx):
        """Filtered against the pack *before* the margin is applied. Ranking first would let a
        deleted metric's score sit next to the live one and trip the ambiguity check."""
        candidates = FakeCandidates(
            [
                MetricCandidate("lg_deleted_metric", 0.99),
                MetricCandidate("lg_fees_billed", 0.70),
            ]
        )
        match = matcher(candidates).match("money charged", ctx, settings())
        assert match.metric.metric_id == "lg_fees_billed"


class TestBothEndpointsAgree:
    """Two code paths disagreeing about whether a question is governed is a bug class this repo
    has already been bitten by. `/query` and `/query/compose` call the same helper."""

    QUESTION = "Whats the money charged for a matter ?"

    def _matcher(self):
        return matcher(
            FakeCandidates([MetricCandidate("lg_fees_billed", PRODUCTION_SIMILARITY)])
        )

    def test_the_resolver_reaches_tier_1(self, ctx):
        res = Resolver(metric_matcher=self._matcher()).resolve(ctx, self.QUESTION, settings())
        assert res.tier is Tier.GOVERNED_METRIC
        assert res.metric_selection["metric_id"] == "lg_fees_billed"

    def test_the_planner_reaches_the_metric_lane(self, ctx):
        answer = Planner(metric_matcher=self._matcher()).plan(ctx, self.QUESTION, settings())
        assert [p.lane for p in answer.parts] == [Lane.METRIC]
        assert answer.parts[0].metric_selection["metric_id"] == "lg_fees_billed"

    def test_both_report_the_same_selection(self, ctx):
        res = Resolver(metric_matcher=self._matcher()).resolve(ctx, self.QUESTION, settings())
        answer = Planner(metric_matcher=self._matcher()).plan(ctx, self.QUESTION, settings())
        assert res.metric_selection == answer.parts[0].metric_selection

    def test_neither_calls_the_metric_governed_without_qualification(self, ctx):
        res = Resolver(metric_matcher=self._matcher()).resolve(ctx, self.QUESTION, settings())
        answer = Planner(metric_matcher=self._matcher()).plan(ctx, self.QUESTION, settings())
        assert res.selected_deterministically is False
        assert answer.is_fully_deterministic is False
        assert answer.governance_label != "governed"


class TestWhatTheApiSays:
    """A reader gating on reproducibility has to be able to see this without reading the logs."""

    QUESTION = "Whats the money charged for a matter ?"

    def _resolution(self, ctx):
        candidates = FakeCandidates([MetricCandidate("lg_fees_billed", PRODUCTION_SIMILARITY)])
        return Resolver(metric_matcher=matcher(candidates)).resolve(ctx, self.QUESTION, settings())

    def test_the_response_carries_the_selection(self, ctx):
        out = self._resolution(ctx).to_dict()
        assert out["deterministic_selection"] is False
        assert out["metric_selection"]["selected_by"] == SELECTED_BY_ROUTER
        assert out["metric_selection"]["similarity"] == pytest.approx(PRODUCTION_SIMILARITY)

    def test_the_explanation_does_not_promise_the_same_answer_every_time(self, ctx):
        """The stock tier-1 wording does, which is true of the SQL and false of which metric was
        picked. One sentence per claim."""
        explanation = self._resolution(ctx).explanation
        assert "the same answer every time" not in explanation
        assert "similarity" in explanation

    def test_the_warning_reaches_the_caller(self, ctx):
        """A field alone is not enough. Someone reading the number has to be told the choice was
        a model's without going looking for it."""
        assert any("similarity" in w for w in self._resolution(ctx).warnings)

    def test_a_keyword_answer_keeps_the_stock_explanation(self, ctx):
        res = Resolver(metric_matcher=matcher()).resolve(ctx, "fees billed by matter", settings())
        assert "the same answer every time" in res.explanation
        assert res.to_dict()["deterministic_selection"] is True
        assert res.warnings == []

    def test_the_composed_part_is_model_selected_not_deterministic(self, ctx):
        candidates = FakeCandidates([MetricCandidate("lg_fees_billed", PRODUCTION_SIMILARITY)])
        answer = Planner(metric_matcher=matcher(candidates)).plan(ctx, self.QUESTION, settings())
        assert answer.parts[0].provenance is Provenance.MODEL_SELECTED

    def test_a_model_selected_part_is_not_labelled_inferred(self, ctx):
        """The opposite overstatement. Nothing in the figure is a model's reading -- it is an
        exact aggregate -- so calling it inferred would understate it as badly as calling it
        deterministic overstates it."""
        candidates = FakeCandidates([MetricCandidate("lg_fees_billed", PRODUCTION_SIMILARITY)])
        answer = Planner(metric_matcher=matcher(candidates)).plan(ctx, self.QUESTION, settings())
        assert answer.parts[0].provenance is not Provenance.INFERRED
        assert "inferred" not in answer.governance_label


class TestTheRouterOffersAndNeverSelects:
    """`metric_candidates` searches only the metric layer, and the matter wall still applies."""

    def test_only_the_metric_layer_is_searched(self, ctx):
        """The caller is tier 1. An entity label is a subject name behind the ethical wall, and a
        candidate list that could contain one would put it in front of the metric matcher."""
        from src.query.router import TierRouter
        from src.query.router_index import KIND_METRIC

        seen: dict = {}

        class Index:
            def search(self, index, vector, **kw):
                seen.update(kw)
                return []

        class Embedder:
            def embed_query(self, text):
                return [0.1, 0.2]

        TierRouter(routing_index=Index(), embedder=Embedder()).metric_candidates(ctx, "q")
        assert seen["kinds"] == frozenset({KIND_METRIC})

    def test_the_matter_wall_is_passed_to_the_search(self):
        """Not queried beats queried and discarded, the same ordering `route` depends on."""
        from src.query.router import TierRouter

        seen: dict = {}

        class Index:
            def search(self, index, vector, **kw):
                seen.update(kw)
                return []

        class Embedder:
            def embed_query(self, text):
                return [0.1]

        screened = AuthContext(
            user_id="bob",
            tenant_id=TENANT,
            matter_allowlist=frozenset({"m-1"}),
            matter_denylist=frozenset({"m-2"}),
        )
        TierRouter(routing_index=Index(), embedder=Embedder()).metric_candidates(screened, "q")
        assert seen["matter_allowlist"] == frozenset({"m-1"})
        assert seen["matter_denylist"] == frozenset({"m-2"})

    def test_an_unreachable_index_returns_no_candidates(self):
        from src.query.router import TierRouter

        class Broken:
            def search(self, index, vector, **kw):
                raise RuntimeError("collection unreachable")

        class Embedder:
            def embed_query(self, text):
                return [0.1]

        router = TierRouter(routing_index=Broken(), embedder=Embedder())
        assert router.metric_candidates(AuthContext(user_id="u", tenant_id=TENANT), "q") == []

    def test_no_index_returns_no_candidates(self):
        from src.query.router import TierRouter

        router = TierRouter()
        assert router.metric_candidates(AuthContext(user_id="u", tenant_id=TENANT), "q") == []

    def test_candidates_come_back_as_cosines_not_raw_scores(self):
        """`cosinesimil` reports `1 / (2 - cos)`. An unconverted 0.833 read as a cosine would
        clear the 0.45 selection floor on a hit whose cosine is 0.8 -- and a raw 0.5 would fail
        it on a hit whose cosine is 0.0."""
        from src.query.router import TierRouter
        from src.query.router_index import KIND_METRIC

        class Record:
            kind = KIND_METRIC
            item_id = "lg_fees_billed"
            label = "fees_billed"

        class Hit:
            record = Record()
            raw_score = 0.833

        class Index:
            def search(self, index, vector, **kw):
                return [Hit()]

        class Embedder:
            def embed_query(self, text):
                return [0.1]

        router = TierRouter(routing_index=Index(), embedder=Embedder())
        found = router.metric_candidates(AuthContext(user_id="u", tenant_id=TENANT), "q")
        assert found[0].similarity == pytest.approx(0.8, abs=0.01)
