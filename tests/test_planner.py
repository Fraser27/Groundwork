"""Multi-lane composition, and the properties that make it defensible.

The design in one sentence: the graph grounds what Athena and OpenSearch return. So the tests
that matter are not "does it fan out" but:

- a governed metric that matches short-circuits, because fanning out adds nothing and costs
  three round trips
- blocks are applied deterministically, and a blocked fact never reaches the model
- a blocked fact is reported rather than silently dropped, which is the failure `scope.py`
  exists to prevent
- lanes are never merged, and an answer containing a model's reading is never labelled
  plain "governed"
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from src.governance import GovernanceSettings
from src.graph.scope import AuthContext
from src.query.planner import Lane, Planner, Provenance, Tier
from src.query.resolver import UNGOVERNED_BLOCKED
from src.query.router import RouterDecision

TENANT = "demo-firm"

#: The default cap is vector-first, where the graph is reached by walking out of a retrieved
#: passage. A test about the graph's lexical term search has to ask for the direction that runs it:
#: tier 2 and tier 3 are one search in opposite directions and a tenant gets one of them.
GRAPH_FIRST = GovernanceSettings(allowed_tiers=frozenset({1, 2}))


@pytest.fixture
def ctx() -> AuthContext:
    return AuthContext(tenant_id=TENANT, user_id="alice")


#: So `rows=None` can mean "ran and returned nothing" rather than "use the default".
_DEFAULT_ROWS = object()


class FakeMatcher:
    def __init__(
        self,
        sql: str = "SELECT 1",
        rows: Any = _DEFAULT_ROWS,
        matches: bool = True,
        warnings: list[str] | None = None,
        runnable: bool = True,
    ) -> None:
        self.sql = sql
        self.rows = [{"total": 42}] if rows is _DEFAULT_ROWS else rows
        self.matches = matches
        self.ran = False
        # Both mirror `MetricMatch`, which populates `warnings` in `compile` and again in `run`.
        # Their absence here is what let the planner drop them unnoticed.
        self.warnings = warnings if warnings is not None else []
        self.is_runnable = runnable

    def match(self, question: str) -> Any:
        return self if self.matches else None

    def compile(self) -> str:
        return self.sql

    def run(self, sql: str) -> Any:
        self.ran = True
        return self.rows


class FakeGraph:
    def __init__(
        self,
        hits: list[dict] | None = None,
        blocking: list[dict] | None = None,
        walked: list[dict] | None = None,
    ) -> None:
        self.hits = hits or []
        self.blocking = blocking or []
        self.walked = walked or []
        self.searched = False
        self.expanded_from: list[str] | None = None
        self.expanded_with: dict[str, Any] = {}

    def search(self, ctx: AuthContext, question: str, **kw: Any) -> list[dict]:
        self.searched = True
        return self.hits

    def expand(self, ctx: AuthContext, seeds: list[str], **kw: Any) -> list[dict]:
        self.expanded_from = seeds
        self.expanded_with = dict(kw)
        return self.walked

    def blocking_facts(self, ctx: AuthContext, seeds: list[str], **kw: Any) -> list[dict]:
        return self.blocking


class FakeVectors:
    def __init__(self, passages: list[dict] | None = None) -> None:
        self.passages = passages or []
        self.searched = False

    def search(self, ctx: AuthContext, question: str, **kw: Any) -> list[dict]:
        self.searched = True
        return self.passages


class FakeSynthesiser:
    def __init__(self, text: str = "A summary.") -> None:
        self.text = text
        self.saw_parts: list[dict] = []
        self.saw_blocks: list[dict] = []

    def summarise(self, question: str, *, parts: list[dict], blocks: list[dict]) -> str:
        self.saw_parts = parts
        self.saw_blocks = blocks
        return self.text


class TestGovernedMetricShortCircuits:
    def test_a_matching_metric_is_the_whole_answer(self, ctx):
        """Fanning out would pay Athena plus Neptune plus OpenSearch to add nothing: the
        metric is exact and the question named it."""
        graph, vectors = (
            FakeGraph(hits=[{"assertion_id": "a1"}]),
            FakeVectors([{"document_id": "d"}]),
        )
        planner = Planner(metric_matcher=FakeMatcher(), graph_reader=graph, vector_search=vectors)

        answer = planner.plan(ctx, "fees billed last quarter", GovernanceSettings())

        assert [p.lane for p in answer.parts] == [Lane.METRIC]
        assert graph.searched is False
        assert vectors.searched is False

    def test_a_metric_answer_is_deterministic_and_governed(self, ctx):
        planner = Planner(metric_matcher=FakeMatcher())
        answer = planner.plan(ctx, "fees billed", GovernanceSettings())
        assert answer.is_fully_deterministic
        assert answer.governance_label == "governed"

    def test_execute_false_returns_sql_without_running_it(self, ctx):
        """The reviewability that makes a governed metric governed."""
        matcher = FakeMatcher()
        planner = Planner(metric_matcher=matcher)
        answer = planner.plan(ctx, "fees billed", GovernanceSettings(), execute=False)
        assert answer.parts[0].sql == "SELECT 1"
        assert matcher.ran is False

    def test_the_matchs_own_warnings_reach_the_answer(self, ctx):
        """`Part` has no field for a warning, so these were dropped for the life of this lane:
        the compiler's fan-out and non-additive-aggregation findings, and the error a failed
        query names. A metric carrying a fan-out risk reported none of it."""
        matcher = FakeMatcher(warnings=["Joining these tables inflates the total."])
        answer = Planner(metric_matcher=matcher).plan(ctx, "fees billed", GovernanceSettings())

        assert "Joining these tables inflates the total." in answer.warnings

    def test_a_metric_with_nowhere_to_run_says_so(self, ctx):
        """`/query` and `/query/compose` have disagreed before, so this is the same claim
        `TestWhyThereIsNoFigure` makes of the resolver."""
        matcher = FakeMatcher(rows=None, runnable=False)
        answer = Planner(metric_matcher=matcher).plan(ctx, "fees billed", GovernanceSettings())

        assert any("no query engine" in w for w in answer.warnings)

    def test_a_failed_query_is_not_reported_as_a_missing_engine(self, ctx):
        """The distinction that cost a morning: the engine was configured and the warehouse
        refused the read. The refusal is already in `warnings`; substituting for it loses it."""
        matcher = FakeMatcher(
            rows=None, warnings=["The metric compiled but the query did not run (403)."]
        )
        answer = Planner(metric_matcher=matcher).plan(ctx, "fees billed", GovernanceSettings())

        assert any("did not run (403)" in w for w in answer.warnings)
        assert not any("no query engine" in w for w in answer.warnings)


class TestFanOut:
    def test_no_metric_match_runs_the_other_lanes(self, ctx):
        planner = Planner(
            metric_matcher=FakeMatcher(matches=False),
            graph_reader=FakeGraph(walked=[{"assertion_id": "a1", "subject_id": "s1"}]),
            vector_search=FakeVectors([{"document_id": "d1", "page": 2}]),
        )
        answer = planner.plan(ctx, "who represents northwind", GovernanceSettings())
        assert {p.lane for p in answer.parts} == {Lane.GRAPH, Lane.PASSAGES}

    def test_lanes_are_reported_separately(self, ctx):
        """Merging them would need a common score, and the three retrievers use incompatible
        scales: weighted term overlap, cosine, and structural reachability with no score."""
        planner = Planner(
            graph_reader=FakeGraph(walked=[{"assertion_id": "a1"}]),
            vector_search=FakeVectors([{"document_id": "d1"}]),
        )
        answer = planner.plan(ctx, "q", GovernanceSettings())
        assert len(answer.parts) == 2
        assert answer.parts[0].content is not answer.parts[1].content

    def test_a_missing_collaborator_disables_its_lane_without_failing(self, ctx):
        """A partial answer that names what it could not reach beats no answer."""
        planner = Planner(graph_reader=FakeGraph(hits=[{"assertion_id": "a1"}]))
        answer = planner.plan(ctx, "q", GRAPH_FIRST)
        assert [p.lane for p in answer.parts] == [Lane.GRAPH]

    def test_nothing_matching_says_so(self, ctx):
        planner = Planner(metric_matcher=FakeMatcher(matches=False))
        answer = planner.plan(ctx, "q", GovernanceSettings())
        assert answer.parts == []
        assert any("Nothing matched" in w for w in answer.warnings)


class TestComposeWalksOutFromThePassagesToo:
    """`/query` and `/query/compose` must agree about what a question found.

    Compose ran the graph lane as a term search alone, so a fact reachable only from its source
    document reached `/query` via `Resolver._try_hybrid` and never reached compose -- the same
    question, two answers, depending on which endpoint you asked. That is a bug class this repo
    has hit repeatedly, and compose is the view whose whole purpose is showing what was found.
    """

    def test_the_graph_lane_walks_out_from_the_retrieved_passages(self, ctx):
        graph = FakeGraph(walked=[{"assertion_id": "walked-1", "hops": 1}])
        answer = Planner(
            graph_reader=graph, vector_search=FakeVectors([{"document_id": "doc-1"}])
        ).plan(ctx, "who represents halveston", GovernanceSettings(), allow_synthesis=False)

        facts = next(p for p in answer.parts if p.lane == Lane.GRAPH).content
        assert [f["assertion_id"] for f in facts] == ["walked-1"]

    def test_the_walk_runs_even_when_tier_2_is_forbidden(self, ctx):
        """The walk is tier 3's own work, so tier 2 must not gate it.

        `TIER_EXPLANATION[HYBRID]` promises "following verified relationships out from them", and
        `Resolver._try_hybrid` expands with no tier-2 check. Compose guarded both halves of the
        graph lane behind `if 2 in runnable`, so a tenant permitted only tier 3 -- which is the
        deployed configuration -- got passages and no `assertion_ids` at all. Every provenance
        call then had nothing valid to be called with.
        """
        graph = FakeGraph(
            hits=[{"assertion_id": "term-search-only"}],
            walked=[{"assertion_id": "walked-1", "hops": 1}],
        )
        answer = Planner(
            graph_reader=graph, vector_search=FakeVectors([{"document_id": "doc-1"}])
        ).plan(
            ctx,
            "does acting for calder create a conflict",
            GovernanceSettings(allowed_tiers=frozenset({3})),
            allow_synthesis=False,
        )

        part = next(p for p in answer.parts if p.lane == Lane.GRAPH)
        ids = [f["assertion_id"] for f in part.content]
        # The walk ran; the term search, which is tier 2's, did not.
        assert ids == ["walked-1"]
        assert part.assertion_ids == ["walked-1"]
        assert Lane.GRAPH in answer.lanes_run

    def test_a_walk_only_part_is_not_stamped_with_the_forbidden_tier(self, ctx):
        """Reporting tier 2 on a part authorised by tier 3 would say a tier the tenant forbade
        had run."""
        graph = FakeGraph(walked=[{"assertion_id": "walked-1"}])
        answer = Planner(
            graph_reader=graph, vector_search=FakeVectors([{"document_id": "doc-1"}])
        ).plan(ctx, "q", GovernanceSettings(allowed_tiers=frozenset({3})), allow_synthesis=False)

        assert next(p for p in answer.parts if p.lane == Lane.GRAPH).tier == Tier.HYBRID

    def test_the_term_search_still_needs_tier_2(self, ctx):
        """The other half of the split. Running the term search without tier 2 would be a cap
        bypass, so with nothing to walk from the lane contributes nothing and says why."""
        graph = FakeGraph(hits=[{"assertion_id": "term-search-only"}])
        answer = Planner(graph_reader=graph, vector_search=FakeVectors([])).plan(
            ctx, "q", GovernanceSettings(allowed_tiers=frozenset({3})), allow_synthesis=False
        )

        assert Lane.GRAPH not in answer.lanes_run
        assert "graph" in answer.lanes_skipped

    def test_it_seeds_the_walk_the_way_the_resolver_does(self, ctx):
        """Both id forms. Seeding only the bare id matched nothing on the first frontier, which is
        how tier 3 came to return zero related facts silently."""
        graph = FakeGraph()
        Planner(
            graph_reader=graph,
            vector_search=FakeVectors([{"document_id": "doc-1", "matter_id": "M-1"}]),
        ).plan(ctx, "q", GovernanceSettings(), allow_synthesis=False)

        assert set(graph.expanded_from or []) == {"doc-1", "document:doc-1", "M-1", "matter:M-1"}

    def test_the_walk_runs_under_the_governed_edge_cap(self, ctx):
        """The cap and the depth are one control. `expand` applies its limit to the whole walk, so a
        cap the tenant did not choose -- the hardcoded 50 this replaced -- silently makes
        `graph_expand_depth` inert once hop 1 fills it. Measured on the retail demo at 50: every
        returned edge was `hops=1`, so a firm that had asked for two hops was getting one."""
        graph = FakeGraph()
        settings = GovernanceSettings(graph_expand_limit=137, graph_expand_depth=3)
        Planner(
            graph_reader=graph, vector_search=FakeVectors([{"document_id": "doc-1"}])
        ).plan(ctx, "q", settings, allow_synthesis=False)

        assert graph.expanded_with["limit"] == 137
        assert graph.expanded_with["depth"] == 3

    def test_a_walked_fact_is_not_reported_twice(self, ctx):
        """The term search and the walk can both find one fact, and a reader counting facts would
        be told the graph holds two."""
        graph = FakeGraph(
            hits=[{"assertion_id": "a1"}],
            walked=[{"assertion_id": "a1"}, {"assertion_id": "a2"}],
        )
        answer = Planner(
            graph_reader=graph, vector_search=FakeVectors([{"document_id": "doc-1"}])
        ).plan(ctx, "q", GovernanceSettings(), allow_synthesis=False)

        facts = next(p for p in answer.parts if p.lane == Lane.GRAPH).content
        assert [f["assertion_id"] for f in facts] == ["a1", "a2"]

    def test_walked_facts_alone_are_enough_for_a_graph_part(self, ctx):
        """The production case: the question named no entity the term search could match, and every
        fact worth having was reachable only from the document."""
        graph = FakeGraph(hits=[], walked=[{"assertion_id": "walked-1"}])
        answer = Planner(
            graph_reader=graph, vector_search=FakeVectors([{"document_id": "doc-1"}])
        ).plan(ctx, "q", GovernanceSettings(), allow_synthesis=False)

        assert Lane.GRAPH in answer.lanes_run

    def test_a_walked_fact_is_screened_like_any_other(self, ctx):
        """A block applies to the evidence, not to how it was retrieved."""
        screened = AuthContext(
            tenant_id=TENANT, user_id="alice", matter_denylist=frozenset({"M-9"})
        )
        graph = FakeGraph(walked=[{"assertion_id": "a1", "matter_id": "M-9"}])
        answer = Planner(
            graph_reader=graph, vector_search=FakeVectors([{"document_id": "doc-1"}])
        ).plan(screened, "q", GovernanceSettings(), allow_synthesis=False)

        facts = next((p.content for p in answer.parts if p.lane == Lane.GRAPH), [])
        assert facts == []

    def test_no_passage_leaves_vector_first_with_no_graph_lane_and_says_why(self, ctx):
        """The other side of the trade. Vector-first reaches the graph through the passages it
        found, so nothing retrieved means nothing walked -- and the term search that would have
        found a fact anyway is the direction this tenant declined. Named rather than silent: an
        empty graph lane otherwise reads as an empty graph.
        """
        answer = Planner(graph_reader=FakeGraph(hits=[{"assertion_id": "a1"}])).plan(
            ctx, "q", GovernanceSettings(), allow_synthesis=False
        )
        assert answer.parts == []
        assert "no passage was retrieved" in answer.lanes_skipped["graph"]

    def test_graph_first_needs_no_passage_to_reach_the_graph(self, ctx):
        """Same reader, same empty index, the other direction: the term search is step 1 here, so
        retrieval finding nothing costs the passages and not the facts."""
        answer = Planner(graph_reader=FakeGraph(hits=[{"assertion_id": "a1"}])).plan(
            ctx, "q", GRAPH_FIRST, allow_synthesis=False
        )
        assert [p.lane for p in answer.parts] == [Lane.GRAPH]

    def test_a_reader_that_cannot_expand_still_contributes_its_term_matches(self, ctx):
        """A partial answer beats no answer, and the graph lane predates `expand`."""

        class SearchOnly:
            def search(self, ctx: AuthContext, question: str, **kw: Any) -> list[dict]:
                return [{"assertion_id": "a1", "source": {"document_id": "doc-1"}}]

        answer = Planner(
            graph_reader=SearchOnly(), vector_search=FakeVectors([{"document_id": "doc-1"}])
        ).plan(ctx, "q", GRAPH_FIRST, allow_synthesis=False)

        assert {p.lane for p in answer.parts} == {Lane.GRAPH, Lane.PASSAGES}

    def test_a_capped_passage_lane_costs_the_walk_not_the_question(self, ctx):
        """Tier 3 forbidden means no retrieval, so there are no passages to walk from -- and the
        walk must not be the thing that reaches OpenSearch behind the cap."""
        vectors = FakeVectors([{"document_id": "doc-1"}])
        graph = FakeGraph(hits=[{"assertion_id": "a1"}], walked=[{"assertion_id": "walked-1"}])
        answer = Planner(graph_reader=graph, vector_search=vectors).plan(
            ctx, "q", GovernanceSettings(allowed_tiers=frozenset({2})), allow_synthesis=False
        )

        assert vectors.searched is False
        assert graph.expanded_from is None
        assert [p.lane for p in answer.parts] == [Lane.GRAPH]


class TestTheGraphPartCarriesItsChains:
    """Beside the edges, never instead of them.

    The join a chain performs is otherwise left to whatever reads the flat list, and on the answer
    path the only reader is a language model. So the part carries both: `content` for a consumer
    that predates chains, `paths` for one that does not have to guess at the join.
    """

    def test_connected_facts_arrive_joined_up(self, ctx):
        graph = FakeGraph(
            walked=[
                {
                    "assertion_id": "a1",
                    "subject_id": "customer:sam",
                    "predicate": "PLACED_ORDER",
                    "object_id": "order:o-1",
                    "confidence": 0.95,
                },
                {
                    "assertion_id": "a2",
                    "subject_id": "associate:curtis",
                    "predicate": "APPROVED_RETURN",
                    "object_id": "order:o-1",
                    "confidence": 0.9,
                },
            ]
        )
        answer = Planner(
            graph_reader=graph, vector_search=FakeVectors([{"document_id": "doc-1"}])
        ).plan(ctx, "who helped sam", GovernanceSettings(), allow_synthesis=False)

        part = next(p for p in answer.parts if p.lane == Lane.GRAPH)
        assert [s["assertion_id"] for s in part.paths[0]["steps"]] == ["a2", "a1"]
        # The flat list is untouched, so nothing that ignores `paths` loses anything.
        assert [f["assertion_id"] for f in part.content] == ["a1", "a2"]

    def test_unconnected_facts_leave_the_field_empty_rather_than_guessing(self, ctx):
        """An empty `paths` is the honest report that the graph holds no route between these facts.
        Inventing one is the single step in this system nobody could audit."""
        graph = FakeGraph(
            walked=[
                {"assertion_id": "a1", "subject_id": "party:a", "object_id": "matter:m-1"},
                {"assertion_id": "a2", "subject_id": "party:b", "object_id": "matter:m-2"},
            ]
        )
        answer = Planner(
            graph_reader=graph, vector_search=FakeVectors([{"document_id": "doc-1"}])
        ).plan(ctx, "q", GovernanceSettings(), allow_synthesis=False)

        assert next(p for p in answer.parts if p.lane == Lane.GRAPH).paths == []

    def test_the_chains_survive_serialisation(self, ctx):
        graph = FakeGraph(
            walked=[
                {"assertion_id": "a1", "subject_id": "party:a", "object_id": "matter:m-1"},
                {"assertion_id": "a2", "subject_id": "party:b", "object_id": "matter:m-1"},
            ]
        )
        answer = Planner(
            graph_reader=graph, vector_search=FakeVectors([{"document_id": "doc-1"}])
        ).plan(ctx, "q", GovernanceSettings(), allow_synthesis=False)

        part = next(p for p in answer.to_dict()["parts"] if p["lane"] == "graph")
        assert len(part["paths"]) == 1


class TestTierGating:
    def test_a_forbidden_tier_is_skipped_with_a_reason(self, ctx):
        # Graph traversal alone. `{2, 3}` is refused outright now -- the two are one search in
        # opposite directions -- so "tier 1 is forbidden" is expressed by permitting only tier 2.
        settings = GovernanceSettings(allowed_tiers=frozenset({2}))
        planner = Planner(
            metric_matcher=FakeMatcher(),
            graph_reader=FakeGraph(hits=[{"assertion_id": "a1"}]),
        )
        answer = planner.plan(ctx, "fees billed", settings)

        assert Lane.METRIC.value in answer.lanes_skipped
        assert [p.lane for p in answer.parts] == [Lane.GRAPH]

    def test_a_metric_only_tenant_never_touches_documents(self, ctx):
        settings = GovernanceSettings(allowed_tiers=frozenset({1}))
        vectors = FakeVectors([{"document_id": "d1"}])
        planner = Planner(metric_matcher=FakeMatcher(matches=False), vector_search=vectors)

        answer = planner.plan(ctx, "q", settings)
        assert vectors.searched is False
        assert Lane.PASSAGES.value in answer.lanes_skipped


class TestDeterministicBlocking:
    def test_an_ethical_screen_becomes_a_block(self, ctx):
        screened = AuthContext(
            tenant_id=TENANT,
            user_id="alice",
            matter_denylist=frozenset({"M-9"}),
            screen_reasons={"M-9": "Acting for the counterparty."},
        )
        planner = Planner(graph_reader=FakeGraph(hits=[{"assertion_id": "a1"}]))
        answer = planner.plan(screened, "q", GovernanceSettings())

        assert [b.matter_id for b in answer.blocks] == ["M-9"]
        assert answer.blocks[0].reason == "Acting for the counterparty."

    def test_a_block_names_its_reason(self, ctx):
        """A silent block is the failure scope.py exists to prevent: an answer looks clean
        only because the inconvenient part was invisible."""
        screened = AuthContext(
            tenant_id=TENANT, user_id="alice", matter_denylist=frozenset({"M-9"})
        )
        planner = Planner(graph_reader=FakeGraph(hits=[{"assertion_id": "a1"}]))
        answer = planner.plan(screened, "q", GovernanceSettings())
        assert answer.blocks[0].reason

    def test_a_blocked_row_is_removed_from_its_part(self, ctx):
        screened = AuthContext(
            tenant_id=TENANT, user_id="alice", matter_denylist=frozenset({"M-9"})
        )
        graph = FakeGraph(
            hits=[
                {"assertion_id": "a1", "matter_id": "M-1"},
                {"assertion_id": "a2", "matter_id": "M-9"},
            ]
        )
        answer = Planner(graph_reader=graph).plan(screened, "q", GRAPH_FIRST)

        kept = answer.parts[0].content
        assert [row["assertion_id"] for row in kept] == ["a1"]

    def test_the_part_survives_even_when_rows_are_blocked(self, ctx):
        """Dropping the whole part would hide that anything matched at all."""
        screened = AuthContext(
            tenant_id=TENANT, user_id="alice", matter_denylist=frozenset({"M-9"})
        )
        graph = FakeGraph(hits=[{"assertion_id": "a1", "matter_id": "M-1"}])
        answer = Planner(graph_reader=graph).plan(screened, "q", GRAPH_FIRST)
        assert len(answer.parts) == 1

    def test_a_rule_block_is_reported(self, ctx):
        graph = FakeGraph(
            hits=[{"assertion_id": "a1", "subject_id": "party:acme"}],
            blocking=[
                {
                    "subject_id": "party:acme",
                    "reason": "Potential conflict",
                    "rule": "conflict_check",
                }
            ],
        )
        answer = Planner(graph_reader=graph).plan(ctx, "q", GRAPH_FIRST)
        assert answer.blocks[0].rule == "conflict_check"

    def test_a_reader_that_cannot_report_blocks_says_so_in_the_response(self, ctx):
        """Screens still apply, so the answer survives -- but it must not read as a clean wall.

        This test used to assert only that a part came back, which is why a reader with no
        `blocking_facts` at all looked like correct degradation for the life of the feature. The
        real `GraphReader` had never had the method, so this was the production path.
        """

        class NoBlockCheck:
            def search(self, ctx, question, **kw):
                # `subject_id` matters: it is what `seeds_from` reads, and with no seed at all
                # there is nothing to veto and so nothing degraded.
                return [{"assertion_id": "a1", "subject_id": "party:acme"}]

        answer = Planner(graph_reader=NoBlockCheck()).plan(ctx, "q", GRAPH_FIRST)
        assert len(answer.parts) == 1
        assert answer.gate["degraded"]
        assert any("could not be checked for conflicts" in w for w in answer.warnings)

    def test_an_answer_touching_no_ids_is_not_degraded(self, ctx):
        """There is nothing for a veto to match on, so skipping the check is a complete
        result rather than a failed one."""

        class NoIds:
            def search(self, ctx, question, **kw):
                return [{"assertion_id": "a1"}]

        answer = Planner(graph_reader=NoIds()).plan(ctx, "q", GRAPH_FIRST)
        assert answer.gate["degraded"] is None
        assert answer.gate["seeds_considered"] == 0


class TestSynthesisNeverDecides:
    def test_the_model_never_sees_a_blocked_fact(self, ctx):
        """The heart of the design. If a model could reason about a blocked fact, a
        hallucination would reinstate it."""
        screened = AuthContext(
            tenant_id=TENANT, user_id="alice", matter_denylist=frozenset({"M-9"})
        )
        graph = FakeGraph(
            hits=[
                {"assertion_id": "a1", "matter_id": "M-1"},
                {"assertion_id": "secret", "matter_id": "M-9"},
            ]
        )
        synth = FakeSynthesiser()
        Planner(graph_reader=graph, synthesiser=synth).plan(screened, "q", GRAPH_FIRST)

        seen = str(synth.saw_parts)
        assert "secret" not in seen
        assert "a1" in seen

    def test_the_model_is_told_what_was_blocked(self, ctx):
        """So it can say "some matters are withheld" instead of implying completeness."""
        screened = AuthContext(
            tenant_id=TENANT, user_id="alice", matter_denylist=frozenset({"M-9"})
        )
        synth = FakeSynthesiser()
        Planner(graph_reader=FakeGraph(hits=[{"assertion_id": "a1"}]), synthesiser=synth).plan(
            screened, "q", GRAPH_FIRST
        )
        assert synth.saw_blocks

    def test_synthesis_can_be_declined(self, ctx):
        synth = FakeSynthesiser()
        answer = Planner(
            graph_reader=FakeGraph(hits=[{"assertion_id": "a1"}]), synthesiser=synth
        ).plan(ctx, "q", GRAPH_FIRST, allow_synthesis=False)
        assert answer.synthesis is None

    def test_a_failed_synthesis_keeps_the_parts(self, ctx):
        """The evidence is the answer; the prose is a convenience over it."""

        class Broken:
            def summarise(self, *a, **kw):
                raise RuntimeError("model unavailable")

        answer = Planner(
            graph_reader=FakeGraph(hits=[{"assertion_id": "a1"}]), synthesiser=Broken()
        ).plan(ctx, "q", GRAPH_FIRST)

        assert answer.synthesis is None
        assert len(answer.parts) == 1
        assert any("failed" in w for w in answer.warnings)

    def test_no_synthesiser_is_reported_rather_than_silent(self, ctx):
        answer = Planner(graph_reader=FakeGraph(hits=[{"assertion_id": "a1"}])).plan(
            ctx, "q", GRAPH_FIRST
        )
        assert any("No synthesis model" in w for w in answer.warnings)


class TestGovernanceLabel:
    def test_a_synthesised_answer_is_never_plain_governed(self, ctx):
        """`Resolution.is_governed` returns True for tiers 1 to 3, which would label a
        composed answer containing a model's reading as governed."""
        answer = Planner(
            graph_reader=FakeGraph(hits=[{"assertion_id": "a1"}]), synthesiser=FakeSynthesiser()
        ).plan(ctx, "q", GRAPH_FIRST)

        assert answer.governance_label != "governed"
        assert "synthesised" in answer.governance_label
        assert answer.is_fully_deterministic is False

    def test_a_model_extracted_assertion_makes_a_part_inferred(self, ctx):
        graph = FakeGraph(
            hits=[{"assertion_id": "a1", "epistemic_class": "EXTRACTED_MODEL", "confidence": 0.72}]
        )
        answer = Planner(graph_reader=graph).plan(ctx, "q", GRAPH_FIRST, allow_synthesis=False)
        assert answer.parts[0].provenance is Provenance.INFERRED
        assert answer.parts[0].confidence == 0.72

    def test_a_declared_assertion_stays_deterministic(self, ctx):
        graph = FakeGraph(hits=[{"assertion_id": "a1", "epistemic_class": "DECLARED"}])
        answer = Planner(graph_reader=graph).plan(ctx, "q", GRAPH_FIRST, allow_synthesis=False)
        assert answer.parts[0].provenance is Provenance.DETERMINISTIC
        assert answer.is_fully_deterministic

    def test_a_passage_is_verbatim_not_inferred(self, ctx):
        """The text is quoted exactly. Similarity chose it, but nothing rewrote it, so it is
        not the same kind of claim as a model's reading."""
        answer = Planner(vector_search=FakeVectors([{"document_id": "d1", "page": 3}])).plan(
            ctx, "q", GovernanceSettings(), allow_synthesis=False
        )
        assert answer.parts[0].provenance is Provenance.VERBATIM

    def test_no_parts_is_not_called_governed(self, ctx):
        answer = Planner().plan(ctx, "q", GovernanceSettings())
        assert answer.governance_label == "no answer"
        assert answer.is_fully_deterministic is False


class FakeRouter:
    """A router that returns whatever decision the test wants, and records that it was asked."""

    def __init__(self, decision: Any = None, fail: bool = False) -> None:
        self.decision = decision
        self.fail = fail
        self.calls: list[str] = []

    def route(self, ctx: AuthContext, question: str, settings: GovernanceSettings) -> Any:
        self.calls.append(question)
        if self.fail:
            raise RuntimeError("router exploded")
        return self.decision


def decision(tiers: list[int], **over: Any) -> RouterDecision:
    return RouterDecision(tiers=tiers, **over)


class TestComposeRecordsWhyItLookedWhereItDid:
    """The gap this closes: compose ran lanes and its trace could not say why those lanes."""

    def test_the_router_is_asked(self, ctx):
        router = FakeRouter(decision([1, 2, 3]))
        Planner(router=router, graph_reader=FakeGraph(hits=[{"assertion_id": "a1"}])).plan(
            ctx, "who represents northwind", GovernanceSettings()
        )
        assert router.calls == ["who represents northwind"]

    def test_the_decision_reaches_the_response_body(self, ctx):
        """`ComposedResult.router` is what the trace diagram reads. Compose sent no such field, so
        step 1 said no routing trace was recorded -- honest, and the wrong answer."""
        router = FakeRouter(decision([2], best_score=0.71))
        answer = Planner(router=router, graph_reader=FakeGraph(hits=[{"assertion_id": "a1"}])).plan(
            ctx, "q", GovernanceSettings()
        )

        body = answer.to_dict()
        assert body["router"] is not None
        assert body["router"]["tiers_selected"] == [2]

    def test_no_router_still_sends_the_field_as_null(self, ctx):
        """The UI reads `router` on every response. A missing key and a null are the same to it,
        but the key existing is what makes "no router" a statement rather than a gap."""
        answer = Planner(graph_reader=FakeGraph(hits=[{"assertion_id": "a1"}])).plan(
            ctx, "q", GovernanceSettings()
        )
        assert answer.to_dict()["router"] is None

    def test_the_trace_says_the_decision_was_not_acted_on(self, ctx):
        """Without this the diagram labels a tier "not selected" while the step below it shows what
        that tier returned -- the page contradicting the system it is describing."""
        router = FakeRouter(decision([1], dropped={"2": "entity: scored 0.11"}))
        answer = Planner(router=router, graph_reader=FakeGraph(hits=[{"assertion_id": "a1"}])).plan(
            ctx, "q", GovernanceSettings()
        )
        assert answer.to_dict()["router"]["applied"] is False


class TestRoutingNeverNarrowsCompose:
    """Compose exists so a reader can see everything the system found. A lane dropped on a score
    would be invisible in the one view whose purpose is visibility."""

    def test_a_lane_the_router_did_not_select_still_runs(self, ctx):
        graph, vectors = (
            FakeGraph(hits=[{"assertion_id": "a1"}]),
            FakeVectors([{"document_id": "d1"}]),
        )
        router = FakeRouter(decision([2], dropped={"3": "passages: scored 0.09"}))
        answer = Planner(router=router, graph_reader=graph, vector_search=vectors).plan(
            ctx, "q", GovernanceSettings()
        )

        assert vectors.searched is True
        assert Lane.PASSAGES in answer.lanes_run

    def test_a_metric_still_short_circuits_when_the_router_dropped_tier_1(self, ctx):
        matcher = FakeMatcher()
        router = FakeRouter(decision([3], dropped={"1": "metric: scored 0.10"}))
        answer = Planner(router=router, metric_matcher=matcher, graph_reader=FakeGraph()).plan(
            ctx, "fees billed", GovernanceSettings()
        )
        assert [p.lane for p in answer.parts] == [Lane.METRIC]

    def test_a_router_that_selects_nothing_costs_no_lane(self, ctx):
        """The precedent for getting this wrong is real, and an optimisation that can refuse to
        answer is a liability."""
        vectors = FakeVectors([{"document_id": "d1"}])
        router = FakeRouter(decision([], degraded=True, reason="nothing resembled the question"))
        answer = Planner(
            router=router,
            graph_reader=FakeGraph(walked=[{"assertion_id": "a1"}]),
            vector_search=vectors,
        ).plan(ctx, "q", GovernanceSettings())

        assert {p.lane for p in answer.parts} == {Lane.GRAPH, Lane.PASSAGES}

    def test_a_router_that_raises_costs_no_lane(self, ctx):
        """`TierRouter.route` is documented never to raise. This is the guard against a future one
        that does, because a 500 from an optimisation is the worst outcome available."""
        vectors = FakeVectors([{"document_id": "d1"}])
        answer = Planner(router=FakeRouter(fail=True), vector_search=vectors).plan(
            ctx, "q", GovernanceSettings()
        )

        assert Lane.PASSAGES in answer.lanes_run
        assert answer.to_dict()["router"] is None


class TestTheRouterCannotWidenTheTenantCap:
    """The router narrows; it must never expand. Compose gates on `allowed_tiers`, and a router
    reporting a tier outside the cap must not be the thing that lets it run."""

    def test_a_forbidden_tier_the_router_selected_is_still_not_queried(self, ctx):
        vectors = FakeVectors([{"document_id": "d1"}])
        router = FakeRouter(decision([1, 2, 3]))
        answer = Planner(
            router=router, metric_matcher=FakeMatcher(matches=False), vector_search=vectors
        ).plan(ctx, "q", GovernanceSettings(allowed_tiers=frozenset({1})))

        assert vectors.searched is False
        assert Lane.PASSAGES.value in answer.lanes_skipped

    def test_an_empty_cap_runs_nothing_however_the_router_scored(self, ctx):
        graph, vectors = (
            FakeGraph(hits=[{"assertion_id": "a1"}]),
            FakeVectors([{"document_id": "d"}]),
        )
        matcher = FakeMatcher()

        class Catalog:
            def __init__(self) -> None:
                self.calls = 0

            def tables(self, tenant_id: str) -> list[Any]:
                self.calls += 1
                return []

        catalog = Catalog()
        answer = Planner(
            router=FakeRouter(decision([1, 2, 3])),
            metric_matcher=matcher,
            graph_reader=graph,
            vector_search=vectors,
            catalog=catalog,
        ).plan(ctx, "q", GovernanceSettings(allowed_tiers=frozenset()))

        assert answer.parts == []
        assert (graph.searched, vectors.searched, catalog.calls) == (False, False, 0)

    def test_a_capped_lane_is_named_as_the_cap_not_as_a_low_score(self, ctx):
        """ "Your administrator turned this off" and "this did not look relevant" are different
        facts, and a UI that cannot tell them apart tells the user to rephrase a question no
        rephrasing will help."""
        router = FakeRouter(decision([2], dropped={"1": "tier 1 is not permitted for this tenant"}))
        answer = Planner(
            router=router, metric_matcher=FakeMatcher(), graph_reader=FakeGraph()
        ).plan(ctx, "q", GovernanceSettings(allowed_tiers=frozenset({2})))

        assert answer.lanes_skipped[Lane.METRIC.value] == "tier 1 is not permitted for this tenant"


class TestCitations:
    def test_a_passage_part_carries_page_and_span(self, ctx):
        """Provenance is file plus page plus quote, so a citation without a page cannot be
        checked by hand."""
        answer = Planner(
            vector_search=FakeVectors(
                [{"document_id": "d1", "page": 4, "char_start": 10, "char_end": 90}]
            )
        ).plan(ctx, "q", GovernanceSettings(), allow_synthesis=False)

        citation = answer.parts[0].citations[0]
        assert citation["document_id"] == "d1"
        assert citation["page"] == 4

    def test_the_response_explains_why_parts_are_separate(self, ctx):
        answer = Planner(metric_matcher=FakeMatcher()).plan(ctx, "q", GovernanceSettings())
        assert "not the same kind of claim" in answer.to_dict()["note"]


class FakeCatalogTable:
    def __init__(self, full_name: str, description: str = "") -> None:
        self.full_name = full_name
        # `relevant_tables` matches on `name`, not `full_name`, so a fake without one can never be
        # selected by word overlap and every test of that selection would pass vacuously.
        self.name = full_name.rpartition(".")[2] or full_name
        self.description = description
        self.columns: list[Any] = []


class FakeCatalog:
    def __init__(self, *names: str) -> None:
        self._tables = [FakeCatalogTable(n) for n in names]

    def tables(self, tenant_id: str) -> list[FakeCatalogTable]:
        return self._tables


class CatalogGraph(FakeGraph):
    """A graph whose term search also reaches catalogued tables, which is graph-first's step 1."""

    def __init__(self, *nodes: str, **kw: Any) -> None:
        super().__init__(**kw)
        self.nodes = list(nodes)

    def catalog_search(self, ctx: AuthContext, question: str, **kw: Any) -> list[str]:
        return self.nodes


class FakeSqlLane:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def run(self, question: str, **kw: Any) -> Any:
        self.calls.append(kw)
        return SimpleNamespace(
            generated=SimpleNamespace(
                sql="SELECT count(*) FROM legal_db.matters",
                tables_offered=["legal_db.matters"],
            ),
            rows={"columns": ["n"], "rows": [[3]]},
            error=None,
            error_code=None,
        )


class TestGraphFirstReportsWhereItLanded:
    """The one thing a reader cannot infer from a graph-first answer. An empty passage list means
    either that no document was reached or that the documents held no matching passage, and those
    are different facts about the same corpus. See `query/graph_first.py`.
    """

    def test_a_question_reaching_no_fact_names_the_term_search(self, ctx):
        answer = Planner(graph_reader=FakeGraph()).plan(ctx, "q", GRAPH_FIRST)
        assert "matches this question's words" in answer.lanes_skipped[Lane.GRAPH.value]

    def test_reaching_no_document_is_not_reported_as_no_matching_passage(self, ctx):
        """A fact with no source document has a proof tree rather than a page, so there was nothing
        to read -- which is not the documents failing to match."""
        answer = Planner(
            graph_reader=FakeGraph(hits=[{"assertion_id": "a1", "subject_id": "s1"}]),
            vector_search=FakeVectors([{"document_id": "d1"}]),
        ).plan(ctx, "q", GRAPH_FIRST, allow_synthesis=False)

        assert "reached none" in answer.lanes_skipped[Lane.PASSAGES.value]

    def test_passages_come_only_from_documents_a_fact_came_from(self, ctx):
        """The claim the direction is entitled to make: every passage is in a document some verified
        fact came out of, so "why was I shown this page" answers with an assertion id."""
        vectors = FakeVectors([{"document_id": "d1", "page": 2}])
        answer = Planner(
            graph_reader=FakeGraph(
                hits=[{"assertion_id": "a1", "source": {"document_id": "d1"}}]
            ),
            vector_search=vectors,
        ).plan(ctx, "q", GRAPH_FIRST, allow_synthesis=False)

        assert {p.lane for p in answer.parts} == {Lane.GRAPH, Lane.PASSAGES}
        assert all(p.tier is Tier.GRAPH_TRAVERSAL for p in answer.parts)

    def test_a_table_the_graph_reached_becomes_a_query(self, ctx):
        """The inversion of tier 3's SQL lane. The graph named the table through DECLARED schema
        edges, so the generated query is restricted to what the traversal landed on."""
        sql = FakeSqlLane()
        answer = Planner(
            graph_reader=CatalogGraph("table:glue:legal_db.matters"),
            catalog=FakeCatalog("legal_db.matters"),
            sql_lane=sql,
        ).plan(ctx, "how many matters", GRAPH_FIRST, allow_synthesis=False)

        lanes = {p.lane: p for p in answer.parts}
        assert set(lanes) == {Lane.CATALOG, Lane.SQL}
        assert lanes[Lane.SQL].provenance is Provenance.MODEL_WRITTEN
        assert lanes[Lane.SQL].tier is Tier.GRAPH_TRAVERSAL

    def test_no_table_reached_names_the_sql_lane_for_the_same_reason(self, ctx):
        answer = Planner(graph_reader=CatalogGraph(), catalog=FakeCatalog()).plan(
            ctx, "q", GRAPH_FIRST
        )
        assert "catalogued table" in answer.lanes_skipped[Lane.SQL.value]

    def test_the_kill_switch_outranks_the_graph_reaching_no_table(self, ctx):
        """Two true reasons, and only one is actionable. An administrator turning the lane off is a
        different fact from the graph reaching nothing, and the refusal is the one to report."""
        planner = Planner(graph_reader=CatalogGraph(), catalog=FakeCatalog())
        answer = planner.plan(
            ctx,
            "q",
            GovernanceSettings(allowed_tiers=frozenset({1, 2}), block_ungoverned_queries=True),
        )

        assert answer.lanes_skipped[Lane.SQL.value] == UNGOVERNED_BLOCKED
        assert planner.blocked, "a refused ungoverned query never reached the Governance screen"

    def test_no_graph_leaves_every_traversal_lane_named(self, ctx):
        """Graph-first has nothing to traverse from, so the passage and table lanes go with it --
        they are downstream of the traversal here rather than beside it."""
        answer = Planner(vector_search=FakeVectors([{"document_id": "d1"}])).plan(
            ctx, "q", GRAPH_FIRST
        )

        for lane in (Lane.GRAPH, Lane.PASSAGES, Lane.CATALOG, Lane.SQL):
            assert "no graph is configured" in answer.lanes_skipped[lane.value], lane

    def test_a_note_from_the_lane_reaches_the_reader(self, ctx):
        answer = Planner(
            graph_reader=FakeGraph(hits=[{"assertion_id": "a1", "source": {"document_id": "d1"}}]),
            vector_search=FakeVectors([]),
        ).plan(ctx, "q", GRAPH_FIRST, allow_synthesis=False)

        assert any("no passage" in w for w in answer.warnings)


class SilentSqlLane:
    """A generator that produced nothing usable, which `SqlLane.run` reports as None."""

    def run(self, question: str, **kw: Any) -> None:
        return None


class TestVectorFirstReportsWhyALaneWasQuiet:
    """Graph-first has always explained itself. Vector-first did not: a lane that contributed
    nothing was in neither `lanes_run` nor `lanes_skipped`, so it left the trace entirely. On a
    tenant whose catalogued tables share no word with the question, that read as tier 3 having no
    SQL lane at all rather than as a lane with nothing to query.
    """

    def test_a_question_no_table_shares_a_word_with_names_both_lanes(self, ctx):
        sql = FakeSqlLane()
        answer = Planner(catalog=FakeCatalog("retail_db.orders"), sql_lane=sql).plan(
            ctx, "who is adverse to Acme", GovernanceSettings()
        )

        for lane in (Lane.CATALOG, Lane.SQL):
            assert "shares a word" in answer.lanes_skipped[lane.value], lane
        assert not sql.calls, "the generator was paid for with no schema to write over"

    def test_an_empty_catalogue_is_not_the_same_fact_as_a_question_that_missed(self, ctx):
        """Different jobs. One is scanning a data source, the other is approving a synonym."""
        answer = Planner(catalog=FakeCatalog(), sql_lane=FakeSqlLane()).plan(
            ctx, "how many orders", GovernanceSettings()
        )

        assert "no table has been catalogued" in answer.lanes_skipped[Lane.CATALOG.value]

    def test_no_catalogue_at_all_is_not_the_same_fact_either(self, ctx):
        answer = Planner(sql_lane=FakeSqlLane()).plan(ctx, "how many orders", GovernanceSettings())

        assert "no data catalogue is configured" in answer.lanes_skipped[Lane.CATALOG.value]

    def test_a_generator_that_wrote_nothing_says_so_rather_than_disappearing(self, ctx):
        """The schema was there and the lane still produced no part. Silence here is the case that
        looks most like the feature being absent."""
        answer = Planner(catalog=FakeCatalog("retail_db.orders"), sql_lane=SilentSqlLane()).plan(
            ctx, "how many orders", GovernanceSettings()
        )

        assert answer.lanes_skipped[Lane.SQL.value] == (
            "the model wrote no query over the schema it was offered"
        )
        assert Lane.CATALOG in answer.lanes_run, "the schema lane ran, so it is not skipped"

    def test_no_sql_generator_configured_is_reported_against_the_lane_it_disables(self, ctx):
        answer = Planner(catalog=FakeCatalog("retail_db.orders")).plan(
            ctx, "how many orders", GovernanceSettings()
        )

        assert "no SQL generator is configured" in answer.lanes_skipped[Lane.SQL.value]
        assert Lane.CATALOG in answer.lanes_run

    def test_the_schema_shown_is_the_schema_the_query_was_written_over(self, ctx):
        """The reason both lanes select from one function. A reader shown one list of tables while
        the query was written over another cannot check the query against it."""
        sql = FakeSqlLane()
        answer = Planner(
            catalog=FakeCatalog("retail_db.orders", "retail_db.returns"), sql_lane=sql
        ).plan(ctx, "how many orders", GovernanceSettings(), allow_synthesis=False)

        catalog = next(p for p in answer.parts if p.lane is Lane.CATALOG)
        shown = {row["full_name"] for row in catalog.content}
        assert shown == {"retail_db.orders"}
        assert {t.full_name for t in sql.calls[0]["candidates"]} == shown

    def test_the_kill_switch_still_outranks_a_question_reaching_no_table(self, ctx):
        """Same precedence graph-first applies. An administrator turning the lane off is the
        actionable fact, and it must not be overwritten by the question having missed."""
        planner = Planner(catalog=FakeCatalog(), sql_lane=FakeSqlLane())
        answer = planner.plan(
            ctx, "q", GovernanceSettings(block_ungoverned_queries=True), allow_synthesis=False
        )

        assert answer.lanes_skipped[Lane.SQL.value] == UNGOVERNED_BLOCKED
        assert planner.blocked, "a refused ungoverned query never reached the Governance screen"

    def test_no_document_search_configured_names_the_passage_lane(self, ctx):
        answer = Planner().plan(ctx, "q", GovernanceSettings())

        assert "no document search is configured" in answer.lanes_skipped[Lane.PASSAGES.value]

    def test_a_retrieval_that_matched_nothing_is_not_reported_as_unconfigured(self, ctx):
        """Two facts a reader acts on differently: one is a deployment gap, the other is a corpus
        that does not cover the question."""
        answer = Planner(vector_search=FakeVectors([])).plan(ctx, "q", GovernanceSettings())

        assert "similar enough" in answer.lanes_skipped[Lane.PASSAGES.value]


class TestTheDirectionIsReported:
    """A skipped lane carries no part and therefore no tier of its own, so the trace has to name the
    direction those lanes belonged to. Without it the UI labelled them from a hardcoded map and put
    tier 3 on the lanes a graph-first tenant had declined.
    """

    def test_graph_first_says_graph_first(self, ctx):
        answer = Planner(graph_reader=FakeGraph()).plan(ctx, "q", GRAPH_FIRST)
        assert answer.to_dict()["retrieval_direction"] == "graph_first"

    def test_vector_first_says_vector_first(self, ctx):
        answer = Planner(vector_search=FakeVectors([])).plan(ctx, "q", GovernanceSettings())
        assert answer.to_dict()["retrieval_direction"] == "vector_first"

    def test_metrics_only_names_no_direction(self, ctx):
        answer = Planner().plan(ctx, "q", GovernanceSettings(allowed_tiers=frozenset({1})))
        assert answer.to_dict()["retrieval_direction"] == "metrics_only"

    def test_a_run_with_no_parts_still_names_its_direction(self, ctx):
        """The case the field exists for. Graph-first with no graph skips every traversal lane, so
        there is no part's tier left to infer from and the skip reasons would be labelled with
        whichever direction the reader's UI happens to default to."""
        answer = Planner().plan(ctx, "q", GRAPH_FIRST)

        assert answer.parts == []
        assert answer.to_dict()["retrieval_direction"] == "graph_first"

    def test_a_cap_naming_both_reports_the_direction_that_ran(self, ctx):
        """Read from the coerced cap, not the raw one. `validate()` refuses both at construction, so
        this can only arrive by assignment -- which is how a settings object gets mutated in
        practice, and `plan()` coerces for exactly that reason. A field naming a direction other
        than the one the lanes ran in would be worse than no field."""
        settings = GovernanceSettings()
        settings.allowed_tiers = frozenset({1, 2, 3})
        answer = Planner(
            graph_reader=FakeGraph(hits=[{"assertion_id": "a1"}]),
            vector_search=FakeVectors([{"document_id": "d1"}]),
        ).plan(ctx, "q", settings, allow_synthesis=False)

        assert answer.to_dict()["retrieval_direction"] == "vector_first"
        assert Lane.PASSAGES in answer.lanes_run


class OneMetricHalf(FakeMatcher):
    """Matches only the half a metric covers, so the other half has to be searched."""

    def __init__(self, word: str) -> None:
        super().__init__()
        self.word = word

    def match(self, question: str) -> Any:
        return self if self.word in question else None


class FakeSplitter:
    def __init__(self, *parts: str, raises: bool = False) -> None:
        self.parts = list(parts)
        self.raises = raises

    def split(self, question: str) -> list[str]:
        if self.raises:
            raise RuntimeError("bedrock unreachable")
        return self.parts or [question]


class TestDecomposition:
    """A compound question run whole reaches one lane well and the other badly, and the half that
    loses is reported as nothing found rather than as unanswered. Split, each half reaches the store
    that can answer it -- and the split is disclosed rather than folded into the label.
    """

    def test_each_part_names_the_question_it_answers(self, ctx):
        """Two parts from the same lane are otherwise indistinguishable: a reader looking at two
        results needs to know they answer different questions, not that the lane ran twice."""
        answer = Planner(
            question_splitter=FakeSplitter("our exposure on Northwind", "does it exclude tax"),
            vector_search=FakeVectors([{"document_id": "d1"}]),
        ).plan(ctx, "both at once", GovernanceSettings(), allow_synthesis=False)

        assert [p.sub_question for p in answer.parts] == [
            "our exposure on Northwind",
            "does it exclude tax",
        ]

    def test_the_split_is_disclosed_with_the_original(self, ctx):
        answer = Planner(question_splitter=FakeSplitter("half one", "half two")).plan(
            ctx, "both at once", GovernanceSettings()
        )

        assert answer.decomposition == {
            "question": "both at once",
            "parts": ["half one", "half two"],
        }

    def test_an_unsplit_question_is_not_disclosed_as_split(self, ctx):
        """`sub_question` stays None too. A question asked whole must not read as one half of
        itself."""
        answer = Planner(
            question_splitter=FakeSplitter(),
            vector_search=FakeVectors([{"document_id": "d1"}]),
        ).plan(ctx, "one thing", GovernanceSettings(), allow_synthesis=False)

        assert answer.decomposition is None
        assert [p.sub_question for p in answer.parts] == [None]

    def test_a_splitter_that_raises_costs_the_split_not_the_answer(self, ctx):
        """Splitting is an improvement to how a question is searched, and an improvement may not
        cost the answer."""
        answer = Planner(
            question_splitter=FakeSplitter(raises=True),
            vector_search=FakeVectors([{"document_id": "d1"}]),
        ).plan(ctx, "q", GovernanceSettings(), allow_synthesis=False)

        assert answer.decomposition is None
        assert [p.lane for p in answer.parts] == [Lane.PASSAGES]

    def test_a_metric_half_no_longer_short_circuits_the_other_half(self, ctx):
        """Unsplit, a matching metric is the whole answer. Split, it answers one of two questions
        and the other still has to be searched -- which is the whole reason to split: the document
        half would otherwise be reported as nothing found rather than as never asked."""
        answer = Planner(
            question_splitter=FakeSplitter("fees billed last quarter", "does it exclude tax"),
            metric_matcher=OneMetricHalf("fees"),
            vector_search=FakeVectors([{"document_id": "d1"}]),
        ).plan(ctx, "both at once", GovernanceSettings(), allow_synthesis=False)

        assert {p.lane for p in answer.parts} == {Lane.METRIC, Lane.PASSAGES}
        assert answer.governance_label != "governed", "a quoted passage is not a compiled metric"

    def test_a_lane_that_ran_for_one_half_is_not_also_reported_skipped(self, ctx):
        """Both would be true of different halves, and a trace saying both contradicts the parts
        sitting next to it."""
        answer = Planner(
            question_splitter=FakeSplitter("half one", "half two"),
            vector_search=FakeVectors([{"document_id": "d1"}]),
        ).plan(ctx, "both at once", GovernanceSettings(), allow_synthesis=False)

        assert Lane.PASSAGES in answer.lanes_run
        assert Lane.PASSAGES.value not in answer.lanes_skipped

    def test_splitting_can_be_declined(self, ctx):
        answer = Planner(question_splitter=FakeSplitter("half one", "half two")).plan(
            ctx, "both at once", GovernanceSettings(), decompose=False
        )
        assert answer.decomposition is None

    def test_a_split_answer_from_metrics_alone_is_still_governed(self, ctx):
        """The label describes what produced the answer. Every part still comes from the same
        compiled metric it would have come from unsplit, so downgrading it because a model chose
        where the sentence divided would report the wrong thing as ungoverned."""
        answer = Planner(
            question_splitter=FakeSplitter("fees billed last quarter", "hours billed last quarter"),
            metric_matcher=FakeMatcher(),
        ).plan(ctx, "both at once", GovernanceSettings(), allow_synthesis=False)

        assert len(answer.parts) == 2
        assert answer.governance_label == "governed"

    def test_the_router_is_asked_once_for_the_whole_question(self, ctx):
        """The decision is advisory unless the router narrows, so routing each half would buy a
        longer trace and a model call per part while changing nothing about what runs."""
        router = FakeRouter(decision([1, 3]))
        Planner(
            router=router, question_splitter=FakeSplitter("half one", "half two")
        ).plan(ctx, "both at once", GovernanceSettings())

        assert router.calls == ["both at once"]
