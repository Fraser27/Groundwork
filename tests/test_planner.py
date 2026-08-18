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

from typing import Any

import pytest

from src.governance import GovernanceSettings
from src.graph.scope import AuthContext
from src.query.planner import Lane, Planner, Provenance
from src.query.router import RouterDecision

TENANT = "demo-firm"


@pytest.fixture
def ctx() -> AuthContext:
    return AuthContext(tenant_id=TENANT, user_id="alice")


class FakeMatcher:
    def __init__(self, sql: str = "SELECT 1", rows: Any = None, matches: bool = True) -> None:
        self.sql = sql
        self.rows = rows if rows is not None else [{"total": 42}]
        self.matches = matches
        self.ran = False

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

    def search(self, ctx: AuthContext, question: str, **kw: Any) -> list[dict]:
        self.searched = True
        return self.hits

    def expand(self, ctx: AuthContext, seeds: list[str], **kw: Any) -> list[dict]:
        self.expanded_from = seeds
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


class TestFanOut:
    def test_no_metric_match_runs_the_other_lanes(self, ctx):
        planner = Planner(
            metric_matcher=FakeMatcher(matches=False),
            graph_reader=FakeGraph(hits=[{"assertion_id": "a1", "subject_id": "s1"}]),
            vector_search=FakeVectors([{"document_id": "d1", "page": 2}]),
        )
        answer = planner.plan(ctx, "who represents northwind", GovernanceSettings())
        assert {p.lane for p in answer.parts} == {Lane.GRAPH, Lane.PASSAGES}

    def test_lanes_are_reported_separately(self, ctx):
        """Merging them would need a common score, and the three retrievers use incompatible
        scales: weighted term overlap, cosine, and structural reachability with no score."""
        planner = Planner(
            graph_reader=FakeGraph(hits=[{"assertion_id": "a1"}]),
            vector_search=FakeVectors([{"document_id": "d1"}]),
        )
        answer = planner.plan(ctx, "q", GovernanceSettings())
        assert len(answer.parts) == 2
        assert answer.parts[0].content is not answer.parts[1].content

    def test_a_missing_collaborator_disables_its_lane_without_failing(self, ctx):
        """A partial answer that names what it could not reach beats no answer."""
        planner = Planner(graph_reader=FakeGraph(hits=[{"assertion_id": "a1"}]))
        answer = planner.plan(ctx, "q", GovernanceSettings())
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

    def test_it_seeds_the_walk_the_way_the_resolver_does(self, ctx):
        """Both id forms. Seeding only the bare id matched nothing on the first frontier, which is
        how tier 3 came to return zero related facts silently."""
        graph = FakeGraph()
        Planner(
            graph_reader=graph,
            vector_search=FakeVectors([{"document_id": "doc-1", "matter_id": "M-1"}]),
        ).plan(ctx, "q", GovernanceSettings(), allow_synthesis=False)

        assert set(graph.expanded_from or []) == {"doc-1", "document:doc-1", "M-1", "matter:M-1"}

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

    def test_the_graph_lane_still_runs_when_no_passage_was_retrieved(self, ctx):
        answer = Planner(graph_reader=FakeGraph(hits=[{"assertion_id": "a1"}])).plan(
            ctx, "q", GovernanceSettings(), allow_synthesis=False
        )
        assert [p.lane for p in answer.parts] == [Lane.GRAPH]

    def test_a_reader_that_cannot_expand_still_contributes_its_term_matches(self, ctx):
        """A partial answer beats no answer, and the graph lane predates `expand`."""

        class SearchOnly:
            def search(self, ctx: AuthContext, question: str, **kw: Any) -> list[dict]:
                return [{"assertion_id": "a1"}]

        answer = Planner(
            graph_reader=SearchOnly(), vector_search=FakeVectors([{"document_id": "doc-1"}])
        ).plan(ctx, "q", GovernanceSettings(), allow_synthesis=False)

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


class TestTierGating:
    def test_a_forbidden_tier_is_skipped_with_a_reason(self, ctx):
        settings = GovernanceSettings(allowed_tiers=frozenset({2, 3}))
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
        answer = Planner(graph_reader=graph).plan(screened, "q", GovernanceSettings())

        kept = answer.parts[0].content
        assert [row["assertion_id"] for row in kept] == ["a1"]

    def test_the_part_survives_even_when_rows_are_blocked(self, ctx):
        """Dropping the whole part would hide that anything matched at all."""
        screened = AuthContext(
            tenant_id=TENANT, user_id="alice", matter_denylist=frozenset({"M-9"})
        )
        graph = FakeGraph(hits=[{"assertion_id": "a1", "matter_id": "M-1"}])
        answer = Planner(graph_reader=graph).plan(screened, "q", GovernanceSettings())
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
        answer = Planner(graph_reader=graph).plan(ctx, "q", GovernanceSettings())
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

        answer = Planner(graph_reader=NoBlockCheck()).plan(ctx, "q", GovernanceSettings())
        assert len(answer.parts) == 1
        assert answer.gate["degraded"]
        assert any("could not be checked for conflicts" in w for w in answer.warnings)

    def test_an_answer_touching_no_ids_is_not_degraded(self, ctx):
        """There is nothing for a veto to match on, so skipping the check is a complete
        result rather than a failed one."""

        class NoIds:
            def search(self, ctx, question, **kw):
                return [{"assertion_id": "a1"}]

        answer = Planner(graph_reader=NoIds()).plan(ctx, "q", GovernanceSettings())
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
        Planner(graph_reader=graph, synthesiser=synth).plan(screened, "q", GovernanceSettings())

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
            screened, "q", GovernanceSettings()
        )
        assert synth.saw_blocks

    def test_synthesis_can_be_declined(self, ctx):
        synth = FakeSynthesiser()
        answer = Planner(
            graph_reader=FakeGraph(hits=[{"assertion_id": "a1"}]), synthesiser=synth
        ).plan(ctx, "q", GovernanceSettings(), allow_synthesis=False)
        assert answer.synthesis is None

    def test_a_failed_synthesis_keeps_the_parts(self, ctx):
        """The evidence is the answer; the prose is a convenience over it."""

        class Broken:
            def summarise(self, *a, **kw):
                raise RuntimeError("model unavailable")

        answer = Planner(
            graph_reader=FakeGraph(hits=[{"assertion_id": "a1"}]), synthesiser=Broken()
        ).plan(ctx, "q", GovernanceSettings())

        assert answer.synthesis is None
        assert len(answer.parts) == 1
        assert any("failed" in w for w in answer.warnings)

    def test_no_synthesiser_is_reported_rather_than_silent(self, ctx):
        answer = Planner(graph_reader=FakeGraph(hits=[{"assertion_id": "a1"}])).plan(
            ctx, "q", GovernanceSettings()
        )
        assert any("No synthesis model" in w for w in answer.warnings)


class TestGovernanceLabel:
    def test_a_synthesised_answer_is_never_plain_governed(self, ctx):
        """`Resolution.is_governed` returns True for tiers 1 to 3, which would label a
        composed answer containing a model's reading as governed."""
        answer = Planner(
            graph_reader=FakeGraph(hits=[{"assertion_id": "a1"}]), synthesiser=FakeSynthesiser()
        ).plan(ctx, "q", GovernanceSettings())

        assert answer.governance_label != "governed"
        assert "synthesised" in answer.governance_label
        assert answer.is_fully_deterministic is False

    def test_a_model_extracted_assertion_makes_a_part_inferred(self, ctx):
        graph = FakeGraph(
            hits=[{"assertion_id": "a1", "epistemic_class": "EXTRACTED_MODEL", "confidence": 0.72}]
        )
        answer = Planner(graph_reader=graph).plan(
            ctx, "q", GovernanceSettings(), allow_synthesis=False
        )
        assert answer.parts[0].provenance is Provenance.INFERRED
        assert answer.parts[0].confidence == 0.72

    def test_a_declared_assertion_stays_deterministic(self, ctx):
        graph = FakeGraph(hits=[{"assertion_id": "a1", "epistemic_class": "DECLARED"}])
        answer = Planner(graph_reader=graph).plan(
            ctx, "q", GovernanceSettings(), allow_synthesis=False
        )
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
            router=router, graph_reader=FakeGraph(hits=[{"assertion_id": "a1"}]),
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
        graph, vectors = FakeGraph(hits=[{"assertion_id": "a1"}]), FakeVectors([{"document_id": "d"}])
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
        """"Your administrator turned this off" and "this did not look relevant" are different
        facts, and a UI that cannot tell them apart tells the user to rephrase a question no
        rephrasing will help."""
        router = FakeRouter(decision([2, 3], dropped={"1": "tier 1 is not permitted for this tenant"}))
        answer = Planner(
            router=router, metric_matcher=FakeMatcher(), graph_reader=FakeGraph()
        ).plan(ctx, "q", GovernanceSettings(allowed_tiers=frozenset({2, 3})))

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
