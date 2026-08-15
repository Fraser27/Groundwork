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
    def __init__(self, hits: list[dict] | None = None, blocking: list[dict] | None = None) -> None:
        self.hits = hits or []
        self.blocking = blocking or []
        self.searched = False

    def search(self, ctx: AuthContext, question: str, **kw: Any) -> list[dict]:
        self.searched = True
        return self.hits

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

    def test_a_reader_without_blocking_facts_degrades(self, ctx):
        """Screens still apply, so grounding weakens rather than disappearing."""

        class OldReader:
            def search(self, ctx, question, **kw):
                return [{"assertion_id": "a1"}]

        answer = Planner(graph_reader=OldReader()).plan(ctx, "q", GovernanceSettings())
        assert len(answer.parts) == 1


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
