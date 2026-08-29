"""The summariser writes over evidence; it never chooses the evidence.

The property worth more than the rest of this file: **a blocked fact's content never reaches the
model, while the fact that something was blocked does.** Both halves matter. Sending the content
would defeat the block, because a model asked to summarise can paraphrase. Hiding the block
entirely would produce a confident answer that silently omits something -- which for an ethical
screen is the exact harm the screen exists to prevent.

No Bedrock. The client is injected, so these run with no credentials.
"""

from __future__ import annotations

import json

import pytest

from src.query.synthesis import (
    MAX_PART_CHARS,
    Synthesiser,
    SynthesisFailed,
    build_prompt,
)

SECRET = "Meridian holds 18 per cent of Calder"


class FakeBedrock:
    """Records what it was asked, which is the point of most of these tests."""

    def __init__(self, text: str = "Northwind Trading is the client.") -> None:
        self.text = text
        self.request: dict | None = None

    def converse(self, **kwargs):
        self.request = kwargs
        return {"output": {"message": {"content": [{"text": self.text}]}}}


class Unreachable:
    def converse(self, **kwargs):
        raise RuntimeError("no route to bedrock")


def _part(**over):
    base = {
        "lane": "graph",
        "provenance": "extracted_model",
        "tier": 2,
        "content": "Thorne Vaux represents Northwind Trading.",
        "sql": None,
        "citations": [{"document_id": "d1", "filename": "engagement.pdf", "page": 1}],
        "assertion_ids": ["a1"],
        "confidence": 0.9,
    }
    return {**base, **over}


def _sent(bedrock: FakeBedrock) -> str:
    """Everything the model was given, as one string to search."""
    return json.dumps(bedrock.request)


class TestABlockedFactStaysBlocked:
    def test_the_content_never_reaches_the_model(self):
        bedrock = FakeBedrock()
        Synthesiser(model_id="m", bedrock=bedrock).summarise(
            "is there a conflict?",
            parts=[_part()],
            blocks=[{"reason": "ethical screen on MBC-2024-0431", "content": SECRET}],
        )
        assert SECRET not in _sent(bedrock)

    def test_the_reason_does_reach_the_model(self):
        """So it can say the answer may be incomplete. A silent omission is the worse failure."""
        bedrock = FakeBedrock()
        Synthesiser(model_id="m", bedrock=bedrock).summarise(
            "is there a conflict?",
            parts=[_part()],
            blocks=[{"reason": "ethical screen on MBC-2024-0431", "content": SECRET}],
        )
        sent = _sent(bedrock)
        assert "ethical screen on MBC-2024-0431" in sent
        assert "incomplete" in sent

    def test_no_withheld_section_when_nothing_was_blocked(self):
        """Otherwise every answer hedges, and a hedge that is always there carries no information."""
        prompt = build_prompt("q", parts=[_part()], blocks=[])
        assert "withheld" not in prompt

    def test_a_notify_finding_is_not_reported_as_withheld(self):
        """`effect: notify` suppresses nothing, so telling the model content was withheld makes it
        write that the answer may be incomplete when the evidence is complete. Every legal rule
        finding is `notify`, so this fired on any answer that reached a conflict."""
        prompt = build_prompt(
            "q",
            parts=[_part()],
            blocks=[{"reason": "Potential conflict involving matter:NTL", "effect": "notify"}],
        )
        assert "withheld" not in prompt
        assert "incomplete" not in prompt

    def test_a_withholding_finding_among_advisories_is_still_reported(self):
        """The filter is on effect, not on emptiness: one real refusal alongside advisories must
        still tell the model the answer is partial."""
        prompt = build_prompt(
            "q",
            parts=[_part()],
            blocks=[
                {"reason": "Potential conflict involving matter:NTL", "effect": "notify"},
                {"reason": "ethical screen on MBC-2024-0431", "effect": "withhold"},
            ],
        )
        assert "ethical screen on MBC-2024-0431" in prompt
        assert "Potential conflict" not in prompt
        assert "incomplete" in prompt


class TestWhatTheModelIsGiven:
    def test_the_parts_are_passed_through(self):
        bedrock = FakeBedrock()
        Synthesiser(model_id="m", bedrock=bedrock).summarise("q", parts=[_part()])
        assert "Thorne Vaux represents Northwind Trading." in _sent(bedrock)

    def test_a_part_is_described_the_way_a_reader_would_cite_it(self):
        """Attribution has to land in the reader's language, because checking it is their next
        move. "engagement.pdf" is findable; "provenance: extracted_model" is not."""
        prompt = build_prompt("q", parts=[_part()])
        assert "engagement.pdf" in prompt

    def test_a_query_result_is_described_as_one(self):
        prompt = build_prompt("q", parts=[_part(sql="SELECT 1", citations=[])])
        assert "query result" in prompt

    def test_an_oversized_part_is_clipped_and_says_so(self):
        """A metric returning thousands of rows would crowd out every other lane. The rows are on
        screen anyway, so the clip is visible rather than silent."""
        prompt = build_prompt("q", parts=[_part(content="x" * (MAX_PART_CHARS + 500))])
        assert "truncated" in prompt
        assert len(prompt) < MAX_PART_CHARS * 2

    def test_the_system_prompt_forbids_inventing_facts(self):
        bedrock = FakeBedrock()
        Synthesiser(model_id="m", bedrock=bedrock).summarise("q", parts=[_part()])
        system = bedrock.request["system"][0]["text"]
        assert "ONLY the parts" in system
        assert "disagree" in system

    def test_temperature_is_omitted_by_default(self):
        """This test used to assert `== 0.0` and so pinned the bug in place. Sonnet 5 rejects the
        field outright, so sending it failed every summary the deployment ever attempted while
        the parts beside it were complete -- the failure looked like a flaky model, not a body
        the API refuses. Determinism is not what is traded away: the field is gone because the
        model no longer accepts being asked."""
        bedrock = FakeBedrock()
        Synthesiser(model_id="m", bedrock=bedrock).summarise("q", parts=[_part()])
        assert "temperature" not in bedrock.request["inferenceConfig"]

    def test_temperature_is_sent_when_explicitly_set(self):
        """Kept settable for an older model that does accept one."""
        bedrock = FakeBedrock()
        Synthesiser(model_id="m", bedrock=bedrock, temperature=0.0).summarise(
            "q", parts=[_part()]
        )
        assert bedrock.request["inferenceConfig"]["temperature"] == 0.0


PATHS = [
    {
        "nodes": ["associate:curtis", "order:o-1", "customer:sam"],
        "steps": [
            {"assertion_id": "a2", "predicate": "APPROVED_RETURN", "reversed": False},
            {"assertion_id": "a1", "predicate": "PLACED_ORDER", "reversed": True},
        ],
        "confidence": 0.9,
    }
]


class TestTheModelIsNotAskedToDoTheJoin:
    """Chaining a flat edge list is the one step in this system nobody can audit.

    Everything else the model is given carries an `assertion_id` back to a page. A link the model
    inferred between two facts carries nothing, and reads exactly like a link the graph found. So
    the chains are assembled deterministically and handed over, and the prompt closes off the
    alternative.
    """

    def test_the_chains_reach_the_model(self):
        prompt = build_prompt("who helped sam", parts=[_part(paths=PATHS)])
        assert "APPROVED_RETURN" in prompt
        assert "connections" in prompt

    def test_the_chains_come_before_the_flat_facts(self):
        """Order is the intervention. A model that reads two hundred loose edges first has usually
        committed to a reading of them before it ever reaches the joined-up version."""
        prompt = build_prompt("q", parts=[_part(paths=PATHS)])
        assert prompt.index("connections") < prompt.index("Thorne Vaux")

    def test_a_part_with_no_chains_carries_no_empty_section(self):
        """An empty `connections` invites the model to fill it. Absent says the graph found no
        route, which is the honest report."""
        assert "connections" not in build_prompt("q", parts=[_part(paths=[])])
        assert "connections" not in build_prompt("q", parts=[_part()])

    def test_the_system_prompt_forbids_chaining_the_flat_facts(self):
        bedrock = FakeBedrock()
        Synthesiser(model_id="m", bedrock=bedrock).summarise("q", parts=[_part(paths=PATHS)])
        system = bedrock.request["system"][0]["text"]
        assert "connections" in system
        assert "the graph did not find it" in system


class TestFailureModes:
    def test_no_parts_means_no_summary(self):
        """None, not an apology: the caller already renders the empty state and its warnings."""
        bedrock = FakeBedrock()
        assert Synthesiser(model_id="m", bedrock=bedrock).summarise("q", parts=[]) is None
        assert bedrock.request is None, "a model was called with nothing to summarise"

    def test_an_unreachable_model_raises_rather_than_returning_prose(self):
        """The planner catches this and keeps the parts. A fabricated summary would be worse than
        no summary, so failing loudly here is what protects the answer."""
        with pytest.raises(SynthesisFailed):
            Synthesiser(model_id="m", bedrock=Unreachable()).summarise("q", parts=[_part()])

    def test_an_empty_response_raises(self):
        with pytest.raises(SynthesisFailed):
            Synthesiser(model_id="m", bedrock=FakeBedrock(text="   ")).summarise(
                "q", parts=[_part()]
            )


class TestThePlannerKeepsTheOrdering:
    """Grounding runs before synthesis, so a model cannot see what the graph refused.

    Asserted through the planner rather than the synthesiser, because the ordering is the planner's
    guarantee: `_blocks_for` then `_without_blocked` then `_synthesise`. A synthesiser that ran
    first would be handed evidence that had not been filtered yet.
    """

    def test_synthesis_receives_only_surviving_parts(self):
        from src.governance import GovernanceSettings
        from src.graph.scope import AuthContext
        from src.query.planner import Planner

        seen: dict = {}

        class Recorder:
            def summarise(self, question, *, parts, blocks=()):
                seen["parts"] = parts
                seen["blocks"] = blocks
                return "written"

        class Metrics:
            def match(self, question):
                return None

        planner = Planner(metric_matcher=Metrics(), synthesiser=Recorder())
        answer = planner.plan(
            AuthContext(user_id="u", tenant_id="test-tenant"),
            "anything",
            GovernanceSettings(),
        )
        # No lane produced anything, so synthesis must not have been attempted at all.
        assert answer.synthesis is None
        assert seen == {}
