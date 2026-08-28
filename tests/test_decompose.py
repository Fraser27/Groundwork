"""Splitting a compound question, and the two ways that can go wrong.

The split is a partition of the reader's own words. A model proposes the boundaries and a
mechanical check confirms them, which is the same division of labour as
`documents/extractors/model.py`: a model proposes, and something that cannot hallucinate either
confirms it or does not. So the tests that matter are the rejections.

- **A dropped word** means half the question vanished, and the half that vanished is reported as
  nothing found rather than as not asked.
- **An added word** means the model rewrote rather than split, and a query built from a word the
  reader never used is unanswerable in the only sense that counts: they cannot check it against
  what they asked.

And the property that keeps the feature from costing anything: **failure always falls back to the
whole question.** Splitting is an improvement to how a question is searched, and an improvement
that can fail a question is not worth having.

No AWS. The Bedrock client is injected.
"""

from __future__ import annotations

import json
from typing import Any

from src.query.decompose import (
    MAX_PARTS,
    MIN_PART_CHARS,
    QuestionSplitter,
    covers,
    looks_compound,
)

COMPOUND = "What is our exposure on Northwind and does the letter exclude tax advice?"


class FakeBedrock:
    """Returns whatever text it was given, as a Converse response."""

    def __init__(self, text: str = "[]", raises: bool = False) -> None:
        self.text = text
        self.raises = raises
        self.calls: list[dict[str, Any]] = []

    def converse(self, **kw: Any) -> dict[str, Any]:
        if self.raises:
            raise RuntimeError("bedrock unreachable")
        self.calls.append(kw)
        return {"output": {"message": {"content": [{"text": self.text}]}}}


def splitter(text: str = "[]", **over: Any) -> QuestionSplitter:
    kw: dict[str, Any] = {"model_id": "test-model", "bedrock": FakeBedrock(text)}
    kw.update(over)
    return QuestionSplitter(**kw)


def proposing(*parts: str) -> QuestionSplitter:
    return splitter(json.dumps(list(parts)))


class TestTheCheapCheckComesFirst:
    """A Bedrock hop per question to be told "this is one question" is latency spent on the common
    case to help the rare one."""

    def test_a_single_question_never_reaches_the_model(self):
        bedrock = FakeBedrock(json.dumps(["a", "b"]))
        parts = QuestionSplitter(model_id="m", bedrock=bedrock).split("who represents Northwind")

        assert parts == ["who represents Northwind"]
        assert bedrock.calls == []

    def test_a_conjunction_is_worth_asking_about(self):
        assert looks_compound("our exposure and the letter's terms")

    def test_a_second_question_mark_is_worth_asking_about(self):
        assert looks_compound("Who represents them? What did they agree?")

    def test_a_semicolon_is_worth_asking_about(self):
        assert looks_compound("our exposure; the letter's terms")

    def test_a_plain_question_is_not(self):
        assert not looks_compound("which matters involve Calder")

    def test_a_trailing_question_mark_alone_is_not_compound(self):
        """`\\?\\s*\\S` has to mean "a question mark with a question after it", not "a question
        mark", or every question in the product pays for a model call."""
        assert not looks_compound("who represents Northwind?")

    def test_an_empty_question_is_no_questions(self):
        assert splitter().split("   ") == []


class TestASplitNeverInventsAQuestion:
    def test_a_faithful_split_is_kept(self):
        parts = proposing(
            "What is our exposure on Northwind?", "Does the letter exclude tax advice?"
        ).split(COMPOUND)

        assert len(parts) == 2

    def test_a_rephrased_part_is_rejected(self):
        """ "exposure" silently becoming "financial risk" would search for something the reader did
        not ask about while the answer still carried their wording."""
        parts = proposing(
            "What is our financial risk on Northwind?", "Does the letter exclude tax advice?"
        ).split(COMPOUND)

        assert parts == [COMPOUND]

    def test_a_dropped_half_is_rejected(self):
        """The dangerous direction. A half that vanished comes back as nothing found rather than as
        a question nobody asked."""
        parts = proposing("What is our exposure on Northwind?", "What is our exposure?").split(
            COMPOUND
        )

        assert parts == [COMPOUND]

    def test_a_plural_becoming_singular_is_still_faithful(self):
        """Singularised on both sides, or every plural question in the product takes the fallback
        path -- "which matters" splitting into "the matter" is a split, not a rewrite."""
        assert covers(
            "which matters involve Calder and Acme",
            ["which matter involves Calder", "which matter involves Acme"],
        )

    def test_one_subject_of_two_is_a_dropped_half(self):
        assert not covers("which matters involve Calder and Acme", ["which matter involves Calder"])

    def test_shared_context_repeated_verbatim_is_faithful(self):
        """Carrying the matter name into the second part is required, not an addition: the part is
        useless without it."""
        assert covers(
            "our exposure on the Northwind matter and how many hours were billed",
            [
                "What is our exposure on the Northwind matter?",
                "How many hours were billed on the Northwind matter?",
            ],
        )

    def test_a_question_with_no_content_words_covers_nothing(self):
        assert covers("and or the", ["and or the"]) is False


class TestFailureFallsBackToTheWholeQuestion:
    def test_an_unreachable_model_asks_the_question_whole(self):
        parts = splitter(bedrock=FakeBedrock(raises=True)).split(COMPOUND)
        assert parts == [COMPOUND]

    def test_output_that_is_not_json_asks_the_question_whole(self):
        assert splitter("I think this asks two things.").split(COMPOUND) == [COMPOUND]

    def test_output_that_is_json_but_not_a_list_asks_the_question_whole(self):
        assert splitter('{"parts": ["a", "b"]}').split(COMPOUND) == [COMPOUND]

    def test_a_single_part_asks_the_question_whole(self):
        """Not the model's part, the reader's. A model that judged the question single may have
        rephrased it while doing so."""
        assert proposing("Northwind exposure and the letter").split(COMPOUND) == [COMPOUND]

    def test_a_fenced_array_is_still_read(self):
        """The one tolerated deviation, because "no markdown fence" is the instruction models
        ignore most often and the content behind it is perfectly good."""
        body = json.dumps(
            ["What is our exposure on Northwind?", "Does the letter exclude tax advice?"]
        )
        parts = splitter(f"```json\n{body}\n```").split(COMPOUND)

        assert len(parts) == 2


class TestWhatComesBackIsBounded:
    def test_a_fragment_is_not_a_part(self):
        """A fragment retrieves noise, and it would also fail coverage for the half it dropped."""
        parts = proposing("tax", "Does the letter exclude tax advice?").split(COMPOUND)
        assert parts == [COMPOUND]
        assert MIN_PART_CHARS > len("tax")

    def test_a_repeated_part_is_offered_once(self):
        """Deduplicated case-insensitively, so a model echoing a part does not double the fan-out
        that part costs."""
        question = "exposure on Northwind and the letter's terms"
        parts = proposing(
            "exposure on Northwind", "Exposure on Northwind", "the letter's terms"
        ).split(question)

        assert len(parts) == 2

    def test_the_cap_itself_is_reachable(self):
        asked = [f"what happened on the topic{i} matter" for i in range(MAX_PARTS)]
        parts = proposing(*asked).split(" and ".join(asked))

        assert len(parts) == MAX_PARTS

    def test_a_model_that_enumerates_gets_the_question_whole(self):
        """Beyond the cap the model has stopped splitting and started enumerating, and every part
        costs a full fan-out across three stores. Truncating is what makes the rest of the split
        unfaithful, so the coverage check turns the cap into the fallback rather than into a
        silently half-asked question."""
        asked = [f"what happened on the topic{i} matter" for i in range(MAX_PARTS + 3)]
        question = " and ".join(asked)
        parts = proposing(*asked).split(question)

        assert parts == [question]

    def test_the_client_is_asked_with_converse(self):
        """Converse, not InvokeModel: the model is admin-selectable across vendors and their native
        request bodies are mutually invalid."""
        bedrock = FakeBedrock(json.dumps(["a" * 20, "b" * 20]))
        QuestionSplitter(model_id="m", bedrock=bedrock).split(COMPOUND)

        assert bedrock.calls[0]["modelId"] == "m"
        assert bedrock.calls[0]["system"]


class TestTheSplitIsDisclosed:
    def test_the_method_names_the_model(self):
        """A split changes what was searched, so which model proposed it belongs in the trace."""
        assert splitter().method == "llm:test-model"
