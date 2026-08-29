"""Splitting a compound question into the questions it actually asks.

"What is our exposure on Northwind and does the engagement letter exclude tax advice" is two
questions. Run whole, it reaches one lane well and the other badly: similarity is dominated by
whichever half has more distinctive words, and the half that loses is not reported as unanswered,
it is reported as nothing found. Split, each half reaches the store that can answer it.

**A split never invents a question.** The model proposes the boundaries and a mechanical check
confirms them: every content word of the original has to appear in some part, and no part may
contain a content word the original did not. So a split is a partition of the reader's own words,
never a rephrasing -- "exposure" silently becoming "financial risk" is rejected, because the
rewritten question would search for something the reader did not ask about while the answer still
carried their wording. This is the same division of labour as `documents/extractors/model.py`: a
model proposes, and something that cannot hallucinate either confirms it or does not.

**Failure is never fatal, and always falls back to the whole question.** No model, no credentials,
unparseable output, a part that fails the coverage check: every one of those returns
`[question]`, which is exactly what the planner did before this module existed. A decomposer that
could fail a question would be trading an occasional better answer for an occasional missing one.

The cheap check comes first. Most questions have no conjunction and no second question mark, and
those skip the model entirely -- a Bedrock hop per question to be told "this is one question" is
latency spent on the common case to help the rare one.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from src.query.graph_reader import NOISE_WORDS, terms_of

logger = logging.getLogger(__name__)

MAX_PARTS = 4
"""Sub-questions one question may become. Beyond this the model has stopped splitting and started
enumerating: every part costs a full fan-out across three stores, and a reader handed nine parts
is reading a search log rather than an answer."""

MAX_TOKENS = 400
"""A list of short questions. More than this is the model explaining itself, which is not wanted."""

MIN_PART_CHARS = 10
"""Shorter than this is a fragment, not a question, and a fragment retrieves noise."""

#: What makes a question worth showing a model. Deliberately loose -- a false positive costs one
#: Bedrock call that returns a single part, a false negative costs the split entirely.
_COMPOUND = re.compile(r"\b(and|also|plus|then|as well as)\b|[;]|\?\s*\S", re.IGNORECASE)

SYSTEM_PROMPT = f"""You split a lawyer's question into the separate questions it contains, so \
each can be searched where it can actually be answered.

Rules:
- Return a JSON array of strings and nothing else. No markdown fence, no commentary.
- If the question asks one thing, return an array with that one question, unchanged.
- Never return an empty array. Every reply has at least one part, because the question itself is \
always a valid answer.
- Use at most {MAX_PARTS} parts.
- Use the asker's own words. Do not rephrase, do not substitute synonyms, do not add a word they \
did not use. A part containing wording they did not write is rejected outright.
- Carry shared context into every part that needs it, repeating it verbatim. "Our exposure on the \
Northwind matter and how many hours were billed" becomes ["What is our exposure on the Northwind \
matter?", "How many hours were billed on the Northwind matter?"] -- the second part is useless \
without the matter name.
- Do not split a single question that merely lists things. "Which parties and dates appear in the \
engagement letter" is one question about one document.
- Do not add a part for something the question implies but does not ask.

Example: "What is our total exposure on Northwind and does the engagement letter exclude tax \
advice?" -> ["What is our total exposure on Northwind?", "Does the Northwind engagement letter \
exclude tax advice?"]"""


class BedrockLike(Protocol):
    def converse(self, **kwargs: Any) -> dict[str, Any]: ...


_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE | re.MULTILINE)


def looks_compound(question: str) -> bool:
    """Whether a question is worth asking a model to split. Cheap, and wrong in the safe direction."""
    return bool(_COMPOUND.search(question))


def _normalised(text: str) -> set[str]:
    """Content words, singularised, for comparing a split against its original.

    Singularised because "matters" becoming "matter" in a part is a faithful split, and failing it
    would send every plural question down the fallback path. `sql_generation._terms` strips the
    same trailing `s` for the same reason.

    **Noise is re-checked after singularising, not before.** `terms_of` filters against
    `NOISE_WORDS` first, so an unpunctuated contraction gets through: "whats" is not a noise word,
    survives, and only then loses its `s` to become "what" -- which is. The original then carried a
    content word no grammatical part could ever cover, so a model that split correctly *and wrote
    proper English* was rejected while one that copied the typo passed. Measured on
    "whats the accomplices in the fraud and whats the total turnover": three correct splits from
    Haiku, all three refused, missing word "what".

    Over-filtering here is safe in a way under-filtering is not: this set is only ever compared
    against another set from the same function, so a word dropped from both sides costs a little
    strictness. A word present on one side alone costs the split.
    """
    stems = {w.replace("'", "").rstrip("s") for w in terms_of(text)}
    return {s for s in stems if s not in NOISE_WORDS and len(s) > 1}


def covers(question: str, parts: list[str]) -> bool:
    """Whether `parts` say the same thing the original did, word for word.

    Both directions, and each catches a different failure. A missing word means the split dropped
    half the question, and the half that vanished would be reported as nothing found rather than
    as not asked. An added word means the model rewrote rather than split, and a query built from a
    word the reader never used is unanswerable in the only sense that matters: they cannot check it
    against what they asked.
    """
    original = _normalised(question)
    if not original:
        return False
    covered: set[str] = set()
    for part in parts:
        terms = _normalised(part)
        if terms - original:
            return False
        covered |= terms
    return not original - covered


@dataclass
class QuestionSplitter:
    """Splits a compound question, or returns it whole. Injectable client, so tests need no AWS."""

    model_id: str
    bedrock: BedrockLike | None = None
    bedrock_factory: Callable[[], BedrockLike] | None = None
    max_tokens: int = MAX_TOKENS
    max_parts: int = MAX_PARTS

    @property
    def client(self) -> BedrockLike:
        if self.bedrock is None:
            factory = self.bedrock_factory
            if factory is None:
                import boto3

                def factory() -> BedrockLike:
                    return boto3.client("bedrock-runtime")

            self.bedrock = factory()
        return self.bedrock

    @property
    def method(self) -> str:
        return f"llm:{self.model_id}"

    def split(self, question: str) -> list[str]:
        """The questions this question contains. Never empty, and never raises.

        A single-element list means "ask this whole", and the caller cannot tell whether that was
        the model's judgement, a refused split or an outage. It does not need to: all three mean
        the same thing to it, and the reason is in the log.
        """
        question = question.strip()
        if not question or not looks_compound(question):
            return [question] if question else []

        parts = self._propose(question)
        if len(parts) < 2:
            return [question]
        if not covers(question, parts):
            # Named at info, because a model that consistently rephrases is a prompt problem and
            # the only way to see it is the count of questions that reached here.
            logger.info("split rejected, parts do not match the question's words: %r", parts)
            return [question]
        return parts

    def _propose(self, question: str) -> list[str]:
        """What the model suggests, cleaned but not yet checked against the original."""
        try:
            response = self.client.converse(
                modelId=self.model_id,
                system=[{"text": SYSTEM_PROMPT}],
                messages=[{"role": "user", "content": [{"text": question}]}],
                # Converse, not InvokeModel: the model is admin-selectable across Nova and
                # Anthropic, and their native request bodies are mutually invalid.
                inferenceConfig={"maxTokens": self.max_tokens},
            )
        except Exception as e:  # noqa: BLE001
            logger.info("question not split, could not reach %s: %s", self.model_id, e)
            return []

        blocks = response.get("output", {}).get("message", {}).get("content") or []
        text = _FENCE.sub("", "".join(b.get("text", "") for b in blocks)).strip()
        try:
            raw = json.loads(text)
        except ValueError:
            logger.info("question not split, model returned no JSON array: %r", text[:200])
            return []
        if not isinstance(raw, list):
            logger.info("question not split, model returned %s not an array", type(raw).__name__)
            return []
        if not raw:
            # An empty array is a contract violation, not a verdict: the prompt guarantees at least
            # the question itself. It was the majority answer from Nova 2 Lite -- four of six calls
            # on a question it split correctly the other two -- and it used to return silently,
            # indistinguishable in the log from "this asks one thing". A model returning it half the
            # time is a model to replace, and that is only visible if it is said.
            logger.info("question not split, model returned an empty array")
            return []

        seen: set[str] = set()
        parts: list[str] = []
        for item in raw:
            part = item.strip() if isinstance(item, str) else ""
            if len(part) < MIN_PART_CHARS or part.lower() in seen:
                continue
            seen.add(part.lower())
            parts.append(part)
            if len(parts) >= self.max_parts:
                break
        if not parts:
            logger.info("question not split, nothing the model returned was a usable part: %r", raw)
        return parts
