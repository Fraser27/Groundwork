"""Prose over evidence that has already been grounded.

The ordering is the whole design, and `Planner.plan` enforces it: blocking runs first and
deterministically, then this writes over whatever survived. **The model never decides what
survives.** It cannot reason about a screened matter even by accident, because it is never shown
one -- so a summary cannot leak what the graph refused.

That is also why this is a separate module rather than a method on the planner. A synthesiser that
could reach the graph would be one refactor away from filtering its own inputs.

What it is told, in the prompt, is as important as what it is given:

- **Say when the parts disagree.** A metric and an extracted fact can conflict, and smoothing that
  over is the failure mode that makes a summary less trustworthy than the rows it summarises.
- **Never introduce a fact.** Every claim has to be in the parts. There is no retrieval here.
- **Name the basis.** "Per the engagement letter" beats an assertive sentence with no source, since
  the reader's next move is to check it.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

logger = logging.getLogger(__name__)

MAX_TOKENS = 1500
"""Enough for a few paragraphs. A summary longer than the evidence is not a summary."""

MAX_PART_CHARS = 4000
"""Per part. A metric returning hundreds of rows would otherwise crowd out every other lane, and
the rows are on screen anyway -- this text exists to explain them, not to repeat them."""

SYSTEM_PROMPT = """You write short factual summaries for lawyers, over evidence that has \
already been assembled and checked. You are the last step, not the researcher.

Absolute rules:
- Use ONLY the parts given to you. If something is not in them, it does not go in the answer, \
however obvious it seems. You have no other source and no way to look anything up.
- Where parts disagree, say so plainly and give both. Do not reconcile them, do not average \
them, do not pick the more likely one. A contradiction is a finding.
- Attribute each claim to the part it came from, in the reader's terms: "the engagement \
letter", "the fees billed metric", "the matters table". The reader's next move is to check you.
- If some evidence was withheld, say that the answer may be incomplete. Never speculate about \
what it was.
- No preamble, no restating the question, no offers of further help. Lead with the answer.
- If the parts do not answer the question, say exactly that. That is a useful answer.

Write plainly. Two or three short paragraphs at most, and fewer where fewer will do."""


class BedrockLike(Protocol):
    def invoke_model(self, **kwargs: Any) -> dict[str, Any]: ...


class SynthesisFailed(RuntimeError):
    """The model could not be reached or returned nothing usable."""


def _clip(value: Any, limit: int = MAX_PART_CHARS) -> Any:
    """Bound a part's payload without hiding that it was bounded."""
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    if len(text) <= limit:
        return value
    return text[:limit] + f"… [truncated, {len(text)} chars total]"


def build_prompt(
    question: str,
    *,
    parts: Sequence[dict[str, Any]],
    blocks: Sequence[dict[str, Any]] = (),
) -> str:
    """The user turn: the question, the surviving evidence, and the fact of any withholding.

    Blocks are named as *count and reason*, never as content. The reader is entitled to know an
    answer is partial -- that is the whole argument for disclosing a screen within a firm -- but a
    model that saw the blocked fact could paraphrase it, which would defeat the block it was told
    about.
    """
    payload: dict[str, Any] = {
        "question": question,
        "parts": [
            {
                "kind": p.get("provenance") or p.get("lane"),
                "tier": p.get("tier"),
                "how_to_cite_it": _describe(p),
                "content": _clip(p.get("content")),
                "sql": p.get("sql"),
                "confidence": p.get("confidence"),
            }
            for p in parts
        ],
    }
    if blocks:
        payload["withheld"] = {
            "count": len(blocks),
            "reasons": sorted({str(b.get("reason") or "not stated") for b in blocks}),
            "note": "Content withheld deliberately. Say the answer may be incomplete.",
        }
    return json.dumps(payload, indent=2, default=str)


def _describe(part: dict[str, Any]) -> str:
    """How a reader would refer to this part, so attribution lands in their language."""
    kind = str(part.get("provenance") or part.get("lane") or "evidence")
    if part.get("sql"):
        return f"the {kind.replace('_', ' ')} query result"
    citations = part.get("citations") or []
    names = sorted({str(c.get("filename") or c.get("document_id")) for c in citations if c})
    if names:
        return "the document " + ", ".join(names[:3])
    return f"the {kind.replace('_', ' ')} result"


@dataclass
class Synthesiser:
    """Writes the summary. Injectable client, so tests need no AWS."""

    model_id: str
    bedrock: BedrockLike | None = None
    bedrock_factory: Callable[[], BedrockLike] | None = None
    max_tokens: int = MAX_TOKENS
    temperature: float | None = None
    """None by default: newer Anthropic models reject `temperature` outright, and Sonnet 5 --
    the configured default -- answers `ValidationException: temperature is deprecated for this
    model`, which failed every summary this deployment attempted. `parse.py` already omits it
    for the same reason. Still settable for an older model that accepts one."""

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

    def summarise(
        self,
        question: str,
        *,
        parts: Sequence[dict[str, Any]],
        blocks: Sequence[dict[str, Any]] = (),
    ) -> str | None:
        """Prose over the parts, or None when there is nothing to write about.

        None rather than an apology: the caller already shows the parts and their warnings, and a
        paragraph explaining that no evidence was found is worse than the empty state beside it.
        """
        if not parts:
            return None

        body: dict[str, Any] = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": self.max_tokens,
            "system": SYSTEM_PROMPT,
            "messages": [
                {"role": "user", "content": build_prompt(question, parts=parts, blocks=blocks)}
            ],
        }
        if self.temperature is not None:
            body["temperature"] = self.temperature

        try:
            response = self.client.invoke_model(modelId=self.model_id, body=json.dumps(body))
            data = json.loads(response["body"].read())
        except Exception as e:
            raise SynthesisFailed(f"could not reach {self.model_id}: {e}") from e

        text = "".join(part.get("text", "") for part in data.get("content", [])).strip()
        if not text:
            raise SynthesisFailed("the model returned no text")
        return text
