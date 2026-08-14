"""Tier 2 and 3: reading assertions to answer a question.

Every read goes through `scope.edge_scope`, so an answer can only ever rest on facts
the caller is allowed to see and that clear the trust floor. That is what makes the
`assertions_used` list on a `Resolution` a real audit trail rather than a label: each
id in it resolves to a document span or a proof tree.

Term matching here is lexical — entity ids and predicates, not embeddings. That is
deliberate for tier 2: "which matters involve Acme Corporation" should find
`party:acme-corporation` by name, and doing it lexically means the result is
reproducible and explainable. Tier 3 is where semantics come in.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from src.documents.review import ReviewQueue
from src.graph.assertions import EpistemicClass
from src.graph.scope import AuthContext

logger = logging.getLogger(__name__)

#: Words that carry no signal for entity matching. Kept small on purpose — an
#: aggressive stop list drops real search terms ("the Crown", "In re Smith").
_NOISE = frozenset(
    {
        "a", "an", "and", "any", "are", "our", "all", "about", "by", "did", "do",
        "does", "for", "from", "give", "has", "have", "how", "in", "involve",
        "involved", "involves", "is", "it", "list", "many", "me", "much", "of",
        "on", "or", "show", "that", "their", "the", "to", "us", "was", "we",
        "what", "when", "where", "which", "who", "whom", "with", "you",
    }
)

_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'&.-]*")


def terms_of(question: str) -> list[str]:
    """Content words, lowercased, order preserved."""
    return [
        w.lower()
        for w in _WORD.findall(question)
        if w.lower() not in _NOISE and len(w) > 1
    ]


def _entity_label(entity_id: str) -> str:
    """`party:acme-corporation` -> `acme corporation`.

    Matching happens on the label rather than the raw id so a question phrased in
    ordinary words can reach an id built by slugging.
    """
    _, _, rest = entity_id.partition(":")
    return (rest or entity_id).replace("-", " ").replace("_", " ").lower()


@dataclass
class Hit:
    """One assertion that matched, with why it matched."""

    assertion_id: str
    subject_id: str
    predicate: str
    object_id: str
    epistemic_class: str
    confidence: float
    matter_id: str | None
    matched_on: list[str]
    source: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "assertion_id": self.assertion_id,
            "subject_id": self.subject_id,
            "predicate": self.predicate,
            "object_id": self.object_id,
            "epistemic_class": self.epistemic_class,
            "confidence": self.confidence,
            "matter_id": self.matter_id,
            "matched_on": self.matched_on,
            "source": self.source,
        }


class GraphReader:
    """Reads assertions for the resolver.

    Backed by `ReviewQueue` for now, which is where assertions currently live. When
    the graph writer lands this becomes `GraphClient.read_scoped` — the interface the
    resolver depends on does not change, which is the point of keeping it narrow.
    """

    def __init__(self, review_queue: ReviewQueue) -> None:
        self._queue = review_queue

    def _readable(self, ctx: AuthContext, min_confidence: float) -> list[Any]:
        """Current, in-scope, trusted-enough assertions.

        `visible()` already applies the tenant and matter walls. The rest of the
        filtering mirrors `scope.edge_scope` so tier 2 and a direct graph read agree
        about what counts as usable.
        """
        trusted = {
            EpistemicClass.DECLARED.value,
            EpistemicClass.EXTRACTED_DET.value,
            EpistemicClass.EXTRACTED_MODEL.value,
            EpistemicClass.INFERRED.value,
        }
        if ctx.include_suggestions:
            trusted.add(EpistemicClass.PREDICTED.value)

        out = []
        for record in self._queue.visible(ctx):
            a = record.assertion
            if not record.is_current:
                continue
            if a.epistemic_class.value not in trusted:
                continue
            if a.confidence < min_confidence:
                continue
            if a.review_state.value not in ("AUTO_ASSERTED", "APPROVED"):
                continue
            out.append(record)
        return out

    def search(
        self,
        ctx: AuthContext,
        question: str,
        *,
        min_confidence: float = 0.8,
        limit: int = 25,
    ) -> list[dict[str, Any]]:
        """Assertions whose entities or predicate match the question.

        Ranked by how many query terms matched, then by confidence. Returning the
        matched terms is what lets the UI say *why* a fact was included.
        """
        terms = terms_of(question)
        if not terms:
            return []

        scored: list[tuple[int, float, Hit]] = []
        for record in self._readable(ctx, min_confidence):
            a = record.assertion
            haystack = " ".join(
                (
                    _entity_label(a.subject_id),
                    _entity_label(a.object_id),
                    a.predicate.replace("_", " ").lower(),
                )
            )
            matched = [t for t in terms if t in haystack]
            if not matched:
                continue
            scored.append(
                (
                    len(matched),
                    a.confidence,
                    Hit(
                        assertion_id=a.assertion_id,
                        subject_id=a.subject_id,
                        predicate=a.predicate,
                        object_id=a.object_id,
                        epistemic_class=a.epistemic_class.value,
                        confidence=a.confidence,
                        matter_id=a.matter_id,
                        matched_on=matched,
                        source=a.source_locator.to_dict(),
                    ),
                )
            )

        scored.sort(key=lambda row: (-row[0], -row[1]))
        return [hit.to_dict() for _, _, hit in scored[:limit]]

    def expand(
        self,
        ctx: AuthContext,
        seed_ids: list[str],
        *,
        depth: int = 2,
        min_confidence: float = 0.8,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Walk out from seed entities or documents along trusted edges.

        Used by tier 3: vector search supplies the passages, this supplies the
        verified relationships around them. Breadth-first and depth-capped, because
        an unbounded walk on a well-connected graph returns the whole tenant.
        """
        records = self._readable(ctx, min_confidence)
        frontier = {s for s in seed_ids if s}
        seen_nodes: set[str] = set()
        out: dict[str, dict[str, Any]] = {}

        for _ in range(max(1, depth)):
            next_frontier: set[str] = set()
            for record in records:
                a = record.assertion
                if a.subject_id not in frontier and a.object_id not in frontier:
                    continue
                out[a.assertion_id] = Hit(
                    assertion_id=a.assertion_id,
                    subject_id=a.subject_id,
                    predicate=a.predicate,
                    object_id=a.object_id,
                    epistemic_class=a.epistemic_class.value,
                    confidence=a.confidence,
                    matter_id=a.matter_id,
                    matched_on=[],
                    source=a.source_locator.to_dict(),
                ).to_dict()
                next_frontier.update({a.subject_id, a.object_id})
                if len(out) >= limit:
                    return list(out.values())
            seen_nodes |= frontier
            frontier = next_frontier - seen_nodes
            if not frontier:
                break

        return list(out.values())
