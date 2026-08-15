"""Multi-lane query planning: retrieve widely, ground on the graph, compose honestly.

The resolver answers from the first tier that can. That is right for a question a governed
metric answers exactly, and wrong for a question whose answer needs both a number and a
qualifier. "What is our exposure on the Northwind matter" wants the figure from the
warehouse *and* the fact that a document says the engagement excludes tax advice.

The shape, in the user's own analogy: penicillin cures the infection, but the patient is
allergic. OpenSearch surfaces penicillin. The graph knows about the allergy. So **the graph's
job here is grounding what the other stores return**, not being the primary retriever.

    1. A governed metric matches           -> compile, run, return. Nothing else needed.
    2. Otherwise, traverse for candidates  -> documents and catalog schema
    3. Retrieve                            -> passages from OpenSearch
    4. Apply blocks                        -> DETERMINISTICALLY, never by a model
    5. Synthesise                          -> a model writes prose over what survived

**Step 4 is deterministic and that is the whole design.** `ontologies/legal.yaml` already
settled it: `conflict_check` carries `min_premise_class: EXTRACTED_DET` with the comment "a
conflict flag resting on an LLM guess would be worse than none". If a model decides whether
the patient is allergic, a hallucination prescribes penicillin. If the graph decides and the
model only writes the note, a hallucination costs wording.

**Lanes are never collapsed into one score.** The three retrievers use incompatible scales:
weighted term overlap, cosine similarity, and structural reachability with no score at all.
Normalising them would make an invented constant the most important governance decision in
the system, so each lane keeps its own provenance and the caller sees which is which.

**An Athena SUM is exact; a model-extracted assertion has a confidence.** There is no
defensible arithmetic combining those, so a composed answer reports parts separately and is
never labelled plain "governed" when a model contributed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.governance import GovernanceSettings
from src.graph.scope import AuthContext
from src.query.resolver import Tier

logger = logging.getLogger(__name__)


class Lane(str, Enum):
    """Where one part of an answer came from."""

    METRIC = "metric"
    GRAPH = "graph"
    PASSAGES = "passages"
    CATALOG = "catalog"


#: How much of an answer a model wrote. Kept separate from `Lane` because provenance and
#: trustworthiness are different questions: a passage is quoted verbatim (so the text is
#: exact even though retrieval was fuzzy), whereas an inference is a model's opinion.
class Provenance(str, Enum):
    DETERMINISTIC = "deterministic"
    """Compiled SQL or a DECLARED fact. Same inputs, same output, no model."""

    VERBATIM = "verbatim"
    """Text quoted exactly from a document. Retrieval chose it; nothing rewrote it."""

    INFERRED = "inferred"
    """A model's reading. Carries a confidence and sits under the review gate."""


@dataclass
class Part:
    """One lane's contribution. Never merged with another lane's."""

    lane: Lane
    provenance: Provenance
    tier: Tier
    content: Any
    sql: str | None = None
    citations: list[dict[str, Any]] = field(default_factory=list)
    assertion_ids: list[str] = field(default_factory=list)
    confidence: float | None = None
    """None for a deterministic part. A number here means the part is a model's reading, and
    the absence of a number is not the same as certainty about a fuzzy thing."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "lane": self.lane.value,
            "provenance": self.provenance.value,
            "tier": int(self.tier),
            "content": self.content,
            "sql": self.sql,
            "citations": self.citations,
            "assertion_ids": self.assertion_ids,
            "confidence": self.confidence,
        }


@dataclass
class Block:
    """Something the graph refuses to let through, and why.

    A block is a recorded fact, not a judgement: an ethical screen, or a rule that fired on
    DECLARED or EXTRACTED_DET premises. It names its reason because a silent block is the
    failure `scope.py` exists to prevent, where a clean-looking answer is clean only because
    the inconvenient part was invisible.
    """

    subject: str
    reason: str
    rule: str = ""
    matter_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "reason": self.reason,
            "rule": self.rule,
            "matter_id": self.matter_id,
        }


@dataclass
class ComposedAnswer:
    """Several lanes' worth of answer, kept apart on purpose."""

    parts: list[Part] = field(default_factory=list)
    blocks: list[Block] = field(default_factory=list)
    lanes_run: list[Lane] = field(default_factory=list)
    lanes_skipped: dict[str, str] = field(default_factory=dict)
    synthesis: str | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def is_fully_deterministic(self) -> bool:
        """True only when no part and no synthesis involved a model.

        The reason `Resolution.is_governed` is not reused: it returns True for tiers 1 to 3,
        which is fine for a single-tier answer but would label a composed answer containing
        a model-extracted assertion as governed.
        """
        if self.synthesis is not None:
            return False
        return bool(self.parts) and all(
            p.provenance is Provenance.DETERMINISTIC for p in self.parts
        )

    @property
    def governance_label(self) -> str:
        """What to call this answer. Never plain "governed" when a model contributed."""
        if not self.parts:
            return "no answer"
        if self.is_fully_deterministic:
            return "governed"
        kinds = sorted({p.provenance.value for p in self.parts})
        label = " + ".join(kinds)
        return f"{label} + synthesised" if self.synthesis else label

    def to_dict(self) -> dict[str, Any]:
        return {
            "parts": [p.to_dict() for p in self.parts],
            "blocks": [b.to_dict() for b in self.blocks],
            "lanes_run": [lane.value for lane in self.lanes_run],
            "lanes_skipped": self.lanes_skipped,
            "synthesis": self.synthesis,
            "governance": self.governance_label,
            "fully_deterministic": self.is_fully_deterministic,
            "warnings": self.warnings,
            "note": (
                "Parts are reported separately because they are not the same kind of claim. "
                "A compiled metric is exact; a quoted passage is exact text chosen by "
                "similarity; an inference is a model's reading and carries a confidence. "
                "Anything the graph blocked is listed with its reason rather than silently "
                "dropped."
            ),
        }


class Planner:
    """Runs the lanes a question needs and composes what comes back.

    Collaborators are injected and every one is optional. A missing collaborator disables its
    lane and says so in `lanes_skipped`, rather than failing the question: a partial answer
    that names what it could not reach beats no answer.
    """

    def __init__(
        self,
        *,
        metric_matcher: Any | None = None,
        graph_reader: Any | None = None,
        vector_search: Any | None = None,
        catalog: Any | None = None,
        synthesiser: Any | None = None,
    ) -> None:
        self._metrics = metric_matcher
        self._graph = graph_reader
        self._vectors = vector_search
        self._catalog = catalog
        self._synthesiser = synthesiser

    def plan(
        self,
        ctx: AuthContext,
        question: str,
        settings: GovernanceSettings,
        *,
        execute: bool = True,
        allow_synthesis: bool = True,
    ) -> ComposedAnswer:
        answer = ComposedAnswer()
        allowed = {int(t) for t in settings.allowed_tiers}

        # A governed metric that matches is the whole answer. Fanning out anyway would pay
        # Athena plus Neptune plus OpenSearch latency to add nothing: the metric is exact and
        # the question named it.
        if 1 in allowed:
            part = self._metric_part(question, execute=execute)
            if part is not None:
                answer.parts.append(part)
                answer.lanes_run.append(Lane.METRIC)
                return answer
        else:
            answer.lanes_skipped[Lane.METRIC.value] = "tier 1 is not permitted for this tenant"

        # No metric matched. Traverse for candidates, then retrieve against them.
        seeds: list[str] = []
        if 2 in allowed:
            graph_part = self._graph_part(ctx, question, settings)
            if graph_part is not None:
                answer.parts.append(graph_part)
                answer.lanes_run.append(Lane.GRAPH)
                seeds = self._seeds_from(graph_part)
        else:
            answer.lanes_skipped[Lane.GRAPH.value] = "tier 2 is not permitted for this tenant"

        if 3 in allowed:
            passage_part = self._passage_part(ctx, question, settings)
            if passage_part is not None:
                answer.parts.append(passage_part)
                answer.lanes_run.append(Lane.PASSAGES)
                seeds.extend(self._seeds_from(passage_part))
        else:
            answer.lanes_skipped[Lane.PASSAGES.value] = "tier 3 is not permitted for this tenant"

        catalog_part = self._catalog_part(ctx, question)
        if catalog_part is not None:
            answer.parts.append(catalog_part)
            answer.lanes_run.append(Lane.CATALOG)

        # Grounding. Deterministic, and it runs before synthesis so a model never sees
        # evidence the graph refused.
        answer.blocks = self._blocks_for(ctx, seeds, settings)
        if answer.blocks:
            answer.parts = [self._without_blocked(p, answer.blocks) for p in answer.parts]

        if allow_synthesis and answer.parts and self._synthesiser is not None:
            answer.synthesis = self._synthesise(question, answer)
        elif allow_synthesis and answer.parts and self._synthesiser is None:
            answer.warnings.append(
                "No synthesis model is configured, so the parts are returned unsummarised."
            )

        if not answer.parts:
            answer.warnings.append(
                "Nothing matched. No approved metric covers this question and nothing "
                "relevant was found in the graph or the documents."
            )
        return answer

    # ── Lanes ────────────────────────────────────────────────────────────────

    def _metric_part(self, question: str, *, execute: bool) -> Part | None:
        if self._metrics is None:
            return None
        match = self._metrics.match(question)
        if match is None:
            return None
        sql = match.compile()
        return Part(
            lane=Lane.METRIC,
            provenance=Provenance.DETERMINISTIC,
            tier=Tier.GOVERNED_METRIC,
            content=match.run(sql) if execute else None,
            sql=sql,
        )

    def _graph_part(
        self, ctx: AuthContext, question: str, settings: GovernanceSettings
    ) -> Part | None:
        if self._graph is None:
            return None
        hits = self._graph.search(ctx, question, min_confidence=settings.min_confidence_floor)
        if not hits:
            return None
        # A traversal returns whatever classes `edge_scope` admits, so the part is only
        # deterministic if nothing in it was model-extracted.
        model_written = any(
            str(h.get("epistemic_class", "")).upper() == "EXTRACTED_MODEL" for h in hits
        )
        confidences = [
            h["confidence"] for h in hits if isinstance(h.get("confidence"), int | float)
        ]
        return Part(
            lane=Lane.GRAPH,
            provenance=Provenance.INFERRED if model_written else Provenance.DETERMINISTIC,
            tier=Tier.GRAPH_TRAVERSAL,
            content=hits,
            assertion_ids=[h["assertion_id"] for h in hits if "assertion_id" in h],
            confidence=min(confidences) if confidences and model_written else None,
        )

    def _passage_part(
        self, ctx: AuthContext, question: str, settings: GovernanceSettings
    ) -> Part | None:
        if self._vectors is None:
            return None
        passages = self._vectors.search(ctx, question, top_k=settings.vector_top_k)
        if not passages:
            return None
        return Part(
            lane=Lane.PASSAGES,
            # The text is quoted exactly. Similarity chose which passage, but nothing
            # rewrote it, so this is not the same kind of claim as an inference.
            provenance=Provenance.VERBATIM,
            tier=Tier.HYBRID,
            content=passages,
            citations=[
                {
                    "document_id": p.get("document_id"),
                    "page": p.get("page"),
                    "char_start": p.get("char_start"),
                    "char_end": p.get("char_end"),
                }
                for p in passages
            ],
        )

    def _catalog_part(self, ctx: AuthContext, question: str) -> Part | None:
        """Schema that might answer the structured half.

        Schemas only. Rows stay in Athena and are queried in place, which is what lets one
        graph hold structured metadata and unstructured content without copying a warehouse
        into it.
        """
        if self._catalog is None:
            return None
        try:
            tables = self._catalog.tables(ctx.tenant_id)
        except Exception as e:
            logger.debug("catalog lane unavailable: %s", e)
            return None
        if not tables:
            return None

        terms = {t for t in question.lower().split() if len(t) > 3}
        relevant = [
            {
                "full_name": t.full_name,
                "description": t.description,
                "columns": [c.name for c in t.columns],
            }
            for t in tables
            if terms & {w for w in f"{t.full_name} {t.description}".lower().split() if len(w) > 3}
        ]
        if not relevant:
            return None
        return Part(
            lane=Lane.CATALOG,
            provenance=Provenance.DETERMINISTIC,
            tier=Tier.GRAPH_TRAVERSAL,
            content=relevant,
        )

    # ── Grounding ────────────────────────────────────────────────────────────

    def _blocks_for(
        self, ctx: AuthContext, seeds: list[str], settings: GovernanceSettings
    ) -> list[Block]:
        """What the graph refuses to let through.

        Two sources, both deterministic. An ethical screen is a recorded decision on
        `AuthContext`. A rule block is an inference, but one the ontology restricts to
        DECLARED and EXTRACTED_DET premises, so no model opinion reaches it.

        Never asks a model. That is the design decision the whole module rests on.
        """
        blocks: list[Block] = []

        for matter_id in sorted(ctx.matter_denylist):
            blocks.append(
                Block(
                    subject=matter_id,
                    reason=ctx.screen_reasons.get(matter_id)
                    or "You are screened from this matter.",
                    rule="ethical_screen",
                    matter_id=matter_id,
                )
            )

        if self._graph is not None and seeds:
            try:
                for found in self._graph.blocking_facts(
                    ctx, seeds, min_confidence=settings.min_confidence_floor
                ):
                    blocks.append(
                        Block(
                            subject=str(found.get("subject_id", "")),
                            reason=str(found.get("reason") or found.get("predicate") or "blocked"),
                            rule=str(found.get("rule") or found.get("predicate") or ""),
                            matter_id=found.get("matter_id"),
                        )
                    )
            except AttributeError:
                # An older reader has no `blocking_facts`. Screens still apply, so grounding
                # degrades rather than disappearing, and the gap is visible in the response.
                logger.debug("graph reader cannot report blocking facts")
            except Exception as e:
                logger.warning("could not read blocking facts: %s", e)

        return blocks

    def _without_blocked(self, part: Part, blocks: list[Block]) -> Part:
        """Drop blocked subjects from a part, keeping the part itself.

        Removing the whole part would hide that anything matched at all, which is the silent
        failure this design exists to avoid. The blocks are reported alongside.
        """
        blocked = {b.subject for b in blocks if b.subject}
        blocked_matters = {b.matter_id for b in blocks if b.matter_id}
        if not blocked and not blocked_matters:
            return part
        if not isinstance(part.content, list):
            return part

        kept = [
            row
            for row in part.content
            if not (
                isinstance(row, dict)
                and (
                    row.get("subject_id") in blocked
                    or row.get("document_id") in blocked
                    or row.get("matter_id") in blocked_matters
                )
            )
        ]
        if len(kept) == len(part.content):
            return part

        # The id lists have to be filtered too. Removing a row from `content` while leaving
        # its assertion id or citation in place would still hand the blocked subject to the
        # synthesiser, which is exactly the leak this step exists to prevent.
        kept_ids = {row.get("assertion_id") for row in kept if isinstance(row, dict)}
        kept_docs = {row.get("document_id") for row in kept if isinstance(row, dict)}
        return Part(
            lane=part.lane,
            provenance=part.provenance,
            tier=part.tier,
            content=kept,
            sql=part.sql,
            citations=[c for c in part.citations if c.get("document_id") in kept_docs]
            if kept_docs
            else part.citations,
            assertion_ids=[a for a in part.assertion_ids if a in kept_ids],
            confidence=part.confidence,
        )

    # ── Synthesis ────────────────────────────────────────────────────────────

    def _synthesise(self, question: str, answer: ComposedAnswer) -> str | None:
        """Ask a model to write prose over what survived grounding.

        The model sees only the evidence that passed step 4, so it cannot reason about a
        blocked fact even accidentally. It writes; it does not decide.
        """
        try:
            return self._synthesiser.summarise(
                question,
                parts=[p.to_dict() for p in answer.parts],
                blocks=[b.to_dict() for b in answer.blocks],
            )
        except Exception as e:
            logger.warning("synthesis failed: %s", e)
            answer.warnings.append(
                "The parts below are complete, but writing a summary of them failed."
            )
            return None

    @staticmethod
    def _seeds_from(part: Part) -> list[str]:
        """Entity and document ids a part touched, for grounding.

        `matter_id` is the join key rather than a fuzzy name match. It already exists on
        every chunk and every assertion, so no normalisation is needed and a mis-join is not
        possible.
        """
        if not isinstance(part.content, list):
            return []
        seeds: list[str] = []
        for row in part.content:
            if not isinstance(row, dict):
                continue
            for key in ("document_id", "subject_id", "object_id", "matter_id"):
                value = row.get(key)
                if isinstance(value, str) and value:
                    seeds.append(value)
        return seeds
