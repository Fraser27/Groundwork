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
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from src.governance import GovernanceSettings
from src.graph.scope import AuthContext
from src.query.blocks import (
    DEGRADED_WARNING,
    Block,
    Screen,
    advisory_warning,
    blocks_for,
    seeds_from,
)
from src.query.graph_reader import passage_seeds
from src.query.metric_matcher import chosen_deterministically, match_metric, selection_of
from src.query.resolver import UNGOVERNED_BLOCKED, BlockedQuery, Tier
from src.query.sql_generation import relevant_tables

logger = logging.getLogger(__name__)

#: Whether the router may remove a lane from a composed answer, or is only recorded in the trace.
#:
#: Recorded. `/query` exists to answer from the first tier that can, and compose exists so a reader
#: can see everything the system found -- a lane the router quietly dropped would be invisible in
#: the one place whose entire purpose is visibility, and "the router scored it low" is not a fact
#: the reader gets to check if the lane never ran. The cost is a round trip on a lane that scored
#: badly. Flip this to True to trade that completeness for the saved latency; the trace then says
#: `applied: true` and the UI reads the router's tiers as what ran rather than as advice.
ROUTER_NARROWS_LANES = False


class Lane(str, Enum):
    """Where one part of an answer came from."""

    METRIC = "metric"
    GRAPH = "graph"
    PASSAGES = "passages"
    CATALOG = "catalog"
    SQL = "sql"


#: How much of an answer a model wrote. Kept separate from `Lane` because provenance and
#: trustworthiness are different questions: a passage is quoted verbatim (so the text is
#: exact even though retrieval was fuzzy), whereas an inference is a model's opinion.
class Provenance(str, Enum):
    DETERMINISTIC = "deterministic"
    """Compiled SQL or a DECLARED fact. Same inputs, same output, no model."""

    MODEL_SELECTED = "model_selected"
    """Compiled SQL, but a model chose *which* approved definition to compile.

    A metric no question word matched, reached by similarity to its description. The figure is an
    exact aggregate over a definition a human approved -- nothing in it is a model's reading, so it
    is not `INFERRED` -- but the same question worded differently could reach a different metric, so
    it is not `DETERMINISTIC` either. Folding it into `DETERMINISTIC` would let a fully-governed
    label rest on a cosine, which is the distinction this enum exists to hold."""

    VERBATIM = "verbatim"
    """Text quoted exactly from a document. Retrieval chose it; nothing rewrote it."""

    INFERRED = "inferred"
    """A model's reading. Carries a confidence and sits under the review gate."""

    MODEL_WRITTEN = "model_written"
    """A model wrote the query. Distinct from both of the above, because it fails differently.

    `INFERRED` is a model reading a document, and it fails by misreading text a reader can go and
    check. `MODEL_SELECTED` is a model choosing between definitions a human approved, so the
    arithmetic is still someone's. This is a model choosing the arithmetic: the figure is exact
    over whatever the query happened to ask, and the query is the thing to check. Nothing approved
    it, so no part carrying this can be called governed."""


#: What each provenance is called in `governance_label`, which a lawyer reads. The enum values are
#: wire identifiers, and "model_selected" is not a phrase to put in front of a client.
_PROVENANCE_LABEL = {
    Provenance.DETERMINISTIC: "deterministic",
    Provenance.MODEL_SELECTED: "deterministic SQL, metric chosen by similarity",
    Provenance.VERBATIM: "verbatim",
    Provenance.INFERRED: "inferred",
    Provenance.MODEL_WRITTEN: "query written by AI, not from an approved metric",
}


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

    metric_selection: dict[str, Any] | None = None
    """Metric lane only: which metric, and whether the choice of it was deterministic. The same
    shape `Resolution.metric_selection` carries, from `metric_matcher.selection_of`."""

    error: str | None = None
    """Why this part has no content, when the reason is a refusal or a failure rather than absence.

    The SQL lane needs it: the firewall validates tables, not columns, so a hallucinated column
    reaches Athena and errors. Reported as `content=None` with an error rather than as an empty
    list, because an empty list reads as "no data" -- the silent failure `scope.py` exists to
    prevent -- and "the query was wrong" is a fact the reader can act on."""

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
            "metric_selection": self.metric_selection,
            "error": self.error,
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

    router: Any | None = None
    """Why these lanes and not others. None means no router was wired, which is a different
    statement from a router that ran and could not choose -- see `RouterDecision.degraded`."""

    gate: dict[str, Any] | None = None
    """What step 4 considered, cleared and withheld, and whether both its sources ran.

    Reported even when nothing was refused. A wall visible only when it blocks cannot be
    distinguished from one that never ran, which is exactly how the rule-block half stayed
    missing."""

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
        kinds = sorted({_PROVENANCE_LABEL[p.provenance] for p in self.parts})
        label = " + ".join(kinds)
        return f"{label} + synthesised" if self.synthesis else label

    def to_dict(self) -> dict[str, Any]:
        return {
            "parts": [p.to_dict() for p in self.parts],
            "blocks": [b.to_dict() for b in self.blocks],
            "lanes_run": [lane.value for lane in self.lanes_run],
            "lanes_skipped": self.lanes_skipped,
            "gate": self.gate,
            "router": self.router.to_dict() if self.router is not None else None,
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


def _row_count(content: Any) -> int:
    """Rows in a part, or 0 for a part that is not a list. A metric's content is a dict or None,
    and counting it as one item would report a withheld row that never existed."""
    return len(content) if isinstance(content, list) else 0


def _skipped(tier: int, allowed: set[int], decision: Any | None) -> str:
    """Why a lane did not run. The tenant cap is named first, and named as the cap.

    "Your administrator turned this off" and "this did not look relevant" are different facts, and
    a lane skipped for the first reason must never be described in the words of the second.
    """
    if tier not in allowed:
        return f"tier {tier} is not permitted for this tenant"
    reason = (getattr(decision, "dropped", None) or {}).get(str(tier))
    if reason:
        return f"the router did not select tier {tier}: {reason}"
    return f"tier {tier} was not selected by the router"


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
        router: Any | None = None,
        sql_lane: Any | None = None,
        synonyms_for: Callable[[AuthContext], Mapping[str, Sequence[str]]] | None = None,
    ) -> None:
        self._metrics = metric_matcher
        self._graph = graph_reader
        self._vectors = vector_search
        self._catalog = catalog
        self._synthesiser = synthesiser
        # Optional, and absent means what compose did before it existed: every permitted lane runs
        # and the trace cannot say why those lanes.
        self._router = router
        # The `SqlLane` from `sql_generation`, which `Resolver` is also given. One module, so the
        # two endpoints cannot disagree about whether a question got model-written SQL.
        self._sql = sql_lane
        # Injected so this layer never learns where approved synonyms live. Absent means table
        # selection is word overlap alone, which is what it was before synonyms were readable.
        self._synonyms_for = synonyms_for
        self.blocked: list[BlockedQuery] = []

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

        decision = self._route(ctx, question, settings)
        answer.router = decision
        runnable = self._runnable(allowed, decision)

        # A governed metric that matches is the whole answer. Fanning out anyway would pay
        # Athena plus Neptune plus OpenSearch latency to add nothing: the metric is exact and
        # the question named it.
        if 1 in runnable:
            part = self._metric_part(ctx, question, settings, execute=execute)
            if part is not None:
                answer.parts.append(part)
                answer.lanes_run.append(Lane.METRIC)
                return answer
        else:
            answer.lanes_skipped[Lane.METRIC.value] = _skipped(1, allowed, decision)

        # No metric matched. Retrieve first, so the graph lane can walk out from the passages the
        # way `Resolver._try_hybrid` does -- compose used to run the term search alone, so the same
        # question returned walked facts on `/query` and none here. The reported order is unchanged:
        # a verified relationship is the stronger claim and a reader should meet it first.
        passage_part = self._passage_part(ctx, question, settings) if 3 in runnable else None

        # The graph part has two halves gated by different tiers, and conflating them was a bug.
        #
        # The term search over assertions is tier 2, and asking for it when tier 2 is forbidden
        # would be a cap bypass. Walking out from the passages just retrieved is **tier 3's own
        # work** -- it is what `TIER_EXPLANATION[HYBRID]` promises ("following verified
        # relationships out from them") and what `Resolver._try_hybrid` does with no tier-2 check.
        #
        # While one `if 2 in runnable` guarded both, a tenant permitted only tier 3 got passages
        # and nothing else: on `demo-firm` the walk finds 30 citable facts against the term
        # search's 8, so the lane reported no `assertion_ids` at all and hybrid was not hybrid.
        seeds: list[str] = []
        graph_part = self._graph_part(
            ctx,
            question,
            settings,
            passages=passage_part.content if passage_part is not None else None,
            include_search=2 in runnable,
        )
        if graph_part is not None:
            answer.parts.append(graph_part)
            answer.lanes_run.append(Lane.GRAPH)
            seeds = self._seeds_from(graph_part)
        elif 2 not in runnable:
            answer.lanes_skipped[Lane.GRAPH.value] = _skipped(2, allowed, decision)

        if 3 in runnable:
            if passage_part is not None:
                answer.parts.append(passage_part)
                answer.lanes_run.append(Lane.PASSAGES)
                seeds.extend(self._seeds_from(passage_part))
        else:
            answer.lanes_skipped[Lane.PASSAGES.value] = _skipped(3, allowed, decision)

        # Gated on tier 3, like the passage lane. It ran with no gate at all, so a tenant who had
        # forbidden every tier still received catalog schema -- a small cap bypass -- and it stamped
        # tier 2 on a part nothing checked against tier 2. Catalog is part of what tier 3 means now:
        # passages, the relationships around them, and the schema of the tables involved.
        if 3 in runnable:
            catalog_part = self._catalog_part(ctx, question)
            if catalog_part is not None:
                answer.parts.append(catalog_part)
                answer.lanes_run.append(Lane.CATALOG)
        else:
            answer.lanes_skipped[Lane.CATALOG.value] = _skipped(3, allowed, decision)

        # The catalog lane finds the schema; this is what it is for. Gated on tier 3 like the other
        # two, and additionally on the kill switch -- which refuses this lane alone. Refusing the
        # whole tier would take passages and graph facts down with it, turning a switch that
        # removes an ungoverned capability into one that removes governed answers.
        if 3 not in runnable:
            answer.lanes_skipped[Lane.SQL.value] = _skipped(3, allowed, decision)
        elif settings.block_ungoverned_queries:
            answer.lanes_skipped[Lane.SQL.value] = UNGOVERNED_BLOCKED
            self.blocked.append(
                BlockedQuery(
                    tenant_id=ctx.tenant_id,
                    user_id=ctx.user_id,
                    question=question,
                    reason=UNGOVERNED_BLOCKED,
                    at=datetime.now(UTC).isoformat(),
                )
            )
        else:
            sql_part = self._sql_part(ctx, question)
            if sql_part is not None:
                answer.parts.append(sql_part)
                answer.lanes_run.append(Lane.SQL)

        # Grounding. Deterministic, and it runs before synthesis so a model never sees
        # evidence the graph refused.
        screen = blocks_for(ctx, graph_reader=self._graph, seeds=seeds)
        answer.blocks = screen.blocks
        withheld = 0
        if screen:
            kept_parts = [self._without_blocked(p, screen.blocks) for p in answer.parts]
            withheld = sum(
                _row_count(before.content) - _row_count(after.content)
                for before, after in zip(answer.parts, kept_parts, strict=True)
            )
            answer.parts = kept_parts
        answer.gate = screen.trace(items_withheld=withheld)
        if screen.degraded:
            answer.warnings.append(DEGRADED_WARNING)
        if screen.advisories:
            # Both endpoints, one wording. A reader told about a conflict on `/query` and not on
            # `/query/compose` would get two different accounts of the same graph.
            answer.warnings.append(advisory_warning(screen))

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

    # ── Routing ──────────────────────────────────────────────────────────────

    def _route(self, ctx: AuthContext, question: str, settings: GovernanceSettings) -> Any | None:
        """The routing decision, or None when there is no router. Never raises."""
        if self._router is None:
            return None
        try:
            decision = self._router.route(ctx, question, settings)
        except Exception as e:  # noqa: BLE001
            # `TierRouter.route` is documented never to raise, so this is the guard against a
            # future one that does. An optimisation must not be able to fail a question.
            logger.warning("router failed, composing every permitted lane: %s", e)
            return None
        # Whoever consumes a decision declares whether it acted on it. The resolver does; compose
        # does not, and a trace that did not say so would have the UI report a lane as unsearched
        # while the lane below it shows its results.
        decision.applied = ROUTER_NARROWS_LANES
        return decision

    @staticmethod
    def _runnable(allowed: set[int], decision: Any | None) -> set[int]:
        """Which tiers may run. Intersection only -- the router narrows, never widens."""
        if not ROUTER_NARROWS_LANES or decision is None:
            return allowed
        # A degraded decision carries every permitted tier, so this is already a no-op for it.
        # Intersecting rather than trusting `decision.tiers` keeps the tenant cap authoritative
        # even if a router ever reports a tier outside it.
        narrowed = allowed & {int(t) for t in getattr(decision, "tiers", ()) or ()}
        # Empty means routing would have cost the answer entirely, which it may never do.
        return narrowed or allowed

    # ── Lanes ────────────────────────────────────────────────────────────────

    def _metric_part(
        self,
        ctx: AuthContext,
        question: str,
        settings: GovernanceSettings,
        *,
        execute: bool,
    ) -> Part | None:
        # Same call `Resolver._try_metric` makes, with the same two arguments, so a question that
        # is governed on one endpoint is governed on the other. The two have disagreed before.
        match = match_metric(self._metrics, question, ctx, settings) if self._metrics else None
        if match is None:
            return None
        sql = match.compile()
        return Part(
            lane=Lane.METRIC,
            provenance=Provenance.DETERMINISTIC
            if chosen_deterministically(match)
            else Provenance.MODEL_SELECTED,
            tier=Tier.GOVERNED_METRIC,
            content=match.run(sql) if execute else None,
            sql=sql,
            metric_selection=selection_of(match),
        )

    def _graph_part(
        self,
        ctx: AuthContext,
        question: str,
        settings: GovernanceSettings,
        *,
        passages: Any = None,
        include_search: bool = True,
    ) -> Part | None:
        """Verified relationships, from a term search and from walking out of the passages.

        `include_search` is False when tier 2 is not permitted. The walk still runs, because it
        belongs to tier 3: the passages it starts from were retrieved under tier 3 and the
        relationships around them are what "hybrid" means. Only the term search is tier 2's.
        """
        if self._graph is None:
            return None
        hits = (
            self._graph.search(ctx, question, min_confidence=settings.min_confidence_floor)
            if include_search
            else []
        )
        hits = [*hits, *self._walked(ctx, passages, settings, already=hits)]
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
            # The tier that authorised this part, which is not always tier 2. With the term
            # search off, every fact here was walked out from a tier 3 retrieval, and stamping
            # it tier 2 would report a tier the tenant has forbidden as having run.
            tier=Tier.GRAPH_TRAVERSAL if include_search else Tier.HYBRID,
            content=hits,
            assertion_ids=[h["assertion_id"] for h in hits if "assertion_id" in h],
            confidence=min(confidences) if confidences and model_written else None,
        )

    def _walked(
        self,
        ctx: AuthContext,
        passages: Any,
        settings: GovernanceSettings,
        *,
        already: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Verified relationships around the retrieved passages, deduplicated against the search.

        The same `expand()` call `Resolver._try_hybrid` makes, from the same seeds, because two
        endpoints disagreeing about what a question found is a bug class this repo keeps hitting.

        A reader with no `expand` contributes its term matches and nothing more. Checked rather
        than caught, and a failure inside `expand` is deliberately not swallowed: quietly
        returning fewer facts is the silent empty this whole path exists to stop being possible.
        """
        expand = getattr(self._graph, "expand", None)
        seeds = passage_seeds(passages)
        if expand is None or not seeds:
            return []
        edges = expand(
            ctx,
            seeds,
            depth=settings.graph_expand_depth,
            min_confidence=settings.min_confidence_floor,
        )
        seen = {h.get("assertion_id") for h in already}
        return [e for e in edges if e.get("assertion_id") not in seen]

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

        # The same selection the SQL lane makes, from the same function. A reader shown one list of
        # schema while a query was written over another could not check the query against it.
        relevant = [
            {
                "full_name": t.full_name,
                "description": t.description,
                "columns": [c.name for c in t.columns],
            }
            for t in relevant_tables(question, tables, synonyms=self._synonyms(ctx))
        ]
        if not relevant:
            return None
        return Part(
            lane=Lane.CATALOG,
            provenance=Provenance.DETERMINISTIC,
            # Tier 3, matching the gate above. It claimed tier 2 while nothing checked it against
            # tier 2, so the reported tier was unbacked by any permission check.
            tier=Tier.HYBRID,
            content=relevant,
        )

    def _sql_part(self, ctx: AuthContext, question: str) -> Part | None:
        """A query a model wrote over the catalogued schema, run if it cleared the firewall.

        Never `DETERMINISTIC`, whatever came back. The figure may be an exact aggregate, but
        nothing approved the arithmetic, so `MODEL_WRITTEN` is what makes `governance_label` stop
        saying governed -- which is the point of the distinction.
        """
        if self._sql is None or self._catalog is None:
            return None
        try:
            tables = self._catalog.tables(ctx.tenant_id)
        except Exception as e:  # noqa: BLE001
            logger.debug("SQL lane unavailable, no catalog: %s", e)
            return None
        if not tables:
            return None

        result = self._sql.run(question, tables=tables, synonyms=self._synonyms(ctx))
        if result is None:
            return None
        return Part(
            lane=Lane.SQL,
            provenance=Provenance.MODEL_WRITTEN,
            tier=Tier.HYBRID,
            content=result.rows,
            sql=result.generated.sql,
            error=result.error,
        )

    def _synonyms(self, ctx: AuthContext) -> Mapping[str, Sequence[str]] | None:
        """Approved synonyms per table `full_name`, or None.

        Degrades rather than raises: synonyms widen table selection, so a graph that cannot be
        reached must cost the widening and nothing else. The catalog and SQL lanes both read this,
        because a reader shown one list of schema while a query was written over another could not
        check the query against it.
        """
        if self._synonyms_for is None:
            return None
        try:
            return self._synonyms_for(ctx)
        except Exception as e:  # noqa: BLE001
            logger.warning("no approved synonyms for table selection: %s", e)
            return None

    # ── Grounding ────────────────────────────────────────────────────────────

    def _without_blocked(self, part: Part, blocks: list[Block]) -> Part:
        """Drop blocked subjects from a part, keeping the part itself.

        Removing the whole part would hide that anything matched at all, which is the silent
        failure this design exists to avoid. The blocks are reported alongside.
        """
        screen = Screen(blocks=blocks)
        if not screen:
            return part
        if not isinstance(part.content, list):
            return part

        kept = screen.keep(part.content)
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
            metric_selection=part.metric_selection,
            error=part.error,
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
            # The reason travels with the warning. It reached only the log for the life of this
            # route, so an API body the model refuses outright looked exactly like a flaky
            # model, and the one detail that identified it took a CloudWatch search to find.
            answer.warnings.append(
                f"The parts below are complete, but writing a summary of them failed: {e}"
            )
            return None

    @staticmethod
    def _seeds_from(part: Part) -> list[str]:
        return seeds_from(part.content)
