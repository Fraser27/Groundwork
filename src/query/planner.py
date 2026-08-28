"""Multi-lane query planning: retrieve widely, ground on the graph, compose honestly.

The resolver answers from the first tier that can. That is right for a question a governed
metric answers exactly, and wrong for a question whose answer needs both a number and a
qualifier. "What is our exposure on the Northwind matter" wants the figure from the
warehouse *and* the fact that a document says the engagement excludes tax advice.

The shape, in the user's own analogy: penicillin cures the infection, but the patient is
allergic. OpenSearch surfaces penicillin. The graph knows about the allergy.

    1. A governed metric matches  -> compile, run, return. Nothing else needed.
    2. Otherwise, one traversal   -> either direction, never both
    3. Apply blocks               -> DETERMINISTICALLY, never by a model
    4. Synthesise                 -> a model writes prose over what survived

**Step 2 has two directions and the tenant picks one**, in `allowed_tiers`. Vector first (tier 3)
retrieves passages by similarity, walks the graph out of them, and offers the catalogued schema:
the graph's job is grounding what the other stores returned. Graph first (tier 2) inverts it --
match the graph, then read only the documents a matched fact came from and query only the tables
its schema edges reached. Same three stores, opposite order, and the choice is what the platform
sells rather than a heuristic. `governance.validate` refuses both at once, because a lane that ran
under whichever tier the loop reached first would make provenance depend on iteration order.

A compound question is split before step 2 and merged after it, so each half reaches the store
that can answer it (`decompose.py`). Steps 3 and 4 run **once** over the merged parts: two screens
could disagree, and two summaries would leave the reader to reconcile them.

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

from src.governance import (
    GRAPH_FIRST_TIER,
    VECTOR_FIRST_TIER,
    GovernanceSettings,
    coerce_allowed_tiers,
    direction_of,
)
from src.graph.scope import AuthContext
from src.query.blocks import (
    DEGRADED_WARNING,
    Block,
    Screen,
    advisory_warning,
    blocks_for,
    seeds_from,
)
from src.query.graph_first import GraphFirstLane
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

    sub_question: str | None = None
    """Which half of a split question this answers. None when the question was asked whole.

    Carried per part rather than only on `ComposedAnswer.decomposition`, because two parts from the
    same lane are otherwise indistinguishable: a reader looking at two SQL results needs to know
    they answer different questions, not that the lane ran twice."""

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
            "sub_question": self.sub_question,
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

    retrieval_direction: str | None = None
    """`graph_first`, `vector_first` or `metrics_only`: which direction these lanes were run in.

    A skipped lane carries no part and therefore no tier, so without this the UI had to guess one
    from a hardcoded map -- and it guessed tier 3, labelling a graph-first tenant's skipped lanes
    with the direction they declined."""

    router: Any | None = None
    """Why these lanes and not others. None means no router was wired, which is a different
    statement from a router that ran and could not choose -- see `RouterDecision.degraded`."""

    gate: dict[str, Any] | None = None
    """What the block step considered, cleared and withheld, and whether both its sources ran.

    Reported even when nothing was refused. A wall visible only when it blocks cannot be
    distinguished from one that never ran, which is exactly how the rule-block half stayed
    missing."""

    decomposition: dict[str, Any] | None = None
    """The original question and the sub-questions it was split into, or None if it was asked whole.

    Disclosed rather than folded into `governance_label`. A model chose the boundaries, so a reader
    is entitled to see them and object -- but the label describes what produced the *answer*, and
    every part still comes from the same compiled metric, verified fact or quoted passage it would
    have come from unsplit. Downgrading a governed metric's figure because a model decided which
    half of the sentence to send it would report the wrong thing as ungoverned."""

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
            "retrieval_direction": self.retrieval_direction,
            "decomposition": self.decomposition,
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


def _skipped_traversal(allowed: set[int], decision: Any | None) -> str:
    """Why a retrieval lane did not run, naming the direction this tenant actually chose.

    Passages, catalog and SQL all run under whichever traversal is permitted, so naming a fixed
    tier would tell a graph-first tenant that "tier 3 is not permitted" about a lane tier 2 runs --
    true, and an answer to a question nobody asked.
    """
    for tier in (GRAPH_FIRST_TIER, VECTOR_FIRST_TIER):
        if tier in allowed:
            return _skipped(tier, allowed, decision)
    return "neither graph-first nor vector-first retrieval is permitted for this tenant"


#: Lanes that belong to whichever traversal direction is permitted, in reporting order. A verified
#: relationship comes first because it is the stronger claim and a reader should meet it before a
#: passage chosen by similarity.
_TRAVERSAL_LANES = (Lane.GRAPH, Lane.PASSAGES, Lane.CATALOG, Lane.SQL)


@dataclass
class _Lanes:
    """One question's lanes, before grounding and synthesis.

    Exists because a split question runs this whole set per part and is screened once: without it
    `plan` would either screen each sub-answer separately, which lets two screens disagree, or
    thread five accumulators through the loop.
    """

    parts: list[Part] = field(default_factory=list)
    run: list[Lane] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)
    seeds: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    metric_only: bool = False
    """A governed metric answered this outright, so no other lane was tried."""


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
        question_splitter: Any | None = None,
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
        # Absent means every question is asked whole, which is what compose did before splitting
        # existed. Not wired into `Resolver`: `/query` answers from one tier, and half a question
        # answered by one tier is not an answer.
        self._splitter = question_splitter
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
        decompose: bool = True,
    ) -> ComposedAnswer:
        answer = ComposedAnswer()
        # Coerced, not read raw. `validate()` already refuses both directions at once, but these
        # settings can also arrive from a stored row written before that rule existed, and one
        # definition of which direction wins beats this module inventing a second.
        allowed = set(coerce_allowed_tiers(settings.allowed_tiers))
        # From the coerced cap rather than from what ran, because the case that needs it is a run
        # where nothing ran: graph-first with no graph configured skips all four traversal lanes and
        # produces no part, leaving a reader nothing to infer the direction from.
        answer.retrieval_direction = direction_of(allowed)

        # Routed once on the whole question, even when it is split. The decision is advisory unless
        # `ROUTER_NARROWS_LANES`, so routing each half would buy a longer trace and a model call
        # per part while changing nothing about what runs.
        decision = self._route(ctx, question, settings)
        answer.router = decision
        runnable = self._runnable(allowed, decision)

        asked = self._split(question) if decompose else [question]
        if len(asked) > 1:
            answer.decomposition = {"question": question, "parts": asked}

        seeds: list[str] = []
        for sub in asked:
            lanes = self._lanes_for(
                ctx,
                sub,
                settings,
                execute=execute,
                allowed=allowed,
                runnable=runnable,
                decision=decision,
            )
            for part in lanes.parts:
                part.sub_question = sub if len(asked) > 1 else None
            answer.parts.extend(lanes.parts)
            answer.lanes_run.extend(la for la in lanes.run if la not in answer.lanes_run)
            # First reason wins: two halves skipping the same lane skip it for the same reason,
            # since every gate here is per-tenant rather than per-question.
            for lane_name, reason in lanes.skipped.items():
                answer.lanes_skipped.setdefault(lane_name, reason)
            seeds.extend(lanes.seeds)
            answer.warnings.extend(w for w in lanes.warnings if w not in answer.warnings)

            # A metric answers its question outright, and when there is only one question that is
            # the whole answer -- no screen, no synthesis, no other lane paid for. Split, it is one
            # part among several and the rest of the pipeline still has work to do.
            if lanes.metric_only and len(asked) == 1:
                return answer

        # A lane that ran for one half is not skipped, whatever the other half found. Reporting it
        # both ways would have the trace contradict the parts sitting next to it.
        for lane in answer.lanes_run:
            answer.lanes_skipped.pop(lane.value, None)

        # Grounding. Deterministic, once over every part, and before synthesis so a model never
        # sees evidence the graph refused.
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

    def _split(self, question: str) -> list[str]:
        """The questions this question contains, or just itself. Never raises, never empty.

        Falls back to the whole question on anything unexpected, including a splitter that raises
        despite documenting that it does not. Splitting is an improvement to how a question is
        searched, and an improvement may not cost the answer.
        """
        if self._splitter is None:
            return [question]
        try:
            parts = [p for p in self._splitter.split(question) if p and p.strip()]
        except Exception as e:  # noqa: BLE001
            logger.warning("question not split, asking it whole: %s", e)
            return [question]
        return parts if len(parts) > 1 else [question]

    def _lanes_for(
        self,
        ctx: AuthContext,
        question: str,
        settings: GovernanceSettings,
        *,
        execute: bool,
        allowed: set[int],
        runnable: set[int],
        decision: Any | None,
    ) -> _Lanes:
        """Every lane one question gets. No screening and no synthesis: those happen once, above."""
        lanes = _Lanes()

        # A governed metric that matches is the whole answer to this question. Fanning out anyway
        # would pay Athena plus Neptune plus OpenSearch latency to add nothing: the metric is exact
        # and the question named it.
        if 1 in runnable:
            part = self._metric_part(ctx, question, settings, execute=execute)
            if part is not None:
                lanes.parts.append(part)
                lanes.run.append(Lane.METRIC)
                lanes.metric_only = True
                return lanes
        else:
            lanes.skipped[Lane.METRIC.value] = _skipped(1, allowed, decision)

        # One direction or the other, never both, and `allowed` was coerced so it cannot name both.
        if GRAPH_FIRST_TIER in runnable:
            self._graph_first(ctx, question, settings, lanes)
        elif VECTOR_FIRST_TIER in runnable:
            self._vector_first(ctx, question, settings, lanes)
        else:
            reason = _skipped_traversal(allowed, decision)
            for lane in _TRAVERSAL_LANES:
                lanes.skipped[lane.value] = reason
        return lanes

    # ── Directions ───────────────────────────────────────────────────────────

    def _vector_first(
        self, ctx: AuthContext, question: str, settings: GovernanceSettings, lanes: _Lanes
    ) -> None:
        """Tier 3: retrieve, then ground on the graph, then offer the schema.

        Retrieval runs first so the graph lane can walk out from the passages the way
        `Resolver._try_hybrid` does. The graph's term search is **not** here: matching the graph
        lexically is the first step of the other direction, and running it under tier 3 would give a
        tenant who chose vector-first the graph-first behaviour they declined.

        The reported order still leads with the graph. A verified relationship is the stronger
        claim, whatever order the stores were called in.
        """
        passage_part = self._passage_part(ctx, question, settings)
        passages = passage_part.content if passage_part is not None else None

        # Tier 3's own work, and stamped as such. `TIER_EXPLANATION[HYBRID]` promises "following
        # verified relationships out from them", and on `demo-firm` the walk finds 30 citable facts
        # where the term search finds 8 -- so a lane that skipped it reported no `assertion_ids` at
        # all and hybrid was not hybrid.
        walked = self._walked(ctx, passages, settings)
        graph_part = self._facts_part(walked, tier=Tier.HYBRID)
        if graph_part is not None:
            lanes.parts.append(graph_part)
            lanes.run.append(Lane.GRAPH)
            lanes.seeds.extend(self._seeds_from(graph_part))
        else:
            lanes.skipped[Lane.GRAPH.value] = self._no_walk_reason(passages)

        if passage_part is not None:
            lanes.parts.append(passage_part)
            lanes.run.append(Lane.PASSAGES)
            lanes.seeds.extend(self._seeds_from(passage_part))

        catalog_part = self._catalog_part(ctx, question)
        if catalog_part is not None:
            lanes.parts.append(catalog_part)
            lanes.run.append(Lane.CATALOG)

        # Gated on the kill switch, which refuses this lane alone. Refusing the whole tier would
        # take passages and graph facts down with it, turning a switch that removes an ungoverned
        # capability into one that removes governed answers.
        if settings.block_ungoverned_queries:
            self._refuse_sql(ctx, question, lanes)
            return
        sql_part = self._sql_part(ctx, question)
        if sql_part is not None:
            lanes.parts.append(sql_part)
            lanes.run.append(Lane.SQL)

    def _no_walk_reason(self, passages: Any) -> str:
        """Why vector-first's graph lane contributed nothing.

        Three different facts, and a reader can act on only one of them. Vector-first reaches the
        graph through the passages it retrieved, so "nothing was retrieved" is a statement about
        the documents and not about the graph -- and the term search that would have found a fact
        anyway belongs to the other direction, which this tenant declined.
        """
        if self._graph is None:
            return "no graph is configured, so nothing could be walked out from"
        if not passages:
            return (
                "no passage was retrieved, and vector-first reaches the graph through the "
                "passages it finds"
            )
        return "no verified relationship near the retrieved passages cleared the trust floor"

    def _graph_first(
        self, ctx: AuthContext, question: str, settings: GovernanceSettings, lanes: _Lanes
    ) -> None:
        """Tier 2: match the graph, then query only what it landed on. See `graph_first.py`.

        The lane object is built here from collaborators this planner already holds, so the lane
        `/query/compose` runs cannot differ from the one `/query` runs.
        """
        if self._graph is None:
            for lane in _TRAVERSAL_LANES:
                lanes.skipped[lane.value] = (
                    "no graph is configured, and graph-first has nothing to traverse from"
                )
            return

        sql_allowed = not settings.block_ungoverned_queries
        if not sql_allowed:
            self._refuse_sql(ctx, question, lanes)

        result = GraphFirstLane(
            graph_reader=self._graph,
            vector_search=self._vectors,
            catalog=self._catalog,
            sql_lane=self._sql,
        ).run(ctx, question, settings, sql_allowed=sql_allowed, synonyms=self._synonyms(ctx))
        lanes.warnings.extend(result.notes)

        # `landed` becomes the skip reasons. Which lanes had nothing to search is the one thing a
        # reader cannot infer from a graph-first answer: an empty passage list means either that no
        # document was reached or that the documents held no matching passage, and those are
        # different facts about the same corpus.
        if result.facts:
            part = self._facts_part(result.facts, tier=Tier.GRAPH_TRAVERSAL)
            if part is not None:
                lanes.parts.append(part)
                lanes.run.append(Lane.GRAPH)
                lanes.seeds.extend(self._seeds_from(part))
        else:
            lanes.skipped[Lane.GRAPH.value] = (
                "no verified fact in the graph matches this question's words"
            )

        if "documents" not in result.landed:
            lanes.skipped[Lane.PASSAGES.value] = (
                "no document was searched: graph-first reads only documents a verified fact came "
                "from, and this question reached none"
            )
        elif result.passages:
            part = self._passages_part(result.passages, tier=Tier.GRAPH_TRAVERSAL)
            lanes.parts.append(part)
            lanes.run.append(Lane.PASSAGES)
            lanes.seeds.extend(self._seeds_from(part))

        payload = result.to_dict()
        if payload["tables"]:
            lanes.parts.append(self._catalog_part_of(payload["tables"], tier=Tier.GRAPH_TRAVERSAL))
            lanes.run.append(Lane.CATALOG)
        else:
            unreached = "this question's words do not reach a catalogued table through the graph"
            lanes.skipped[Lane.CATALOG.value] = unreached
            # `setdefault`: a refusal already recorded above is the more actionable reason, and an
            # administrator turning the lane off is not the same fact as the graph reaching no table.
            lanes.skipped.setdefault(Lane.SQL.value, unreached)

        generated = payload.get("generated")
        if generated is not None:
            lanes.parts.append(
                Part(
                    lane=Lane.SQL,
                    provenance=Provenance.MODEL_WRITTEN,
                    tier=Tier.GRAPH_TRAVERSAL,
                    content=generated["rows"],
                    sql=generated["sql"],
                    error=generated["error"],
                )
            )
            lanes.run.append(Lane.SQL)

    def _refuse_sql(self, ctx: AuthContext, question: str, lanes: _Lanes) -> None:
        """Record the kill switch refusing model-written SQL, for the Governance screen.

        One implementation for both directions. Two would let the two disagree about what was
        refused, and a refusal nobody logged is indistinguishable from a lane that found nothing.
        """
        lanes.skipped[Lane.SQL.value] = UNGOVERNED_BLOCKED
        self.blocked.append(
            BlockedQuery(
                tenant_id=ctx.tenant_id,
                user_id=ctx.user_id,
                question=question,
                reason=UNGOVERNED_BLOCKED,
                at=datetime.now(UTC).isoformat(),
            )
        )

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

    @staticmethod
    def _facts_part(hits: Sequence[dict[str, Any]], *, tier: Tier) -> Part | None:
        """Verified relationships, however they were reached.

        `tier` is the tier that authorised them and is the caller's to state, not this method's to
        guess: the same facts are tier 2's when a term search found them and tier 3's when they
        were walked out of a retrieved passage. Stamping one tier on both would report a tier the
        tenant has forbidden as having run.
        """
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
            tier=tier,
            content=list(hits),
            assertion_ids=[h["assertion_id"] for h in hits if "assertion_id" in h],
            confidence=min(confidences) if confidences and model_written else None,
        )

    def _walked(
        self,
        ctx: AuthContext,
        passages: Any,
        settings: GovernanceSettings,
        *,
        already: Sequence[dict[str, Any]] = (),
    ) -> list[dict[str, Any]]:
        """Verified relationships around the retrieved passages.

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
        return self._passages_part(passages, tier=Tier.HYBRID)

    @staticmethod
    def _passages_part(passages: Sequence[dict[str, Any]], *, tier: Tier) -> Part:
        """Quoted text, and the citations that let a reader go and check it."""
        return Part(
            lane=Lane.PASSAGES,
            # The text is quoted exactly. Similarity chose which passage, but nothing
            # rewrote it, so this is not the same kind of claim as an inference.
            provenance=Provenance.VERBATIM,
            tier=tier,
            content=list(passages),
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
        # Tier 3, matching the gate that let this lane run. It claimed tier 2 while nothing checked
        # it against tier 2, so the reported tier was unbacked by any permission check.
        return self._catalog_part_of(relevant, tier=Tier.HYBRID)

    @staticmethod
    def _catalog_part_of(rows: Sequence[dict[str, Any]], *, tier: Tier) -> Part:
        """Schema, already selected. The selection differs by direction, the reporting does not."""
        return Part(
            lane=Lane.CATALOG,
            provenance=Provenance.DETERMINISTIC,
            tier=tier,
            content=list(rows),
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
            sub_question=part.sub_question,
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
