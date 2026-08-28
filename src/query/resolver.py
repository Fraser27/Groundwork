"""Tiered query resolution.

Three tiers, tried most-precise-first, each falling through on a miss:

1. **Governed metric** — YAML compiled to SQL. Deterministic, no model involved.
2. **Graph traversal** — openCypher over assertions that pass `scope.edge_scope`.
3. **Hybrid** — vector retrieval, expansion along verified edges, the catalogued schema of
   the tables involved, and a query a model writes over that schema.

The tier is part of the answer, not an implementation detail. "This came from an
approved metric" and "a model read this out of a document" are different claims about
trustworthiness, and the caller is entitled to know which one they are getting.

There was a fourth tier where a model wrote SQL. It was never implemented -- the generator was
`None` at its only construction site -- so the system documented a capability it did not have,
which is the wrong direction for a governance claim to be wrong in. SQL generation lives inside
tier 3 instead, where the catalogued schema it needs already is, and it is refusable on its own
without taking the passages and graph facts beside it down.

Whatever a tier returns is then screened by `src.query.blocks`, the same veto the planner
applies. Matter scoping already runs inside the reader, so this is defence in depth — but the
two endpoints must agree about screened data, or "why does the system believe this?" has two
answers depending on which one you asked.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import IntEnum
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

logger = logging.getLogger(__name__)


class Tier(IntEnum):
    GOVERNED_METRIC = 1
    GRAPH_TRAVERSAL = 2
    HYBRID = 3

    # 4 was LLM_SQL, and it is retired rather than reused. Renumbering would make the
    # append-only question log lie: rows already recorded `tier: 4`, and a 4 that came to mean
    # something else would misdescribe every answer given before the change. A gap in a sequence
    # is a smaller cost than a log that cannot be read literally.
    #
    # It was also never implemented -- the generator was `None` at its only construction site --
    # so the tier documented a capability the system did not have. SQL generation returns inside
    # tier 3, where the catalog it needs already lives.


TIER_EXPLANATION = {
    Tier.GOVERNED_METRIC: (
        "Answered from an approved metric definition. The SQL was compiled from that "
        "definition with no AI involved, so this question returns the same answer every time."
    ),
    Tier.GRAPH_TRAVERSAL: (
        "Answered by following verified relationships in the knowledge graph. Only facts "
        "above the confidence floor, and approved where approval was required, were used."
    ),
    Tier.HYBRID: (
        "Answered by finding relevant passages, following verified relationships out from "
        "them, and reading the catalogued schema of the tables involved. Sources are cited. "
        "Where the question needed a figure no approved metric covers, an AI may also have "
        "written a query over that schema; if it did, the query is shown and labelled."
    ),
}


class QueryBlocked(PermissionError):
    """Raised when a tenant's tier cap refuses a question outright."""


UNGOVERNED_BLOCKED = "ungoverned queries are disabled for this tenant"
"""Why the SQL lane did not run when `block_ungoverned_queries` is on.

A recorded skip and never an exception. The switch removes one lane of tier 3, not the tier: a
raise would take passages and graph facts down with it, so a control meant to remove an ungoverned
capability would remove governed answers instead."""

GENERATED_SQL_WARNING = (
    "Part of this answer comes from a query an AI wrote, not from an approved metric. It was "
    "checked against the tables it was allowed to read and required to be an aggregate, but the "
    "figures are only as right as the question it asked. Read the SQL before relying on them."
)
"""Carried in `warnings`, not only in `governed`. A reader looking at a number needs telling in
words; a boolean in the response body is not a disclosure."""


@dataclass
class Resolution:
    tier: Tier
    answer: Any
    sql: str | None = None
    citations: list[dict[str, Any]] = field(default_factory=list)
    assertions_used: list[str] = field(default_factory=list)
    tiers_attempted: list[Tier] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    blocks: list[Block] = field(default_factory=list)
    """What was withheld, named. Reported rather than dropped: an answer that looks clean only
    because the inconvenient part was invisible is the failure `scope.py` exists to prevent."""

    gate: dict[str, Any] | None = None
    """What the wall considered, cleared and withheld, and whether it ran whole.

    Counts and not only refusals: a step that is visible only when it blocks reads as an
    exception rather than as a gate everything passed through. None for tier 1, which is exempt.
    The UI has declared this type since before anything sent it."""

    router: Any | None = None
    """How the tiers were chosen, when a router chose them. None means they were tried in order."""

    generated_sql: Any | None = None
    """The `GeneratedSQL` when a model wrote part of the query, which is what makes an answer
    ungoverned. Set by tier 3's SQL lane; None everywhere else, including when the kill switch
    refused the lane -- a refused lane produced no model-written SQL, so the answer beside it is
    governed and says so."""

    metric_selection: dict[str, Any] | None = None
    """Tier 1 only: how the metric was chosen, and whether that choice was deterministic.

    Separate from `is_governed`, which asks who *wrote* the answer. A metric reached by similarity
    was still compiled from a definition a human approved, so nothing about the SQL is a model's --
    but the choice of which approved metric to compile was, and a reader gating on reproducibility
    needs that fact in the response rather than in a log line."""

    @property
    def explanation(self) -> str:
        if self.tier is Tier.GOVERNED_METRIC and not self.selected_deterministically:
            # The stock tier-1 wording promises the same answer every time, which is true of the
            # SQL and not of which metric was picked. One sentence per claim.
            return (
                "Answered from an approved metric definition. The SQL was compiled from that "
                "definition with no AI involved, but no approved metric shared a word with the "
                "question, so which metric to use was chosen by similarity -- a differently "
                "worded question could reach a different metric."
            )
        return TIER_EXPLANATION[self.tier]

    @property
    def is_governed(self) -> bool:
        """Whether a model wrote any part of the answer.

        Judged on what contributed rather than on which tier answered. It used to be
        `tier is not LLM_SQL`, which was equivalent while exactly one tier could involve a
        model -- but tier 3 will gain SQL generation, so a tier number stops being a proxy for
        governance and a per-answer check is the only one that stays true.

        A metric selected by similarity is still governed by this test, and deliberately: its SQL
        came from an approved definition with no model in the path. What a model did was *choose*
        between approved definitions, which is a different and weaker claim -- so it is reported in
        `metric_selection` rather than folded in here, where it would either overstate that choice
        as reproducible or understate the SQL as model-written.
        """
        return self.generated_sql is None

    @property
    def selected_deterministically(self) -> bool:
        """Whether the same question would reach the same tier-1 metric with no model involved."""
        if self.metric_selection is None:
            return True
        return bool(self.metric_selection.get("deterministic"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": int(self.tier),
            "tier_name": self.tier.name,
            "governed": self.is_governed,
            "deterministic_selection": self.selected_deterministically,
            "metric_selection": self.metric_selection,
            "explanation": self.explanation,
            "answer": self.answer,
            "sql": self.sql,
            "citations": self.citations,
            "assertions_used": self.assertions_used,
            "tiers_attempted": [int(t) for t in self.tiers_attempted],
            "warnings": self.warnings,
            "blocks": [b.to_dict() for b in self.blocks],
            "gate": self.gate,
            "router": self.router.to_dict() if self.router is not None else None,
        }


@dataclass
class BlockedQuery:
    """A refused ungoverned query.

    Recorded rather than discarded: a question people keep asking is a governed
    metric waiting to be written, so the refusal log is a backlog.
    """

    tenant_id: str
    user_id: str
    question: str
    reason: str
    at: str


def _evidence(answer: Any) -> list[Any]:
    """The id-bearing rows in an answer, whatever shape the tier gave it.

    Tier 2 answers with a list of hits and tier 3 with `{passages, related}`. Athena columns and
    rows carry no ids at all and so cannot be screened row-wise, which is why a tier that
    executes SQL needs its safety from somewhere other than this veto.
    """
    if isinstance(answer, list):
        return answer
    if isinstance(answer, dict):
        return [*(answer.get("passages") or []), *(answer.get("related") or [])]
    return []


def _apply(result: Resolution, screen: Screen) -> int:
    """Strip blocked evidence from a resolution in place. Returns how much went.

    The citation and assertion lists are filtered too. Removing a row while leaving its id
    behind would still hand the blocked subject to whatever reads the audit trail, which is
    exactly the leak the veto exists to prevent.
    """
    removed = 0
    answer = result.answer
    if isinstance(answer, list):
        kept = screen.keep(answer)
        removed += len(answer) - len(kept)
        result.answer = kept
    elif isinstance(answer, dict) and ("passages" in answer or "related" in answer):
        for key in ("passages", "related"):
            rows = answer.get(key)
            if isinstance(rows, list):
                kept = screen.keep(rows)
                removed += len(rows) - len(kept)
                answer[key] = kept

    surviving = _evidence(result.answer)
    kept_assertions = {row.get("assertion_id") for row in surviving if isinstance(row, dict)}
    kept_documents = {row.get("document_id") for row in surviving if isinstance(row, dict)}
    result.assertions_used = [a for a in result.assertions_used if a in kept_assertions]
    result.citations = [c for c in result.citations if c.get("document_id") in kept_documents]
    return removed


class Resolver:
    """Walks the tiers. Each `_try_*` returns None on a miss."""

    def __init__(
        self,
        *,
        metric_matcher: Any | None = None,
        graph_reader: Any | None = None,
        vector_search: Any | None = None,
        catalog: Any | None = None,
        sql_lane: Any | None = None,
        router: Any | None = None,
        synonyms_for: Callable[[AuthContext], Mapping[str, Sequence[str]]] | None = None,
    ) -> None:
        self._metrics = metric_matcher
        self._graph = graph_reader
        self._vectors = vector_search
        # Schemas for the SQL lane. Not a lane of its own here: `/query` returns one tier's answer,
        # so schema with no query over it would be an answer no lawyer asked for.
        self._catalog = catalog
        # The `SqlLane` from `sql_generation`, shared with `Planner`.
        self._sql = sql_lane
        # Injected so this layer never learns where approved synonyms live. Absent means table
        # selection is word overlap alone, which is what it was before synonyms were readable.
        self._synonyms_for = synonyms_for
        # Optional, and absent means the previous behaviour: try every permitted tier in order.
        self._router = router
        self.blocked: list[BlockedQuery] = []

    def resolve(
        self,
        ctx: AuthContext,
        question: str,
        settings: GovernanceSettings,
        *,
        tier_override: Tier | None = None,
        tiers_requested: Sequence[Tier] | None = None,
        execute: bool = True,
    ) -> Resolution:
        """Try tiers most-precise-first and return the first that answers.

        Three things narrow which tiers run, and they are not interchangeable:

        `settings.allowed_tiers` is the tenant's hard cap. A tier outside it never runs,
        and asking for one explicitly is refused rather than silently answered at another
        tier, because "answered the way you asked" and "answered some other way" must not
        look identical to the caller.

        `tiers_requested` is the caller's chosen subset, honoured only within the cap.

        `tier_override` pins exactly one tier. Kept for callers that want a single tier
        and for the existing API shape.
        """
        allowed = {int(t) for t in settings.allowed_tiers}

        decision = None
        if tier_override is not None:
            requested = [tier_override]
        elif tiers_requested:
            requested = sorted(set(tiers_requested))
        elif self._router is not None:
            # Only in this branch. An explicit tier is an instruction, and routing around one
            # would make testing a single tier impossible and "answered the way you asked" false.
            decision = self._router.route(ctx, question, settings)
            requested = [Tier(t) for t in decision.tiers]
        else:
            requested = list(Tier)

        refused = [t for t in requested if int(t) not in allowed]
        if refused and (tier_override is not None or tiers_requested):
            names = ", ".join(str(int(t)) for t in refused)
            raise QueryBlocked(
                f"tier {names} is not permitted for this tenant. Permitted tiers: "
                f"{', '.join(str(t) for t in sorted(allowed))}."
            )

        tiers = [t for t in requested if int(t) in allowed]
        if not tiers:
            raise QueryBlocked(
                "no resolution tier is permitted for this tenant, so no question can be "
                "answered. An administrator controls this in governance settings."
            )

        attempted: list[Tier] = []

        for tier in tiers:
            attempted.append(tier)
            result = self._attempt(ctx, question, settings, tier, execute=execute)
            if result is not None:
                result.tiers_attempted = attempted
                result.router = decision
                return self._screened(ctx, result, settings)

        # A question no tier could answer is a governed metric waiting to be written, so it is
        # recorded for an administrator to read. This used to happen only when the kill switch
        # refused it, which meant the backlog was empty for every tenant that had the switch off
        # -- the majority, and the ones most likely to need a new metric.
        self.blocked.append(
            BlockedQuery(
                tenant_id=ctx.tenant_id,
                user_id=ctx.user_id,
                question=question,
                reason="no tier could answer this question",
                at=datetime.now(UTC).isoformat(),
            )
        )
        logger.info("no tier answered for %s: %s", ctx.tenant_id, question)

        # Screened even though nothing matched. "Nothing found" while a wall is in force is the
        # exact shape of the harm: a conflict check reads as clean when it is only incomplete.
        return self._screened(
            ctx,
            Resolution(
                tier=Tier.GRAPH_TRAVERSAL,
                answer=None,
                tiers_attempted=attempted,
                router=decision,
                warnings=[
                    (
                        "No approved metric matched and nothing relevant was found in the "
                        "graph. Rephrasing may help, or this may need a new metric definition."
                    )
                ],
            ),
            settings,
        )

    def _screened(
        self, ctx: AuthContext, result: Resolution, settings: GovernanceSettings
    ) -> Resolution:
        """Apply the deterministic veto to whatever the tier returned.

        Same `blocks_for` the planner calls, so the two endpoints cannot disagree about
        screened data. It runs after the tier rather than inside it because a block is about
        the evidence, not the retrieval strategy.
        """
        # Tier 1 is exempt by decision, not oversight. A compiled metric is an Athena-side
        # aggregate: its rows carry no assertion, document or matter id, so there is nothing to
        # veto, and dropping rows from a total would misreport the figure rather than withhold
        # it. A metric that must exclude a matter says so in its own definition.
        if result.tier is Tier.GOVERNED_METRIC:
            return result

        screen = blocks_for(
            ctx,
            graph_reader=self._graph,
            seeds=seeds_from(_evidence(result.answer)),
        )
        removed = _apply(result, screen) if screen else 0
        result.blocks = screen.blocks
        # Recorded even for a clean wall. "Nothing refused" over zero seeds and over forty are
        # different facts, and the trace claimed no count of either was kept.
        result.gate = screen.trace(items_withheld=removed)
        if screen.degraded:
            result.warnings = [*result.warnings, DEGRADED_WARNING]
        if removed:
            result.warnings = [
                *result.warnings,
                (
                    f"{removed} matching {'item' if removed == 1 else 'items'} of evidence "
                    f"{'was' if removed == 1 else 'were'} withheld. What was withheld, and "
                    "why, is listed with the answer."
                ),
            ]
        if screen.advisories:
            result.warnings = [*result.warnings, advisory_warning(screen)]
        return result

    def _attempt(
        self,
        ctx: AuthContext,
        question: str,
        settings: GovernanceSettings,
        tier: Tier,
        *,
        execute: bool,
    ) -> Resolution | None:
        if tier is Tier.GOVERNED_METRIC:
            return self._try_metric(ctx, question, settings, execute=execute)
        if tier is Tier.GRAPH_TRAVERSAL:
            return self._try_graph(ctx, question, settings)
        return self._try_hybrid(ctx, question, settings)

    def _try_metric(
        self,
        ctx: AuthContext,
        question: str,
        settings: GovernanceSettings,
        *,
        execute: bool,
    ) -> Resolution | None:
        if self._metrics is None:
            return None
        # `ctx` and `settings` reach the matcher so it can fall back to routing candidates when no
        # metric word matched. `Planner._metric_part` passes the same two, because a question being
        # governed must not depend on which endpoint asked.
        match = match_metric(self._metrics, question, ctx, settings)
        if match is None:
            return None
        sql = match.compile()
        # `execute=False` returns the SQL for review without running it — the point
        # of a governed metric is that a human can read what it will do.
        answer = match.run(sql) if execute else None
        # A warning rather than only a field, because a caller reading the number needs to be
        # told the choice was a model's without going looking for it.
        warnings = [] if chosen_deterministically(match) else [getattr(match, "selection_note", "")]
        # `run` returns None only when no executor is wired, never for an empty result, so this is
        # "nothing to run it against". Unwarned it reads as "the metric ran and found nothing".
        if execute and answer is None:
            warnings.append(
                "The SQL compiled but no query engine is configured, so this is the query rather "
                "than the figure. Nothing was run."
            )
        return Resolution(
            tier=Tier.GOVERNED_METRIC,
            answer=answer,
            sql=sql,
            metric_selection=selection_of(match),
            warnings=warnings,
        )

    def _try_graph(
        self, ctx: AuthContext, question: str, settings: GovernanceSettings
    ) -> Resolution | None:
        if self._graph is None:
            return None
        hits = self._graph.search(ctx, question, min_confidence=settings.min_confidence_floor)
        if not hits:
            return None
        return Resolution(
            tier=Tier.GRAPH_TRAVERSAL,
            answer=hits,
            assertions_used=[h["assertion_id"] for h in hits if "assertion_id" in h],
        )

    def _try_hybrid(
        self, ctx: AuthContext, question: str, settings: GovernanceSettings
    ) -> Resolution | None:
        passages: list[dict[str, Any]] = []
        expanded: list[dict[str, Any]] = []
        if self._vectors is not None and self._graph is not None:
            passages = self._vectors.search(ctx, question, top_k=settings.vector_top_k)
            if passages:
                # Ids the passages already carry, both bare and prefixed, and no model between
                # retrieval and traversal -- see `passage_seeds`. Shared with
                # `Planner._graph_part` so the two endpoints walk from the same frontier.
                expanded = self._graph.expand(
                    ctx,
                    passage_seeds(passages),
                    depth=settings.graph_expand_depth,
                    min_confidence=settings.min_confidence_floor,
                )

        # Attempted even with no passages, and that is the point: "how much have we billed" is
        # answered by the warehouse and may match no document at all. Returning None here because
        # retrieval found nothing would make the SQL lane unreachable on `/query` for exactly the
        # questions it exists to answer, while compose ran it -- the endpoints would disagree.
        generated = self._generated_sql(ctx, question, settings)
        if not passages and generated is None:
            return None

        answer: dict[str, Any] = {"passages": passages, "related": expanded}
        if generated is not None:
            # Reported beside the passages rather than as the answer. The passages are quoted and
            # the facts are verified; this is neither, so merging them would make one label cover
            # two kinds of claim.
            answer["generated"] = {
                "sql": generated.generated.sql,
                "tables_offered": list(generated.generated.tables_offered),
                "rows": generated.rows,
                "error": generated.error,
                "error_code": generated.error_code,
            }
        return Resolution(
            tier=Tier.HYBRID,
            answer=answer,
            citations=[
                {
                    "document_id": p["document_id"],
                    "page": p.get("page"),
                    "char_start": p.get("char_start"),
                    "char_end": p.get("char_end"),
                }
                for p in passages
            ],
            assertions_used=[e["assertion_id"] for e in expanded if "assertion_id" in e],
            generated_sql=generated.generated if generated is not None else None,
            warnings=[GENERATED_SQL_WARNING] if generated is not None else [],
        )

    def _generated_sql(
        self, ctx: AuthContext, question: str, settings: GovernanceSettings
    ) -> Any | None:
        """Model-written SQL for the structured half, or None.

        The same `SqlLane` the planner runs, so `/query` and `/query/compose` cannot disagree about
        whether a question got model-written SQL -- which would make `governed` mean different
        things depending on which endpoint was asked.

        The kill switch is checked here and not in `_attempt`: it refuses this lane, so tier 3 still
        returns its passages and its walked facts. Refusing the tier would trade an ungoverned
        capability for the governed answers beside it.
        """
        if self._sql is None or self._catalog is None:
            return None
        if settings.block_ungoverned_queries:
            self.blocked.append(
                BlockedQuery(
                    tenant_id=ctx.tenant_id,
                    user_id=ctx.user_id,
                    question=question,
                    reason=UNGOVERNED_BLOCKED,
                    at=datetime.now(UTC).isoformat(),
                )
            )
            return None
        try:
            tables = self._catalog.tables(ctx.tenant_id)
        except Exception as e:  # noqa: BLE001
            logger.debug("SQL lane unavailable, no catalog: %s", e)
            return None
        if not tables:
            return None
        return self._sql.run(question, tables=tables, synonyms=self._synonyms(ctx))

    def _synonyms(self, ctx: AuthContext) -> Mapping[str, Sequence[str]] | None:
        """Approved synonyms per table `full_name`, or None.

        Degrades rather than raises: synonyms widen table selection, so a graph that cannot be
        reached must cost the widening and nothing else. Failing the question instead would let an
        unrelated outage turn an answer word overlap already found into an error.
        """
        if self._synonyms_for is None:
            return None
        try:
            return self._synonyms_for(ctx)
        except Exception as e:  # noqa: BLE001
            logger.warning("no approved synonyms for table selection: %s", e)
            return None
