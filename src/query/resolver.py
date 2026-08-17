"""Tiered query resolution.

Three tiers, tried most-precise-first, each falling through on a miss:

1. **Governed metric** — YAML compiled to SQL. Deterministic, no model involved.
2. **Graph traversal** — openCypher over assertions that pass `scope.edge_scope`.
3. **Hybrid** — vector retrieval, expansion along verified edges, and the catalogued
   schema of the tables involved.

The tier is part of the answer, not an implementation detail. "This came from an
approved metric" and "a model read this out of a document" are different claims about
trustworthiness, and the caller is entitled to know which one they are getting.

There was a fourth tier where a model wrote SQL. It was never implemented -- the generator was
`None` at its only construction site -- so the system documented a capability it did not have,
which is the wrong direction for a governance claim to be wrong in. SQL generation returns
inside tier 3, where the catalogued schema it needs already lives.

Whatever a tier returns is then screened by `src.query.blocks`, the same veto the planner
applies. Matter scoping already runs inside the reader, so this is defence in depth — but the
two endpoints must agree about screened data, or "why does the system believe this?" has two
answers depending on which one you asked.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import IntEnum
from typing import Any

from src.governance import GovernanceSettings
from src.graph.scope import AuthContext
from src.query.blocks import Block, Screen, blocks_for, seeds_from

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
        "them, and reading the catalogued schema of the tables involved. Sources are cited."
    ),
}


class QueryBlocked(PermissionError):
    """Raised when the kill switch refuses an ungoverned query."""


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

    router: Any | None = None
    """How the tiers were chosen, when a router chose them. None means they were tried in order."""

    generated_sql: Any | None = None
    """Set when a model wrote the query, which is what makes an answer ungoverned. Nothing
    populates it yet -- SQL generation lands in tier 3 in a later stage -- so `is_governed` has a
    single definition from the start rather than being rewritten when the generator arrives."""

    @property
    def explanation(self) -> str:
        return TIER_EXPLANATION[self.tier]

    @property
    def is_governed(self) -> bool:
        """Whether a model wrote any part of the answer.

        Judged on what contributed rather than on which tier answered. It used to be
        `tier is not LLM_SQL`, which was equivalent while exactly one tier could involve a
        model -- but tier 3 will gain SQL generation, so a tier number stops being a proxy for
        governance and a per-answer check is the only one that stays true.
        """
        return self.generated_sql is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tier": int(self.tier),
            "tier_name": self.tier.name,
            "governed": self.is_governed,
            "explanation": self.explanation,
            "answer": self.answer,
            "sql": self.sql,
            "citations": self.citations,
            "assertions_used": self.assertions_used,
            "tiers_attempted": [int(t) for t in self.tiers_attempted],
            "warnings": self.warnings,
            "blocks": [b.to_dict() for b in self.blocks],
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
        sql_generator: Any | None = None,
        firewall: Any | None = None,
        router: Any | None = None,
    ) -> None:
        self._metrics = metric_matcher
        self._graph = graph_reader
        self._vectors = vector_search
        self._sql_gen = sql_generator
        self._firewall = firewall
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
            min_confidence=settings.min_confidence_floor,
        )
        if not screen:
            return result

        result.blocks = screen.blocks
        removed = _apply(result, screen)
        if removed:
            result.warnings = [
                *result.warnings,
                (
                    f"{removed} matching {'item' if removed == 1 else 'items'} of evidence "
                    f"{'was' if removed == 1 else 'were'} withheld. What was withheld, and "
                    "why, is listed with the answer."
                ),
            ]
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
            return self._try_metric(ctx, question, execute=execute)
        if tier is Tier.GRAPH_TRAVERSAL:
            return self._try_graph(ctx, question, settings)
        return self._try_hybrid(ctx, question, settings)

    def _try_metric(self, ctx: AuthContext, question: str, *, execute: bool) -> Resolution | None:
        if self._metrics is None:
            return None
        match = self._metrics.match(question)
        if match is None:
            return None
        sql = match.compile()
        # `execute=False` returns the SQL for review without running it — the point
        # of a governed metric is that a human can read what it will do.
        answer = match.run(sql) if execute else None
        return Resolution(tier=Tier.GOVERNED_METRIC, answer=answer, sql=sql)

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
        if self._vectors is None or self._graph is None:
            return None
        passages = self._vectors.search(ctx, question, top_k=settings.vector_top_k)
        if not passages:
            return None
        # Seeded with the graph's node id, not the bare document id. An assertion's subject is
        # `document:<id>` (`DocumentMeta.entity_id`), so seeding the raw id matched nothing on the
        # first frontier and every hybrid answer returned zero related facts -- a silent empty
        # rather than an error, so tier 3 looked like it had simply found nothing to connect.
        # Both forms are passed: a seed that no assertion carries costs one set lookup, and being
        # wrong in the other direction costs the whole graph half of the answer.
        seeds = [f"document:{p['document_id']}" for p in passages if p.get("document_id")]
        seeds += [p["document_id"] for p in passages if p.get("document_id")]
        expanded = self._graph.expand(
            ctx,
            seeds,
            depth=settings.graph_expand_depth,
            min_confidence=settings.min_confidence_floor,
        )
        return Resolution(
            tier=Tier.HYBRID,
            answer={"passages": passages, "related": expanded},
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
        )
