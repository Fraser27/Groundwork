"""Tiered query resolution.

Four tiers, tried most-precise-first, each falling through on a miss:

1. **Governed metric** — YAML compiled to SQL. Deterministic, no model involved.
2. **Graph traversal** — openCypher over assertions that pass `scope.edge_scope`.
3. **Hybrid** — vector retrieval, then expansion along verified edges.
4. **LLM SQL** — ad-hoc, firewalled, and refusable by the kill switch.

The tier is part of the answer, not an implementation detail. "This came from an
approved metric" and "an LLM wrote this SQL" are different claims about
trustworthiness, and the caller is entitled to know which one they are getting.

Tier 4 is the only tier that can be switched off, because it is the only one where
the system is guessing at intent rather than executing something a human approved.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC
from enum import IntEnum
from typing import Any

from src.governance import GovernanceSettings
from src.graph.scope import AuthContext

logger = logging.getLogger(__name__)


class Tier(IntEnum):
    GOVERNED_METRIC = 1
    GRAPH_TRAVERSAL = 2
    HYBRID = 3
    LLM_SQL = 4


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
        "Answered by finding relevant passages and then following verified relationships "
        "out from them. Sources are cited."
    ),
    Tier.LLM_SQL: (
        "No approved metric matched, so an AI model wrote the SQL. It was checked against "
        "the query firewall before running, but it is not a governed answer, treat it as "
        "a starting point."
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

    @property
    def explanation(self) -> str:
        return TIER_EXPLANATION[self.tier]

    @property
    def is_governed(self) -> bool:
        return self.tier is not Tier.LLM_SQL

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
    ) -> None:
        self._metrics = metric_matcher
        self._graph = graph_reader
        self._vectors = vector_search
        self._sql_gen = sql_generator
        self._firewall = firewall
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

        if tier_override is not None:
            requested = [tier_override]
        elif tiers_requested:
            requested = sorted(set(tiers_requested))
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
                return result

        return Resolution(
            tier=Tier.GRAPH_TRAVERSAL,
            answer=None,
            tiers_attempted=attempted,
            warnings=[
                (
                    "No approved metric matched and nothing relevant was found in the graph. "
                    "Rephrasing may help, or this may need a new metric definition."
                )
            ],
        )

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
        if tier is Tier.HYBRID:
            return self._try_hybrid(ctx, question, settings)
        return self._try_llm_sql(ctx, question, settings, execute=execute)

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

    def _try_llm_sql(
        self,
        ctx: AuthContext,
        question: str,
        settings: GovernanceSettings,
        *,
        execute: bool,
    ) -> Resolution | None:
        if settings.block_ungoverned_queries:
            from datetime import datetime

            reason = "ungoverned queries are disabled for this tenant"
            self.blocked.append(
                BlockedQuery(
                    tenant_id=ctx.tenant_id,
                    user_id=ctx.user_id,
                    question=question,
                    reason=reason,
                    at=datetime.now(UTC).isoformat(),
                )
            )
            logger.info("blocked ungoverned query for %s: %s", ctx.tenant_id, question)
            raise QueryBlocked(
                "This question could not be answered from an approved metric or the "
                "knowledge graph, and ungoverned queries are switched off. An "
                "administrator can review blocked questions in Governance."
            )

        if self._sql_gen is None:
            return None

        sql = self._sql_gen.generate(question)
        if self._firewall is not None:
            self._firewall.validate(sql)

        answer = self._sql_gen.run(sql) if execute else None
        return Resolution(
            tier=Tier.LLM_SQL,
            answer=answer,
            sql=sql,
            warnings=[
                (
                    "This SQL was written by an AI model, not compiled from an approved "
                    "metric. Check it before relying on the result."
                )
            ],
        )
