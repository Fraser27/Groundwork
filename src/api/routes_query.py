"""Query endpoint.

Reports which tier answered, because "an approved metric produced this" and "an LLM
wrote this SQL" are different claims and the caller is entitled to know which.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Body, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from src.api.deps import ServicesDep, TenantDep, build_metric_matcher
from src.query.planner import Planner
from src.query.resolver import QueryBlocked, Tier
from src.query.vector_search import VectorSearch
from src.query_audit import event_for

router = APIRouter(tags=["query"])


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    execute: bool = True
    """False returns the SQL without running it — the reviewability that makes a
    governed metric governed."""
    tier_override: int | None = Field(default=None, ge=1, le=4)
    """Pin exactly one tier. Kept for callers that want a single tier."""

    tiers: list[int] | None = Field(default=None)
    """Which tiers may answer, as a subset. Honoured only within the tenant's
    `allowed_tiers` cap: asking for a forbidden tier is refused rather than answered a
    different way, so the caller can tell the two apart."""

    max_results: int | None = Field(default=None, ge=1, le=500)

    @field_validator("tiers")
    @classmethod
    def _valid_tiers(cls, v: list[int] | None) -> list[int] | None:
        if v is None:
            return v
        bad = [t for t in v if t not in (1, 2, 3, 4)]
        if bad:
            raise ValueError(f"tiers must be in 1-4, got {bad}")
        return sorted(set(v))


@router.post("/tenants/{tenant}/query")
async def run_query(
    services: ServicesDep,
    principal: TenantDep,
    body: Annotated[QueryRequest, Body()],
) -> dict[str, Any]:
    ctx, _ = principal
    settings = services.settings_for(ctx.tenant_id)

    # Wired to whatever is actually available — a missing collaborator disables its
    # tier rather than erroring, so the answer degrades instead of failing.
    resolver = services.build_resolver(ctx.tenant_id)

    try:
        resolution = resolver.resolve(
            ctx,
            body.query,
            settings,
            tier_override=Tier(body.tier_override) if body.tier_override else None,
            tiers_requested=[Tier(t) for t in body.tiers] if body.tiers else None,
            execute=body.execute,
        )
    except QueryBlocked as e:
        # Recorded before raising. The resolver keeps its own list, but a resolver is built per
        # request and discarded, so that list died with it and the Governance screen could only
        # ever show an empty backlog. A refusal is the signal the kill switch exists to produce.
        for entry in resolver.blocked:
            services.record_blocked(
                ctx.tenant_id,
                {
                    "question": entry.question,
                    "user_id": entry.user_id,
                    "reason": entry.reason,
                    "at": entry.at,
                },
            )
        # 403, not 400: the request was well-formed and deliberately refused.
        raise HTTPException(status.HTTP_403_FORBIDDEN, str(e)) from e

    # A read that leaves no trace cannot answer "what did we tell the client, and on what
    # basis?". Refusals are not recorded here: `record_blocked` above already has them, and a
    # refusal produced no answer, so it is a backlog signal rather than a basis for advice.
    recorded = services.record_question(
        event_for(ctx.tenant_id, ctx.user_id, body.query, resolution)
    )
    out = resolution.to_dict()
    if not recorded:
        out["warnings"] = [
            *out["warnings"],
            "This question was answered but not recorded in the audit trail.",
        ]
    return out


class ComposeRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    execute: bool = True
    synthesise: bool = True
    """False returns the evidence without asking a model to write over it, which is the
    reviewable form: every part is there with its own provenance."""


@router.post("/tenants/{tenant}/query/compose")
async def compose_query(
    services: ServicesDep,
    principal: TenantDep,
    body: Annotated[ComposeRequest, Body()],
) -> dict[str, Any]:
    """Answer from several sources at once, grounded on the graph.

    Different from `/query`, which returns the first tier that can answer. This runs the
    lanes a question needs and keeps their results apart: a compiled metric, a graph
    traversal, and quoted passages are not the same kind of claim, so merging them into one
    number and one confidence would be inventing a statistic.

    A governed metric that matches still short-circuits, because it is exact and fanning out
    would add nothing but latency.

    What the graph blocks is applied deterministically and before any model sees the
    evidence. The synthesis model writes prose over what survived; it never decides what
    survives.
    """
    ctx, _ = principal
    settings = services.settings_for(ctx.tenant_id)

    planner = Planner(
        metric_matcher=build_metric_matcher(services, ctx.tenant_id),
        graph_reader=services.graph_reader,
        vector_search=VectorSearch(services.embedder) if services.embedder else None,
        catalog=services.catalog,
        synthesiser=None,
    )
    answer = planner.plan(
        ctx,
        body.query,
        settings,
        execute=body.execute,
        allow_synthesis=body.synthesise,
    )
    return answer.to_dict()
