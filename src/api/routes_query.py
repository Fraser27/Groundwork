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
    tier_override: int | None = Field(default=None, ge=1, le=3)
    """Pin exactly one tier. Kept for callers that want a single tier."""

    tiers: list[int] | None = Field(default=None)
    """Which tiers may answer, as a subset. Honoured only within the tenant's
    `allowed_tiers` cap: asking for a forbidden tier is refused rather than answered a
    different way, so the caller can tell the two apart."""

    max_results: int | None = Field(default=None, ge=1, le=500)

    min_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    """Raise the confidence floor for this question only.

    The Ask page has sent this since the trust-floor control was built. Nothing declared it, and
    Pydantic ignores unknown fields, so it was dropped in silence -- the page then reported "no
    fact cleared the trust floor of 0.85" using its own local number, naming a floor that had
    never been applied. Below the tenant's floor it is ignored rather than refused, because
    `with_raised_floor` treats a request as able to narrow and never to widen.
    """

    @field_validator("tiers")
    @classmethod
    def _valid_tiers(cls, v: list[int] | None) -> list[int] | None:
        if v is None:
            return v
        bad = [t for t in v if t not in (1, 2, 3)]
        if bad:
            # 422 rather than a 500 from `Tier(4)` deep inside the resolver. 4 was a tier once,
            # so a stale caller asking for it must be told, not crashed at.
            raise ValueError(f"tiers must be in 1-3, got {bad}")
        return sorted(set(v))


def _drain_blocked(services: Any, tenant_id: str, blocked: list[Any]) -> None:
    """Move a request's refusals onto `Services`, which outlives it.

    A resolver and a planner are both built per request and discarded, so their lists died with
    them and the Governance screen could only ever show an empty backlog. A refusal is the signal
    the kill switch exists to produce: a question people keep asking is a metric waiting to be
    written.
    """
    for entry in blocked:
        services.record_blocked(
            tenant_id,
            {
                "question": entry.question,
                "user_id": entry.user_id,
                "reason": entry.reason,
                "at": entry.at,
            },
        )


@router.post("/tenants/{tenant}/query")
async def run_query(
    services: ServicesDep,
    principal: TenantDep,
    body: Annotated[QueryRequest, Body()],
) -> dict[str, Any]:
    ctx, _ = principal
    settings = services.settings_for(ctx.tenant_id).with_raised_floor(body.min_confidence)

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
        _drain_blocked(services, ctx.tenant_id, resolver.blocked)
        # 403, not 400: the request was well-formed and deliberately refused.
        raise HTTPException(status.HTTP_403_FORBIDDEN, str(e)) from e

    # Also drained on the success path, and that is the whole of what the kill switch does: it
    # skips the SQL lane and records the refusal, and tier 3 still answers with its passages and
    # its graph facts. A refusal that raised would have taken those down with it.
    _drain_blocked(services, ctx.tenant_id, resolver.blocked)

    # A read that leaves no trace cannot answer "what did we tell the client, and on what
    # basis?". Refusals are not recorded here: `record_blocked` above already has them, and a
    # refusal produced no answer, so it is a backlog signal rather than a basis for advice.
    recorded = services.record_question(
        event_for(ctx.tenant_id, ctx.user_id, body.query, resolution)
    )
    out = resolution.to_dict()
    # The floor that was actually applied, so the page stops asserting its own. It rendered "no
    # fact cleared the trust floor of 0.85" from a local variable while the request field was
    # being dropped, which is the page claiming a control it never exercised.
    out["min_confidence"] = settings.min_confidence_floor
    if not recorded:
        out["warnings"] = [
            *out["warnings"],
            "This question was answered but not recorded in the audit trail.",
        ]
    return out


def _synthesiser_for(services: Any) -> Any | None:
    """The synthesis model, or None when the deployment has no Bedrock access.

    None rather than raising: without a model the parts and their citations are still the answer,
    and refusing the question outright would trade a complete result for no result.
    """
    model_id = getattr(getattr(services, "config", None), "models", None)
    model_id = getattr(model_id, "synthesis_model", "")
    if not model_id:
        return None

    from src.query.synthesis import Synthesiser

    return Synthesiser(model_id=model_id)


class ComposeRequest(BaseModel):
    query: str = Field(min_length=1, max_length=4000)
    execute: bool = True
    synthesise: bool = True
    """False returns the evidence without asking a model to write over it, which is the
    reviewable form: every part is there with its own provenance."""

    min_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    """Raise the confidence floor for this question only. Same one-way clamp as `/query`: the two
    endpoints must not disagree about how strict a question was."""


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
    settings = services.settings_for(ctx.tenant_id).with_raised_floor(body.min_confidence)

    planner = Planner(
        metric_matcher=build_metric_matcher(services, ctx.tenant_id),
        graph_reader=services.graph_reader,
        vector_search=VectorSearch(services.embedder) if services.embedder else None,
        catalog=services.catalog,
        # Built per request from the tenant's configured model, so changing the model is a settings
        # change rather than a release. This was `None` for the life of the route, so compose
        # always answered "no synthesis model is configured": the seam existed with nothing in it.
        synthesiser=_synthesiser_for(services) if body.synthesise else None,
        # Recorded, not obeyed: `ROUTER_NARROWS_LANES` is False, so every permitted lane still
        # runs. Compose had no router at all, which meant the one page whose purpose is showing
        # everything the system found could not say why it looked where it looked.
        router=services.build_tier_router(),
        # The same lane `/query` gets, from `Services`, so the two endpoints cannot disagree about
        # whether a question got model-written SQL.
        sql_lane=services.build_sql_lane(ctx.tenant_id),
    )
    answer = planner.plan(
        ctx,
        body.query,
        settings,
        execute=body.execute,
        allow_synthesis=body.synthesise,
    )
    _drain_blocked(services, ctx.tenant_id, planner.blocked)
    out = answer.to_dict()
    out["min_confidence"] = settings.min_confidence_floor
    return out
