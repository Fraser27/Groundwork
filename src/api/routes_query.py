"""Query endpoint.

Reports which tier answered, because "an approved metric produced this" and "an LLM
wrote this SQL" are different claims and the caller is entitled to know which.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Body, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from src.api.deps import ServicesDep, TenantDep
from src.query.resolver import QueryBlocked, Tier

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
        # 403, not 400: the request was well-formed and deliberately refused.
        raise HTTPException(status.HTTP_403_FORBIDDEN, str(e)) from e

    return resolution.to_dict()
