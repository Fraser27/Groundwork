"""Governance settings — the Admin surface.

Every knob an administrator can turn without a redeploy. `GovernanceSettings.apply`
does the validating, so a refused change returns 422 with the message it raised —
those messages are written to be read by a lawyer-administrator, notably the
cap/floor one, so they are passed through verbatim rather than replaced.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any

from fastapi import APIRouter, Body, HTTPException, status

from src.api.deps import ServicesDep, TenantDep, require_admin
from src.governance import FIELD_HELP, GovernanceError

logger = logging.getLogger(__name__)

router = APIRouter(tags=["governance"])


@router.get("/tenants/{tenant}/governance")
async def get_governance(services: ServicesDep, principal: TenantDep) -> dict[str, Any]:
    """Current settings plus the help text the UI renders as tooltips.

    Help ships with the settings so the explanation cannot drift out of sync with
    the controls.
    """
    ctx, _ = principal
    settings = services.settings_for(ctx.tenant_id)
    return {"settings": settings.to_dict(), "help": FIELD_HELP}


@router.patch("/tenants/{tenant}/governance")
async def patch_governance(
    services: ServicesDep,
    principal: TenantDep,
    patch: Annotated[dict[str, Any], Body()],
) -> dict[str, Any]:
    require_admin(principal)
    ctx, _ = principal
    current = services.settings_for(ctx.tenant_id)

    try:
        updated = current.apply(patch, updated_by=ctx.user_id)
    except GovernanceError as e:
        # 422 with the raw message: it explains *why* the combination is unsafe,
        # which is the part an administrator needs.
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(e)) from e

    # Persisted, not just cached: this used to write only to a per-process dict, so an
    # administrator's change was lost on the next deploy. For the ungoverned-query kill switch
    # that means a firm believing questions are being refused while they are being answered.
    try:
        updated = services.save_settings(ctx.tenant_id, updated)
    except GovernanceError as e:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(e)) from e
    except Exception as e:
        logger.exception("could not persist governance settings for %s", ctx.tenant_id)
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            f"the change was not saved, so nothing has been altered: {e}",
        ) from e

    warnings: list[str] = []
    if patch.get("embedding_model") and patch["embedding_model"] != current.embedding_model:
        warnings.append(
            "Embedding model changed. Existing vectors came from the previous model; "
            "re-process every document or search quality degrades with no visible error."
        )
    if patch.get("enforce_closed_vocabulary") is False:
        warnings.append(
            "Closed vocabulary disabled. The same relationship can now be recorded "
            "several ways, which can make a conflict check return nothing and look clean."
        )
    if patch.get("auto_assert_deterministic") is False:
        warnings.append(
            "Pattern-matched facts now require review. The queue will grow considerably."
        )

    return {"settings": updated.to_dict(), "warnings": warnings}


@router.get("/tenants/{tenant}/governance/blocked")
async def blocked_queries(services: ServicesDep, principal: TenantDep) -> dict[str, Any]:
    """Queries refused by the kill switch, so an admin can see what is being asked.

    A refusal is a signal, not just a denial: a question people keep asking is a
    governed metric waiting to be written.
    """
    require_admin(principal)
    ctx, _ = principal
    settings = services.settings_for(ctx.tenant_id)
    blocked = list(reversed(services.blocked_queries.get(ctx.tenant_id, [])))
    return {
        "blocked": blocked,
        "count": len(blocked),
        "kill_switch_active": settings.block_ungoverned_queries,
        "note": (
            "Newest first, and held in this process rather than stored: a backlog to read, not "
            "an audit record. The audit record is the log line. Capped, so a firm asking many "
            "refused questions keeps the most recent."
        ),
    }
