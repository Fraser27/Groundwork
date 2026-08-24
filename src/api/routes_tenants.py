"""Platform routes: which tenants exist, and creating or deleting one.

**These are the only routes where the tenant is an argument.** Everywhere else it comes from
the caller's own token -- `TenantDep` refuses a path that disagrees with it -- so a firm's
admin has no parameter to tamper with. That cannot work here: the tenant being created does
not exist yet, and the one being deleted is not the caller's. So these hang off `/platform`
with `PrincipalDep` and are gated on `require_home_admin` instead, which is a stronger check
than `require_admin`: being a platform-admin is a role *within* a firm, and one firm's admin
must not be able to reach another's data.

That asymmetry is the isolation boundary. If a later change moves these under
`/tenants/{tenant}` for consistency, the guard goes with them or it is gone.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any

from fastapi import APIRouter, Body, HTTPException, status
from pydantic import BaseModel, Field

from src.api.deps import PrincipalDep, ServicesDep, require_home_admin
from src.tenant_lifecycle import (
    TenantExists,
    TenantLifecycleError,
    create_tenant,
    delete_tenant,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["platform"])


class CreateTenantRequest(BaseModel):
    tenant_id: str = Field(min_length=2, max_length=63)
    admin_email: str = Field(min_length=3, max_length=320)
    """Mandatory. A tenant with no users cannot be signed in to, so creating one without an
    admin produces a namespace nobody can reach.

    A plain string, matching `CreateUserRequest`: the address is validated by `user_admin`,
    which owns the rule Cognito will actually apply."""

    name: str = Field(default="", max_length=200)
    ontology_domain: str = Field(default="legal", min_length=1, max_length=63)


class DeleteTenantRequest(BaseModel):
    """Three confirmations, because three different things are unrecoverable.

    Modelled on `confirm_metric_loss` in the reset route. Typing the id is the one that stops
    a misclick on the wrong row of a list.
    """

    confirm_tenant_id: str = Field(min_length=1)
    confirm_document_loss: bool = False
    confirm_audit_loss: bool = False


@router.get("/platform/tenants")
async def list_tenants(services: ServicesDep, principal: PrincipalDep) -> dict[str, Any]:
    """Every tenant, deleted ones included, so a reused id is visible as reused."""
    require_home_admin(services, principal)
    registry = services.tenant_registry
    if registry is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "no tenant registry is configured")
    records = registry.list(include_deleted=True)
    return {
        "tenants": [r.to_dict() for r in records],
        "home_tenant": services.config.auth.home_tenant,
    }


@router.post("/platform/tenants", status_code=status.HTTP_201_CREATED)
async def create_tenant_route(
    services: ServicesDep,
    principal: PrincipalDep,
    body: Annotated[CreateTenantRequest, Body()],
) -> dict[str, Any]:
    """Create a tenant and invite its first admin, or neither.

    409 on a taken id or an email that already has an account: an address is the username for
    the whole user pool and a user's tenant is immutable, so an existing account cannot be
    moved into a new tenant.
    """
    actor = require_home_admin(services, principal)
    try:
        report = create_tenant(
            services,
            actor,
            tenant_id=body.tenant_id,
            admin_email=body.admin_email,
            name=body.name,
            ontology_domain=body.ontology_domain,
        )
    except TenantExists as e:
        raise HTTPException(status.HTTP_409_CONFLICT, str(e)) from e
    except TenantLifecycleError as e:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(e)) from e
    return report.to_dict()


@router.post("/platform/tenants/{tenant_id}/delete")
async def delete_tenant_route(
    services: ServicesDep,
    principal: PrincipalDep,
    tenant_id: str,
    body: Annotated[DeleteTenantRequest, Body()],
) -> dict[str, Any]:
    """Delete a tenant and everything belonging to it.

    POST rather than DELETE because it carries a body of confirmations, matching
    `/admin/reset`. 200 with a report even when a step failed: a partial delete is a fact the
    operator needs, and every step is idempotent so the answer is to run it again.
    """
    actor = require_home_admin(services, principal)

    if body.confirm_tenant_id != tenant_id:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"confirm_tenant_id does not match. Type {tenant_id!r} to confirm -- this deletes "
            "every document, fact and user belonging to it.",
        )
    if not body.confirm_document_loss:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "Deleting a tenant erases its documents from S3, every version of them. Nothing "
            "replays afterwards, because S3 was what a replay read from. Set "
            "confirm_document_loss to proceed.",
        )
    if not body.confirm_audit_loss:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "This also destroys the audit trail: who changed what the system believed, and "
            "which questions were answered on what basis. Set confirm_audit_loss to proceed.",
        )
    if tenant_id == services.config.auth.home_tenant:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "refusing to delete the operator tenant. Its admins are the only identities that "
            "can create or delete tenants, so this would lock the platform out of itself.",
        )

    try:
        report = delete_tenant(services, actor, tenant_id)
    except TenantLifecycleError as e:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(e)) from e
    return report.to_dict()
