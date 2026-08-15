"""The grants API — assignments, ethical screens, and the trail they leave.

Every mutation is `platform-admin` only. Deciding who may see a matter is the most
consequential act in the product, and it is also the one an attacker most wants: a
self-service assignment endpoint is a privilege escalation with a REST interface.

Two shapes here are deliberate rather than incidental:

- **Nothing is deleted.** Unassigning revokes, lifting a screen lifts. Both write a new
  row and an event, so "who screened whom, when, and why" survives.
- **A screen's `reason` is validated by `AccessManager`, not by the request model.** A
  blank or whitespace-only reason has to fail with the domain's message — an
  administrator needs to be told why an unexplained wall is refused, not that a string
  failed a length check.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any

from fastapi import APIRouter, Body, HTTPException, Query, status
from pydantic import BaseModel, Field

from src.access import AccessEvent, MatterAssignment, MatterScreen
from src.api.deps import Services, ServicesDep, TenantDep, require_admin
from src.user_admin import UserAdmin, UserAdminError

logger = logging.getLogger(__name__)

router = APIRouter(tags=["access"])


def _require_user_admin(services: Services) -> UserAdmin:
    if services.user_admin is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "user administration needs a Cognito user pool (COGNITO_USER_POOL_ID unset)",
        )
    return services.user_admin


class AssignRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    matter_id: str = Field(min_length=1, max_length=256)
    role: str = "member"


class UnassignRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    matter_id: str = Field(min_length=1, max_length=256)
    reason: str = ""
    """Optional here, unlike a screen: coming off a matter is routine staffing."""


class ScreenRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    matter_id: str = Field(min_length=1, max_length=256)
    reason: str = ""
    """Required, but not enforced here — `AccessManager.screen` raises the message an
    administrator can act on, and a pydantic length error would replace it."""

    contact: str | None = Field(default=None, max_length=256)
    """Shown verbatim to the screened user. Without it they are sent to "your firm's
    risk team", which is a route but not a person."""


class LiftScreenRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=256)
    matter_id: str = Field(min_length=1, max_length=256)
    reason: str = ""


def _assignment_out(a: MatterAssignment) -> dict[str, Any]:
    return {
        "user_id": a.user_id,
        "matter_id": a.matter_id,
        "role": a.role,
        "granted_by": a.granted_by,
        "granted_at": a.granted_at,
        "revoked_at": a.revoked_at,
        "revoked_by": a.revoked_by,
    }


def _screen_out(s: MatterScreen) -> dict[str, Any]:
    return {
        "user_id": s.user_id,
        "matter_id": s.matter_id,
        "reason": s.reason,
        "contact": s.contact,
        "screened_by": s.screened_by,
        "screened_at": s.screened_at,
        "lifted_at": s.lifted_at,
        "lifted_by": s.lifted_by,
    }


def _event_out(e: AccessEvent) -> dict[str, Any]:
    return {
        "at": e.at,
        "actor": e.actor,
        "action": e.action,
        "subject_user": e.subject_user,
        "matter_id": e.matter_id,
        "reason": e.reason,
        "detail": e.detail,
    }


def _refuse(e: ValueError) -> HTTPException:
    """422 carrying the domain's own words. The message is the actionable part."""
    return HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(e))


@router.get("/tenants/{tenant}/access/users/{user_id}")
async def user_access(services: ServicesDep, principal: TenantDep, user_id: str) -> dict[str, Any]:
    """One user's assignments, screens, and the decision each matter resolves to.

    Admin-only like the mutations: a user's screens name the matters they are walled off
    from, which is exactly what a wall exists to keep from spreading around a firm.
    """
    require_admin(principal)
    ctx, _ = principal
    store = services.access.store

    assignments = store.assignments_for(ctx.tenant_id, user_id)
    screens = store.screens_for(ctx.tenant_id, user_id)
    # Resolved without platform-admin: this answers "what did the grants say", and
    # folding a role in would make an admin's own row read as access to everything.
    access = services.access.resolve(ctx.tenant_id, user_id)

    matters = sorted({a.matter_id for a in assignments} | {s.matter_id for s in screens})
    return {
        "user_id": user_id,
        "assignments": [_assignment_out(a) for a in assignments],
        "screens": [_screen_out(s) for s in screens],
        "decisions": [
            {
                "matter_id": mid,
                "decision": access.decide(mid).value,
                "explanation": access.explain(mid),
            }
            for mid in matters
        ],
    }


@router.get("/tenants/{tenant}/access/matters/{matter_id}")
async def matter_access(
    services: ServicesDep, principal: TenantDep, matter_id: str
) -> dict[str, Any]:
    """The team on a matter, plus who is screened off it.

    Both halves or neither: a roster that omits the screens reads as complete when a
    wall is standing in the middle of the matter.
    """
    require_admin(principal)
    ctx, _ = principal
    store = services.access.store
    return {
        "matter_id": matter_id,
        "team": [_assignment_out(a) for a in store.team_of(ctx.tenant_id, matter_id)],
        "screened": [_screen_out(s) for s in store.screens_on(ctx.tenant_id, matter_id)],
    }


@router.post("/tenants/{tenant}/access/assignments", status_code=status.HTTP_201_CREATED)
async def create_assignment(
    services: ServicesDep, principal: TenantDep, body: AssignRequest
) -> dict[str, Any]:
    require_admin(principal)
    ctx, _ = principal
    assignment = services.access.assign(
        ctx.tenant_id, body.user_id, body.matter_id, actor=ctx.user_id, role=body.role
    )
    return {"assignment": _assignment_out(assignment)}


@router.delete("/tenants/{tenant}/access/assignments")
async def revoke_assignment(
    services: ServicesDep, principal: TenantDep, body: UnassignRequest
) -> dict[str, Any]:
    """Revoke, never delete. The response says so, because "deleted" would be a lie."""
    require_admin(principal)
    ctx, _ = principal
    services.access.unassign(
        ctx.tenant_id, body.user_id, body.matter_id, actor=ctx.user_id, reason=body.reason
    )
    return {
        "user_id": body.user_id,
        "matter_id": body.matter_id,
        "revoked": True,
        "history_preserved": True,
    }


@router.post("/tenants/{tenant}/access/screens", status_code=status.HTTP_201_CREATED)
async def create_screen(
    services: ServicesDep, principal: TenantDep, body: ScreenRequest
) -> dict[str, Any]:
    """Raise an ethical wall. Beats an assignment, and beats platform-admin."""
    require_admin(principal)
    ctx, _ = principal
    try:
        screen = services.access.screen(
            ctx.tenant_id,
            body.user_id,
            body.matter_id,
            actor=ctx.user_id,
            reason=body.reason,
            contact=body.contact,
        )
    except ValueError as e:
        raise _refuse(e) from e
    return {"screen": _screen_out(screen)}


@router.delete("/tenants/{tenant}/access/screens")
async def lift_screen(
    services: ServicesDep, principal: TenantDep, body: LiftScreenRequest
) -> dict[str, Any]:
    """Lift a wall. A reason is required here too — taking a screen down is the change
    a regulator would ask about."""
    require_admin(principal)
    ctx, _ = principal
    try:
        services.access.lift_screen(
            ctx.tenant_id,
            body.user_id,
            body.matter_id,
            actor=ctx.user_id,
            reason=body.reason,
        )
    except ValueError as e:
        raise _refuse(e) from e
    return {
        "user_id": body.user_id,
        "matter_id": body.matter_id,
        "lifted": True,
        "history_preserved": True,
    }


@router.get("/tenants/{tenant}/access/audit")
async def audit(
    services: ServicesDep,
    principal: TenantDep,
    matter_id: Annotated[str | None, Query()] = None,
    user_id: Annotated[str | None, Query()] = None,
) -> dict[str, Any]:
    """The append-only trail, by matter or by user.

    One or the other is required. An unfiltered read would be a table scan on the store
    that backs an authorization path, and "show me every grant change in the firm" is
    not a question the audit surface needs to answer in one request.
    """
    require_admin(principal)
    ctx, _ = principal
    if bool(matter_id) == bool(user_id):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "pass exactly one of matter_id or user_id, the trail is read per matter or per person",
        )

    store = services.access.store
    events = (
        store.events_for_matter(ctx.tenant_id, matter_id)
        if matter_id
        else store.events_for_user(ctx.tenant_id, user_id or "")
    )
    return {
        "matter_id": matter_id,
        "user_id": user_id,
        "events": [_event_out(e) for e in sorted(events, key=lambda e: e.at)],
    }


class CreateUserRequest(BaseModel):
    email: str = Field(min_length=3, max_length=320)
    is_admin: bool = False
    """Whether to also add the new user to `platform-admin`. Off by default: an admin
    inviting a colleague should have to say so deliberately."""


@router.post("/tenants/{tenant}/users", status_code=status.HTTP_201_CREATED)
async def create_user(
    services: ServicesDep,
    principal: TenantDep,
    body: Annotated[CreateUserRequest, Body()],
) -> dict[str, Any]:
    """Invite a user into this tenant. Cognito emails them a temporary password.

    The tenant is taken from the caller's own context, never from the request: an admin
    can only create users inside their own firm, so there is no parameter to tamper with.

    The temporary password is minted and mailed by Cognito, so it never passes through
    this process and cannot land in a log or a response body.
    """
    require_admin(principal)
    ctx, _ = principal
    admin = _require_user_admin(services)

    try:
        entry = admin.create_user(
            email=body.email.strip().lower(),
            tenant_id=ctx.tenant_id,
            admin_sub=ctx.user_id,
            is_admin=body.is_admin,
        )
    except UserAdminError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e)) from e

    # The token cannot carry the tenant, so the API reads it from here. Written after
    # Cognito succeeds, because a binding for a user who does not exist is worse than a
    # user with no binding — the latter is a clear refusal, the former is a phantom.
    if services.tenant_directory is not None:
        services.tenant_directory.put_user(entry.user_id, ctx.tenant_id, email=entry.email)

    return {
        "user_id": entry.user_id,
        "email": entry.email,
        "display_name": entry.display_name,
        "status": entry.status,
        "tenant_id": ctx.tenant_id,
        "note": (
            "Cognito has emailed a temporary password. They must change it at first "
            "sign-in, and their tenant is fixed at creation and cannot be changed."
        ),
    }


@router.get("/tenants/{tenant}/users")
async def list_users(
    services: ServicesDep,
    principal: TenantDep,
    scope: Annotated[str, Query(pattern="^(mine|tenant)$")] = "mine",
) -> dict[str, Any]:
    """Users this admin created, or everyone in the tenant.

    `mine` is a `ListUsersInGroup` on the admin's ownership group, which is a real query.
    `tenant` exists because the ownership model has a gap worth naming: if the admin who
    invited someone leaves, their users are in a group nobody lists, and this is how they
    stay reachable.
    """
    require_admin(principal)
    ctx, _ = principal
    admin = _require_user_admin(services)

    # Reconcile the cache on read. Cognito owns whether a user exists, DynamoDB makes
    # "who is in this tenant" a query rather than a scan of the whole pool, and doing this
    # here means a user deleted straight from the Cognito console disappears on the next page
    # load instead of lingering until a scheduled sweep runs.
    synced: dict[str, int] = {}
    if scope == "tenant" and services.tenant_directory is not None:
        try:
            synced = admin.sync_from_cognito(ctx.tenant_id, services.tenant_directory)
        except UserAdminError as e:
            logger.warning("could not sync the user cache: %s", e)

    try:
        entries = (
            admin.list_my_users(ctx.user_id)
            if scope == "mine"
            else admin.list_tenant_users(ctx.tenant_id)
        )
    except UserAdminError as e:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(e)) from e

    return {
        "scope": scope,
        "synced": synced,
        "users": [
            {
                "user_id": e.user_id,
                "email": e.email,
                "display_name": e.display_name,
                "status": e.status,
                "enabled": e.enabled,
                "created_at": e.created_at,
            }
            for e in entries
        ],
    }


@router.delete("/tenants/{tenant}/users/{email}")
async def delete_user(services: ServicesDep, principal: TenantDep, email: str) -> dict[str, Any]:
    """Remove a user from Cognito and from the tenant cache.

    Cognito first, then the cache. If Cognito fails the account still exists and must stay
    listed, whereas dropping the cache row first would hide a live account from the only
    screen that can manage it.

    An address outside the caller's own tenant is refused the same way as one that does not
    exist: an admin must not be able to discover another firm's users by guessing.
    """
    require_admin(principal)
    ctx, _ = principal
    admin = _require_user_admin(services)

    try:
        admin.delete_user(email, tenant_id=ctx.tenant_id, directory=services.tenant_directory)
    except UserAdminError as e:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(e)) from e

    return {
        "email": email,
        "deleted": True,
        "note": (
            "Removed from Cognito and from the tenant directory. Any documents they uploaded "
            "remain, and the audit trail of what they did is unchanged."
        ),
    }
