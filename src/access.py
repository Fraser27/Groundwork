"""Who may see which matters.

Two halves, deliberately kept apart because they change at different rates and have
different failure modes:

- **Roles** come from Cognito groups, in the verified JWT. "Is this person a reviewer?"
- **Matter assignments** come from here, read per request. "Is Alice on the Halveston
  matter?"

Cognito is the wrong home for the second. It would need a group per matter, and group
claims only refresh when a token is reissued — so revoking someone's access would not
bite until they logged out. A screen has to take effect now.

**Allowlist-primary.** A user sees only matters they are assigned to; the default posture
is closed. A `MatterScreen` is a separate, stronger refusal layered on top: a screen
denies a matter even if an assignment exists, which is what an ethical wall must do to be
worth anything.

**Screens are loud.** Within one firm, a lawyer is told which matter they are screened
from and who to ask. The alternative — silently filtering — is how a wall causes the harm
it exists to prevent: a conflict check comes back clean because the matching matters were
invisible, and someone proceeds. Cross-*tenant* isolation stays silent, because that is a
confidentiality boundary between firms rather than a documented screen inside one.

Every change is appended, never overwritten. The record of who screened whom, when, and
why *is* the compliance artifact; a mutable grants table would destroy it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Protocol

logger = logging.getLogger(__name__)

#: Used when a screen names no contact. A wall with nowhere to appeal is not a documented
#: screen, so there is always a route even when whoever raised it left the field blank.
DEFAULT_SCREEN_CONTACT = "your firm's risk team"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def screen_message(matter_id: str, reason: str | None, contact: str | None) -> str:
    """The sentence a screened user reads. Lives here so `scope.py` cannot word it differently
    — one wall must not have two explanations."""
    parts = [f"You are screened from {matter_id}."]
    if reason:
        parts.append(f"Reason recorded: {reason}.")
    parts.append(f"Contact {contact or DEFAULT_SCREEN_CONTACT} to discuss.")
    return " ".join(parts)


def not_assigned_message(matter_id: str) -> str:
    return (
        f"You are not assigned to {matter_id}. Ask the matter owner to add you if you "
        "need access."
    )


class AccessDecision(str, Enum):
    """Why a matter is or is not readable.

    A bool would be enough to enforce access and not enough to explain it. The UI has to
    distinguish "you are screened from this" — which names a contact — from "you are not
    on this matter", which is an ordinary assignment gap.
    """

    ALLOWED = "ALLOWED"
    SCREENED = "SCREENED"
    """An ethical wall. Named to the user, with a route to the risk team."""

    NOT_ASSIGNED = "NOT_ASSIGNED"
    """No assignment exists. Not a wall, nobody decided anything about this user."""

    PLATFORM_ADMIN = "PLATFORM_ADMIN"
    """Allowed by role rather than assignment. Recorded distinctly so an audit can see
    that access came from a role, which is the thing a reviewer would question."""

    def __bool__(self) -> bool:
        return self in (AccessDecision.ALLOWED, AccessDecision.PLATFORM_ADMIN)


@dataclass(frozen=True)
class MatterAssignment:
    """A user is on a matter."""

    tenant_id: str
    user_id: str
    matter_id: str
    granted_by: str
    granted_at: str = field(default_factory=_now)
    role: str = "member"
    revoked_at: str | None = None
    revoked_by: str | None = None

    @property
    def is_active(self) -> bool:
        return self.revoked_at is None


@dataclass(frozen=True)
class MatterScreen:
    """An ethical wall: this user must not see this matter.

    Beats an assignment. A lawyer moving off a matter mid-dispute keeps their assignment
    row in history and gains a screen — which is the honest record of what happened, and
    the reason revocation is not simply a delete.
    """

    tenant_id: str
    user_id: str
    matter_id: str
    reason: str
    """Required. Six months on, "why is Bob screened from Halveston" is the only question
    anyone asks, and a blank reason makes the wall indefensible."""

    screened_by: str
    screened_at: str = field(default_factory=_now)
    contact: str | None = None
    """Who the screened user should ask. Shown to them verbatim."""

    lifted_at: str | None = None
    lifted_by: str | None = None

    @property
    def is_active(self) -> bool:
        return self.lifted_at is None


@dataclass(frozen=True)
class AccessEvent:
    """One append-only audit row."""

    tenant_id: str
    actor: str
    action: str
    subject_user: str
    matter_id: str
    at: str = field(default_factory=_now)
    reason: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)


class AccessStore(Protocol):
    """Persistence for assignments, screens and the audit trail.

    DynamoDB in deployment; `InMemoryAccessStore` for tests and local development.
    """

    def assignments_for(self, tenant_id: str, user_id: str) -> list[MatterAssignment]: ...
    def screens_for(self, tenant_id: str, user_id: str) -> list[MatterScreen]: ...
    def team_of(self, tenant_id: str, matter_id: str) -> list[MatterAssignment]: ...
    def screens_on(self, tenant_id: str, matter_id: str) -> list[MatterScreen]: ...
    """Who is screened off a matter. The team alone would let an administrator read a
    matter's roster as complete when a wall is standing in the middle of it."""


    def put_assignment(self, assignment: MatterAssignment) -> None: ...
    def put_screen(self, screen: MatterScreen) -> None: ...
    def append_event(self, event: AccessEvent) -> None: ...
    def events_for_matter(self, tenant_id: str, matter_id: str) -> list[AccessEvent]: ...
    def events_for_user(self, tenant_id: str, user_id: str) -> list[AccessEvent]: ...


@dataclass
class InMemoryAccessStore:
    """Reference implementation. Enforces the same tenant scoping as the real store."""

    _assignments: dict[tuple[str, str, str], MatterAssignment] = field(default_factory=dict)
    _screens: dict[tuple[str, str, str], MatterScreen] = field(default_factory=dict)
    _events: list[AccessEvent] = field(default_factory=list)

    def assignments_for(self, tenant_id: str, user_id: str) -> list[MatterAssignment]:
        return [
            a
            for (t, u, _), a in self._assignments.items()
            if t == tenant_id and u == user_id and a.is_active
        ]

    def screens_for(self, tenant_id: str, user_id: str) -> list[MatterScreen]:
        return [
            s
            for (t, u, _), s in self._screens.items()
            if t == tenant_id and u == user_id and s.is_active
        ]

    def team_of(self, tenant_id: str, matter_id: str) -> list[MatterAssignment]:
        return [
            a
            for (t, _, m), a in self._assignments.items()
            if t == tenant_id and m == matter_id and a.is_active
        ]

    def screens_on(self, tenant_id: str, matter_id: str) -> list[MatterScreen]:
        return [
            s
            for (t, _, m), s in self._screens.items()
            if t == tenant_id and m == matter_id and s.is_active
        ]

    def put_assignment(self, assignment: MatterAssignment) -> None:
        key = (assignment.tenant_id, assignment.user_id, assignment.matter_id)
        self._assignments[key] = assignment

    def put_screen(self, screen: MatterScreen) -> None:
        self._screens[(screen.tenant_id, screen.user_id, screen.matter_id)] = screen

    def append_event(self, event: AccessEvent) -> None:
        self._events.append(event)

    def events_for_matter(self, tenant_id: str, matter_id: str) -> list[AccessEvent]:
        return [e for e in self._events if e.tenant_id == tenant_id and e.matter_id == matter_id]

    def events_for_user(self, tenant_id: str, user_id: str) -> list[AccessEvent]:
        return [
            e
            for e in self._events
            if e.tenant_id == tenant_id and e.subject_user == user_id
        ]


@dataclass(frozen=True)
class MatterAccess:
    """A user's resolved matter access, plus enough context to explain a refusal."""

    tenant_id: str
    user_id: str
    assigned: frozenset[str]
    screened: frozenset[str]
    screen_reasons: dict[str, str] = field(default_factory=dict)
    screen_contacts: dict[str, str | None] = field(default_factory=dict)
    is_platform_admin: bool = False

    def decide(self, matter_id: str) -> AccessDecision:
        """A screen beats everything, including platform-admin.

        That ordering is the point of a wall. An admin who can read through a screen is
        not screened, and a regulator would say so.
        """
        if matter_id in self.screened:
            return AccessDecision.SCREENED
        if self.is_platform_admin:
            return AccessDecision.PLATFORM_ADMIN
        if matter_id in self.assigned:
            return AccessDecision.ALLOWED
        return AccessDecision.NOT_ASSIGNED

    def explain(self, matter_id: str) -> str:
        """A sentence for the user. Named, and routed where a screen applies."""
        decision = self.decide(matter_id)
        if decision is AccessDecision.SCREENED:
            return screen_message(
                matter_id,
                self.screen_reasons.get(matter_id),
                self.screen_contacts.get(matter_id),
            )
        if decision is AccessDecision.NOT_ASSIGNED:
            return not_assigned_message(matter_id)
        if decision is AccessDecision.PLATFORM_ADMIN:
            return "Visible because you hold platform-admin, not because of an assignment."
        return f"You are assigned to {matter_id}."

    def to_scope(self) -> tuple[frozenset[str] | None, frozenset[str]]:
        """Allowlist and denylist for `AuthContext`.

        A platform admin gets `None` — every matter in the tenant — because their role is
        the grant. Screens are still applied, which is why they are returned separately
        rather than subtracted here: `scope.py` must apply the denylist last.
        """
        allowlist = None if self.is_platform_admin else self.assigned
        return allowlist, self.screened


class AccessManager:
    """Resolves access, and records every change to it."""

    def __init__(self, store: AccessStore | None = None) -> None:
        self.store = store or InMemoryAccessStore()

    def resolve(
        self, tenant_id: str, user_id: str, *, is_platform_admin: bool = False
    ) -> MatterAccess:
        assignments = self.store.assignments_for(tenant_id, user_id)
        screens = self.store.screens_for(tenant_id, user_id)
        return MatterAccess(
            tenant_id=tenant_id,
            user_id=user_id,
            assigned=frozenset(a.matter_id for a in assignments),
            screened=frozenset(s.matter_id for s in screens),
            screen_reasons={s.matter_id: s.reason for s in screens},
            screen_contacts={s.matter_id: s.contact for s in screens},
            is_platform_admin=is_platform_admin,
        )

    def assign(
        self,
        tenant_id: str,
        user_id: str,
        matter_id: str,
        *,
        actor: str,
        role: str = "member",
    ) -> MatterAssignment:
        assignment = MatterAssignment(
            tenant_id=tenant_id,
            user_id=user_id,
            matter_id=matter_id,
            granted_by=actor,
            role=role,
        )
        self.store.put_assignment(assignment)
        self.store.append_event(
            AccessEvent(
                tenant_id=tenant_id,
                actor=actor,
                action="ASSIGN",
                subject_user=user_id,
                matter_id=matter_id,
                detail={"role": role},
            )
        )
        logger.info("%s assigned %s to %s", actor, user_id, matter_id)
        return assignment

    def unassign(
        self, tenant_id: str, user_id: str, matter_id: str, *, actor: str, reason: str = ""
    ) -> None:
        """Mark an assignment revoked. Never deletes the row.

        A deleted assignment erases the fact that access once existed, which is exactly
        what an audit needs to see.
        """
        for a in self.store.assignments_for(tenant_id, user_id):
            if a.matter_id == matter_id:
                self.store.put_assignment(
                    MatterAssignment(
                        tenant_id=a.tenant_id,
                        user_id=a.user_id,
                        matter_id=a.matter_id,
                        granted_by=a.granted_by,
                        granted_at=a.granted_at,
                        role=a.role,
                        revoked_at=_now(),
                        revoked_by=actor,
                    )
                )
        self.store.append_event(
            AccessEvent(
                tenant_id=tenant_id,
                actor=actor,
                action="UNASSIGN",
                subject_user=user_id,
                matter_id=matter_id,
                reason=reason or None,
            )
        )

    def screen(
        self,
        tenant_id: str,
        user_id: str,
        matter_id: str,
        *,
        actor: str,
        reason: str,
        contact: str | None = None,
    ) -> MatterScreen:
        """Raise an ethical wall. The reason is mandatory."""
        if not reason.strip():
            raise ValueError(
                "a screen requires a reason, an unexplained wall cannot be defended "
                "when someone asks why it exists"
            )
        screen = MatterScreen(
            tenant_id=tenant_id,
            user_id=user_id,
            matter_id=matter_id,
            reason=reason.strip(),
            screened_by=actor,
            contact=contact,
        )
        self.store.put_screen(screen)
        self.store.append_event(
            AccessEvent(
                tenant_id=tenant_id,
                actor=actor,
                action="SCREEN",
                subject_user=user_id,
                matter_id=matter_id,
                reason=reason.strip(),
                detail={"contact": contact} if contact else {},
            )
        )
        logger.info("%s screened %s from %s: %s", actor, user_id, matter_id, reason)
        return screen

    def lift_screen(
        self, tenant_id: str, user_id: str, matter_id: str, *, actor: str, reason: str
    ) -> None:
        if not reason.strip():
            raise ValueError("lifting a screen requires a reason")
        for s in self.store.screens_for(tenant_id, user_id):
            if s.matter_id == matter_id:
                self.store.put_screen(
                    MatterScreen(
                        tenant_id=s.tenant_id,
                        user_id=s.user_id,
                        matter_id=s.matter_id,
                        reason=s.reason,
                        screened_by=s.screened_by,
                        screened_at=s.screened_at,
                        contact=s.contact,
                        lifted_at=_now(),
                        lifted_by=actor,
                    )
                )
        self.store.append_event(
            AccessEvent(
                tenant_id=tenant_id,
                actor=actor,
                action="LIFT_SCREEN",
                subject_user=user_id,
                matter_id=matter_id,
                reason=reason.strip(),
            )
        )
