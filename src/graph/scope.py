"""Tenant and matter scoping for every graph read.

The design rule this file exists to enforce: *it must be impossible to express an
unscoped query*. Not "discouraged", not "code-reviewed for" — impossible. One
forgotten `WHERE tenant_id = $t` in one code path is a cross-firm privilege
breach, and privilege breaches are not the kind of bug you patch and move on from.

So no module outside `src/graph/` builds Cypher by hand. Callers describe the shape
of the traversal they want; this module owns the scoping clause, and there is no
parameter for "skip scoping".

A correction worth recording. An earlier draft of this design claimed tenants were
isolated *structurally*, one named graph each. That was wrong: named graphs are an
RDF/SPARQL concept, and Neptune holds a single property graph per cluster, so
openCypher has no equivalent. Tenant and matter isolation are **both** property
filters, differing only in who is allowed to change them.

Which makes this module the only thing standing between two law firms' data — hence
the belt-and-braces posture: no Cypher outside `src/graph/`, reads go through
`GraphClient.read_scoped` which rejects any query lacking a `{scope}` token, and no
argument anywhere means "skip scoping".

If a customer ever demands physically separate storage, the escape hatch is a
cluster per tenant keyed off `AuthContext.cluster_key()`. Cheap to add later because
every read already carries tenant identity; near-impossible if these filters were
scattered across call sites.

Two levels, differing in how often they change:

- **Tenant** — fixed at authentication from the verified JWT. Never read from a
  request parameter, or a caller could widen their own scope.
- **Matter** — driven by the caller's Cedar grants. Ethical walls change weekly as
  staffing changes, so they need to be policy, not a migration.

The two levels also refuse differently, and an earlier draft of this file was wrong
about that. It argued for one vague refusal everywhere, on the grounds that
distinguishing "walled off" from "does not exist" confirms a matter exists. Inside a
single firm that is the wrong threat model: an ethical wall is a documented,
acknowledged screen, and its whole purpose is that people know it is there. Silent
filtering causes the harm the wall exists to prevent — a conflict check returns "no
conflicts" while three screened matters matched, and someone proceeds.

So a matter-level refusal **names the matter and gives a contact**; a tenant-level one
stays silent, because a boundary between two firms is confidentiality rather than a
screen, and nothing there was agreed with the person being refused.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from src.access import AccessDecision, MatterAccess, not_assigned_message, screen_message
from src.graph.assertions import (
    SIGNED_OFF_STATES,
    SUGGESTION_ONLY_CLASSES,
    EpistemicClass,
    ReviewState,
)

#: Tenant ids reach resource names (S3 prefixes, vector index names, and a cluster
#: identifier if we ever split storage), so they are validated to a strict shape
#: rather than escaped. Rejecting odd input beats sanitising it.
_TENANT_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,62}$")

#: Default trust floor for retrieval. Deliberately conservative: below this,
#: assertions are visible in the review queue but do not shape answers.
DEFAULT_MIN_CONFIDENCE = 0.8

#: Classes that may inform an answer by default. PREDICTED is absent by design.
DEFAULT_TRUSTED_CLASSES = frozenset(
    {
        EpistemicClass.DECLARED,
        EpistemicClass.EXTRACTED_DET,
        EpistemicClass.EXTRACTED_MODEL,
        EpistemicClass.INFERRED,
    }
)


class ScopeViolation(PermissionError):
    """Raised when a query would read outside the caller's authorisation.

    Carries the decision as a field, not only in the message: the HTTP and MCP layers
    have to tell a named in-tenant screen (403, disclosed) from a cross-tenant refusal
    (404, silent), and string-parsing an error message to make that choice is how the
    two eventually diverge.
    """

    def __init__(
        self,
        message: str,
        *,
        decision: AccessDecision | None = None,
        matter_id: str | None = None,
        contact: str | None = None,
        reason: str | None = None,
    ) -> None:
        super().__init__(message)
        self.decision = decision
        self.matter_id = matter_id
        self.contact = contact
        self.reason = reason

    @property
    def is_screen(self) -> bool:
        """True only for a documented in-tenant ethical wall, the disclosable case."""
        return self.decision is AccessDecision.SCREENED


@dataclass(frozen=True)
class AuthContext:
    """Who is asking, and what they are allowed to see.

    Built from the verified Cognito JWT plus the caller's grants. Never assembled
    from request parameters — that would let a caller widen their own scope.
    """

    user_id: str
    tenant_id: str

    matter_allowlist: frozenset[str] | None = None
    """None means every matter in the tenant. A set means exactly those. Empty set
    means nothing — which is a valid, if useless, grant."""

    matter_denylist: frozenset[str] = frozenset()
    """Ethical walls. Applied after the allowlist and always wins, mirroring
    Cedar's `forbid`-overrides-`permit` semantics: a wall must not be defeatable by
    also holding a broad role."""

    include_suggestions: bool = False
    """Opt-in to PREDICTED assertions. Only the research/suggestions surface sets
    this; ordinary retrieval must not present guesses as findings."""

    # `compare=False` on both: they are explanatory detail hanging off the denylist, not
    # part of the scope, and a dict field would otherwise make this frozen context
    # unhashable.
    screen_reasons: dict[str, str] = field(default_factory=dict, compare=False)
    """Why each denied matter is denied, so a refusal can say. Empty for a context built
    from bare grants, which is why the message degrades rather than fails."""

    screen_contacts: dict[str, str | None] = field(default_factory=dict, compare=False)
    """Who to ask, per matter. Shown to the screened user verbatim."""

    def __post_init__(self) -> None:
        if not _TENANT_ID_RE.match(self.tenant_id):
            # No naming here, and none in the tenant-mismatch path either: a firm must
            # not learn anything about another firm's identifiers.
            raise ScopeViolation(
                f"tenant_id {self.tenant_id!r} is not a valid identifier "
                "(lowercase alphanumeric and hyphens, 2-63 chars)"
            )

    @classmethod
    def from_access(
        cls,
        access: MatterAccess,
        *,
        user_id: str | None = None,
        include_suggestions: bool = False,
    ) -> AuthContext:
        """Build a context that can explain its refusals, from resolved access."""
        allowlist, denylist = access.to_scope()
        return cls(
            user_id=user_id or access.user_id,
            tenant_id=access.tenant_id,
            matter_allowlist=allowlist,
            matter_denylist=denylist,
            include_suggestions=include_suggestions,
            screen_reasons=dict(access.screen_reasons),
            screen_contacts=dict(access.screen_contacts),
        )

    def can_read_matter(self, matter_id: str) -> AccessDecision:
        """A decision rather than a bool, so callers can explain a refusal.

        `AccessDecision.__bool__` keeps every existing `if not can_read_matter(...)`
        working, which is why this could change type without a sweep of call sites.
        """
        if matter_id in self.matter_denylist:
            return AccessDecision.SCREENED
        if self.matter_allowlist is None or matter_id in self.matter_allowlist:
            return AccessDecision.ALLOWED
        return AccessDecision.NOT_ASSIGNED

    def assert_can_read_matter(self, matter_id: str) -> None:
        decision = self.can_read_matter(matter_id)
        if decision:
            return
        if decision is AccessDecision.SCREENED:
            reason = self.screen_reasons.get(matter_id)
            contact = self.screen_contacts.get(matter_id)
            raise ScopeViolation(
                screen_message(matter_id, reason, contact),
                decision=decision,
                matter_id=matter_id,
                contact=contact,
                reason=reason,
            )
        raise ScopeViolation(
            not_assigned_message(matter_id), decision=decision, matter_id=matter_id
        )

    def withheld_matters(self) -> list[dict[str, str | None]]:
        """The screens in force, named, for a caller that must report an incomplete view."""
        return [
            {
                "matter_id": mid,
                "reason": self.screen_reasons.get(mid),
                "contact": self.screen_contacts.get(mid),
            }
            for mid in sorted(self.matter_denylist)
        ]

    def cluster_key(self) -> str:
        """Logical storage key for this tenant.

        Today every tenant shares one Neptune cluster and this is used for resource
        naming (S3 prefixes, vector index names). It exists as a seam: if a customer
        contract ever requires physically separate storage, this becomes the cluster
        selector without touching a single call site.
        """
        return f"tenant-{self.tenant_id}"


@dataclass(frozen=True)
class ScopedQuery:
    """A Cypher fragment with its parameters, guaranteed scoped."""

    where: str
    params: dict[str, object]

    def and_(self, clause: str) -> ScopedQuery:
        """Add a caller predicate. Cannot widen scope, only narrow it."""
        return ScopedQuery(where=f"({self.where}) AND ({clause})", params=self.params)


@dataclass(frozen=True)
class TrustFilter:
    """"Is this fact trusted enough to inform an answer?" — asked once, answered two ways.

    The same three conditions were written twice: as Cypher in `edge_scope` and as Python
    `if` statements in `GraphReader._readable`, whose docstring promised it "mirrors
    edge_scope". Nothing enforced the mirror, and the halves had already drifted in effect —
    `edge_scope`'s confidence floor is dead for retrieval, because both its callers pass
    `min_confidence=0.0` deliberately, leaving `_readable` as the sole live gate. So the
    policy that decides what a lawyer sees lived in the copy nobody was pointing at.

    One definition, rendered twice: `matches()` for a Python collection, `clauses()` for a
    traversal. Adding a condition to one is now impossible without the other getting it.
    """

    classes: frozenset[EpistemicClass]
    min_confidence: float
    states: frozenset[ReviewState] | None
    """None admits every state, which is what the review queue needs — it exists to show
    exactly the claims retrieval will not use."""

    @classmethod
    def for_context(
        cls,
        ctx: AuthContext,
        *,
        min_confidence: float = DEFAULT_MIN_CONFIDENCE,
        trusted_classes: frozenset[EpistemicClass] | None = None,
        include_pending: bool = False,
    ) -> TrustFilter:
        classes = trusted_classes or DEFAULT_TRUSTED_CLASSES
        if ctx.include_suggestions:
            classes = classes | SUGGESTION_ONLY_CLASSES
        else:
            # Belt and braces: even a caller passing PREDICTED explicitly does not get it
            # without the flag, because `trusted_classes` is caller-supplied.
            classes = classes - SUGGESTION_ONLY_CLASSES
        return cls(
            classes=frozenset(classes),
            min_confidence=min_confidence,
            states=None if include_pending else frozenset(SIGNED_OFF_STATES),
        )

    def matches(self, assertion: Any) -> bool:
        """The Python rendering, for a store that is not a graph traversal."""
        if EpistemicClass(assertion.epistemic_class) not in self.classes:
            return False
        if assertion.confidence < self.min_confidence:
            return False
        return self.states is None or ReviewState(assertion.review_state) in self.states

    def clauses(self, edge_var: str) -> tuple[list[str], dict[str, object]]:
        """The Cypher rendering, for a filtered traversal."""
        clauses = [
            f"{edge_var}.epistemic_class IN $scope_classes",
            f"{edge_var}.confidence >= $scope_min_conf",
        ]
        params: dict[str, object] = {
            "scope_classes": [c.value for c in self.classes],
            "scope_min_conf": self.min_confidence,
        }
        if self.states is not None:
            clauses.append(f"{edge_var}.review_state IN $scope_states")
            params["scope_states"] = [s.value for s in self.states]
        return clauses, params


def edge_scope(
    ctx: AuthContext,
    *,
    edge_var: str = "r",
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
    trusted_classes: frozenset[EpistemicClass] | None = None,
    as_of: str | None = None,
    include_pending: bool = False,
) -> ScopedQuery:
    """Build the WHERE clause every traversal must carry.

    Composes five independent filters. Each one is a decision we made explicitly in
    design, so each is named here rather than buried in a query string:

    tenancy, matter walls, epistemic trust, confidence floor, and time.

    `as_of` is the bitemporal read: it reconstructs what the graph asserted on a
    given date, which is the question that actually matters when someone asks what
    a file showed at the time advice was given.

    The trust half of that — classes, floor, review state — is `TrustFilter`, shared with
    `GraphReader._readable` so the two cannot answer the same question differently.
    """
    trust = TrustFilter.for_context(
        ctx,
        min_confidence=min_confidence,
        trusted_classes=trusted_classes,
        include_pending=include_pending,
    )

    clauses = [f"{edge_var}.tenant_id = $scope_tenant"]
    params: dict[str, object] = {"scope_tenant": ctx.tenant_id}

    if ctx.matter_allowlist is not None:
        clauses.append(
            f"({edge_var}.matter_id IS NULL OR {edge_var}.matter_id IN $scope_matters)"
        )
        params["scope_matters"] = list(ctx.matter_allowlist)

    if ctx.matter_denylist:
        clauses.append(
            f"({edge_var}.matter_id IS NULL OR NOT {edge_var}.matter_id IN $scope_denied)"
        )
        params["scope_denied"] = list(ctx.matter_denylist)

    trust_clauses, trust_params = trust.clauses(edge_var)
    clauses.extend(trust_clauses)
    params.update(trust_params)

    if as_of is None:
        clauses.append(f"{edge_var}.superseded_at IS NULL")
    else:
        # Transaction time: what we believed then, including facts later retracted.
        clauses.append(
            f"{edge_var}.recorded_at <= $scope_as_of "
            f"AND ({edge_var}.superseded_at IS NULL OR {edge_var}.superseded_at > $scope_as_of)"
        )
        params["scope_as_of"] = as_of

    return ScopedQuery(where=" AND ".join(clauses), params=params)


def node_scope(ctx: AuthContext, *, node_var: str = "n") -> ScopedQuery:
    """Tenant/matter scoping for node lookups."""
    clauses = [f"{node_var}.tenant_id = $scope_tenant"]
    params: dict[str, object] = {"scope_tenant": ctx.tenant_id}

    if ctx.matter_allowlist is not None:
        clauses.append(
            f"({node_var}.matter_id IS NULL OR {node_var}.matter_id IN $scope_matters)"
        )
        params["scope_matters"] = list(ctx.matter_allowlist)

    if ctx.matter_denylist:
        clauses.append(
            f"({node_var}.matter_id IS NULL OR NOT {node_var}.matter_id IN $scope_denied)"
        )
        params["scope_denied"] = list(ctx.matter_denylist)

    return ScopedQuery(where=" AND ".join(clauses), params=params)
