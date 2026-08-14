"""Cognito JWT verification, and the only place an `AuthContext` is built.

The single rule that matters: **`tenant_id` comes from the verified token and from
nowhere else.** Not a header, not a query parameter, not the URL path. A path
parameter is checked *against* the token and a mismatch is a 403 — it exists for
routing and readable logs, not for authorization.

The dev bypass is the obvious hole, so it is closed twice: `LexGraphConfig.validate`
refuses to start if it is set outside local, and `_dev_context` re-checks the
environment at request time. One of those would be enough; both is cheap.

Roles come from the token; matter access comes from `AccessManager` on every request.
The split is the reason a screen bites immediately rather than at next login.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Any

import httpx
import jwt
from jwt import PyJWKClient

from src.access import AccessManager
from src.config import LexGraphConfig
from src.graph.scope import AuthContext, ScopeViolation
from src.tenant_directory import StaticTenantDirectory, TenantDirectory, UnknownUser

logger = logging.getLogger(__name__)


class AuthError(PermissionError):
    """Authentication or authorization failure. Maps to 401/403 at the boundary."""


@dataclass
class Grants:
    """What a user may see, resolved from their groups and matter assignments.

    Deliberately separate from the token: Cognito groups say what role someone
    holds, but matter walls change weekly as staffing changes, so they live in
    DynamoDB and are read per-request. A wall must take effect without waiting for
    a token to expire.
    """

    tenant_id: str
    roles: frozenset[str] = frozenset()
    matter_allowlist: frozenset[str] | None = None
    matter_denylist: frozenset[str] = frozenset()

    @property
    def is_platform_admin(self) -> bool:
        return "platform-admin" in self.roles

    @property
    def can_review(self) -> bool:
        """Approving an LLM's claim is a professional judgement, so it is not open
        to every authenticated user."""
        return bool(self.roles & {"platform-admin", "matter-owner", "reviewer"})


class TokenVerifier:
    """Verifies Cognito access tokens against the pool's JWKS.

    Signature, issuer, expiry and `token_use` are all checked. Skipping any one of
    them turns this into base64 decoding wearing a hat.
    """

    def __init__(self, config: LexGraphConfig) -> None:
        self._cfg = config
        self._jwks: PyJWKClient | None = None

    def _client(self) -> PyJWKClient:
        if self._jwks is None:
            issuer = self._cfg.auth.issuer_url.rstrip("/")
            self._jwks = PyJWKClient(f"{issuer}/.well-known/jwks.json")
        return self._jwks

    def verify(self, token: str) -> dict[str, Any]:
        try:
            key = self._client().get_signing_key_from_jwt(token).key
            claims = jwt.decode(
                token,
                key,
                algorithms=["RS256"],
                issuer=self._cfg.auth.issuer_url.rstrip("/"),
                options={"require": ["exp", "iss", "sub"]},
            )
        except jwt.ExpiredSignatureError as e:
            raise AuthError("token expired") from e
        except (jwt.InvalidTokenError, httpx.HTTPError) as e:
            raise AuthError(f"token verification failed: {e}") from e

        # An id token has a different audience and different claims; accepting one
        # here would accept a token minted for a different purpose.
        if claims.get("token_use") not in (None, "access"):
            raise AuthError("expected an access token")

        return claims


def _tenant_from_claims(claims: dict[str, Any]) -> str | None:
    """The tenant claim, if this token happens to carry one.

    Returns None rather than raising, because for an **access** token the answer is
    normally None and that is not an error: Cognito puts custom attributes only on the id
    token, and this API deliberately accepts access tokens only. The binding therefore
    comes from `TenantDirectory`; this is checked first purely so a deployment that does
    put the claim on the token (a pre-token-generation trigger, or a non-Cognito issuer)
    keeps working without a table read.
    """
    tenant = claims.get("custom:tenant_id") or claims.get("tenant_id")
    return str(tenant) if tenant else None


def _roles_from_claims(claims: dict[str, Any]) -> frozenset[str]:
    groups = claims.get("cognito:groups") or claims.get("groups") or []
    if isinstance(groups, str):
        groups = [g.strip() for g in groups.split(",") if g.strip()]
    return frozenset(str(g) for g in groups)


DEV_USER = "dev@localhost"


def _dev_grants(config: LexGraphConfig) -> Grants:
    """Unauthenticated local development.

    Re-checks the environment even though `config.validate()` already did, because
    this is the function that actually hands out access.
    """
    if config.environment != "local":
        raise AuthError("dev bypass is only available in local development")

    return Grants(
        tenant_id=config.auth.dev_bypass_tenant,
        roles=frozenset({"platform-admin", "reviewer"}),
    )


class Authenticator:
    """Turns a bearer token into an `AuthContext` plus `Grants`."""

    def __init__(
        self,
        config: LexGraphConfig,
        access: AccessManager | None = None,
        tenants: TenantDirectory | StaticTenantDirectory | None = None,
    ) -> None:
        self._cfg = config
        self._verifier = TokenVerifier(config) if config.auth.issuer_url else None
        self.access = access or AccessManager()
        self.tenants = tenants

    @property
    def dev_mode(self) -> bool:
        return bool(self._cfg.auth.dev_bypass_tenant) and self._cfg.environment == "local"

    def authenticate(
        self, bearer: str | None, *, include_suggestions: bool = False
    ) -> tuple[AuthContext, Grants]:
        if not bearer:
            if self.dev_mode:
                return self._resolve(
                    DEV_USER, _dev_grants(self._cfg), include_suggestions=include_suggestions
                )
            raise AuthError("missing bearer token")

        if self._verifier is None:
            raise AuthError("no token issuer configured")

        claims = self._verifier.verify(bearer)
        user_id = str(claims.get("sub") or claims.get("username") or "unknown")

        # Roles come from the token; the tenant comes from the directory. Cognito signs
        # `cognito:groups`, so a role is a verified fact, but it cannot put a custom
        # attribute on an access token — see `tenant_directory`.
        tenant_id = _tenant_from_claims(claims) or self._tenant_from_directory(user_id)
        grants = Grants(tenant_id=tenant_id, roles=_roles_from_claims(claims))
        return self._resolve(user_id, grants, include_suggestions=include_suggestions)

    def _tenant_from_directory(self, user_id: str) -> str:
        if self.tenants is None:
            raise AuthError(
                "no tenant directory configured — an access token carries no tenant_id "
                "claim, so there is nothing to scope this request by"
            )
        try:
            return self.tenants.tenant_for(user_id)
        except UnknownUser as e:
            # Not defaulted. A user with no binding must not land in another firm's data;
            # the fix is an operator creating the record.
            raise AuthError("user is not bound to a tenant") from e

    def _resolve(
        self, user_id: str, grants: Grants, *, include_suggestions: bool
    ) -> tuple[AuthContext, Grants]:
        """Read matter access now, and let `MatterAccess` decide the scope.

        The allowlist/denylist split matters here: a platform admin's allowlist is None,
        so a screen can only bite if the denylist is applied separately and last. That is
        what `to_scope()` returns and what `from_access` wires in.
        """
        access = self.access.resolve(
            grants.tenant_id, user_id, is_platform_admin=grants.is_platform_admin
        )
        allowlist, denylist = access.to_scope()
        grants = replace(grants, matter_allowlist=allowlist, matter_denylist=denylist)
        try:
            ctx = AuthContext.from_access(
                access, user_id=user_id, include_suggestions=include_suggestions
            )
        except ScopeViolation as e:
            # A malformed tenant_id in a signed token means the pool is issuing
            # tokens we cannot scope safely.
            raise AuthError(f"token tenant_id is not usable: {e}") from e
        return ctx, grants

    def assert_tenant_matches(self, ctx: AuthContext, path_tenant: str) -> None:
        """The path tenant is for routing; the token decides.

        Vague on purpose: confirming that some other tenant exists is itself a leak.
        """
        if path_tenant != ctx.tenant_id:
            logger.warning(
                "tenant mismatch: token=%s path=%s user=%s",
                ctx.tenant_id,
                path_tenant,
                ctx.user_id,
            )
            raise AuthError("not found")


def bearer_from_header(header: str | None) -> str | None:
    if not header:
        return None
    parts = header.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1].strip() or None
