"""Creating and listing users, without anybody opening the Cognito console.

Cognito is the user store, so this is a thin wrapper over `cognito-idp` rather than a
second directory. Two things are added on top:

**A tenant binding.** `custom:tenant_id` is immutable and cannot be set after creation,
so it is written here at create time and also recorded in `TenantDirectory` — the access
token cannot carry it, so the table is what the API reads. Both are written in one call
so they cannot disagree.

**An ownership group.** Each admin gets a Cognito group named `owner-{sub}`, and every
user they create joins it. Listing "my users" is then `ListUsersInGroup`, which is a real
query rather than a scan over the pool.

A caveat, stated because it will eventually matter: the ownership group is keyed to the
admin's `sub`, so if that admin is deleted their users remain in a group nobody lists.
Tenant-scoped listing would not have that property. This is the chosen behaviour, not an
oversight — `list_tenant_users` exists as the escape hatch.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Protocol

logger = logging.getLogger(__name__)

#: Prefix for the per-admin ownership group. Cognito group names allow letters, digits and
#: `+=,.@-_`, which a raw uuid sub satisfies.
OWNER_GROUP_PREFIX = "owner-"

#: Cognito's own cap on a page of users.
MAX_PAGE = 60

_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class CognitoLike(Protocol):
    """The slice of the cognito-idp client this module uses."""

    def admin_create_user(self, **kwargs: Any) -> dict[str, Any]: ...
    def admin_add_user_to_group(self, **kwargs: Any) -> dict[str, Any]: ...
    def admin_get_user(self, **kwargs: Any) -> dict[str, Any]: ...
    def admin_delete_user(self, **kwargs: Any) -> dict[str, Any]: ...
    def list_users_in_group(self, **kwargs: Any) -> dict[str, Any]: ...
    def list_users(self, **kwargs: Any) -> dict[str, Any]: ...
    def create_group(self, **kwargs: Any) -> dict[str, Any]: ...


class UserAdminError(RuntimeError):
    """A user could not be created or listed. Message is safe to show an admin."""


@dataclass(frozen=True)
class DirectoryEntry:
    """A user as an admin sees them. Deliberately not the raw Cognito record."""

    user_id: str
    email: str
    status: str
    created_at: str
    enabled: bool
    tenant_id: str = ""

    @property
    def display_name(self) -> str:
        return self.email.split("@")[0].replace(".", " ").title() or self.email


def owner_group(admin_sub: str) -> str:
    return f"{OWNER_GROUP_PREFIX}{admin_sub}"


def _attr(user: dict[str, Any], name: str) -> str:
    for a in user.get("Attributes") or user.get("UserAttributes") or []:
        if a.get("Name") == name:
            return str(a.get("Value", ""))
    return ""


def _entry(user: dict[str, Any]) -> DirectoryEntry:
    created = user.get("UserCreateDate")
    return DirectoryEntry(
        # The `sub` attribute, not Username: with email aliasing Username may itself be a
        # uuid, and `sub` is what the token carries and what the tenant table keys on.
        user_id=_attr(user, "sub") or str(user.get("Username", "")),
        email=_attr(user, "email"),
        status=str(user.get("UserStatus", "")),
        created_at=created.isoformat() if hasattr(created, "isoformat") else str(created or ""),
        enabled=bool(user.get("Enabled", True)),
        tenant_id=_attr(user, "custom:tenant_id"),
    )


class UserAdmin:
    def __init__(
        self,
        user_pool_id: str,
        *,
        client: CognitoLike | None = None,
        region: str = "us-east-1",
    ) -> None:
        self.user_pool_id = user_pool_id
        self.region = region
        self._client = client

    @property
    def client(self) -> CognitoLike:
        if self._client is None:
            import boto3

            self._client = boto3.client("cognito-idp", region_name=self.region)
        return self._client

    def ensure_owner_group(self, admin_sub: str) -> str:
        """Create the admin's ownership group if it is not there yet.

        Idempotent: an existing group is the normal case and not an error.
        """
        name = owner_group(admin_sub)
        try:
            self.client.create_group(
                UserPoolId=self.user_pool_id,
                GroupName=name,
                Description=f"Users created by {admin_sub}",
            )
            logger.info("created ownership group %s", name)
        except Exception as e:
            if "GroupExistsException" not in type(e).__name__ and "already exists" not in str(e):
                raise UserAdminError(f"could not create ownership group: {e}") from e
        return name

    def create_user(
        self,
        *,
        email: str,
        tenant_id: str,
        admin_sub: str,
        is_admin: bool = False,
    ) -> DirectoryEntry:
        """Invite a user. Cognito emails them a temporary password.

        The temporary password is never generated here and never passes through this
        process, so it cannot end up in a log or an HTTP response — Cognito mints it and
        mails it, and the user is forced to change it at first sign-in.
        """
        if not _EMAIL.match(email or ""):
            raise UserAdminError(f"{email!r} is not a valid email address")
        if not tenant_id:
            raise UserAdminError("a user must be created inside a tenant")

        group = self.ensure_owner_group(admin_sub)

        try:
            created = self.client.admin_create_user(
                UserPoolId=self.user_pool_id,
                Username=email,
                UserAttributes=[
                    {"Name": "email", "Value": email},
                    # Trusted because an admin typed it; without this the user must verify
                    # before the invite email's password works.
                    {"Name": "email_verified", "Value": "true"},
                    # Immutable, so this is the only chance to set it.
                    {"Name": "custom:tenant_id", "Value": tenant_id},
                ],
                DesiredDeliveryMediums=["EMAIL"],
            )
        except Exception as e:
            if "UsernameExistsException" in type(e).__name__ or "already exists" in str(e):
                raise UserAdminError(f"{email} already has an account") from e
            raise UserAdminError(f"could not create {email}: {e}") from e

        entry = _entry(created.get("User", {}))

        # Ownership and role are separate calls, so a failure here leaves a real user who
        # is simply not yet listed. Reported rather than swallowed: the admin needs to know
        # the invite went out even though the grouping did not.
        try:
            self.client.admin_add_user_to_group(
                UserPoolId=self.user_pool_id, Username=email, GroupName=group
            )
            if is_admin:
                self.client.admin_add_user_to_group(
                    UserPoolId=self.user_pool_id, Username=email, GroupName="platform-admin"
                )
        except Exception as e:
            raise UserAdminError(
                f"{email} was invited but could not be added to a group: {e}"
            ) from e

        logger.info("invited %s into tenant %s (owner %s)", email, tenant_id, admin_sub)
        return entry

    def sync_from_cognito(self, tenant_id: str, directory: Any) -> dict[str, int]:
        """Reconcile the DynamoDB cache against Cognito, in both directions.

        Cognito is the source of truth for *existence*; DynamoDB is a cache keyed by tenant so
        a list does not mean paging the whole pool. Reconciling on read rather than on a
        schedule means a user deleted straight from the Cognito console disappears on the next
        admin page load, instead of lingering until a sweep runs.

        The paging caveat that makes this necessary: `ListUsers` returns up to 60 users of the
        *whole pool*, so filtering to one tenant afterwards can return fewer of a firm's users
        than exist once several tenants share the pool. The cache is what makes a tenant-scoped
        list correct rather than merely fast.
        """
        counts = {"added": 0, "removed": 0, "unchanged": 0}
        live = {e.user_id: e for e in self._all_pool_users() if e.tenant_id == tenant_id}
        cached = {e.sub for e in directory.users_for_tenant(tenant_id)}

        for sub, entry in live.items():
            if sub in cached:
                counts["unchanged"] += 1
                continue
            directory.put_user(sub, tenant_id, email=entry.email)
            counts["added"] += 1

        # Deleted in Cognito, so the cache row is a ghost: it would list a user who cannot
        # sign in and whose sub resolves to nothing.
        for sub in cached - set(live):
            directory.forget_user(sub)
            counts["removed"] += 1

        if counts["added"] or counts["removed"]:
            logger.info("synced %s users for %s: %s", len(live), tenant_id, counts)
        return counts

    def _all_pool_users(self) -> list[DirectoryEntry]:
        """Every user in the pool, paged. Used only by the sync, never on a request path."""
        entries: list[DirectoryEntry] = []
        token: str | None = None
        while True:
            kwargs: dict[str, Any] = {"UserPoolId": self.user_pool_id, "Limit": MAX_PAGE}
            if token:
                kwargs["PaginationToken"] = token
            got = self.client.list_users(**kwargs)
            entries.extend(_entry(u) for u in got.get("Users", []))
            token = got.get("PaginationToken")
            if not token:
                return entries

    def delete_user(self, email: str, *, tenant_id: str, directory: Any | None = None) -> None:
        """Remove a user from Cognito and from the cache.

        Cognito first. If that fails the user still exists and must stay listed, whereas
        dropping the cache row first would hide a live account from the only screen that can
        manage it.
        """
        try:
            found = self.client.admin_get_user(UserPoolId=self.user_pool_id, Username=email)
        except Exception as e:
            raise UserAdminError(f"no account for {email}: {e}") from e

        sub = _attr(found, "sub") or str(found.get("Username", ""))
        if _attr(found, "custom:tenant_id") not in ("", tenant_id):
            # Refused rather than reported as missing: an admin acting on their own tenant
            # should never be able to delete another firm's user by guessing an address.
            raise UserAdminError(f"no account for {email}")

        try:
            self.client.admin_delete_user(UserPoolId=self.user_pool_id, Username=email)
        except Exception as e:
            raise UserAdminError(f"could not delete {email}: {e}") from e

        if directory is not None and sub:
            directory.forget_user(sub)
        logger.info("deleted user %s from tenant %s", email, tenant_id)

    def list_my_users(self, admin_sub: str, *, limit: int = MAX_PAGE) -> list[DirectoryEntry]:
        """Users this admin created. Empty when they have created none."""
        try:
            got = self.client.list_users_in_group(
                UserPoolId=self.user_pool_id,
                GroupName=owner_group(admin_sub),
                Limit=min(limit, MAX_PAGE),
            )
        except Exception as e:
            if "ResourceNotFoundException" in type(e).__name__ or "not found" in str(e).lower():
                # No group yet means this admin has created nobody. Not an error.
                return []
            raise UserAdminError(f"could not list users: {e}") from e
        return sorted((_entry(u) for u in got.get("Users", [])), key=lambda u: u.email)

    def list_tenant_users(self, tenant_id: str, *, limit: int = MAX_PAGE) -> list[DirectoryEntry]:
        """Everyone in a tenant, whoever created them.

        The escape hatch for the ownership model: if the admin who invited someone is
        gone, their users are still reachable here. Filtered in this process because
        Cognito cannot filter on a custom attribute.
        """
        try:
            got = self.client.list_users(UserPoolId=self.user_pool_id, Limit=min(limit, MAX_PAGE))
        except Exception as e:
            raise UserAdminError(f"could not list users: {e}") from e
        entries = (_entry(u) for u in got.get("Users", []))
        return sorted((e for e in entries if e.tenant_id == tenant_id), key=lambda u: u.email)
