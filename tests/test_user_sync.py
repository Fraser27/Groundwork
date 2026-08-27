"""Keeping the DynamoDB user cache honest against Cognito.

Cognito owns whether an account exists. DynamoDB is a cache keyed by tenant, and it exists
for a correctness reason rather than only speed: Cognito's `ListUsers` pages over the *whole
pool*, so filtering to one tenant afterwards can return fewer of a firm's users than exist
once several tenants share the pool.

Reconciliation runs on read. The property that matters is that a user deleted straight from
the Cognito console stops being listed, because a cache row for a deleted user names somebody
who cannot sign in and resolves a `sub` that authenticates to nothing.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.tenant_directory import StaticTenantDirectory, TenantDirectory, UserRecord
from src.user_admin import UserAdmin, UserAdminError

TENANT = "demo-firm"
OTHER = "other-firm"


def cognito_user(sub: str, email: str, tenant: str = TENANT, enabled: bool = True) -> dict:
    return {
        "Username": email,
        "UserStatus": "CONFIRMED",
        "Enabled": enabled,
        "Attributes": [
            {"Name": "sub", "Value": sub},
            {"Name": "email", "Value": email},
            {"Name": "custom:tenant_id", "Value": tenant},
        ],
    }


class FakeCognito:
    """Enough of cognito-idp to exercise the sync, including pagination."""

    def __init__(self, users: list[dict] | None = None, page_size: int = 60) -> None:
        self.users = users or []
        self.page_size = page_size
        self.deleted: list[str] = []
        self.list_calls = 0

    def list_users(self, **kw: Any) -> dict[str, Any]:
        self.list_calls += 1
        start = int(kw.get("PaginationToken") or 0)
        page = self.users[start : start + self.page_size]
        out: dict[str, Any] = {"Users": page}
        if start + self.page_size < len(self.users):
            out["PaginationToken"] = str(start + self.page_size)
        return out

    def admin_get_user(self, **kw: Any) -> dict[str, Any]:
        for u in self.users:
            if u["Username"] == kw["Username"]:
                return {"Username": u["Username"], "UserAttributes": u["Attributes"]}
        raise RuntimeError("UserNotFoundException")

    def admin_delete_user(self, **kw: Any) -> dict[str, Any]:
        self.deleted.append(kw["Username"])
        self.users = [u for u in self.users if u["Username"] != kw["Username"]]
        return {}

    def list_users_in_group(self, **kw: Any) -> dict[str, Any]:
        return {"Users": self.users}

    def create_group(self, **kw: Any) -> dict[str, Any]:
        return {}

    def admin_create_user(self, **kw: Any) -> dict[str, Any]:
        return {"User": {"Username": kw["Username"], "Attributes": []}}

    def admin_add_user_to_group(self, **kw: Any) -> dict[str, Any]:
        return {}


@pytest.fixture
def directory() -> StaticTenantDirectory:
    return StaticTenantDirectory()


class TestSyncAddsMissingUsers:
    def test_a_cognito_user_absent_from_the_cache_is_added(self, directory):
        cognito = FakeCognito([cognito_user("s1", "a@firm.example")])
        admin = UserAdmin("pool", client=cognito)

        counts = admin.sync_from_cognito(TENANT, directory)
        assert counts["added"] == 1
        assert directory.tenant_for("s1") == TENANT

    def test_an_already_cached_user_is_left_alone(self, directory):
        directory.put_user("s1", TENANT)
        cognito = FakeCognito([cognito_user("s1", "a@firm.example")])

        counts = UserAdmin("pool", client=cognito).sync_from_cognito(TENANT, directory)
        assert counts == {"added": 0, "removed": 0, "unchanged": 1}


class TestSyncRemovesGhosts:
    def test_a_user_deleted_in_cognito_leaves_the_cache(self, directory):
        """The direction that matters. A row for a deleted user lists somebody who cannot
        sign in, and resolves a sub that authenticates to nothing."""
        directory.put_user("gone", TENANT)
        cognito = FakeCognito([])

        counts = UserAdmin("pool", client=cognito).sync_from_cognito(TENANT, directory)
        assert counts["removed"] == 1
        assert directory.users_for_tenant(TENANT) == []

    def test_sync_is_idempotent(self, directory):
        cognito = FakeCognito([cognito_user("s1", "a@firm.example")])
        admin = UserAdmin("pool", client=cognito)

        admin.sync_from_cognito(TENANT, directory)
        second = admin.sync_from_cognito(TENANT, directory)
        assert second["added"] == 0
        assert second["removed"] == 0


class TestTenantIsolation:
    def test_another_tenants_users_are_not_cached_for_this_tenant(self, directory):
        cognito = FakeCognito(
            [
                cognito_user("mine", "a@firm.example", TENANT),
                cognito_user("theirs", "b@other.example", OTHER),
            ]
        )
        UserAdmin("pool", client=cognito).sync_from_cognito(TENANT, directory)

        assert [u.sub for u in directory.users_for_tenant(TENANT)] == ["mine"]

    def test_syncing_one_tenant_does_not_evict_another(self, directory):
        """The removal pass must be scoped, or syncing firm A empties firm B's cache."""
        directory.put_user("theirs", OTHER)
        cognito = FakeCognito([cognito_user("mine", "a@firm.example", TENANT)])

        UserAdmin("pool", client=cognito).sync_from_cognito(TENANT, directory)
        assert directory.tenant_for("theirs") == OTHER


class TestPagination:
    def test_every_page_is_read(self, directory):
        """The reason the cache exists. A single ListUsers page is capped at 60 users of the
        whole pool, so a firm with users beyond that page would be under-reported."""
        users = [cognito_user(f"s{i}", f"u{i}@firm.example") for i in range(150)]
        cognito = FakeCognito(users, page_size=60)

        counts = UserAdmin("pool", client=cognito).sync_from_cognito(TENANT, directory)
        assert counts["added"] == 150
        assert cognito.list_calls == 3

    def test_a_paging_failure_is_reported_as_a_user_admin_error(self, directory):
        """So the caller can swallow it. The user list route treats a sync failure as
        non-fatal, which it can only do if the failure arrives as a known type rather than
        whatever boto raised."""

        class Broken(FakeCognito):
            def list_users(self, **kw: Any) -> dict[str, Any]:
                raise RuntimeError("throttled")

        with pytest.raises(UserAdminError, match="could not page the user pool"):
            UserAdmin("pool", client=Broken()).sync_from_cognito(TENANT, directory)


class TestDeleteRemovesFromBoth:
    def test_delete_removes_the_cognito_account_and_the_cache_row(self, directory):
        directory.put_user("s1", TENANT)
        cognito = FakeCognito([cognito_user("s1", "a@firm.example")])

        UserAdmin("pool", client=cognito).delete_user(
            "a@firm.example", tenant_id=TENANT, directory=directory
        )
        assert cognito.deleted == ["a@firm.example"]
        assert directory.users_for_tenant(TENANT) == []

    def test_deleting_another_tenants_user_is_refused(self, directory):
        """Refused the same way as a missing address, so an admin cannot discover another
        firm's users by guessing."""
        cognito = FakeCognito([cognito_user("s1", "a@other.example", OTHER)])

        with pytest.raises(UserAdminError, match="no account for"):
            UserAdmin("pool", client=cognito).delete_user(
                "a@other.example", tenant_id=TENANT, directory=directory
            )
        assert cognito.deleted == []

    def test_deleting_a_missing_user_is_refused(self, directory):
        with pytest.raises(UserAdminError, match="no account for"):
            UserAdmin("pool", client=FakeCognito([])).delete_user(
                "nobody@firm.example", tenant_id=TENANT, directory=directory
            )

    def test_the_cache_row_survives_a_failed_cognito_delete(self, directory):
        """Cognito first, deliberately. Dropping the cache row on a failed delete would hide
        a live account from the only screen that can manage it."""

        class Refusing(FakeCognito):
            def admin_delete_user(self, **kw: Any) -> dict[str, Any]:
                raise RuntimeError("throttled")

        directory.put_user("s1", TENANT)
        cognito = Refusing([cognito_user("s1", "a@firm.example")])

        with pytest.raises(UserAdminError):
            UserAdmin("pool", client=cognito).delete_user(
                "a@firm.example", tenant_id=TENANT, directory=directory
            )
        assert directory.tenant_for("s1") == TENANT


class TestDirectoryReads:
    def test_users_for_tenant_queries_the_index(self):
        """A query, not a scan. A scan on an admin page grows with every tenant in the pool."""
        queried: dict[str, Any] = {}

        class FakeTable:
            def query(self, **kw: Any) -> dict[str, Any]:
                queried.update(kw)
                return {"Items": [{"sub": "s1", "tenant": TENANT, "email": "a@firm.example"}]}

            def put_item(self, **kw: Any) -> dict[str, Any]:
                return {}

            def get_item(self, **kw: Any) -> dict[str, Any]:
                return {}

            def delete_item(self, **kw: Any) -> dict[str, Any]:
                return {}

        users = TenantDirectory(table=FakeTable()).users_for_tenant(TENANT)
        assert users == [UserRecord(sub="s1", tenant_id=TENANT, email="a@firm.example")]
        assert queried["IndexName"] == "TenantIndex"

    def test_forget_user_deletes_the_row_and_the_cached_entry(self):
        deleted: list[Any] = []

        class FakeTable:
            def get_item(self, **kw: Any) -> dict[str, Any]:
                return {"Item": {"sub": "s1", "tenant": TENANT}}

            def put_item(self, **kw: Any) -> dict[str, Any]:
                return {}

            def query(self, **kw: Any) -> dict[str, Any]:
                return {"Items": []}

            def delete_item(self, **kw: Any) -> dict[str, Any]:
                deleted.append(kw["Key"])
                return {}

        d = TenantDirectory(table=FakeTable())
        assert d.tenant_for("s1") == TENANT
        d.forget_user("s1")
        assert deleted == [{"tenant_id": "USER#s1"}]


class TestTheListRouteReconciles:
    """The sync is wired into the user list, and the wiring is where it can go wrong.

    Two mistakes are easy here and neither shows up in a unit test of the sync itself: gating
    reconciliation on `scope=tenant` when the admin screen requests `mine`, so the common path
    never reconciles; and letting a sync failure fail the request, so a cache problem takes out
    a user list that Cognito could have answered on its own.
    """

    @pytest.fixture
    def client(self):
        from fastapi.testclient import TestClient

        from src.api.app import create_app
        from src.api.deps import get_services
        from src.config import AuthConfig, GraphConfig, GroundworkConfig

        cfg = GroundworkConfig(
            environment="local",
            auth=AuthConfig(dev_bypass_tenant=TENANT),
            graph=GraphConfig(uri="bolt://127.0.0.1:1", user="none", password="none"),
        )
        cfg.validate()
        c = TestClient(create_app(cfg))
        services = get_services()
        services.tenant_directory = StaticTenantDirectory()
        return c, services

    def test_the_default_scope_still_reconciles(self, client):
        c, services = client
        cognito = FakeCognito([cognito_user("s1", "a@firm.example")])
        services.user_admin = UserAdmin("pool", client=cognito)

        body = c.get(f"/api/tenants/{TENANT}/users").json()
        assert body["scope"] == "mine"
        assert body["synced"]["added"] == 1
        assert services.tenant_directory.tenant_for("s1") == TENANT

    def test_a_sync_failure_does_not_fail_the_list(self, client):
        """Cognito is the source of truth and the route reads it anyway, so a cache error must
        degrade to a stale cache rather than an error page."""
        c, services = client

        class Broken(FakeCognito):
            def list_users(self, **kw: Any) -> dict[str, Any]:
                raise RuntimeError("throttled")

        # `list_users_in_group` still works, so Cognito can answer the list even though the
        # sync cannot run. That asymmetry is the whole point of the test.
        services.user_admin = UserAdmin("pool", client=Broken())

        r = c.get(f"/api/tenants/{TENANT}/users")
        assert r.status_code == 200
        assert r.json()["synced"] == {}
