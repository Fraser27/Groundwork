"""The user-to-tenant binding.

This sits on the authentication path of every request, and it decides which firm's data a
caller sees. The properties worth asserting are therefore about refusal and about not
trusting the client: an unknown subject must never be defaulted into a tenant, and the
binding must never be readable from anything the caller controls.

No AWS. The DynamoDB table is a fake.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.tenant_directory import (
    KEY_ATTR,
    StaticTenantDirectory,
    TenantDirectory,
    UnknownUser,
    user_key,
)

SUB = "84289448-90d1-70f2-1b62-82b68589ac67"
OTHER_SUB = "11111111-2222-3333-4444-555555555555"


class FakeTable:
    def __init__(self) -> None:
        self.items: dict[str, dict[str, Any]] = {}
        self.gets = 0

    def put_item(self, **kw: Any) -> dict[str, Any]:
        item = kw["Item"]
        self.items[item[KEY_ATTR]] = item
        return {}

    def get_item(self, **kw: Any) -> dict[str, Any]:
        self.gets += 1
        item = self.items.get(kw["Key"][KEY_ATTR])
        return {"Item": item} if item else {}

    def query(self, **kw: Any) -> dict[str, Any]:
        return {"Items": []}


@pytest.fixture
def table() -> FakeTable:
    return FakeTable()


@pytest.fixture
def directory(table: FakeTable) -> TenantDirectory:
    return TenantDirectory(table=table)


class TestResolution:
    def test_a_bound_user_resolves_to_their_tenant(self, directory):
        directory.put_user(SUB, "demo-firm", email="lawyer@firm.example")
        assert directory.tenant_for(SUB) == "demo-firm"

    def test_an_unknown_subject_is_refused_not_defaulted(self, directory):
        """The whole point. Defaulting would put a stranger in somebody's matter data."""
        with pytest.raises(UnknownUser):
            directory.tenant_for("nobody")

    def test_a_record_with_no_tenant_is_refused(self, directory, table):
        """A half-written record must not read as a valid binding."""
        table.put_item(Item={KEY_ATTR: user_key(SUB), "sub": SUB})
        with pytest.raises(UnknownUser):
            directory.tenant_for(SUB)

    def test_two_users_do_not_share_a_binding(self, directory):
        directory.put_user(SUB, "firm-a")
        directory.put_user(OTHER_SUB, "firm-b")
        assert directory.tenant_for(SUB) == "firm-a"
        assert directory.tenant_for(OTHER_SUB) == "firm-b"


class TestKeying:
    def test_keyed_on_sub_not_email(self, directory, table):
        """An email can be reassigned to a different person; a sub cannot. The email is
        stored for lookup convenience but is never the identity."""
        directory.put_user(SUB, "demo-firm", email="lawyer@firm.example")
        item = table.items[user_key(SUB)]
        assert item[KEY_ATTR] == f"USER#{SUB}"
        assert item["email"] == "lawyer@firm.example"

    def test_email_is_stored_lowercased(self, directory, table):
        """So an operator searching by a differently-cased address still matches."""
        directory.put_user(SUB, "demo-firm", email="Lawyer@Firm.Example")
        assert table.items[user_key(SUB)]["email"] == "lawyer@firm.example"

    def test_a_user_without_an_email_still_binds(self, directory, table):
        directory.put_user(SUB, "demo-firm")
        assert table.items[user_key(SUB)]["tenant"] == "demo-firm"


class TestCaching:
    def test_a_repeat_lookup_does_not_reread(self, directory, table):
        """This is on the auth path of every request; a read per request is the thing the
        cache exists to avoid."""
        directory.put_user(SUB, "demo-firm")
        directory.tenant_for(SUB)
        before = table.gets
        directory.tenant_for(SUB)
        assert table.gets == before

    def test_the_cache_expires(self, table):
        clock = {"now": 0.0}
        directory = TenantDirectory(table=table, ttl_seconds=60, clock=lambda: clock["now"])
        directory.put_user(SUB, "demo-firm")
        directory.tenant_for(SUB)
        first = table.gets

        clock["now"] = 61.0
        directory.tenant_for(SUB)
        assert table.gets > first

    def test_a_correction_takes_effect_without_waiting(self, directory, table):
        """`put_user` invalidates, so fixing a wrong binding is not a five-minute wait."""
        directory.put_user(SUB, "wrong-firm")
        assert directory.tenant_for(SUB) == "wrong-firm"
        directory.put_user(SUB, "right-firm")
        assert directory.tenant_for(SUB) == "right-firm"

    def test_an_unknown_subject_is_not_cached_as_a_negative(self, directory):
        """Otherwise binding a brand-new user would appear not to work for the TTL."""
        with pytest.raises(UnknownUser):
            directory.tenant_for(SUB)
        directory.put_user(SUB, "demo-firm")
        assert directory.tenant_for(SUB) == "demo-firm"


class TestStaticDirectory:
    def test_it_still_refuses_an_unknown_subject(self):
        """The dev implementation must not be more permissive than the real one, or tests
        pass against behaviour that does not ship."""
        with pytest.raises(UnknownUser):
            StaticTenantDirectory({SUB: "demo-firm"}).tenant_for("nobody")

    def test_it_resolves_a_known_subject(self):
        assert StaticTenantDirectory({SUB: "demo-firm"}).tenant_for(SUB) == "demo-firm"
