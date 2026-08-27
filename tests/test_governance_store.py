"""Governance settings that survive a deploy, and a refusal log an admin can read.

The kill switch is why this matters more than it looks. Settings lived in a per-process dict, so
an administrator could switch off ungoverned queries, watch the UI confirm it, and lose it on
the next deploy — leaving a firm believing questions are being refused while they are quietly
being answered again. A cosmetic bug for a colour preference; not for that switch.

The refusal log had the mirror-image problem: refusals were recorded on the `Resolver`, which is
built per request and discarded, so the Governance screen could only ever show an empty list. A
refusal is the signal the switch exists to produce.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.deps import MAX_BLOCKED_PER_TENANT, get_services
from src.config import AuthConfig, GraphConfig, GroundworkConfig
from src.governance import GovernanceError, GovernanceSettings
from src.governance_store import (
    GOVERNANCE_PREFIX,
    GovernanceStore,
    InMemoryGovernanceStore,
    governance_key,
)

TENANT = "demo-firm"
OTHER = "other-firm"


class FakeTable:
    """A DynamoDB table that records exactly what was stored, so encoding is assertable."""

    def __init__(self) -> None:
        self.items: dict[str, dict[str, Any]] = {}
        self.reads = 0

    def put_item(self, Item: dict[str, Any]) -> dict[str, Any]:
        self.items[Item["tenant_id"]] = Item
        return {}

    def get_item(self, Key: dict[str, Any]) -> dict[str, Any]:
        self.reads += 1
        item = self.items.get(Key["tenant_id"])
        return {"Item": item} if item else {}

    def delete_item(self, Key: dict[str, Any]) -> dict[str, Any]:
        self.items.pop(Key["tenant_id"], None)
        return {}


@pytest.fixture
def table() -> FakeTable:
    return FakeTable()


@pytest.fixture
def store(table: FakeTable) -> GovernanceStore:
    return GovernanceStore(table=table)


class TestSettingsPersist:
    def test_a_stored_setting_is_read_back(self, store):
        store.put(TENANT, GovernanceSettings(block_ungoverned_queries=True))
        store.forget(TENANT)
        assert store.get(TENANT).block_ungoverned_queries is True

    def test_the_kill_switch_survives_a_new_store_over_the_same_table(self, table):
        """The deploy. A second store is a second process reading the same table."""
        GovernanceStore(table=table).put(TENANT, GovernanceSettings(block_ungoverned_queries=True))

        restarted = GovernanceStore(table=table)
        assert restarted.get(TENANT).block_ungoverned_queries is True

    def test_a_tenant_that_never_changed_anything_gets_defaults(self, store):
        assert store.get("never-configured").min_confidence_floor == 0.8

    def test_another_tenants_settings_are_not_returned(self, store):
        store.put(TENANT, GovernanceSettings(block_ungoverned_queries=True))
        assert store.get(OTHER).block_ungoverned_queries is False

    def test_the_key_is_entity_prefixed(self, store, table):
        """Shares the tenant table using the same single-key pattern as tenant_directory, so
        governance items cannot collide with user bindings."""
        store.put(TENANT, GovernanceSettings())
        assert governance_key(TENANT) in table.items
        assert governance_key(TENANT).startswith(GOVERNANCE_PREFIX)


class TestEncoding:
    def test_floats_are_stored_as_decimal(self, store, table):
        """DynamoDB has no float type and boto3 raises rather than rounding silently, which is
        the right behaviour for a confidence floor."""
        store.put(TENANT, GovernanceSettings(min_confidence_floor=0.85))
        stored = table.items[governance_key(TENANT)]
        assert isinstance(stored["min_confidence_floor"], Decimal)

    def test_a_float_round_trips_as_a_float(self, store):
        store.put(TENANT, GovernanceSettings(min_confidence_floor=0.85))
        store.forget(TENANT)
        floor = store.get(TENANT).min_confidence_floor
        assert isinstance(floor, float)
        assert floor == pytest.approx(0.85)

    def test_the_tier_set_is_stored_as_a_list_and_read_as_a_set(self, store, table):
        """A frozenset cannot be stored, the same class of constraint as Neptune refusing
        list-valued properties."""
        store.put(TENANT, GovernanceSettings(allowed_tiers=frozenset({1, 2})))
        assert table.items[governance_key(TENANT)]["allowed_tiers"] == [1, 2]

        store.forget(TENANT)
        assert store.get(TENANT).allowed_tiers == frozenset({1, 2})

    def test_booleans_stay_booleans(self, store):
        store.put(TENANT, GovernanceSettings(auto_assert_deterministic=False))
        store.forget(TENANT)
        assert store.get(TENANT).auto_assert_deterministic is False


class TestItRefusesUnsafeCombinations:
    def test_a_cap_at_or_above_the_floor_is_rejected(self, store):
        """The gap between them is what keeps an unreviewed model claim under the retrieval
        floor. Validated on write as well as in `apply`, because this is a public entry point."""
        with pytest.raises(GovernanceError):
            store.put(
                TENANT, GovernanceSettings(min_confidence_floor=0.5, model_confidence_cap=0.9)
            )

    def test_nothing_is_written_when_validation_fails(self, store, table):
        with pytest.raises(GovernanceError):
            store.put(
                TENANT, GovernanceSettings(min_confidence_floor=0.5, model_confidence_cap=0.9)
            )
        assert table.items == {}


class TestDegradingSafely:
    def test_a_read_failure_falls_back_to_defaults(self):
        """An API that cannot answer a question because it cannot read a policy about answering
        questions is worse than one using the safe defaults, which have the floor and caps on."""

        class Broken(FakeTable):
            def get_item(self, Key: dict[str, Any]) -> dict[str, Any]:
                raise RuntimeError("throttled")

        settings = GovernanceStore(table=Broken()).get(TENANT)
        assert settings.min_confidence_floor == 0.8

    def test_an_unusable_stored_item_falls_back_to_defaults(self, store, table):
        """A field renamed, or a value that has since become invalid. Defaults are the safe
        answer and the warning is how somebody finds out."""
        table.items[governance_key(TENANT)] = {
            "tenant_id": governance_key(TENANT),
            "settings_present": True,
            "min_confidence_floor": Decimal("0.5"),
            "model_confidence_cap": Decimal("0.9"),
        }
        assert store.get(TENANT).min_confidence_floor == 0.8

    def test_an_item_without_the_marker_is_treated_as_absent(self, store, table):
        """Distinguishes "this tenant stored settings" from an item that exists for some other
        reason, so a partial record cannot masquerade as configuration."""
        table.items[governance_key(TENANT)] = {"tenant_id": governance_key(TENANT), "other": 1}
        assert store.get(TENANT).min_confidence_floor == 0.8


class TestCaching:
    def test_a_second_read_does_not_hit_the_table(self, store, table):
        store.get(TENANT)
        before = table.reads
        store.get(TENANT)
        assert table.reads == before

    def test_the_cache_expires(self, table):
        """Bounds how long a stale kill switch can stay stale across tasks."""
        now = [1000.0]
        store = GovernanceStore(table=table, ttl_seconds=30, clock=lambda: now[0])
        store.get(TENANT)
        before = table.reads

        now[0] += 31
        store.get(TENANT)
        assert table.reads > before

    def test_a_write_refreshes_the_cache(self, store):
        store.get(TENANT)
        store.put(TENANT, GovernanceSettings(block_ungoverned_queries=True))
        assert store.get(TENANT).block_ungoverned_queries is True


class TestTheReferenceStore:
    def test_it_offers_the_same_surface(self):
        """A method the real store lacks would fail only in production."""
        reference = {n for n in dir(GovernanceStore) if not n.startswith("_")}
        memory = {n for n in dir(InMemoryGovernanceStore) if not n.startswith("_")}
        assert reference - {"table", "table_name", "ttl_seconds"} <= memory


def _client():
    cfg = GroundworkConfig(
        environment="local",
        auth=AuthConfig(dev_bypass_tenant=TENANT),
        graph=GraphConfig(uri="bolt://127.0.0.1:1", user="none", password="none"),
    )
    cfg.validate()
    return TestClient(create_app(cfg)), get_services()


class TestOverHttp:
    def test_a_patch_is_persisted_not_just_cached(self):
        """The bug: PATCH wrote to a per-process dict, so the change was lost on deploy."""
        client, services = _client()
        client.patch(f"/api/tenants/{TENANT}/governance", json={"block_ungoverned_queries": True})

        # Drop the request-level cache, which is what a new process would not have.
        services.governance.clear()
        assert services.settings_for(TENANT).block_ungoverned_queries is True

    def test_a_refused_change_is_reported_with_its_reason(self):
        client, _ = _client()
        response = client.patch(
            f"/api/tenants/{TENANT}/governance",
            json={"min_confidence_floor": 0.5, "model_confidence_cap": 0.9},
        )
        assert response.status_code == 422
        assert "floor" in response.json()["detail"]

    def test_a_refused_change_alters_nothing(self):
        client, services = _client()
        before = services.settings_for(TENANT).min_confidence_floor
        client.patch(
            f"/api/tenants/{TENANT}/governance",
            json={"min_confidence_floor": 0.5, "model_confidence_cap": 0.9},
        )
        services.governance.clear()
        assert services.settings_for(TENANT).min_confidence_floor == before


class TestTheBlockedList:
    def test_a_refusal_is_visible_to_an_administrator(self):
        """It always returned an empty list, because refusals were recorded on a per-request
        resolver and died with it."""
        client, services = _client()
        services.record_blocked(TENANT, {"question": "total fees last quarter?", "at": "now"})

        body = client.get(f"/api/tenants/{TENANT}/governance/blocked").json()
        assert body["count"] == 1
        assert body["blocked"][0]["question"] == "total fees last quarter?"

    def test_newest_first(self):
        """A recent refusal is the one an administrator can still act on."""
        client, services = _client()
        services.record_blocked(TENANT, {"question": "first", "at": "t1"})
        services.record_blocked(TENANT, {"question": "second", "at": "t2"})

        blocked = client.get(f"/api/tenants/{TENANT}/governance/blocked").json()["blocked"]
        assert [b["question"] for b in blocked] == ["second", "first"]

    def test_another_tenants_refusals_are_not_shown(self):
        client, services = _client()
        services.record_blocked(OTHER, {"question": "their question", "at": "t1"})

        assert client.get(f"/api/tenants/{TENANT}/governance/blocked").json()["count"] == 0

    def test_the_log_is_capped(self):
        """A firm asking thousands of refused questions must not exhaust the task's memory."""
        _, services = _client()
        for i in range(MAX_BLOCKED_PER_TENANT + 25):
            services.record_blocked(TENANT, {"question": f"q{i}", "at": "t"})

        log = services.blocked_queries[TENANT]
        assert len(log) == MAX_BLOCKED_PER_TENANT
        # The oldest went, not the newest.
        assert log[-1]["question"] == f"q{MAX_BLOCKED_PER_TENANT + 24}"

    def test_it_reports_whether_the_switch_is_on(self):
        client, _ = _client()
        client.patch(f"/api/tenants/{TENANT}/governance", json={"block_ungoverned_queries": True})
        body = client.get(f"/api/tenants/{TENANT}/governance/blocked").json()
        assert body["kill_switch_active"] is True

    def test_a_real_refusal_reaches_the_list(self):
        """End to end rather than by calling `record_blocked` directly: the wiring from the
        resolver's per-request list into the process-level one is the part that was missing."""
        client, _ = _client()

        # No switch needed: a question no tier could answer reaches the backlog on the ordinary
        # path now, which is the case that matters -- most tenants have the switch off, and they
        # are the ones whose unanswerable questions should be turning into metrics.
        answered = client.post(
            f"/api/tenants/{TENANT}/query", json={"query": "what were total fees last quarter?"}
        )
        assert answered.status_code == 200

        body = client.get(f"/api/tenants/{TENANT}/governance/blocked").json()
        assert body["count"] >= 1
        assert "total fees" in body["blocked"][0]["question"]
