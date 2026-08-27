"""Who may create and delete a tenant, and what they have to say to do it.

The boundary here is not "is this caller an admin". Every firm has admins, and one firm's
admin must not be able to delete another firm's data, so authority is being an admin **of the
operator tenant**. These tests exist because that distinction is invisible in a diff: both
guards are one line and one of them is wrong.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.deps import get_services
from src.config import AuthConfig, GraphConfig, GroundworkConfig
from src.tenant_registry import InMemoryTenantRegistry, TenantRecord

HOME = "demo-firm"
OTHER = "other-firm"
LIST = "/api/platform/tenants"


def _config(*, home: str = HOME, acting_as: str = HOME) -> GroundworkConfig:
    """`dev_bypass_tenant` is what the caller's token would say, so it is how a test acts as
    an admin of a tenant other than the operator one."""
    cfg = GroundworkConfig(
        environment="local",
        auth=AuthConfig(dev_bypass_tenant=acting_as, home_tenant=home),
        graph=GraphConfig(uri="bolt://127.0.0.1:1", user="none", password="none"),
    )
    cfg.validate()
    return cfg


def _client(**over: Any) -> TestClient:
    client = TestClient(create_app(_config(**over)))
    services = get_services()
    registry = InMemoryTenantRegistry()
    registry.put(TenantRecord(tenant_id=OTHER, name="Other", created_by="someone"))
    services.tenant_registry = registry
    return client


def _delete_body(tenant: str) -> dict[str, Any]:
    return {
        "confirm_tenant_id": tenant,
        "confirm_document_loss": True,
        "confirm_audit_loss": True,
    }


class TestOnlyTheOperatorTenantHasAuthority:
    def test_an_admin_of_another_tenant_cannot_list(self):
        client = _client(acting_as=OTHER)
        assert client.get(LIST).status_code == 403

    def test_an_admin_of_another_tenant_cannot_create(self):
        client = _client(acting_as=OTHER)
        body = {"tenant_id": "new-firm", "admin_email": "a@b.example"}
        assert client.post(LIST, json=body).status_code == 403

    def test_an_admin_of_another_tenant_cannot_delete(self):
        """The one that matters: a customer's own admin reaching another customer."""
        client = _client(acting_as=OTHER)
        r = client.post(f"{LIST}/{HOME}/delete", json=_delete_body(HOME))
        assert r.status_code == 403

    def test_the_refusal_does_not_name_the_operator_tenant(self):
        """A refused caller learning which tenant would qualify is a probe answered."""
        client = _client(acting_as=OTHER)
        assert HOME not in client.get(LIST).json()["detail"]

    def test_no_operator_tenant_configured_closes_the_routes(self):
        """Closed by default. An unset operator tenant is a misconfiguration, and the safe
        reading is that nobody qualifies rather than everybody."""
        client = _client(home="", acting_as=HOME)
        assert client.get(LIST).status_code == 503


class TestDeletionAsksBeforeItDestroys:
    @pytest.fixture
    def client(self) -> TestClient:
        return _client()

    def test_a_mismatched_name_is_refused(self, client):
        r = client.post(
            f"{LIST}/{OTHER}/delete",
            json={**_delete_body(OTHER), "confirm_tenant_id": "wrong"},
        )
        assert r.status_code == 400
        assert "does not match" in r.json()["detail"]

    def test_the_document_loss_must_be_acknowledged(self, client):
        r = client.post(
            f"{LIST}/{OTHER}/delete",
            json={**_delete_body(OTHER), "confirm_document_loss": False},
        )
        assert r.status_code == 400
        assert "S3" in r.json()["detail"]

    def test_the_audit_loss_must_be_acknowledged(self, client):
        r = client.post(
            f"{LIST}/{OTHER}/delete",
            json={**_delete_body(OTHER), "confirm_audit_loss": False},
        )
        assert r.status_code == 400
        assert "audit trail" in r.json()["detail"]

    def test_the_operator_tenant_cannot_delete_itself(self, client):
        """It holds the only identities that can create or delete tenants, so this is a
        one-way lockout of the whole platform."""
        r = client.post(f"{LIST}/{HOME}/delete", json=_delete_body(HOME))
        assert r.status_code == 409
        assert "lock the platform out of itself" in r.json()["detail"]

    def test_a_refusal_deletes_nothing(self, client):
        """Each guard returns before the cascade, so a refused request must leave the tenant
        exactly as it was."""
        services = get_services()
        client.post(
            f"{LIST}/{OTHER}/delete",
            json={**_delete_body(OTHER), "confirm_document_loss": False},
        )
        assert services.tenant_registry.get(OTHER).is_live


class TestListing:
    def test_deleted_tenants_are_listed_too(self):
        """So a reused id is visible as reused rather than looking new."""
        client = _client()
        services = get_services()
        services.tenant_registry.tombstone(OTHER, actor="someone")

        body = client.get(LIST).json()
        assert body["home_tenant"] == HOME
        rows = {t["tenant_id"]: t for t in body["tenants"]}
        assert rows[OTHER]["is_live"] is False
        assert rows[OTHER]["deleted_at"]


class TestCreationValidatesBeforeItWrites:
    @pytest.fixture
    def client(self) -> TestClient:
        return _client()

    def test_a_malformed_id_is_refused(self, client):
        r = client.post(LIST, json={"tenant_id": "Not Valid", "admin_email": "a@b.example"})
        assert r.status_code == 422

    def test_an_admin_email_is_mandatory(self, client):
        """A tenant with no users cannot be signed in to, so creating one without an admin
        produces a namespace nobody can reach."""
        assert client.post(LIST, json={"tenant_id": "new-firm"}).status_code == 422

    def test_a_taken_id_is_a_conflict(self, client):
        r = client.post(LIST, json={"tenant_id": OTHER, "admin_email": "a@b.example"})
        assert r.status_code == 409
