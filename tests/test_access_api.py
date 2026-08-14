"""The grants API over HTTP.

Three properties are the reason this file exists, and each is a silent failure if it
regresses:

- a non-admin cannot mutate anything, so the API is not a privilege-escalation surface;
- a screen without a reason is refused *with the domain's message*, because a pydantic
  length error tells an administrator nothing about why an unexplained wall is unsafe;
- a screen raised through the API bites the next request, including a platform admin's.

The last one is the reason these go through `TestClient` rather than calling the manager:
the ordering trap is in the wiring — an admin's allowlist is None, so a screen can only
apply if the denylist travels separately and is applied last.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.deps import get_services
from src.auth import Grants
from src.config import AuthConfig, GraphConfig, LexGraphConfig

TENANT = "dev-tenant"
MATTER = "M-2291"
ALICE = "alice@firm.com"
BOB = "bob@firm.com"
REASON = "acted for the opposing party in 2024"
CONTACT = "risk@firm.com"

BASE = f"/api/tenants/{TENANT}/access"


def _config() -> LexGraphConfig:
    cfg = LexGraphConfig(
        environment="local",
        auth=AuthConfig(dev_bypass_tenant=TENANT),
        graph=GraphConfig(uri="bolt://127.0.0.1:1", user="none", password="none"),
    )
    cfg.validate()
    return cfg


@pytest.fixture
def client() -> TestClient:
    """The dev bypass grants platform-admin, so this caller may mutate."""
    return TestClient(create_app(_config()))


def _demote(roles: frozenset[str]) -> None:
    """Re-authenticate the dev caller with narrower roles.

    Wraps `authenticate` rather than the store because roles come from the token and
    matter access does not — replacing the roles is the only honest way to test the
    admin gate without also disturbing the access resolution under test.
    """
    services = get_services()
    original = services.authenticator.authenticate

    def demoted(*args: Any, **kwargs: Any) -> Any:
        ctx, grants = original(*args, **kwargs)
        return ctx, Grants(
            tenant_id=grants.tenant_id,
            roles=roles,
            matter_allowlist=grants.matter_allowlist,
            matter_denylist=grants.matter_denylist,
        )

    services.authenticator.authenticate = demoted  # type: ignore[method-assign]


def _assign(client: TestClient, user_id: str = ALICE, matter_id: str = MATTER, **over: Any):
    return client.post(
        f"{BASE}/assignments", json={"user_id": user_id, "matter_id": matter_id, **over}
    )


def _screen(client: TestClient, user_id: str = ALICE, matter_id: str = MATTER, **over: Any):
    body = {
        "user_id": user_id,
        "matter_id": matter_id,
        "reason": REASON,
        "contact": CONTACT,
        **over,
    }
    return client.post(f"{BASE}/screens", json=body)


class TestAssignments:
    def test_assigning_returns_the_row(self, client):
        r = _assign(client, role="matter-owner")
        assert r.status_code == 201
        body = r.json()["assignment"]
        assert body["user_id"] == ALICE
        assert body["role"] == "matter-owner"
        assert body["granted_by"] == "dev@localhost"
        assert body["revoked_at"] is None

    def test_an_unassigned_user_starts_closed(self, client):
        body = client.get(f"{BASE}/users/{ALICE}").json()
        assert body["assignments"] == []
        assert body["screens"] == []
        assert body["decisions"] == []

    def test_an_assignment_shows_as_allowed(self, client):
        _assign(client)
        decisions = client.get(f"{BASE}/users/{ALICE}").json()["decisions"]
        assert decisions == [
            {
                "matter_id": MATTER,
                "decision": "ALLOWED",
                "explanation": f"You are assigned to {MATTER}.",
            }
        ]

    def test_the_team_endpoint_lists_the_matter_roster(self, client):
        _assign(client, ALICE)
        _assign(client, BOB)
        body = client.get(f"{BASE}/matters/{MATTER}").json()
        assert sorted(a["user_id"] for a in body["team"]) == [ALICE, BOB]
        assert body["screened"] == []

    def test_unassigning_revokes_and_says_history_is_kept(self, client):
        _assign(client)
        r = client.request(
            "DELETE",
            f"{BASE}/assignments",
            json={"user_id": ALICE, "matter_id": MATTER, "reason": "rolled off"},
        )
        assert r.status_code == 200
        assert r.json()["history_preserved"] is True
        assert client.get(f"{BASE}/users/{ALICE}").json()["assignments"] == []

    def test_a_revoked_assignment_leaves_the_matter_closed(self, client):
        _assign(client)
        client.request(
            "DELETE",
            f"{BASE}/assignments",
            json={"user_id": ALICE, "matter_id": MATTER, "reason": "rolled off"},
        )
        assert client.get(f"{BASE}/matters/{MATTER}").json()["team"] == []


class TestScreens:
    def test_screening_records_reason_and_contact(self, client):
        r = _screen(client)
        assert r.status_code == 201
        screen = r.json()["screen"]
        assert screen["reason"] == REASON
        assert screen["contact"] == CONTACT
        assert screen["screened_by"] == "dev@localhost"

    def test_a_screen_beats_an_assignment(self, client):
        _assign(client)
        _screen(client)
        decisions = client.get(f"{BASE}/users/{ALICE}").json()["decisions"]
        assert [d["decision"] for d in decisions] == ["SCREENED"]

    def test_the_explanation_names_the_matter_the_reason_and_a_contact(self, client):
        _screen(client)
        explanation = client.get(f"{BASE}/users/{ALICE}").json()["decisions"][0]["explanation"]
        assert MATTER in explanation
        assert REASON in explanation
        assert CONTACT in explanation

    def test_the_matter_view_shows_who_is_screened(self, client):
        """A roster without the screens reads as complete while a wall stands in the
        middle of the matter."""
        _assign(client, ALICE)
        _screen(client, BOB)
        body = client.get(f"{BASE}/matters/{MATTER}").json()
        assert [a["user_id"] for a in body["team"]] == [ALICE]
        assert [s["user_id"] for s in body["screened"]] == [BOB]

    def test_lifting_a_screen_keeps_the_record(self, client):
        _assign(client)
        _screen(client)
        r = client.request(
            "DELETE",
            f"{BASE}/screens",
            json={"user_id": ALICE, "matter_id": MATTER, "reason": "dispute concluded"},
        )
        assert r.status_code == 200
        assert r.json()["history_preserved"] is True
        decisions = client.get(f"{BASE}/users/{ALICE}").json()["decisions"]
        assert [d["decision"] for d in decisions] == ["ALLOWED"]

    def test_the_lift_is_on_the_audit_trail_with_its_reason(self, client):
        _screen(client)
        client.request(
            "DELETE",
            f"{BASE}/screens",
            json={"user_id": ALICE, "matter_id": MATTER, "reason": "dispute concluded"},
        )
        events = client.get(f"{BASE}/audit", params={"user_id": ALICE}).json()["events"]
        lift = next(e for e in events if e["action"] == "LIFT_SCREEN")
        assert lift["reason"] == "dispute concluded"


class TestAScreenNeedsAReason:
    def test_a_missing_reason_is_422(self, client):
        r = client.post(f"{BASE}/screens", json={"user_id": ALICE, "matter_id": MATTER})
        assert r.status_code == 422

    def test_the_422_explains_why_rather_than_naming_a_field(self, client):
        """An administrator has to be told what is wrong with an unexplained wall, not
        that a string failed a length check."""
        r = _screen(client, reason="")
        assert r.status_code == 422
        assert "cannot be defended" in str(r.json()["detail"])

    def test_whitespace_is_not_a_reason(self, client):
        assert _screen(client, reason="   ").status_code == 422

    def test_a_refused_screen_raises_no_wall(self, client):
        _assign(client)
        _screen(client, reason="")
        decisions = client.get(f"{BASE}/users/{ALICE}").json()["decisions"]
        assert [d["decision"] for d in decisions] == ["ALLOWED"]

    def test_lifting_without_a_reason_is_422(self, client):
        _screen(client)
        r = client.request(
            "DELETE", f"{BASE}/screens", json={"user_id": ALICE, "matter_id": MATTER}
        )
        assert r.status_code == 422

    def test_a_refused_lift_leaves_the_wall_standing(self, client):
        _screen(client)
        client.request(
            "DELETE",
            f"{BASE}/screens",
            json={"user_id": ALICE, "matter_id": MATTER, "reason": " "},
        )
        decisions = client.get(f"{BASE}/users/{ALICE}").json()["decisions"]
        assert [d["decision"] for d in decisions] == ["SCREENED"]


class TestAScreenBeatsPlatformAdmin:
    """The important one, end to end.

    A platform admin's allowlist is None — every matter in the tenant — so the screen can
    only bite if the denylist is carried separately and applied last. This is the test
    that a refactor of `to_scope()` or of `Authenticator` cannot quietly break.
    """

    def test_the_admins_own_screen_denies_them(self, client):
        _screen(client, user_id="dev@localhost")
        # A fresh request re-resolves access, which is the whole point of reading matter
        # access per request rather than from the token.
        body = client.get(f"/api/tenants/{TENANT}/matters").json()
        assert [w["matter_id"] for w in body["withheld"]] == [MATTER]
        assert body["withheld_count"] == 1

    def test_the_admin_is_told_the_reason_and_the_contact(self, client):
        _screen(client, user_id="dev@localhost")
        withheld = client.get(f"/api/tenants/{TENANT}/matters").json()["withheld"][0]
        assert withheld["reason"] == REASON
        assert withheld["contact"] == CONTACT

    def test_the_admin_keeps_every_unscreened_matter(self, client):
        """Confirms the wall is a wall and not a blanket revocation of the role."""
        _screen(client, user_id="dev@localhost")
        _, grants = get_services().authenticator.authenticate(None)
        assert grants.matter_allowlist is None
        assert grants.matter_denylist == frozenset({MATTER})

    def test_lifting_the_screen_restores_the_admin(self, client):
        _screen(client, user_id="dev@localhost")
        client.request(
            "DELETE",
            f"{BASE}/screens",
            json={"user_id": "dev@localhost", "matter_id": MATTER, "reason": "concluded"},
        )
        assert client.get(f"/api/tenants/{TENANT}/matters").json()["withheld"] == []


class TestAdminOnly:
    """Deciding who may see a matter is the one thing an attacker most wants. A
    self-service assignment endpoint would be privilege escalation with a REST API."""

    @pytest.mark.parametrize(
        "method,path,body",
        [
            ("POST", "/assignments", {"user_id": ALICE, "matter_id": MATTER}),
            ("DELETE", "/assignments", {"user_id": ALICE, "matter_id": MATTER}),
            (
                "POST",
                "/screens",
                {"user_id": ALICE, "matter_id": MATTER, "reason": REASON},
            ),
            (
                "DELETE",
                "/screens",
                {"user_id": ALICE, "matter_id": MATTER, "reason": REASON},
            ),
        ],
    )
    def test_a_reviewer_gets_403_from_every_mutation(self, client, method, path, body):
        _demote(frozenset({"reviewer"}))
        r = client.request(method, f"{BASE}{path}", json=body)
        assert r.status_code == 403, r.text
        assert "platform-admin" in r.json()["detail"]

    def test_a_reviewer_cannot_read_another_users_screens(self, client):
        """A user's screens name the matters they are walled off from, which is exactly
        what a wall exists to keep from circulating around a firm."""
        _demote(frozenset({"reviewer"}))
        assert client.get(f"{BASE}/users/{ALICE}").status_code == 403

    def test_a_reviewer_cannot_read_the_matter_roster(self, client):
        _demote(frozenset({"reviewer"}))
        assert client.get(f"{BASE}/matters/{MATTER}").status_code == 403

    def test_a_reviewer_cannot_read_the_audit_trail(self, client):
        _demote(frozenset({"reviewer"}))
        r = client.get(f"{BASE}/audit", params={"matter_id": MATTER})
        assert r.status_code == 403

    def test_a_refused_mutation_changes_nothing(self, client):
        _demote(frozenset({"reviewer"}))
        _screen(client)
        _demote(frozenset({"platform-admin"}))
        assert client.get(f"{BASE}/users/{ALICE}").json()["screens"] == []


class TestAuditTrail:
    def test_the_trail_records_actor_subject_matter_and_reason(self, client):
        _screen(client)
        events = client.get(f"{BASE}/audit", params={"matter_id": MATTER}).json()["events"]
        screen = next(e for e in events if e["action"] == "SCREEN")
        assert screen["actor"] == "dev@localhost"
        assert screen["subject_user"] == ALICE
        assert screen["matter_id"] == MATTER
        assert screen["reason"] == REASON
        assert screen["detail"] == {"contact": CONTACT}

    def test_the_trail_is_ordered_and_complete(self, client):
        _assign(client)
        _screen(client)
        client.request(
            "DELETE",
            f"{BASE}/screens",
            json={"user_id": ALICE, "matter_id": MATTER, "reason": "concluded"},
        )
        client.request(
            "DELETE",
            f"{BASE}/assignments",
            json={"user_id": ALICE, "matter_id": MATTER, "reason": "rolled off"},
        )
        events = client.get(f"{BASE}/audit", params={"user_id": ALICE}).json()["events"]
        assert [e["action"] for e in events] == [
            "ASSIGN",
            "SCREEN",
            "LIFT_SCREEN",
            "UNASSIGN",
        ]

    def test_the_trail_by_matter_excludes_other_matters(self, client):
        _screen(client, ALICE, MATTER)
        _screen(client, BOB, "M-OTHER")
        events = client.get(f"{BASE}/audit", params={"matter_id": MATTER}).json()["events"]
        assert [e["subject_user"] for e in events] == [ALICE]

    def test_an_unfiltered_audit_read_is_refused(self, client):
        """An unbounded read would be a scan of the store on an authorization path."""
        r = client.get(f"{BASE}/audit")
        assert r.status_code == 422
        assert "exactly one" in r.json()["detail"]

    def test_both_filters_at_once_is_refused(self, client):
        r = client.get(f"{BASE}/audit", params={"matter_id": MATTER, "user_id": ALICE})
        assert r.status_code == 422


class TestTenantIsolation:
    def test_another_tenants_path_is_404(self, client):
        """404 rather than 403: confirming another firm exists is itself a leak."""
        _assign(client)
        assert client.get(f"/api/tenants/firm-beta/access/users/{ALICE}").status_code == 404

    def test_a_mutation_against_another_tenant_is_404(self, client):
        r = client.post(
            "/api/tenants/firm-beta/access/assignments",
            json={"user_id": ALICE, "matter_id": MATTER},
        )
        assert r.status_code == 404

    def test_grants_are_scoped_to_the_token_tenant_not_the_path(self, client):
        """The path is for routing. A grant written here must land under the token's
        tenant, or a mismatch would be a cross-firm write."""
        _assign(client)
        store = get_services().access.store
        assert [a.user_id for a in store.team_of(TENANT, MATTER)] == [ALICE]
        assert store.team_of("firm-beta", MATTER) == []

    def test_the_audit_trail_does_not_cross_tenants(self, client):
        _screen(client)
        assert get_services().access.store.events_for_matter("firm-beta", MATTER) == []
