"""Matter access: the default posture, the ordering of a screen against everything
else, and whether history survives a revocation.

The test that matters most is `test_a_screen_beats_platform_admin`. Every other
ordering in this module is arguable; that one is what makes an ethical wall a wall
rather than a filter, and an admin who can read through it means the firm has no wall.

DynamoDB is faked. No AWS.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.access import (
    AccessDecision,
    AccessEvent,
    AccessManager,
    InMemoryAccessStore,
    MatterAssignment,
    MatterScreen,
)
from src.access_dynamo import DynamoAccessStore

TENANT = "firm-acme"
OTHER_TENANT = "firm-beta"
MATTER = "M-2291"
ALICE = "alice@acme.com"
BOB = "bob@acme.com"
ADMIN = "admin@acme.com"
REASON = "acted for the opposing party in 2024"
CONTACT = "risk@acme.com"


class FakeTable:
    """A DynamoDB `Table` with enough behaviour to hold the store honest.

    Implements the key-condition forms `DynamoAccessStore` actually issues, honours the
    put condition expression, and records every call so a test can assert that no scan
    happened and that the conditional put is really conditional.
    """

    def __init__(self) -> None:
        self.items: dict[tuple[str, str], dict[str, Any]] = {}
        self.queries: list[dict[str, Any]] = []

    def put_item(self, **kwargs: Any) -> dict[str, Any]:
        item = kwargs["Item"]
        key = (item["PK"], item["SK"])
        if kwargs.get("ConditionExpression") and key in self.items:
            raise RuntimeError("ConditionalCheckFailedException")
        self.items[key] = dict(item)
        return {}

    def query(self, **kwargs: Any) -> dict[str, Any]:
        self.queries.append(kwargs)
        values = kwargs["ExpressionAttributeValues"]
        pk_attr, sk_attr = ("GSI1PK", "GSI1SK") if kwargs.get("IndexName") else ("PK", "SK")
        matched = [
            i
            for i in self.items.values()
            if i.get(pk_attr) == values[":pk"] and str(i.get(sk_attr, "")).startswith(
                values[":prefix"]
            )
        ]
        return {"Items": sorted(matched, key=lambda i: i["SK"])}

    def scan(self, **kwargs: Any) -> dict[str, Any]:  # pragma: no cover
        raise AssertionError("a scan on an authorization path is never acceptable")


@pytest.fixture
def manager() -> AccessManager:
    return AccessManager(InMemoryAccessStore())


@pytest.fixture
def table() -> FakeTable:
    return FakeTable()


@pytest.fixture
def dynamo(table: FakeTable) -> DynamoAccessStore:
    return DynamoAccessStore("lexgraph-grants", table=table)


class TestDefaultIsClosed:
    def test_an_unassigned_user_sees_nothing(self, manager):
        access = manager.resolve(TENANT, ALICE)
        assert access.assigned == frozenset()
        assert not access.decide(MATTER)
        assert access.decide(MATTER) is AccessDecision.NOT_ASSIGNED

    def test_an_unassigned_user_gets_an_empty_allowlist_not_a_wildcard(self, manager):
        """The difference between closed and open is `frozenset()` against `None`, and
        getting it backwards hands every matter in the firm to a new starter."""
        allowlist, _ = manager.resolve(TENANT, ALICE).to_scope()
        assert allowlist == frozenset()

    def test_an_assignment_opens_exactly_one_matter(self, manager):
        manager.assign(TENANT, ALICE, MATTER, actor=ADMIN)
        manager.assign(TENANT, ALICE, "M-OTHER", actor=ADMIN)
        access = manager.resolve(TENANT, ALICE)
        assert access.decide(MATTER) is AccessDecision.ALLOWED
        assert access.decide("M-UNRELATED") is AccessDecision.NOT_ASSIGNED

    def test_the_refusal_tells_a_lawyer_what_to_do(self, manager):
        explanation = manager.resolve(TENANT, ALICE).explain(MATTER)
        assert MATTER in explanation
        assert "matter owner" in explanation


class TestAScreenBeatsEverything:
    def test_a_screen_beats_an_assignment(self, manager):
        manager.assign(TENANT, ALICE, MATTER, actor=ADMIN)
        manager.screen(TENANT, ALICE, MATTER, actor=ADMIN, reason=REASON, contact=CONTACT)
        assert manager.resolve(TENANT, ALICE).decide(MATTER) is AccessDecision.SCREENED

    def test_a_screen_beats_platform_admin(self, manager):
        """The one that matters. A platform admin reads every matter in the tenant by
        role, and a screen must still stop them — otherwise the firm has a filter."""
        manager.screen(TENANT, ADMIN, MATTER, actor=ADMIN, reason=REASON)
        access = manager.resolve(TENANT, ADMIN, is_platform_admin=True)
        assert access.decide(MATTER) is AccessDecision.SCREENED
        assert not access.decide(MATTER)

    def test_platform_admin_still_reads_unscreened_matters(self, manager):
        """Confirms the previous test fails for the right reason rather than because
        admin access is broken."""
        manager.screen(TENANT, ADMIN, MATTER, actor=ADMIN, reason=REASON)
        access = manager.resolve(TENANT, ADMIN, is_platform_admin=True)
        assert access.decide("M-OTHER") is AccessDecision.PLATFORM_ADMIN

    def test_the_denylist_is_returned_separately_from_the_allowlist(self, manager):
        """An admin's allowlist is None, so subtracting screens from it is impossible.
        They have to travel separately and be applied last."""
        manager.screen(TENANT, ADMIN, MATTER, actor=ADMIN, reason=REASON)
        allowlist, denylist = manager.resolve(
            TENANT, ADMIN, is_platform_admin=True
        ).to_scope()
        assert allowlist is None
        assert denylist == frozenset({MATTER})

    def test_a_screen_is_loud_and_names_a_contact(self, manager):
        manager.screen(TENANT, ALICE, MATTER, actor=ADMIN, reason=REASON, contact=CONTACT)
        explanation = manager.resolve(TENANT, ALICE).explain(MATTER)
        assert MATTER in explanation
        assert REASON in explanation
        assert CONTACT in explanation

    def test_a_screen_without_a_contact_still_routes_somewhere(self, manager):
        manager.screen(TENANT, ALICE, MATTER, actor=ADMIN, reason=REASON)
        assert "risk team" in manager.resolve(TENANT, ALICE).explain(MATTER)


class TestAScreenNeedsAReason:
    def test_a_blank_reason_is_refused(self, manager):
        with pytest.raises(ValueError, match="requires a reason"):
            manager.screen(TENANT, ALICE, MATTER, actor=ADMIN, reason="")

    def test_whitespace_is_not_a_reason(self, manager):
        with pytest.raises(ValueError):
            manager.screen(TENANT, ALICE, MATTER, actor=ADMIN, reason="   \n ")

    def test_a_refused_screen_writes_nothing(self, manager):
        """A wall that half-exists is worse than one that does not: the user is denied
        and the record cannot say why."""
        with pytest.raises(ValueError):
            manager.screen(TENANT, ALICE, MATTER, actor=ADMIN, reason="")
        assert manager.store.screens_for(TENANT, ALICE) == []
        assert manager.store.events_for_user(TENANT, ALICE) == []

    def test_lifting_a_screen_also_needs_a_reason(self, manager):
        manager.screen(TENANT, ALICE, MATTER, actor=ADMIN, reason=REASON)
        with pytest.raises(ValueError, match="requires a reason"):
            manager.lift_screen(TENANT, ALICE, MATTER, actor=ADMIN, reason="")

    def test_a_refused_lift_leaves_the_wall_standing(self, manager):
        manager.screen(TENANT, ALICE, MATTER, actor=ADMIN, reason=REASON)
        with pytest.raises(ValueError):
            manager.lift_screen(TENANT, ALICE, MATTER, actor=ADMIN, reason="")
        assert manager.resolve(TENANT, ALICE).decide(MATTER) is AccessDecision.SCREENED


class TestHistorySurvives:
    def test_unassign_revokes_rather_than_deletes(self, manager):
        manager.assign(TENANT, ALICE, MATTER, actor=ADMIN, role="matter-owner")
        manager.unassign(TENANT, ALICE, MATTER, actor=ADMIN, reason="rolled off")

        assert manager.resolve(TENANT, ALICE).decide(MATTER) is AccessDecision.NOT_ASSIGNED
        # The row is still there, marked — which is what an audit needs to see.
        revoked = manager.store._assignments[(TENANT, ALICE, MATTER)]
        assert revoked.revoked_at and revoked.revoked_by == ADMIN
        assert revoked.granted_by == ADMIN
        assert revoked.role == "matter-owner"

    def test_lifting_a_screen_keeps_the_reason_it_was_raised_for(self, manager):
        manager.screen(TENANT, ALICE, MATTER, actor=ADMIN, reason=REASON, contact=CONTACT)
        manager.lift_screen(TENANT, ALICE, MATTER, actor=BOB, reason="dispute concluded")

        lifted = manager.store._screens[(TENANT, ALICE, MATTER)]
        assert lifted.lifted_by == BOB
        assert lifted.reason == REASON
        assert lifted.contact == CONTACT
        assert lifted.screened_by == ADMIN

    def test_a_lifted_screen_stops_denying(self, manager):
        manager.assign(TENANT, ALICE, MATTER, actor=ADMIN)
        manager.screen(TENANT, ALICE, MATTER, actor=ADMIN, reason=REASON)
        manager.lift_screen(TENANT, ALICE, MATTER, actor=ADMIN, reason="concluded")
        assert manager.resolve(TENANT, ALICE).decide(MATTER) is AccessDecision.ALLOWED

    def test_reassigning_after_a_revocation_works(self, manager):
        manager.assign(TENANT, ALICE, MATTER, actor=ADMIN)
        manager.unassign(TENANT, ALICE, MATTER, actor=ADMIN, reason="rolled off")
        manager.assign(TENANT, ALICE, MATTER, actor=ADMIN)
        assert manager.resolve(TENANT, ALICE).decide(MATTER) is AccessDecision.ALLOWED


class TestAuditTrail:
    def test_every_change_lands_on_the_trail(self, manager):
        manager.assign(TENANT, ALICE, MATTER, actor=ADMIN)
        manager.screen(TENANT, ALICE, MATTER, actor=ADMIN, reason=REASON)
        manager.lift_screen(TENANT, ALICE, MATTER, actor=ADMIN, reason="concluded")
        manager.unassign(TENANT, ALICE, MATTER, actor=ADMIN, reason="rolled off")

        actions = [e.action for e in manager.store.events_for_user(TENANT, ALICE)]
        assert actions == ["ASSIGN", "SCREEN", "LIFT_SCREEN", "UNASSIGN"]

    def test_a_screen_event_records_actor_subject_matter_and_reason(self, manager):
        manager.screen(TENANT, ALICE, MATTER, actor=ADMIN, reason=REASON, contact=CONTACT)
        event = manager.store.events_for_matter(TENANT, MATTER)[0]
        assert event.actor == ADMIN
        assert event.subject_user == ALICE
        assert event.matter_id == MATTER
        assert event.reason == REASON
        assert event.detail == {"contact": CONTACT}

    def test_the_trail_is_readable_by_matter(self, manager):
        manager.screen(TENANT, ALICE, MATTER, actor=ADMIN, reason=REASON)
        manager.screen(TENANT, BOB, "M-OTHER", actor=ADMIN, reason=REASON)
        assert [e.subject_user for e in manager.store.events_for_matter(TENANT, MATTER)] == [
            ALICE
        ]


class TestTenantIsolation:
    def test_an_assignment_does_not_cross_tenants(self, manager):
        manager.assign(TENANT, ALICE, MATTER, actor=ADMIN)
        assert manager.resolve(OTHER_TENANT, ALICE).assigned == frozenset()

    def test_a_screen_does_not_cross_tenants(self, manager):
        manager.screen(TENANT, ALICE, MATTER, actor=ADMIN, reason=REASON)
        assert manager.resolve(OTHER_TENANT, ALICE).screened == frozenset()

    def test_the_team_is_per_tenant(self, manager):
        manager.assign(TENANT, ALICE, MATTER, actor=ADMIN)
        assert manager.store.team_of(OTHER_TENANT, MATTER) == []

    def test_the_audit_trail_is_per_tenant(self, manager):
        manager.screen(TENANT, ALICE, MATTER, actor=ADMIN, reason=REASON)
        assert manager.store.events_for_matter(OTHER_TENANT, MATTER) == []


class TestDynamoRoundTrip:
    def test_an_assignment_survives_every_field(self, dynamo):
        original = MatterAssignment(
            tenant_id=TENANT,
            user_id=ALICE,
            matter_id=MATTER,
            granted_by=ADMIN,
            granted_at="2026-01-02T03:04:05+00:00",
            role="matter-owner",
        )
        dynamo.put_assignment(original)
        assert dynamo.assignments_for(TENANT, ALICE) == [original]

    def test_a_screen_survives_every_field(self, dynamo):
        """`reason` and `contact` in particular. Dropping either degrades the wall's
        explanation to "you are screened from something", which is the failure the
        loud-screen design exists to prevent."""
        original = MatterScreen(
            tenant_id=TENANT,
            user_id=ALICE,
            matter_id=MATTER,
            reason=REASON,
            screened_by=ADMIN,
            screened_at="2026-01-02T03:04:05+00:00",
            contact=CONTACT,
        )
        dynamo.put_screen(original)
        assert dynamo.screens_for(TENANT, ALICE) == [original]

    def test_an_event_survives_every_field_including_detail(self, dynamo):
        original = AccessEvent(
            tenant_id=TENANT,
            actor=ADMIN,
            action="SCREEN",
            subject_user=ALICE,
            matter_id=MATTER,
            at="2026-01-02T03:04:05+00:00",
            reason=REASON,
            detail={"contact": CONTACT, "note": "raised by risk"},
        )
        dynamo.append_event(original)
        assert dynamo.events_for_matter(TENANT, MATTER) == [original]
        assert dynamo.events_for_user(TENANT, ALICE) == [original]

    def test_a_revoked_assignment_is_read_back_as_inactive(self, dynamo):
        dynamo.put_assignment(
            MatterAssignment(
                tenant_id=TENANT,
                user_id=ALICE,
                matter_id=MATTER,
                granted_by=ADMIN,
                revoked_at="2026-02-01T00:00:00+00:00",
                revoked_by=ADMIN,
            )
        )
        assert dynamo.assignments_for(TENANT, ALICE) == []

    def test_a_lifted_screen_is_read_back_as_inactive(self, dynamo):
        dynamo.put_screen(
            MatterScreen(
                tenant_id=TENANT,
                user_id=ALICE,
                matter_id=MATTER,
                reason=REASON,
                screened_by=ADMIN,
                lifted_at="2026-02-01T00:00:00+00:00",
                lifted_by=ADMIN,
            )
        )
        assert dynamo.screens_for(TENANT, ALICE) == []

    def test_team_of_uses_the_index_not_a_scan(self, dynamo, table):
        dynamo.put_assignment(
            MatterAssignment(
                tenant_id=TENANT, user_id=ALICE, matter_id=MATTER, granted_by=ADMIN
            )
        )
        dynamo.put_assignment(
            MatterAssignment(
                tenant_id=TENANT, user_id=BOB, matter_id=MATTER, granted_by=ADMIN
            )
        )
        team = dynamo.team_of(TENANT, MATTER)
        assert sorted(a.user_id for a in team) == [ALICE, BOB]
        assert table.queries[-1]["IndexName"] == "GSI1"

    def test_screens_on_a_matter_are_queryable(self, dynamo):
        dynamo.put_screen(
            MatterScreen(
                tenant_id=TENANT,
                user_id=ALICE,
                matter_id=MATTER,
                reason=REASON,
                screened_by=ADMIN,
            )
        )
        assert [s.user_id for s in dynamo.screens_on(TENANT, MATTER)] == [ALICE]

    def test_assignments_and_screens_do_not_bleed_into_each_other(self, dynamo):
        """Same partition, different sort prefixes. A prefix bug would read a screen as
        an assignment, which fails open."""
        dynamo.put_assignment(
            MatterAssignment(
                tenant_id=TENANT, user_id=ALICE, matter_id=MATTER, granted_by=ADMIN
            )
        )
        dynamo.put_screen(
            MatterScreen(
                tenant_id=TENANT,
                user_id=ALICE,
                matter_id="M-OTHER",
                reason=REASON,
                screened_by=ADMIN,
            )
        )
        assert [a.matter_id for a in dynamo.assignments_for(TENANT, ALICE)] == [MATTER]
        assert [s.matter_id for s in dynamo.screens_for(TENANT, ALICE)] == ["M-OTHER"]

    def test_events_are_not_returned_as_assignments(self, dynamo):
        dynamo.append_event(
            AccessEvent(
                tenant_id=TENANT,
                actor=ADMIN,
                action="ASSIGN",
                subject_user=ALICE,
                matter_id=MATTER,
            )
        )
        assert dynamo.assignments_for(TENANT, ALICE) == []
        assert len(dynamo.events_for_user(TENANT, ALICE)) == 1

    def test_tenant_isolation_holds_in_the_key(self, dynamo):
        dynamo.put_assignment(
            MatterAssignment(
                tenant_id=TENANT, user_id=ALICE, matter_id=MATTER, granted_by=ADMIN
            )
        )
        assert dynamo.assignments_for(OTHER_TENANT, ALICE) == []
        assert dynamo.team_of(OTHER_TENANT, MATTER) == []

    def test_appending_is_conditional_so_an_event_cannot_be_overwritten(self, dynamo, table):
        """The condition is what makes this append-only rather than append-shaped: put
        the same event key twice and DynamoDB refuses instead of rewriting the record."""
        written: list[dict[str, Any]] = []
        table.put_item = lambda **kw: written.append(kw) or {}  # type: ignore[method-assign]
        dynamo.append_event(
            AccessEvent(
                tenant_id=TENANT,
                actor=ADMIN,
                action="SCREEN",
                subject_user=ALICE,
                matter_id=MATTER,
            )
        )
        assert "attribute_not_exists" in written[0]["ConditionExpression"]

    def test_a_replayed_event_key_is_refused_rather_than_merged(self, dynamo, table):
        dynamo.append_event(
            AccessEvent(
                tenant_id=TENANT,
                actor=ADMIN,
                action="SCREEN",
                subject_user=ALICE,
                matter_id=MATTER,
            )
        )
        item = next(v for k, v in table.items.items() if k[1].startswith("EVENT#"))
        with pytest.raises(RuntimeError, match="ConditionalCheckFailed"):
            table.put_item(Item=item, ConditionExpression="attribute_not_exists(PK)")

    def test_two_events_at_the_same_instant_both_survive(self, dynamo):
        """`at` alone is not unique enough to key on, so the uuid is load-bearing: a
        collision would silently drop one audit row."""
        for _ in range(2):
            dynamo.append_event(
                AccessEvent(
                    tenant_id=TENANT,
                    actor=ADMIN,
                    action="SCREEN",
                    subject_user=ALICE,
                    matter_id=MATTER,
                    at="2026-01-02T03:04:05+00:00",
                )
            )
        assert len(dynamo.events_for_user(TENANT, ALICE)) == 2

    def test_pagination_is_followed(self, dynamo, table):
        """A truncated read of a denylist is a privilege breach, so every page is read."""
        pages = [
            {
                "Items": [
                    {
                        "PK": "x",
                        "SK": f"{ 'SCREEN#' }{i}",
                        "tenant_id": TENANT,
                        "user_id": ALICE,
                        "matter_id": f"M-{i}",
                        "reason": REASON,
                        "screened_by": ADMIN,
                        "screened_at": "2026-01-01T00:00:00+00:00",
                    }
                ],
                **({"LastEvaluatedKey": {"PK": "x"}} if i == 0 else {}),
            }
            for i in range(2)
        ]
        table.query = lambda **_: pages.pop(0)  # type: ignore[method-assign]
        assert len(dynamo.screens_for(TENANT, ALICE)) == 2


class TestManagerOverDynamo:
    """The manager's semantics must not change with the store underneath it."""

    def test_a_screen_beats_platform_admin_on_dynamo_too(self, dynamo):
        manager = AccessManager(dynamo)
        manager.assign(TENANT, ADMIN, MATTER, actor=ADMIN)
        manager.screen(TENANT, ADMIN, MATTER, actor=ADMIN, reason=REASON, contact=CONTACT)
        access = manager.resolve(TENANT, ADMIN, is_platform_admin=True)
        assert access.decide(MATTER) is AccessDecision.SCREENED
        assert CONTACT in access.explain(MATTER)

    def test_unassign_on_dynamo_preserves_the_row(self, dynamo, table):
        manager = AccessManager(dynamo)
        manager.assign(TENANT, ALICE, MATTER, actor=ADMIN)
        manager.unassign(TENANT, ALICE, MATTER, actor=ADMIN, reason="rolled off")
        stored = next(v for k, v in table.items.items() if k[1].startswith("ASSIGN#"))
        assert stored["revoked_at"] and stored["revoked_by"] == ADMIN
