"""DynamoDB persistence for matter assignments, screens and the audit trail.

`src/access.py` owns the semantics; this module only stores them and reads them back
faithfully. Faithfully is the whole job: a dropped `reason` or `contact` degrades a
wall's explanation to "you are screened from something", which is the failure the
loud-screen design exists to avoid.

Single table, keyed by the two questions asked on a request path:

    PK = TENANT#{t}#USER#{u}     SK = ASSIGN#{matter} | SCREEN#{matter} | EVENT#{at}#{uuid}
    GSI1PK = TENANT#{t}#MATTER#{m}   GSI1SK = ASSIGN#{user} | SCREEN#{user} | EVENT#{at}#{uuid}

The GSI is not a convenience. "Who is on this matter" has to be a query — a scan on an
authorization path would degrade with tenant size and eventually time out, and a store
that fails open under load is not a store you can put a wall behind.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Callable
from typing import Any, Protocol

from src.access import AccessEvent, MatterAssignment, MatterScreen

logger = logging.getLogger(__name__)

#: Must match the index name in `cdk/lib/data-stack.ts`.
GSI1 = "GSI1"

ASSIGN = "ASSIGN#"
SCREEN = "SCREEN#"
EVENT = "EVENT#"


class TableLike(Protocol):
    """The slice of a boto3 DynamoDB `Table` resource this module uses.

    Deliberately narrow — there is no `scan`, so no future edit can quietly add one.
    """

    def put_item(self, **kwargs: Any) -> dict[str, Any]: ...
    def query(self, **kwargs: Any) -> dict[str, Any]: ...


def user_pk(tenant_id: str, user_id: str) -> str:
    return f"TENANT#{tenant_id}#USER#{user_id}"


def matter_pk(tenant_id: str, matter_id: str) -> str:
    return f"TENANT#{tenant_id}#MATTER#{matter_id}"


def _assignment_item(a: MatterAssignment) -> dict[str, Any]:
    return {
        "PK": user_pk(a.tenant_id, a.user_id),
        "SK": f"{ASSIGN}{a.matter_id}",
        "GSI1PK": matter_pk(a.tenant_id, a.matter_id),
        "GSI1SK": f"{ASSIGN}{a.user_id}",
        "tenant_id": a.tenant_id,
        "user_id": a.user_id,
        "matter_id": a.matter_id,
        "granted_by": a.granted_by,
        "granted_at": a.granted_at,
        "role": a.role,
        "revoked_at": a.revoked_at,
        "revoked_by": a.revoked_by,
    }


def _to_assignment(item: dict[str, Any]) -> MatterAssignment:
    return MatterAssignment(
        tenant_id=item["tenant_id"],
        user_id=item["user_id"],
        matter_id=item["matter_id"],
        granted_by=item["granted_by"],
        granted_at=item["granted_at"],
        role=item.get("role") or "member",
        revoked_at=item.get("revoked_at"),
        revoked_by=item.get("revoked_by"),
    )


def _screen_item(s: MatterScreen) -> dict[str, Any]:
    return {
        "PK": user_pk(s.tenant_id, s.user_id),
        "SK": f"{SCREEN}{s.matter_id}",
        "GSI1PK": matter_pk(s.tenant_id, s.matter_id),
        "GSI1SK": f"{SCREEN}{s.user_id}",
        "tenant_id": s.tenant_id,
        "user_id": s.user_id,
        "matter_id": s.matter_id,
        "reason": s.reason,
        "screened_by": s.screened_by,
        "screened_at": s.screened_at,
        "contact": s.contact,
        "lifted_at": s.lifted_at,
        "lifted_by": s.lifted_by,
    }


def _to_screen(item: dict[str, Any]) -> MatterScreen:
    return MatterScreen(
        tenant_id=item["tenant_id"],
        user_id=item["user_id"],
        matter_id=item["matter_id"],
        reason=item["reason"],
        screened_by=item["screened_by"],
        screened_at=item["screened_at"],
        contact=item.get("contact"),
        lifted_at=item.get("lifted_at"),
        lifted_by=item.get("lifted_by"),
    )


def _event_item(e: AccessEvent, event_id: str) -> dict[str, Any]:
    sk = f"{EVENT}{e.at}#{event_id}"
    return {
        "PK": user_pk(e.tenant_id, e.subject_user),
        "SK": sk,
        "GSI1PK": matter_pk(e.tenant_id, e.matter_id),
        "GSI1SK": sk,
        "event_id": event_id,
        "tenant_id": e.tenant_id,
        "actor": e.actor,
        "action": e.action,
        "subject_user": e.subject_user,
        "matter_id": e.matter_id,
        "at": e.at,
        "reason": e.reason,
        # JSON rather than a map: `detail` is `dict[str, Any]` and the resource
        # serialiser rejects floats outright, so an audit write would fail on a value
        # nobody thought about. A refused audit write is the worst failure here.
        "detail": json.dumps(e.detail, sort_keys=True),
    }


def _to_event(item: dict[str, Any]) -> AccessEvent:
    raw = item.get("detail")
    return AccessEvent(
        tenant_id=item["tenant_id"],
        actor=item["actor"],
        action=item["action"],
        subject_user=item["subject_user"],
        matter_id=item["matter_id"],
        at=item["at"],
        reason=item.get("reason"),
        detail=json.loads(raw) if isinstance(raw, str) and raw else {},
    )


class DynamoAccessStore:
    """`AccessStore` over one DynamoDB table.

    boto3 is reached lazily and injectable, so tests need no AWS.
    """

    def __init__(
        self,
        table_name: str = "",
        *,
        table: TableLike | None = None,
        table_factory: Callable[[], TableLike] | None = None,
        index_name: str = GSI1,
    ) -> None:
        self.table_name = table_name
        self.index_name = index_name
        self._table = table
        self._table_factory = table_factory

    @property
    def table(self) -> TableLike:
        if self._table is None:
            factory = self._table_factory
            if factory is None:
                import boto3

                name = self.table_name
                # The resource API rather than the client: it hands back plain Python
                # values, so nothing here has to speak AttributeValue.
                factory = lambda: boto3.resource("dynamodb").Table(name)  # noqa: E731
            self._table = factory()
        return self._table

    def assignments_for(self, tenant_id: str, user_id: str) -> list[MatterAssignment]:
        items = self._by_user(tenant_id, user_id, ASSIGN)
        return [a for a in (_to_assignment(i) for i in items) if a.is_active]

    def screens_for(self, tenant_id: str, user_id: str) -> list[MatterScreen]:
        items = self._by_user(tenant_id, user_id, SCREEN)
        return [s for s in (_to_screen(i) for i in items) if s.is_active]

    def team_of(self, tenant_id: str, matter_id: str) -> list[MatterAssignment]:
        items = self._by_matter(tenant_id, matter_id, ASSIGN)
        return [a for a in (_to_assignment(i) for i in items) if a.is_active]

    def screens_on(self, tenant_id: str, matter_id: str) -> list[MatterScreen]:
        """Everyone currently screened from a matter. Beyond the Protocol, for Admin."""
        items = self._by_matter(tenant_id, matter_id, SCREEN)
        return [s for s in (_to_screen(i) for i in items) if s.is_active]

    def put_assignment(self, assignment: MatterAssignment) -> None:
        self.table.put_item(Item=_assignment_item(assignment))

    def put_screen(self, screen: MatterScreen) -> None:
        self.table.put_item(Item=_screen_item(screen))

    def append_event(self, event: AccessEvent) -> None:
        """Write one audit row, or fail.

        The condition is what makes this append-only rather than merely
        append-shaped: an event id that already exists is refused instead of
        overwritten, so no later write can rewrite the record of who screened whom.
        """
        item = _event_item(event, uuid.uuid4().hex)
        self.table.put_item(
            Item=item,
            ConditionExpression="attribute_not_exists(PK) AND attribute_not_exists(SK)",
        )

    def events_for_matter(self, tenant_id: str, matter_id: str) -> list[AccessEvent]:
        return [_to_event(i) for i in self._by_matter(tenant_id, matter_id, EVENT)]

    def events_for_user(self, tenant_id: str, user_id: str) -> list[AccessEvent]:
        return [_to_event(i) for i in self._by_user(tenant_id, user_id, EVENT)]

    def _by_user(self, tenant_id: str, user_id: str, prefix: str) -> list[dict[str, Any]]:
        return self._query(
            KeyConditionExpression="PK = :pk AND begins_with(SK, :prefix)",
            ExpressionAttributeValues={":pk": user_pk(tenant_id, user_id), ":prefix": prefix},
        )

    def _by_matter(self, tenant_id: str, matter_id: str, prefix: str) -> list[dict[str, Any]]:
        return self._query(
            IndexName=self.index_name,
            KeyConditionExpression="GSI1PK = :pk AND begins_with(GSI1SK, :prefix)",
            ExpressionAttributeValues={
                ":pk": matter_pk(tenant_id, matter_id),
                ":prefix": prefix,
            },
        )

    def _query(self, **kwargs: Any) -> list[dict[str, Any]]:
        """Every page. A truncated read of a denylist is a privilege breach."""
        items: list[dict[str, Any]] = []
        start: dict[str, Any] | None = None
        while True:
            page = self.table.query(
                **kwargs, **({"ExclusiveStartKey": start} if start else {})
            )
            items.extend(page.get("Items", []))
            start = page.get("LastEvaluatedKey")
            if not start:
                return items
