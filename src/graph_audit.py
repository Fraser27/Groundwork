"""Append-only record of who changed the graph, and why.

Distinct from the access audit in `access_dynamo`, which answers "who could read what". This one
answers **"who changed what the system believes"** — a reviewer overriding a model, an
administrator wiping a document. Different question, different key shape: that store is keyed by
user and matter because a screen is about a person, and these events are about a *document*, a
*matter*, or a single assertion.

**Nothing here is ever deleted, and neither is anything it describes.** A wipe sets
`superseded_at` on the assertions rather than removing them, so:

- the current graph stops returning them, which is what "deleted" means to a user;
- an `as_of` read before the timestamp still reconstructs them, which is what makes
  "what did the file show when we advised?" answerable after a correction;
- this log records who did it and when, so the deletion itself is part of the record rather
  than a gap in it.

**No cascade, deliberately, and this is the interesting decision.** When a premise is wiped, the
inference resting on it is left standing rather than retracted. That looks wrong and is not: the
conclusion *was* true, drawn from evidence that was present at the time, and the proof tree still
resolves because the premises are superseded rather than gone. Retracting the conclusion would
assert that the firm never held the belief, which is false and is exactly the kind of rewriting a
compliance record must not do. `retract.retract` remains the tool for the different case —
withdrawing a belief because it was *wrong* — where the cascade is correct.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

logger = logging.getLogger(__name__)

#: Shares the grants table, which already exists and is already RETAIN-on-delete because it holds
#: the compliance artifact. Keys: PK = TENANT#{t}#GRAPH, SK = EVENT#{at}#{uuid}. One partition per
#: tenant, sorted by time, because "what happened to this firm's graph, newest first" is the only
#: question asked of it.
GRAPH_PK_SUFFIX = "#GRAPH"
EVENT_PREFIX = "EVENT#"

#: Actions. Closed, because an audit log whose vocabulary drifts cannot be filtered reliably.
SUPERSEDE = "SUPERSEDE"
"""A reviewer replaced a model's claim with their own."""

WIPE_DOCUMENT = "WIPE_DOCUMENT"
"""Everything derived from one document was withdrawn from the current graph."""

WIPE_MATTER = "WIPE_MATTER"
"""The same, for every document on a matter."""

ACTIONS = frozenset({SUPERSEDE, WIPE_DOCUMENT, WIPE_MATTER})

#: How many assertion ids to store on one event. A DynamoDB item is capped at 400KB and a wipe of
#: a large matter would exceed it. `affected` stays exact, so the count is never wrong even when
#: the list is clipped.
MAX_STORED_IDS = 200


def _now() -> str:
    return datetime.now(UTC).isoformat()


class TableLike(Protocol):
    def put_item(self, **kwargs: Any) -> dict[str, Any]: ...
    def query(self, **kwargs: Any) -> dict[str, Any]: ...


@dataclass(frozen=True)
class GraphEvent:
    """One append-only row: who changed the graph, what, and why."""

    tenant_id: str
    actor: str
    action: str
    at: str = field(default_factory=_now)
    document_id: str | None = None
    matter_id: str | None = None
    assertion_ids: tuple[str, ...] = ()
    reason: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def affected(self) -> int:
        return len(self.assertion_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "at": self.at,
            "actor": self.actor,
            "action": self.action,
            "document_id": self.document_id,
            "matter_id": self.matter_id,
            "assertion_ids": list(self.assertion_ids),
            "affected": self.affected,
            "reason": self.reason,
            "detail": dict(self.detail),
        }


def graph_pk(tenant_id: str) -> str:
    return f"TENANT#{tenant_id}{GRAPH_PK_SUFFIX}"


def _item(event: GraphEvent, event_id: str) -> dict[str, Any]:
    return {
        "PK": graph_pk(event.tenant_id),
        # The timestamp sorts and the uuid disambiguates: two wipes in the same millisecond must
        # be two rows, not one overwriting the other.
        "SK": f"{EVENT_PREFIX}{event.at}#{event_id}",
        "tenant_id": event.tenant_id,
        "actor": event.actor,
        "action": event.action,
        "at": event.at,
        "document_id": event.document_id,
        "matter_id": event.matter_id,
        # Stored so the exact set is recoverable. Capped: a wipe of ten thousand assertions
        # should not produce an item DynamoDB refuses, and the count remains exact either way.
        "assertion_ids": list(event.assertion_ids[:MAX_STORED_IDS]),
        "affected": event.affected,
        "ids_truncated": len(event.assertion_ids) > MAX_STORED_IDS,
        "reason": event.reason,
        "detail": event.detail,
    }


def _to_event(item: dict[str, Any]) -> GraphEvent:
    return GraphEvent(
        tenant_id=str(item.get("tenant_id", "")),
        actor=str(item.get("actor", "")),
        action=str(item.get("action", "")),
        at=str(item.get("at", "")),
        document_id=item.get("document_id"),
        matter_id=item.get("matter_id"),
        assertion_ids=tuple(item.get("assertion_ids") or ()),
        reason=item.get("reason"),
        detail=dict(item.get("detail") or {}),
    )


class InMemoryGraphAudit:
    """Reference store for tests and local dev. Append-only, like the real one."""

    def __init__(self) -> None:
        self._events: list[GraphEvent] = []

    def append(self, event: GraphEvent) -> GraphEvent:
        if event.action not in ACTIONS:
            raise ValueError(f"unknown graph audit action {event.action!r}")
        self._events.append(event)
        return event

    def events(self, tenant_id: str, *, limit: int = 200) -> list[GraphEvent]:
        mine = [e for e in self._events if e.tenant_id == tenant_id]
        return sorted(mine, key=lambda e: e.at, reverse=True)[:limit]


class GraphAudit:
    """The audit log in DynamoDB."""

    def __init__(
        self,
        table_name: str = "",
        *,
        table: TableLike | None = None,
        table_factory: Callable[[], TableLike] | None = None,
    ) -> None:
        self.table_name = table_name
        self._table = table
        self._table_factory = table_factory

    @property
    def table(self) -> TableLike:
        if self._table is None:
            factory = self._table_factory
            if factory is None:
                import boto3

                name = self.table_name
                factory = lambda: boto3.resource("dynamodb").Table(name)
            self._table = factory()
        return self._table

    def append(self, event: GraphEvent) -> GraphEvent:
        """Write one row, or fail.

        The condition is what makes this append-only rather than merely append-shaped: a key that
        already exists is refused instead of overwritten, so no later write can rewrite the record
        of who deleted what.
        """
        if event.action not in ACTIONS:
            raise ValueError(f"unknown graph audit action {event.action!r}")
        self.table.put_item(
            Item=_item(event, uuid.uuid4().hex),
            ConditionExpression="attribute_not_exists(PK) AND attribute_not_exists(SK)",
        )
        logger.info(
            "graph audit: %s %s by %s affecting %d assertions",
            event.action,
            event.document_id or event.matter_id or "-",
            event.actor,
            event.affected,
        )
        return event

    def events(self, tenant_id: str, *, limit: int = 200) -> list[GraphEvent]:
        """Newest first. A recent change is the one somebody is asking about."""
        got = self.table.query(
            KeyConditionExpression="PK = :pk AND begins_with(SK, :prefix)",
            ExpressionAttributeValues={":pk": graph_pk(tenant_id), ":prefix": EVENT_PREFIX},
            ScanIndexForward=False,
            Limit=limit,
        )
        return [_to_event(i) for i in got.get("Items", [])]
