"""Append-only record of what was asked, which tier answered, and on what basis.

`graph_audit` answers "who changed what the system believes". This one answers the read side:
**which questions were answered, and from which facts.** A product whose claim is that every
answer has a basis cannot have reads that leave no trace — without this, "what did we tell the
client, and on what evidence?" has no answer, and neither does its inverse, "this fact turned out
to be wrong, which advice rested on it?".

**A separate partition from the graph log rather than a fifth action on it.** Questions outnumber
belief changes by orders of magnitude, so sharing one partition would make "recent graph changes"
page through thousands of questions to find three wipes. Same table, same append-only guarantee,
different question being asked of it.

**Refusals are deliberately not here.** The kill switch already records those via
`Services.record_blocked`, and a refusal produced no answer, so it is a backlog signal — a
governed metric waiting to be written — rather than a basis for advice.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from src.graph_audit import MAX_STORED_IDS, TableLike

if TYPE_CHECKING:
    from src.query.resolver import Resolution

logger = logging.getLogger(__name__)

#: Shares the grants table with the graph log, which is already RETAIN-on-delete because it holds
#: the compliance artifact. Keys: PK = TENANT#{t}#ASKED, SK = ASK#{at}#{uuid}.
ASKED_PK_SUFFIX = "#ASKED"
ASK_PREFIX = "ASK#"

#: How far back the inverse lookup reads. "Which questions used this fact" has no index — one
#: question cites many assertions, and a GSI row per citation would mean dozens of writes on the
#: read path. Bounded scan of recent questions instead, and the endpoint says the window is
#: bounded rather than implying completeness.
MAX_SCAN = 500


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class QueryEvent:
    """One append-only row: who asked what, which tier answered, and on what basis."""

    tenant_id: str
    actor: str
    question: str
    tier: int
    tier_name: str
    governed: bool
    answered: bool
    at: str = field(default_factory=_now)
    sql: str | None = None
    assertion_ids: tuple[str, ...] = ()
    document_ids: tuple[str, ...] = ()
    ids_truncated: bool = False
    """Carried on the read, unlike `GraphEvent`, because the inverse lookup filters on this list:
    a clipped list would silently answer "no questions used this fact"."""

    @property
    def facts_used(self) -> int:
        return len(self.assertion_ids)

    def uses(self, assertion_id: str) -> bool:
        return assertion_id in self.assertion_ids

    def to_dict(self) -> dict[str, Any]:
        return {
            "at": self.at,
            "actor": self.actor,
            "question": self.question,
            "tier": self.tier,
            "tier_name": self.tier_name,
            "governed": self.governed,
            "answered": self.answered,
            "sql": self.sql,
            "assertion_ids": list(self.assertion_ids),
            "document_ids": list(self.document_ids),
            "facts_used": self.facts_used,
            "ids_truncated": self.ids_truncated,
        }


def asked_pk(tenant_id: str) -> str:
    return f"TENANT#{tenant_id}{ASKED_PK_SUFFIX}"


def event_for(tenant_id: str, actor: str, question: str, resolution: Resolution) -> QueryEvent:
    """What to record for one answered question.

    `answered` is separate from `governed`: a tier can be reached, find nothing and return an
    empty resolution, and "we had no answer" is itself part of the record.
    """
    seen: dict[str, None] = {}
    for c in resolution.citations:
        doc = c.get("document_id")
        if doc:
            seen[str(doc)] = None
    return QueryEvent(
        tenant_id=tenant_id,
        actor=actor,
        question=question,
        tier=int(resolution.tier),
        tier_name=resolution.tier.name,
        governed=resolution.is_governed,
        answered=resolution.answer is not None,
        sql=resolution.sql,
        assertion_ids=tuple(resolution.assertions_used),
        document_ids=tuple(seen),
    )


def _item(event: QueryEvent, event_id: str) -> dict[str, Any]:
    return {
        "PK": asked_pk(event.tenant_id),
        # The timestamp sorts and the uuid disambiguates: two people asking in the same
        # millisecond must be two rows, not one overwriting the other.
        "SK": f"{ASK_PREFIX}{event.at}#{event_id}",
        "tenant_id": event.tenant_id,
        "actor": event.actor,
        "question": event.question,
        "tier": event.tier,
        "tier_name": event.tier_name,
        "governed": event.governed,
        "answered": event.answered,
        "at": event.at,
        "sql": event.sql,
        # Capped for the same reason as the graph log: a hybrid answer over a large matter would
        # exceed the 400KB item limit. `facts_used` stays exact so the count is never wrong.
        "assertion_ids": list(event.assertion_ids[:MAX_STORED_IDS]),
        "document_ids": list(event.document_ids[:MAX_STORED_IDS]),
        "facts_used": event.facts_used,
        "ids_truncated": len(event.assertion_ids) > MAX_STORED_IDS,
    }


def _to_event(item: dict[str, Any]) -> QueryEvent:
    return QueryEvent(
        tenant_id=str(item.get("tenant_id", "")),
        actor=str(item.get("actor", "")),
        question=str(item.get("question", "")),
        # DynamoDB hands numbers back as Decimal.
        tier=int(item.get("tier") or 0),
        tier_name=str(item.get("tier_name", "")),
        governed=bool(item.get("governed")),
        answered=bool(item.get("answered")),
        at=str(item.get("at", "")),
        sql=item.get("sql"),
        assertion_ids=tuple(item.get("assertion_ids") or ()),
        document_ids=tuple(item.get("document_ids") or ()),
        ids_truncated=bool(item.get("ids_truncated")),
    )


class InMemoryQueryAudit:
    """Reference store for tests and local dev. Append-only, like the real one."""

    def __init__(self) -> None:
        self._events: list[QueryEvent] = []

    def append(self, event: QueryEvent) -> QueryEvent:
        self._events.append(event)
        return event

    def questions(self, tenant_id: str, *, limit: int = 200) -> list[QueryEvent]:
        mine = [e for e in self._events if e.tenant_id == tenant_id]
        return sorted(mine, key=lambda e: e.at, reverse=True)[:limit]


class QueryAudit:
    """The question log in DynamoDB."""

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

    def append(self, event: QueryEvent) -> QueryEvent:
        """Write one row, or fail.

        The condition is what makes this append-only rather than merely append-shaped: a key that
        already exists is refused instead of overwritten, so no later write can rewrite the record
        of what was asked.
        """
        self.table.put_item(
            Item=_item(event, uuid.uuid4().hex),
            ConditionExpression="attribute_not_exists(PK) AND attribute_not_exists(SK)",
        )
        logger.info(
            "query audit: tier %d (%s) for %s using %d assertions",
            event.tier,
            "governed" if event.governed else "ungoverned",
            event.actor,
            event.facts_used,
        )
        return event

    def questions(self, tenant_id: str, *, limit: int = 200) -> list[QueryEvent]:
        """Newest first. A recent question is the one somebody is asking about."""
        got = self.table.query(
            KeyConditionExpression="PK = :pk AND begins_with(SK, :prefix)",
            ExpressionAttributeValues={":pk": asked_pk(tenant_id), ":prefix": ASK_PREFIX},
            ScanIndexForward=False,
            Limit=limit,
        )
        return [_to_event(i) for i in got.get("Items", [])]

    def drop_tenant(self, tenant_id: str) -> int:
        """Delete this tenant's question log. Only ever as part of deleting the tenant.

        Same reasoning as `GraphAudit.drop_tenant`: append-only protects the record while there
        is a firm it belongs to. Pages, because `questions` is capped for display and a capped
        delete would leave the oldest rows behind.
        """
        pk = asked_pk(tenant_id)
        deleted = 0
        start: dict[str, Any] | None = None
        while True:
            page = self.table.query(
                KeyConditionExpression="PK = :pk AND begins_with(SK, :prefix)",
                ExpressionAttributeValues={":pk": pk, ":prefix": ASK_PREFIX},
                **({"ExclusiveStartKey": start} if start else {}),
            )
            for item in page.get("Items", []):
                self.table.delete_item(Key={"PK": pk, "SK": item["SK"]})
                deleted += 1
            start = page.get("LastEvaluatedKey")
            if not start:
                logger.info("dropped %d query audit rows for %s", deleted, tenant_id)
                return deleted
