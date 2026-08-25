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
    from src.agent.loop import RunResult
    from src.query.planner import ComposedAnswer
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

#: Which way in produced a row. A person on the Ask page and an agent driving the MCP are not the
#: same event, and a log that cannot tell them apart cannot answer who was answered by what.
SURFACE_QUERY = "query"
SURFACE_COMPOSE = "compose"
SURFACE_AGENT = "retrieval_agent"
SURFACE_MCP = "mcp"
"""A third-party agent calling the tools directly. Distinct from `retrieval_agent`, which is our
own loop and records the whole run: an MCP row is one tool call with nobody claiming the answer."""


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

    surface: str = SURFACE_QUERY
    """Which way in was used. Without it the log cannot answer "was this a person on the Ask page
    or an agent driving the MCP", and those carry different weight when advice is questioned."""

    governance: str = ""
    """What the answer may be called, in the words the UI shows: `ComposedAnswer.governance_label`.

    `governed` cannot express a composed answer. A run mixing a compiled metric with a model's
    reading is neither governed nor ungoverned, and flattening it to one boolean would record a
    claim the answer itself refuses to make. Empty for a single-tier `Resolution`, where the
    boolean is the whole truth."""

    run_id: str | None = None
    """Ties the row to the transcript the agent streamed, so a recorded run stays inspectable."""

    tools_called: tuple[str, ...] = ()
    """The tool ladder in call order, with repeats: an agent calling `compose` three times is a
    different run from one calling it once, and the count is the thing a reader wants."""

    @property
    def facts_used(self) -> int:
        return len(self.assertion_ids)

    def uses(self, assertion_id: str) -> bool:
        return assertion_id in self.assertion_ids

    @property
    def basis(self) -> str:
        """What to call the basis of this answer, whichever surface produced it."""
        if self.governance:
            return self.governance
        return "governed" if self.governed else "ungoverned"

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
            "surface": self.surface,
            "governance": self.governance,
            "basis": self.basis,
            "run_id": self.run_id,
            "tools_called": list(self.tools_called),
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
        surface=SURFACE_QUERY,
    )


def event_for_composed(
    tenant_id: str,
    actor: str,
    question: str,
    answer: ComposedAnswer,
    *,
    surface: str = SURFACE_COMPOSE,
    run_id: str | None = None,
    tools_called: tuple[str, ...] = (),
) -> QueryEvent:
    """What to record for one composed answer.

    Separate from `event_for` because a `ComposedAnswer` has no single tier and no single basis.
    `tier` is the **highest** that ran: the cap a reader wants to check is the furthest the answer
    reached, and reporting the lowest would understate it. The basis goes in `governance` as the
    label the answer itself computes, never flattened into `governed` -- see `QueryEvent.governance`.

    `governed` is still set, because the inverse lookup and the old rows share this field, but it
    is the narrow claim `is_fully_deterministic` makes rather than a guess from the tier.
    """
    seen: dict[str, None] = {}
    assertion_ids: dict[str, None] = {}
    sql: str | None = None
    tiers: list[int] = []

    for part in answer.parts:
        tiers.append(int(part.tier))
        sql = sql or part.sql
        for aid in part.assertion_ids:
            assertion_ids[aid] = None
        for citation in part.citations:
            doc = citation.get("document_id")
            if doc:
                seen[str(doc)] = None

    tier = max(tiers) if tiers else 0
    return QueryEvent(
        tenant_id=tenant_id,
        actor=actor,
        question=question,
        tier=tier,
        tier_name=_tier_name(tier),
        governed=answer.is_fully_deterministic,
        # A part that ran and found nothing is still an answer attempt, so this asks whether any
        # lane produced content rather than whether prose was written over it.
        answered=bool(answer.parts),
        sql=sql,
        assertion_ids=tuple(assertion_ids),
        document_ids=tuple(seen),
        surface=surface,
        governance=answer.governance_label,
        run_id=run_id,
        tools_called=tools_called,
    )


def _tier_name(tier: int) -> str:
    from src.query.resolver import Tier

    try:
        return Tier(tier).name
    except ValueError:
        return ""


def event_for_run(tenant_id: str, actor: str, question: str, result: RunResult) -> QueryEvent:
    """What to record for one agent run: **one row, carrying the tools it called.**

    Not a row per tool call. A run is one question with one answer, and eight rows for eight
    calls would make the log read as eight pieces of advice; `tools_called` keeps the ladder
    without inflating the count.

    Derived from the transcript rather than from a returned object, because that is all a run
    leaves behind -- the loop hands the model tool results and never assembles a `ComposedAnswer`.
    So the basis is read back out of the `compose` and `ask` results the agent actually received,
    which has the useful property that a row can only claim evidence the agent was really shown.
    """
    tools: list[str] = []
    assertion_ids: dict[str, None] = {}
    document_ids: dict[str, None] = {}
    labels: dict[str, None] = {}
    tiers: list[int] = []
    sql: str | None = None
    answered = False

    for event in result.events:
        kind = event.get("kind")
        if kind == "tool_call":
            tools.append(str(event.get("tool", "")))
            continue
        if kind != "tool_result" or event.get("is_error"):
            continue
        payload = event.get("result")
        if not isinstance(payload, dict):
            continue

        governance = payload.get("governance")
        if isinstance(governance, str) and governance:
            labels[governance] = None

        for part in payload.get("parts") or []:
            if not isinstance(part, dict):
                continue
            answered = True
            if isinstance(part.get("tier"), int):
                tiers.append(part["tier"])
            sql = sql or part.get("sql")
            for aid in part.get("assertion_ids") or []:
                assertion_ids[str(aid)] = None
            for citation in part.get("citations") or []:
                if isinstance(citation, dict) and citation.get("document_id"):
                    document_ids[str(citation["document_id"])] = None

        # `ask` returns a single-tier resolution, a different shape with the same standing.
        if isinstance(payload.get("tier"), int) and "parts" not in payload:
            tiers.append(payload["tier"])
            answered = answered or payload.get("answer") is not None
            sql = sql or payload.get("sql")
            for aid in payload.get("assertions_used") or []:
                assertion_ids[str(aid)] = None
            for citation in payload.get("citations") or []:
                if isinstance(citation, dict) and citation.get("document_id"):
                    document_ids[str(citation["document_id"])] = None

    tier = max(tiers) if tiers else 0
    return QueryEvent(
        tenant_id=tenant_id,
        actor=actor,
        question=question,
        tier=tier,
        tier_name=_tier_name(tier),
        # Never governed. The agent's prose is its own, written over whatever the tools returned,
        # so even a run that only touched compiled metrics produced an ungoverned answer.
        governed=False,
        answered=answered,
        sql=sql,
        assertion_ids=tuple(assertion_ids),
        document_ids=tuple(document_ids),
        surface=SURFACE_AGENT,
        governance=_run_governance(sorted(labels), capped=result.was_capped),
        run_id=result.run_id,
        tools_called=tuple(tools),
    )


def _run_governance(labels: list[str], *, capped: bool) -> str:
    """The basis of an agent run, always ending in the agent's own writing.

    Every label the tools reported is kept rather than reduced to the weakest. A run that read a
    compiled metric and a model's reading rested on both, and the reader deciding whether to trust
    the answer needs to see the mixture, not its floor.
    """
    base = " ; ".join(labels) if labels else "no governed source"
    label = f"{base} + written by agent"
    return f"{label} (stopped at a cap)" if capped else label


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
        "surface": event.surface,
        "governance": event.governance,
        "run_id": event.run_id,
        # Capped like the id lists: a capped agent can call tools a dozen times and the ladder is
        # readable long before that.
        "tools_called": list(event.tools_called[:MAX_STORED_IDS]),
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
        # A row written before these existed is an Ask row, which is what it was.
        surface=str(item.get("surface") or SURFACE_QUERY),
        governance=str(item.get("governance") or ""),
        run_id=item.get("run_id") or None,
        tools_called=tuple(item.get("tools_called") or ()),
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
            "query audit: %s tier %d (%s) for %s using %d assertions",
            event.surface,
            event.tier,
            event.basis,
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
