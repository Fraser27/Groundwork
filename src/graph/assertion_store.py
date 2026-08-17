"""Assertions in Neptune, so an approval survives a deploy.

Satisfies the same `AssertionStore` protocol as `InMemoryAssertionStore`, which is why this is
a drop-in: `ReviewQueue` is unchanged, and the in-memory store stays the reference
implementation for tests and local dev.

The gap this closes was the largest one in the system. `promote()` set a lifecycle flag on a
record in a Python dict, so a reviewer could approve a fact, see it go live, and lose it on
the next deploy — while the UI reported success. Every other durable thing was already durable
(documents in S3, jobs and grants in DynamoDB, metric definitions in the graph); assertions,
the actual product, were not.

Writes are node-then-edge-then-premises, in that order and deliberately:

1. The `:Assertion` node is the record. It must exist before anything references it.
2. The traversable edge carries the trust fields `scope.edge_scope` filters on.
3. `PREMISE` links come last, because a premise may be in the same batch and MATCH on both
   ends means a premise that never arrives leaves no edge rather than a dangling one.

Reads go through `GraphClient.read_scoped`, which refuses a query without a `{scope}` token.
Tenant isolation here is a property filter, so that refusal is the mechanism, not a
convention.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from src.documents.review import AssertionRecord, Lifecycle
from src.graph import assertion_queries as q
from src.graph.assertions import Assertion, EpistemicClass, ReviewState, SourceLocator
from src.graph.client import GraphClient
from src.graph.scope import AuthContext, edge_scope

logger = logging.getLogger(__name__)

#: Read cap. A tenant list is an operator/UI surface, not a bulk export; an unbounded read on
#: a firm with a million assertions is a timeout dressed up as a query.
DEFAULT_LIMIT = 5000


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _to_item(record: AssertionRecord) -> dict[str, Any]:
    """Flatten a record into Cypher parameters.

    Nested structures are JSON-encoded because **Neptune rejects list- and map-valued
    properties** ("Property value must be a simple literal") where Neo4j accepts them. That
    constraint already bit the metric store; it applies identically here to `premises` and
    the source locator.

    The locator's three most-read fields are also flattened out as their own properties, so
    "which page did this come from" does not require parsing JSON in the caller.
    """
    a = record.assertion
    loc = a.source_locator
    return {
        "tenant_id": a.tenant_id,
        "assertion_id": a.assertion_id,
        "subject_id": a.subject_id,
        "predicate": a.predicate,
        "object_id": a.object_id,
        "epistemic_class": a.epistemic_class.value,
        "method": a.method,
        "confidence": a.confidence,
        "raw_confidence": a.raw_confidence,
        "matter_id": a.matter_id,
        "review_state": a.review_state.value,
        "reviewed_by": a.reviewed_by,
        "reviewed_at": a.reviewed_at,
        "rule_id": a.rule_id,
        "rule_version": a.rule_version,
        "valid_from": a.valid_from,
        "valid_until": a.valid_until,
        "recorded_at": a.recorded_at,
        "superseded_at": a.superseded_at,
        "lifecycle": record.lifecycle.value,
        # Record-level fields, not assertion fields. `job_id` is load-bearing rather than
        # informational: `promote(job_id=...)` filters on it, so a store that dropped it made
        # every scoped promotion a no-op -- seven auto-asserted facts sat STAGED in production,
        # unpromotable, because nothing else promotes and nothing approves what needs no approval.
        "job_id": record.job_id,
        "review_note": record.review_note,
        "retracted_reason": record.retracted_reason,
        "retracted_by": record.retracted_by,
        "corrects": record.corrects,
        "source_locator_json": json.dumps(loc.to_dict()),
        "premises_json": json.dumps(list(a.premises)),
        "document_id": loc.document_id,
        "filename": loc.filename,
        "page": loc.page,
        "quote": loc.quote,
    }


def _from_node(node: dict[str, Any]) -> AssertionRecord:
    """Rebuild a record from a stored node.

    Reconstructed through `Assertion(...)` directly rather than `build_assertion`: the
    invariants were enforced when the fact was created, and re-running them on read would
    reject a fact whose *pack* has since changed — a vocabulary edit would silently make
    history unreadable. `assertion_id` is passed explicitly so it is never recomputed.
    """
    loc_raw = node.get("source_locator_json") or "{}"
    locator = SourceLocator(**json.loads(loc_raw))
    premises = tuple(json.loads(node.get("premises_json") or "[]"))

    assertion = Assertion(
        tenant_id=str(node["tenant_id"]),
        subject_id=str(node["subject_id"]),
        predicate=str(node["predicate"]),
        object_id=str(node["object_id"]),
        epistemic_class=EpistemicClass(node["epistemic_class"]),
        method=str(node["method"]),
        confidence=float(node["confidence"]),
        source_locator=locator,
        matter_id=node.get("matter_id"),
        raw_confidence=(
            float(raw) if (raw := node.get("raw_confidence")) is not None else None
        ),
        premises=premises,
        rule_id=node.get("rule_id"),
        rule_version=node.get("rule_version"),
        valid_from=node.get("valid_from"),
        valid_until=node.get("valid_until"),
        recorded_at=str(node.get("recorded_at") or ""),
        superseded_at=node.get("superseded_at"),
        review_state=ReviewState(node.get("review_state") or ReviewState.PENDING.value),
        reviewed_by=node.get("reviewed_by"),
        reviewed_at=node.get("reviewed_at"),
        assertion_id=str(node["assertion_id"]),
    )
    return AssertionRecord(
        assertion=assertion,
        lifecycle=Lifecycle(node.get("lifecycle") or Lifecycle.STAGED.value),
        job_id=node.get("job_id"),
        review_note=node.get("review_note"),
        retracted_reason=node.get("retracted_reason"),
        retracted_by=node.get("retracted_by"),
        corrects=node.get("corrects"),
    )


@dataclass
class GraphAssertionStore:
    """Neptune-backed assertion store.

    `admin_ctx` builds the scope for reads that are not on a request path — a store method
    takes a tenant id, not an `AuthContext`, because the matter wall is applied by
    `ReviewQueue.visible` above this layer. Reads here are tenant-scoped and deliberately
    include staged and pending facts: the review queue exists to show exactly those.
    """

    graph: GraphClient
    limit: int = DEFAULT_LIMIT

    def _scope(self, tenant_id: str):
        # Trust filters are off for a store read, not weakened: `visible()` applies the matter
        # wall, and a review queue that hid unreviewed low-confidence claims would hide its
        # own reason for existing. `min_confidence=0.0` with every class admitted is the
        # honest expression of "everything this tenant holds".
        ctx = AuthContext(user_id="system", tenant_id=tenant_id, include_suggestions=True)
        return edge_scope(
            ctx,
            edge_var="a",
            min_confidence=0.0,
            trusted_classes=frozenset(EpistemicClass),
            include_pending=True,
        )

    def put(self, record: AssertionRecord) -> None:
        item = _to_item(record)
        self.graph.write(q.UPSERT_ASSERTION_NODE, item)
        self.graph.write(q.upsert_edge(record.assertion.predicate), item)

        for premise_id in record.assertion.premises:
            self.graph.write(
                q.LINK_PREMISES,
                {
                    "tenant_id": record.assertion.tenant_id,
                    "assertion_id": record.assertion_id,
                    "premise_id": premise_id,
                },
            )

    def get(self, tenant_id: str, assertion_id: str) -> AssertionRecord | None:
        rows = self.graph.read_scoped(
            q.GET_ASSERTION, self._scope(tenant_id), {"assertion_id": assertion_id}
        )
        return _from_node(rows[0]["a"]) if rows else None

    def all_for_tenant(self, tenant_id: str) -> list[AssertionRecord]:
        rows = self.graph.read_scoped(
            q.LIST_ASSERTIONS, self._scope(tenant_id), {"limit": self.limit}
        )
        return [_from_node(r["a"]) for r in rows]

    def dependents_of(self, tenant_id: str, assertion_id: str) -> list[AssertionRecord]:
        """Assertions resting on this one, for the retraction cascade.

        The `PREMISE` edge walked backwards, which is what the in-memory store's reverse index
        exists to simulate.
        """
        rows = self.graph.read_scoped(
            q.DEPENDENTS_OF, self._scope(tenant_id), {"assertion_id": assertion_id}
        )
        return [_from_node(r["a"]) for r in rows]

    def drop_tenant(self, tenant_id: str) -> int:
        """Delete every assertion, edge and entity for one tenant.

        For a rebuild, not a correction: a withdrawal supersedes and leaves a trail, but these
        facts are about to be re-derived from S3, so recording retractions would log an event
        that did not happen. Defensible only because the documents outlive this.
        """
        existing = self.graph.read_scoped(
            q.LIST_ASSERTIONS, self._scope(tenant_id), {"limit": self.limit}
        )
        for statement in (
            q.DELETE_TENANT_EDGES,
            q.DELETE_TENANT_ASSERTIONS,
            q.DELETE_TENANT_ENTITIES,
        ):
            self.graph.write(statement, {"tenant_id": tenant_id})
        logger.info("dropped %d assertions for %s", len(existing), tenant_id)
        return len(existing)

    def set_review_state(self, record: AssertionRecord) -> None:
        """Write a review decision to both the node and the edge.

        Both, because they are read by different paths: `edge_scope` filters traversals on the
        edge copy, so an approval that updated only the node would leave the fact approved and
        invisible to every query that matters.
        """
        a = record.assertion
        self.graph.write(
            q.SET_REVIEW_STATE,
            {
                "tenant_id": a.tenant_id,
                "assertion_id": a.assertion_id,
                "review_state": a.review_state.value,
                "reviewed_by": a.reviewed_by,
                "reviewed_at": a.reviewed_at,
                "lifecycle": record.lifecycle.value,
        # Record-level fields, not assertion fields. `job_id` is load-bearing rather than
        # informational: `promote(job_id=...)` filters on it, so a store that dropped it made
        # every scoped promotion a no-op -- seven auto-asserted facts sat STAGED in production,
        # unpromotable, because nothing else promotes and nothing approves what needs no approval.
        "job_id": record.job_id,
        "review_note": record.review_note,
        "retracted_reason": record.retracted_reason,
        "retracted_by": record.retracted_by,
        "corrects": record.corrects,
            },
        )

    def supersede(
        self,
        tenant_id: str,
        assertion_id: str,
        *,
        review_state: str | None = None,
        lifecycle: str | None = None,
        at: str | None = None,
    ) -> None:
        """Close a fact rather than delete it, on both copies."""
        self.graph.write(
            q.SUPERSEDE_ASSERTION,
            {
                "tenant_id": tenant_id,
                "assertion_id": assertion_id,
                "at": at or _now(),
                "review_state": review_state,
                "lifecycle": lifecycle,
            },
        )
