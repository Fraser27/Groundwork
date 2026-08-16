"""Withdraw everything derived from one document, or one matter, without losing the record.

Two operations that a firm actually needs and that `Reset` is far too blunt for: re-extracting a
single document after improving an extractor, and clearing a matter that should not have been
loaded. Reset takes the whole tenant.

**Soft, always.** Assertions are marked `superseded_at` rather than deleted, so:

- the current graph stops returning them, which is what "deleted" means to a user;
- an `as_of` read before that timestamp still reconstructs them, so "what did the file show when
  we advised?" survives the deletion;
- `graph_audit` records who did it and when, so the deletion is part of the record rather than a
  gap in it.

**No cascade, and this is the decision worth defending.** An inference resting on a wiped premise
is left standing. That looks like a bug and is the opposite: the conclusion *was* drawn, from
evidence that was present at the time, and the proof tree still resolves because the premises are
superseded rather than removed. Retracting it would assert the firm never held the belief, which
is false — and rewriting history to look tidier is precisely what a compliance record must not do.
`retract.retract` is still the right tool for the different case: withdrawing a belief because it
was *wrong*, where cascading is correct because the conclusion never should have been drawn.

Vectors are the exception: those are deleted outright. A vector is not a claim about anything, it
is a derived index entry, and there is no such thing as an audit trail of an embedding. It is
rebuilt by replaying the document.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from src.documents.review import AssertionRecord, Lifecycle, ReviewQueue
from src.graph.scope import AuthContext
from src.graph_audit import WIPE_DOCUMENT, WIPE_MATTER, GraphEvent

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class WipeReport:
    """What a wipe did, in terms an administrator can check."""

    scope: str
    target: str
    assertions_superseded: int = 0
    vectors_deleted: int = 0
    jobs_dropped: int = 0
    documents: tuple[str, ...] = ()
    errors: list[str] = field(default_factory=list)
    at: str = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "target": self.target,
            "assertions_superseded": self.assertions_superseded,
            "vectors_deleted": self.vectors_deleted,
            "jobs_dropped": self.jobs_dropped,
            "documents": list(self.documents),
            "errors": self.errors,
            "at": self.at,
            "note": (
                "Withdrawn from the current graph, not deleted. An as-of read before this "
                "timestamp still shows these facts, and the Audit page records who withdrew "
                "them. Inferences drawn from them are left standing: they were true when "
                "drawn, and their premises are still resolvable."
            ),
        }


def _records_for_document(
    queue: ReviewQueue, ctx: AuthContext, document_id: str
) -> list[AssertionRecord]:
    return [
        r
        for r in queue.visible(ctx)
        if r.assertion.source_locator.document_id == document_id and r.is_current
    ]


def _records_for_matter(
    queue: ReviewQueue, ctx: AuthContext, matter_id: str
) -> list[AssertionRecord]:
    return [r for r in queue.visible(ctx) if r.assertion.matter_id == matter_id and r.is_current]


def _supersede_all(
    queue: ReviewQueue, records: list[AssertionRecord], *, reason: str, by: str, at: str
) -> int:
    for record in records:
        record.assertion.superseded_at = at
        record.retracted_reason = reason
        record.retracted_by = by
        # Lifecycle stays LIVE if it was live, matching `retract`: `superseded_at` is what removes
        # a fact from reads, and rewriting lifecycle would erase that it was once believed and
        # acted upon. Only a staged fact, never believed, becomes DISCARDED.
        if record.lifecycle is Lifecycle.STAGED:
            record.lifecycle = Lifecycle.DISCARDED
        queue.store.put(record)
    return len(records)


def wipe_document(
    services: Any,
    ctx: AuthContext,
    document_id: str,
    *,
    reason: str,
    drop_vectors: bool = True,
    drop_jobs: bool = True,
) -> WipeReport:
    """Withdraw everything derived from one document.

    The document itself stays in S3, deliberately. It is the source of truth and the only thing
    here that is not rebuildable, so a wipe clears what was *derived* from it and leaves a replay
    possible. Deleting the file is a separate, louder act.
    """
    if not reason:
        raise ValueError("a wipe must carry a reason: it is the part the audit log is for")

    report = WipeReport(scope="document", target=document_id, documents=(document_id,))
    at = report.at
    queue: ReviewQueue = services.review_queue

    records = _records_for_document(queue, ctx, document_id)
    report.assertions_superseded = _supersede_all(
        queue, records, reason=reason, by=ctx.user_id, at=at
    )

    if drop_vectors and services.embedder is not None:
        try:
            report.vectors_deleted = services.embedder.forget_document(ctx, document_id)
        except Exception as e:
            # Reported, not raised. A stale vector returns a passage whose facts are withdrawn,
            # which is untidy; failing the whole wipe over it would leave the graph half-cleared,
            # which is worse.
            report.errors.append(f"vectors: {e}")

    if drop_jobs:
        try:
            report.jobs_dropped = _drop_jobs(services, ctx, document_id)
        except Exception as e:
            report.errors.append(f"jobs: {e}")

    _audit(
        services,
        GraphEvent(
            tenant_id=ctx.tenant_id,
            actor=ctx.user_id,
            action=WIPE_DOCUMENT,
            at=at,
            document_id=document_id,
            assertion_ids=tuple(r.assertion_id for r in records),
            reason=reason,
            detail={"vectors_deleted": report.vectors_deleted, "jobs_dropped": report.jobs_dropped},
        ),
        report,
    )
    logger.info(
        "%s wiped document %s: %d assertions superseded",
        ctx.user_id,
        document_id,
        report.assertions_superseded,
    )
    return report


def wipe_matter(
    services: Any,
    ctx: AuthContext,
    matter_id: str,
    *,
    reason: str,
    drop_vectors: bool = True,
    drop_jobs: bool = True,
) -> WipeReport:
    """Withdraw everything on one matter, across every document filed under it.

    Scoped through `visible()`, so a matter the caller is screened from is not wipeable by them —
    an ethical wall has to hold against destructive operations too, or it is not a wall.
    """
    if not reason:
        raise ValueError("a wipe must carry a reason: it is the part the audit log is for")

    ctx.assert_can_read_matter(matter_id)

    report = WipeReport(scope="matter", target=matter_id)
    at = report.at
    queue: ReviewQueue = services.review_queue

    records = _records_for_matter(queue, ctx, matter_id)
    documents = {
        r.assertion.source_locator.document_id
        for r in records
        if r.assertion.source_locator.document_id
    }
    report.documents = tuple(sorted(documents))
    report.assertions_superseded = _supersede_all(
        queue, records, reason=reason, by=ctx.user_id, at=at
    )

    for document_id in report.documents:
        if drop_vectors and services.embedder is not None:
            try:
                report.vectors_deleted += services.embedder.forget_document(ctx, document_id)
            except Exception as e:
                report.errors.append(f"vectors {document_id}: {e}")
        if drop_jobs:
            try:
                report.jobs_dropped += _drop_jobs(services, ctx, document_id)
            except Exception as e:
                report.errors.append(f"jobs {document_id}: {e}")

    _audit(
        services,
        GraphEvent(
            tenant_id=ctx.tenant_id,
            actor=ctx.user_id,
            action=WIPE_MATTER,
            at=at,
            matter_id=matter_id,
            assertion_ids=tuple(r.assertion_id for r in records),
            reason=reason,
            detail={
                "documents": list(report.documents),
                "vectors_deleted": report.vectors_deleted,
                "jobs_dropped": report.jobs_dropped,
            },
        ),
        report,
    )
    logger.info(
        "%s wiped matter %s: %d assertions across %d documents",
        ctx.user_id,
        matter_id,
        report.assertions_superseded,
        len(report.documents),
    )
    return report


def _drop_jobs(services: Any, ctx: AuthContext, document_id: str) -> int:
    """Forget the ingest history for one document, so a replay starts clean.

    Job rows are bookkeeping rather than a claim about the world, so these are deleted outright.
    The audit event records that they went.
    """
    store = services.job_store
    jobs = store.jobs_for_document(ctx.tenant_id, document_id)
    drop = getattr(store, "drop_job", None)
    if drop is None:
        return 0
    for job in jobs:
        drop(ctx.tenant_id, job.job_id)
    return len(jobs)


def _audit(services: Any, event: GraphEvent, report: WipeReport) -> None:
    """Record the wipe, and say so loudly if it could not be recorded.

    An unaudited wipe is the one outcome this feature must not produce silently: the facts are
    gone from the current graph and nothing says who removed them. The wipe is not rolled back —
    superseding is already done and undoing it would need its own audit trail — but the report
    carries the failure so an operator knows the record is incomplete.
    """
    audit = getattr(services, "graph_audit", None)
    if audit is None:
        report.errors.append("no audit log configured: this wipe was not recorded")
        return
    try:
        audit.append(event)
    except Exception as e:
        report.errors.append(f"AUDIT FAILED, this wipe is unrecorded: {e}")
        logger.error("graph audit write failed for %s: %s", event.action, e)
