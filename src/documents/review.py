"""The review queue: stage assertions, approve or reject them, promote to live.

Two separate gates, often confused:

1. **Review state** — did a human sign off on this claim? Derived from epistemic class
   by `build_assertion`, so EXTRACTED_MODEL arrives PENDING and nothing can opt out.
2. **Staged vs live** — is this claim in the graph queries read? An auto-asserted
   citation is signed off the moment it is created but still has to be *written*.

Keeping them separate is what makes a bad extraction run harmless. Staging is a
transaction boundary: a whole document's assertions land together or not at all, so a
half-ingested filing never looks like a complete one.

Approval is a named human act. `reviewed_by` is mandatory and there is no
`approve_all(auto=True)` — a batch approve loop is one line of caller code and at
least it is a line someone wrote deliberately.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Protocol

from src.graph.assertions import (
    AUTO_ASSERT_CLASSES,
    Assertion,
    EpistemicClass,
    ReviewState,
)
from src.graph.scope import AuthContext, ScopeViolation

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Lifecycle(str, Enum):
    STAGED = "STAGED"
    LIVE = "LIVE"
    DISCARDED = "DISCARDED"


class ReviewError(ValueError):
    pass


class AssertionNotFound(ReviewError):
    """No such assertion, or one the caller may not see.

    Split from plain ReviewError so the HTTP layer can answer 404 for absence and
    409 for a state conflict. Without it, approving something nonexistent is
    indistinguishable from approving something already rejected, and a caller cannot
    tell whether retrying would help.

    Deliberately does not distinguish absent from walled-off, matching `scope.py`:
    confirming a matter exists is itself a leak.
    """


@dataclass
class AssertionRecord:
    """A stored assertion plus where it sits in the staged/live lifecycle."""

    assertion: Assertion
    lifecycle: Lifecycle = Lifecycle.STAGED
    job_id: str | None = None
    review_note: str | None = None
    retracted_reason: str | None = None
    retracted_by: str | None = None

    @property
    def assertion_id(self) -> str:
        return self.assertion.assertion_id

    @property
    def is_current(self) -> bool:
        return self.assertion.superseded_at is None


class AssertionStore(Protocol):
    def put(self, record: AssertionRecord) -> None: ...
    def get(self, tenant_id: str, assertion_id: str) -> AssertionRecord | None: ...
    def all_for_tenant(self, tenant_id: str) -> list[AssertionRecord]: ...
    def dependents_of(self, tenant_id: str, assertion_id: str) -> list[AssertionRecord]: ...


@dataclass
class InMemoryAssertionStore:
    """Reference store with the reverse-premise index retraction needs.

    The index is maintained on write rather than computed on read: cascading
    retraction walks it transitively, and a scan per level turns a retraction into an
    O(n·depth) job over the whole tenant graph. In Neptune this is the
    `(:Assertion)-[:PREMISE]->(:Assertion)` edge traversed backwards.
    """

    _records: dict[tuple[str, str], AssertionRecord] = field(default_factory=dict)
    _dependents: dict[tuple[str, str], set[str]] = field(default_factory=dict)

    def put(self, record: AssertionRecord) -> None:
        key = (record.assertion.tenant_id, record.assertion_id)
        self._records[key] = record
        for premise in record.assertion.premises:
            self._dependents.setdefault((record.assertion.tenant_id, premise), set()).add(
                record.assertion_id
            )

    def get(self, tenant_id: str, assertion_id: str) -> AssertionRecord | None:
        return self._records.get((tenant_id, assertion_id))

    def all_for_tenant(self, tenant_id: str) -> list[AssertionRecord]:
        return [r for (t, _), r in self._records.items() if t == tenant_id]

    def dependents_of(self, tenant_id: str, assertion_id: str) -> list[AssertionRecord]:
        ids = self._dependents.get((tenant_id, assertion_id), set())
        return [r for aid in ids if (r := self.get(tenant_id, aid)) is not None]

    def drop_tenant(self, tenant_id: str) -> int:
        """Forget every assertion for one tenant, returning how many went.

        For a rebuild, not a correction. Retracting each assertion would be the right tool
        for withdrawing a belief — it supersedes and leaves a trail — but here the facts are
        about to be re-derived from the same sources, so a trail of retractions would record
        an event that did not happen.
        """
        doomed = [key for key in self._records if key[0] == tenant_id]
        for key in doomed:
            del self._records[key]
        self._dependents = {k: v for k, v in self._dependents.items() if k[0] != tenant_id}
        return len(doomed)


@dataclass(frozen=True)
class QueueItem:
    """One row of the review queue, with what a reviewer needs to decide."""

    assertion_id: str
    subject_id: str
    predicate: str
    object_id: str
    epistemic_class: EpistemicClass
    method: str
    confidence: float
    document_id: str | None
    page: int | None
    char_start: int | None
    char_end: int | None
    matter_id: str | None
    recorded_at: str
    job_id: str | None

    @classmethod
    def of(cls, record: AssertionRecord) -> QueueItem:
        a = record.assertion
        loc = a.source_locator
        return cls(
            assertion_id=a.assertion_id,
            subject_id=a.subject_id,
            predicate=a.predicate,
            object_id=a.object_id,
            epistemic_class=a.epistemic_class,
            method=a.method,
            confidence=a.confidence,
            document_id=loc.document_id,
            page=loc.page,
            char_start=loc.char_start,
            char_end=loc.char_end,
            matter_id=a.matter_id,
            recorded_at=a.recorded_at,
            job_id=record.job_id,
        )


class ReviewQueue:
    def __init__(self, store: AssertionStore | None = None) -> None:
        self.store = store or InMemoryAssertionStore()

    # ── staging ───────────────────────────────────────────────────────────────

    def stage(
        self,
        ctx: AuthContext,
        assertions: Sequence[Assertion],
        *,
        job_id: str | None = None,
    ) -> list[str]:
        """Write a document's assertions into staging, all or nothing.

        Cross-tenant and out-of-scope matters are rejected before anything is written,
        so a single bad assertion cannot leave a document half-staged.
        """
        for a in assertions:
            if a.tenant_id != ctx.tenant_id:
                raise ScopeViolation(
                    f"assertion {a.assertion_id} is tenant {a.tenant_id}, caller is {ctx.tenant_id}"
                )
            if a.matter_id is not None:
                ctx.assert_can_read_matter(a.matter_id)

        for a in assertions:
            self.store.put(AssertionRecord(assertion=a, lifecycle=Lifecycle.STAGED, job_id=job_id))
        logger.info("staged %d assertions for job %s", len(assertions), job_id)
        return [a.assertion_id for a in assertions]

    # ── reading the queue ─────────────────────────────────────────────────────

    def list_pending(
        self,
        ctx: AuthContext,
        *,
        document_id: str | None = None,
        job_id: str | None = None,
        limit: int = 100,
    ) -> list[QueueItem]:
        """Everything awaiting a human decision, scoped to the caller.

        Superseded assertions are excluded. A cascade can retract a claim that was
        still pending, and asking a reviewer to adjudicate something already withdrawn
        wastes the scarcest resource in the system.

        Sorted by confidence ascending: the claims the model was least sure of are the
        ones a reviewer's attention is worth most on.
        """
        items = [
            QueueItem.of(r)
            for r in self._awaiting_review(ctx)
            if (document_id is None or r.assertion.source_locator.document_id == document_id)
            and (job_id is None or r.job_id == job_id)
        ]
        items.sort(key=lambda i: (i.confidence, i.assertion_id))
        return items[:limit]

    def list_staged(self, ctx: AuthContext, *, job_id: str | None = None) -> list[QueueItem]:
        return [
            QueueItem.of(r)
            for r in self.visible(ctx)
            if r.lifecycle is Lifecycle.STAGED
            and r.is_current
            and (job_id is None or r.job_id == job_id)
        ]

    def pending_count(self, ctx: AuthContext) -> int:
        return len(self._awaiting_review(ctx))

    def _awaiting_review(self, ctx: AuthContext) -> list[AssertionRecord]:
        return [
            r
            for r in self.visible(ctx)
            if r.assertion.review_state is ReviewState.PENDING and r.is_current
        ]

    # ── decisions ─────────────────────────────────────────────────────────────

    def approve(
        self, ctx: AuthContext, assertion_id: str, *, note: str | None = None
    ) -> AssertionRecord:
        record = self.fetch(ctx, assertion_id)
        state = record.assertion.review_state
        if not record.is_current:
            # Usually a cascade got here first: the premise this rests on was retracted
            # while it sat in the queue. Approving would revive a withdrawn claim.
            raise ReviewError(
                f"{assertion_id} was retracted at {record.assertion.superseded_at} "
                f"({record.retracted_reason}); it cannot be approved"
            )
        if state is ReviewState.APPROVED:
            return record
        if state is ReviewState.REJECTED:
            raise ReviewError(
                f"{assertion_id} was rejected; re-extract rather than reviving a "
                "rejected assertion, so the audit trail stays honest"
            )
        if state is ReviewState.AUTO_ASSERTED:
            raise ReviewError(
                f"{assertion_id} is {record.assertion.epistemic_class.value} and needs no "
                "approval, approving it would imply a human checked something they did not"
            )
        record.assertion.review_state = ReviewState.APPROVED
        record.assertion.reviewed_by = ctx.user_id
        record.assertion.reviewed_at = _now()
        record.review_note = note
        self.store.put(record)
        logger.info("%s approved %s", ctx.user_id, assertion_id)
        return record

    def reject(self, ctx: AuthContext, assertion_id: str, *, reason: str) -> AssertionRecord:
        """Reject a claim. The reason is mandatory — it is the training signal.

        Does not cascade on its own. A PENDING assertion has no dependent inferences
        (rules fire on live assertions only), but if this ever changes, call
        `retract.retract` instead, which handles the closure.
        """
        if not reason:
            raise ReviewError("a rejection must carry a reason")
        record = self.fetch(ctx, assertion_id)
        if record.assertion.review_state is ReviewState.APPROVED:
            raise ReviewError(
                f"{assertion_id} is already live; use retract.retract so dependent "
                "inferences are invalidated too"
            )
        record.assertion.review_state = ReviewState.REJECTED
        record.assertion.reviewed_by = ctx.user_id
        record.assertion.reviewed_at = _now()
        record.review_note = reason
        # Never deleted: a rejected extraction is evidence about the extractor.
        record.lifecycle = Lifecycle.DISCARDED
        self.store.put(record)
        logger.info("%s rejected %s: %s", ctx.user_id, assertion_id, reason)
        return record

    def approve_many(
        self, ctx: AuthContext, assertion_ids: Iterable[str], *, note: str | None = None
    ) -> list[str]:
        return [self.approve(ctx, aid, note=note).assertion_id for aid in assertion_ids]

    # ── promotion ─────────────────────────────────────────────────────────────

    def promote(self, ctx: AuthContext, *, job_id: str | None = None) -> list[str]:
        """Move signed-off staged assertions into the live graph.

        Signed off means AUTO_ASSERTED or APPROVED — the two states `edge_scope`
        admits by default. PENDING assertions stay staged; that is the gate working,
        not a partial failure.
        """
        promoted: list[str] = []
        for record in self.visible(ctx):
            if record.lifecycle is not Lifecycle.STAGED:
                continue
            if job_id is not None and record.job_id != job_id:
                continue
            if not record.is_current:
                continue
            if record.assertion.review_state not in (
                ReviewState.AUTO_ASSERTED,
                ReviewState.APPROVED,
            ):
                continue
            record.lifecycle = Lifecycle.LIVE
            self.store.put(record)
            promoted.append(record.assertion_id)
        logger.info("promoted %d assertions to live (job %s)", len(promoted), job_id)
        return promoted

    def live_assertions(self, ctx: AuthContext) -> list[AssertionRecord]:
        return [r for r in self.visible(ctx) if r.lifecycle is Lifecycle.LIVE and r.is_current]

    def auto_asserted_ids(self, ctx: AuthContext) -> list[str]:
        return [
            r.assertion_id
            for r in self.visible(ctx)
            if r.assertion.epistemic_class in AUTO_ASSERT_CLASSES
        ]

    # ── scoped access ─────────────────────────────────────────────────────────
    #
    # The only two ways into the store. `src/documents/retract.py` uses them for the
    # same reason this module does: no read of an assertion may skip the matter check.

    def visible(self, ctx: AuthContext) -> list[AssertionRecord]:
        return [
            r
            for r in self.store.all_for_tenant(ctx.tenant_id)
            if r.assertion.matter_id is None or ctx.can_read_matter(r.assertion.matter_id)
        ]

    def fetch(self, ctx: AuthContext, assertion_id: str) -> AssertionRecord:
        record = self.store.get(ctx.tenant_id, assertion_id)
        # Absence and another tenant's id are both reported as absence — a firm learns
        # nothing about another firm. An in-tenant screen refuses distinctly, below.
        if record is None:
            raise AssertionNotFound(f"no assertion {assertion_id!r}")
        if record.assertion.matter_id is not None:
            ctx.assert_can_read_matter(record.assertion.matter_id)
        return record
