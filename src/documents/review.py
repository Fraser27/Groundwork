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
    build_assertion,
)
from src.graph.scope import AuthContext, ScopeViolation

logger = logging.getLogger(__name__)

#: Confidence on a reviewer's correction. High, because a person reading the cited span is as
#: certain as this system gets about what a document says -- but not 1.0, because the document
#: itself can be wrong and nothing here should claim otherwise.
REVIEWER_CONFIDENCE = 0.98


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
    corrects: str | None = None
    """The assertion this one replaces, when a reviewer corrected it.

    Not `premises`: the contract refuses those on a DECLARED assertion, and correctly, because
    premises mean "derived from" and a reviewer read the document rather than deriving anything
    from the model's mistake. This records that a person overrode a specific extraction without
    misstating how the new claim was reached."""

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
        # LIVE here, not in a later `promote` pass. Approving *is* the act that makes a fact live,
        # and separating the two meant an approved fact sat STAGED forever: `promote` is only
        # called during ingest, when nothing has been approved yet, so nothing ever promoted a
        # reviewer's decision. The UI reported success, the store held APPROVED, and every read
        # path filters on lifecycle -- so four approvals were invisible everywhere.
        record.lifecycle = Lifecycle.LIVE
        self.store.put(record)
        logger.info("%s approved %s, now live", ctx.user_id, assertion_id)
        return record

    def supersede(
        self,
        ctx: AuthContext,
        assertion_id: str,
        *,
        predicate: str | None = None,
        subject_id: str | None = None,
        object_id: str | None = None,
        reason: str,
        allowed_predicates: frozenset[str] | None = None,
    ) -> tuple[AssertionRecord, AssertionRecord]:
        """Correct a claim: record what the reviewer says instead, and close the original.

        The third option a reviewer needs. Approve accepts a model's reading, reject discards it,
        and neither fits "the relationship is real but this is the wrong predicate" -- which is
        the common case, because a model that spots two parties and misjudges what connects them
        has found something worth keeping.

        **Never an edit.** An assertion says that a named method, at a named version, reading a
        named span, produced this claim, and `assertion_id` is a hash over exactly that. Changing
        the predicate in place would leave the record asserting that the model extracted something
        it did not, so the provenance would be a lie that still looked authoritative. Instead:

        - a **new DECLARED assertion** carries the reviewer's version. DECLARED because a person
          asserted it, which is why it needs no second review -- and its `method` names the
          reviewer, so "who says so" resolves to a human rather than a model;
        - the original is **superseded, not deleted**, so an `as_of` read before this moment still
          shows what the model said and what the file supported at the time.

        The corrected assertion keeps the original's `source_locator`, deliberately: the reviewer
        is re-reading the same span, and a correction with no citation would be an opinion.

        It does **not** name the original as a premise, and the contract is right to refuse that:
        premises mean "derived from", and this claim is not derived from the model's mistake -- the
        reviewer read the document. The link is recorded as `corrects` on the record instead, so
        the trail still shows a person overrode a specific extraction without misstating how the
        new claim was reached.

        Returns `(corrected, original)`.
        """
        if not reason:
            raise ReviewError("a correction must carry a reason: it is the record of why")

        record = self.fetch(ctx, assertion_id)
        if not record.is_current:
            raise ReviewError(
                f"{assertion_id} was withdrawn at {record.assertion.superseded_at}; "
                "correcting it would revive a retracted claim"
            )

        original = record.assertion
        new_predicate = predicate or original.predicate
        new_subject = subject_id or original.subject_id
        new_object = object_id or original.object_id
        if (new_predicate, new_subject, new_object) == (
            original.predicate,
            original.subject_id,
            original.object_id,
        ):
            raise ReviewError(
                "a correction must change the subject, predicate or object; "
                "to accept the claim as it stands, approve it"
            )

        at = _now()
        corrected = build_assertion(
            tenant_id=original.tenant_id,
            subject_id=new_subject,
            predicate=new_predicate,
            object_id=new_object,
            # DECLARED, not EXTRACTED_MODEL: a person asserted this, and the class is the axis
            # that keeps "a lawyer says so" distinguishable from "a model read it".
            epistemic_class=EpistemicClass.DECLARED,
            method=f"reviewer:{ctx.user_id}",
            # A reviewer correcting a claim is as certain as this system gets about a document.
            # Not 1.0: the underlying document could still be wrong.
            confidence=REVIEWER_CONFIDENCE,
            source_locator=original.source_locator,
            matter_id=original.matter_id,
            valid_from=original.valid_from,
            allowed_predicates=allowed_predicates,
        )

        original.superseded_at = at
        record.retracted_reason = f"corrected by {ctx.user_id}: {reason}"
        record.retracted_by = ctx.user_id
        # Review state records that a person dealt with it. Not APPROVED -- they did not accept
        # the claim -- and not REJECTED, because the extraction found something real.
        original.review_state = ReviewState.REJECTED
        original.reviewed_by = ctx.user_id
        original.reviewed_at = at
        if record.lifecycle is Lifecycle.STAGED:
            record.lifecycle = Lifecycle.DISCARDED
        self.store.put(record)

        new_record = AssertionRecord(
            assertion=corrected, lifecycle=Lifecycle.LIVE, job_id=record.job_id
        )
        new_record.review_note = reason
        new_record.corrects = original.assertion_id
        self.store.put(new_record)

        logger.info(
            "%s corrected %s: %s -> %s",
            ctx.user_id,
            assertion_id,
            original.predicate,
            new_predicate,
        )
        return new_record, record

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
