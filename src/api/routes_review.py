"""Review queue — where a person signs off on what a model claimed.

The most consequential surface in the product. Two things it must get right:

- **Retracted assertions are not reviewable.** A cascade can retract a claim that is
  still pending, and approving it afterwards would revive a withdrawn fact.
  `ReviewQueue` enforces this; the routes surface it as 409 rather than 500.
- **Bulk approval reports what it would cascade.** Approving in bulk is where a
  reviewer stops reading, so the response says what changed rather than just "ok".

A refusal for an in-tenant ethical screen surfaces as 403 naming the matter and a
contact, via `scope_violation_to_http`. Every other scope violation stays 404.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any

from fastapi import APIRouter, Body, HTTPException, Query, status
from pydantic import BaseModel, Field

from src.api.deps import (
    ServicesDep,
    TenantDep,
    require_admin,
    require_reviewer,
    scope_violation_to_http,
)
from src.documents.review import AssertionNotFound, ReviewError
from src.documents.storage import (
    DEFAULT_EXPIRY_SECONDS,
    DocumentNotFound,
    storage_from_config,
)
from src.graph.assertions import AssertionError_, EpistemicClass, ReviewState
from src.graph.scope import AuthContext, ScopeViolation

logger = logging.getLogger(__name__)

router = APIRouter(tags=["review"])


class RejectRequest(BaseModel):
    reason: str = Field(min_length=1, max_length=2000)
    """Required: a rejection lands on the audit trail, and "why" is the part that
    matters six months later."""


class AssertionOut(BaseModel):
    assertion_id: str
    subject_id: str
    predicate: str
    object_id: str
    epistemic_class: str
    method: str
    confidence: float
    review_state: str
    matter_id: str | None = None
    recorded_at: str | None = None
    superseded_at: str | None = None
    tenant_id: str = ""
    # Named to match what the UI dereferences. It reads `source_locator.quote` and
    # `premises.length` with no guard, so omitting either does not render a blank cell, it
    # throws inside the table body and takes the whole page with it.
    source_locator: dict[str, Any] = Field(default_factory=dict)
    premises: list[str] = Field(default_factory=list)
    rule_id: str | None = None
    rule_version: str | None = None
    valid_from: str | None = None
    valid_until: str | None = None
    reviewed_by: str | None = None
    reviewed_at: str | None = None
    below_floor: bool = False
    """Whether this sits under the retrieval trust floor. Shown so a reviewer knows
    the claim is not currently shaping any answer."""


def _settle_document(services: Any, ctx: AuthContext, record: Any) -> None:
    """Move a document off PENDING_REVIEW once nothing of its own is still pending.

    The job state was only ever advanced during ingest, so a document whose facts were all
    approved afterwards sat at "awaiting review" permanently -- on the Documents list, on the
    matter, and everywhere else that reads the job. The review decision is the event that settles
    it, so the reconciliation belongs here.

    Judged on this job's own staged ids, not the tenant's pending count: `pending_review` is
    tenant-wide, and using it would park every document behind one unreviewed claim somewhere
    else. That exact bug is called out in `runner.py`.

    Never raises. A stale badge is a bad outcome; failing an approval that already succeeded is a
    worse one.
    """
    store = getattr(services, "job_store", None)
    document_id = getattr(getattr(record, "assertion", None), "source_locator", None)
    document_id = getattr(document_id, "document_id", None)
    if store is None or not document_id:
        return

    try:
        from src.documents.job_store import JobTracker
        from src.documents.models import JobState

        jobs = store.jobs_for_document(ctx.tenant_id, document_id)
        pending_here = {
            r.assertion.assertion_id
            for r in services.review_queue.visible(ctx)
            if r.assertion.review_state is ReviewState.PENDING
        }
        tracker = JobTracker(store)
        for job in jobs:
            if job.state is not JobState.PENDING_REVIEW:
                continue
            if pending_here & set(job.staged_assertion_ids or ()):
                continue
            tracker.advance(job, JobState.APPROVED)
            tracker.advance(job, JobState.LIVE)
            logger.info("document %s settled to LIVE: nothing of its own is pending", document_id)
    # Broad on purpose: the review decision has already succeeded, and a stale badge is a far
    # better outcome than failing an approval that went through.
    except Exception as e:  # noqa: BLE001
        logger.warning("could not settle the document state for %s: %s", document_id, e)


def _to_out(record: Any, floor: float) -> AssertionOut:
    a = record.assertion
    return AssertionOut(
        assertion_id=a.assertion_id,
        subject_id=a.subject_id,
        predicate=a.predicate,
        object_id=a.object_id,
        epistemic_class=a.epistemic_class.value,
        method=a.method,
        confidence=a.confidence,
        review_state=a.review_state.value,
        matter_id=a.matter_id,
        recorded_at=a.recorded_at,
        superseded_at=a.superseded_at,
        tenant_id=a.tenant_id,
        source_locator=a.source_locator.to_dict(),
        premises=list(a.premises),
        rule_id=a.rule_id,
        rule_version=a.rule_version,
        valid_from=a.valid_from,
        valid_until=a.valid_until,
        reviewed_by=a.reviewed_by,
        reviewed_at=a.reviewed_at,
        below_floor=a.confidence < floor,
    )


@router.get("/tenants/{tenant}/assertions")
async def list_assertions(
    services: ServicesDep,
    principal: TenantDep,
    review_state: Annotated[str | None, Query()] = ReviewState.PENDING.value,
    epistemic_class: Annotated[str | None, Query()] = None,
    matter_id: Annotated[str | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> dict[str, Any]:
    """Defaults to PENDING — this endpoint *is* the review queue.

    Ordered least-confident-first: the claims most likely to be wrong are the ones
    worth a human's attention first.
    """
    ctx, _ = principal
    floor = services.settings_for(ctx.tenant_id).min_confidence_floor

    try:
        records = services.review_queue.visible(ctx)
    except ScopeViolation as e:
        raise scope_violation_to_http(e) from e

    if review_state:
        records = [r for r in records if r.assertion.review_state.value == review_state]
    if epistemic_class:
        records = [r for r in records if r.assertion.epistemic_class.value == epistemic_class]
    if matter_id:
        records = [r for r in records if r.assertion.matter_id == matter_id]

    records.sort(key=lambda r: r.assertion.confidence)

    return {
        "assertions": [_to_out(r, floor) for r in records[:limit]],
        "total": len(records),
        "confidence_floor": floor,
    }


@router.post("/tenants/{tenant}/assertions/{assertion_id}/approve")
async def approve(
    services: ServicesDep,
    principal: TenantDep,
    assertion_id: str,
) -> AssertionOut:
    require_reviewer(principal)
    ctx, _ = principal
    floor = services.settings_for(ctx.tenant_id).min_confidence_floor
    try:
        record = services.review_queue.approve(ctx, assertion_id)
    except ScopeViolation as e:
        raise scope_violation_to_http(e) from e
    except AssertionNotFound as e:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(e)) from e
    except ReviewError as e:
        # A state conflict: the claim exists but cannot move to APPROVED from where it
        # is — already rejected, retracted by a cascade, or auto-asserted.
        raise HTTPException(status.HTTP_409_CONFLICT, str(e)) from e
    _settle_document(services, ctx, record)
    return _to_out(record, floor)


@router.post("/tenants/{tenant}/assertions/{assertion_id}/reject")
async def reject(
    services: ServicesDep,
    principal: TenantDep,
    assertion_id: str,
    body: Annotated[RejectRequest, Body()],
) -> AssertionOut:
    require_reviewer(principal)
    ctx, _ = principal
    floor = services.settings_for(ctx.tenant_id).min_confidence_floor
    try:
        record = services.review_queue.reject(ctx, assertion_id, reason=body.reason)
    except ScopeViolation as e:
        raise scope_violation_to_http(e) from e
    except AssertionNotFound as e:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(e)) from e
    except ReviewError as e:
        raise HTTPException(status.HTTP_409_CONFLICT, str(e)) from e
    # A rejection settles a document as much as an approval does: the reviewer is finished with
    # it either way, and leaving it "awaiting review" would misdescribe a decision that was made.
    _settle_document(services, ctx, record)
    return _to_out(record, floor)


class DocumentProvenance(BaseModel):
    """What a PDF viewer needs to land a reviewer on the cited passage.

    `download_url` is minted per request and never persisted — see
    `src.documents.storage`. Everything else comes off the stored locator, so the
    citation stays checkable by hand when no link can be produced.
    """

    document_id: str
    filename: str | None = None
    page: int | None = None
    quote: str | None = None
    chunk_id: str | None = None
    download_url: str | None = None
    expires_at: str | None = None
    link_unavailable: str | None = None
    """Why there is no URL. Provenance without a link is degraded, not broken, so this
    is a message rather than an error."""


def _document_provenance(
    services: Any, ctx: AuthContext, assertion: Any
) -> DocumentProvenance | None:
    """Resolve an assertion's document locator into something openable.

    The matter check has already happened — `ReviewQueue.fetch` refused the assertion
    otherwise — and is repeated against the *document's* matter here because a
    presigned URL answers to nobody once it exists.
    """
    loc = assertion.source_locator
    if not loc.is_document:
        return None

    out = DocumentProvenance(
        document_id=loc.document_id,
        filename=loc.filename,
        page=loc.page,
        quote=loc.quote,
        chunk_id=loc.chunk_id,
    )

    storage = storage_from_config(services.config)
    if storage is None:
        out.link_unavailable = (
            "no document store configured (DOCUMENT_BUCKET unset), so the file cannot be "
            "linked, the page and quote above still identify the passage"
        )
        return out

    doc = storage.describe(loc.document_id, tenant_id=ctx.tenant_id)
    if doc is None:
        out.link_unavailable = (
            "the source file is not in the document store; it may have been ingested as "
            "text rather than uploaded"
        )
        return out
    out.filename = out.filename or doc.filename

    if doc.matter_id is not None:
        try:
            ctx.assert_can_read_matter(doc.matter_id)
        except ScopeViolation as e:
            # Named for a screen, so a reviewer looking at a proof tree with a missing
            # link knows to ask rather than assuming the file is gone.
            out.link_unavailable = str(e) if e.is_screen else "not available"
            return out

    try:
        link = storage.presign_download(
            loc.document_id, expires_in=DEFAULT_EXPIRY_SECONDS, ctx=ctx, page=loc.page
        )
    except (DocumentNotFound, ScopeViolation):
        out.link_unavailable = "the source file is no longer retrievable"
        return out
    except Exception as e:
        # A signing failure must not take down the explanation: the premise tree and
        # the quote are the substance of the answer, the link is a convenience.
        logger.warning("presigning %s failed: %s", loc.document_id, e)
        out.link_unavailable = "the download link could not be generated"
        return out

    out.download_url = link.url
    out.expires_at = link.expires_at
    return out


@router.get("/tenants/{tenant}/assertions/{assertion_id}/provenance")
async def provenance(
    services: ServicesDep,
    principal: TenantDep,
    assertion_id: str,
) -> dict[str, Any]:
    """Why the system believes this.

    For an extraction, the document page and quote plus a short-lived link to the file.
    For an inference, the premise tree — recursively, because a premise may itself be
    inferred.
    """
    ctx, _ = principal
    try:
        record = services.review_queue.fetch(ctx, assertion_id)
    except ScopeViolation as e:
        raise scope_violation_to_http(e) from e
    except AssertionNotFound as e:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(e)) from e

    a = record.assertion
    floor = services.settings_for(ctx.tenant_id).min_confidence_floor

    premises: list[dict[str, Any]] = []
    for pid in a.premises:
        try:
            p = services.review_queue.fetch(ctx, pid)
        except (AssertionNotFound, ReviewError, ScopeViolation):
            # A premise the caller cannot see, or one that predates this store.
            premises.append({"assertion_id": pid, "visible": False})
            continue
        premises.append({**_to_out(p, floor).model_dump(), "visible": True})

    doc = _document_provenance(services, ctx, a)

    return {
        "assertion": _to_out(record, floor).model_dump(),
        "premises": premises,
        "rule_id": a.rule_id,
        "rule_version": a.rule_version,
        "is_current": a.is_current,
        "explanation": _explain(a),
        "document": doc.model_dump() if doc is not None else None,
    }


def _explain(a: Any) -> str:
    """Plain-language account of where a fact came from, for a non-engineer."""
    cls = a.epistemic_class
    if cls is EpistemicClass.DECLARED:
        return (
            f"Recorded directly from a system of record ({a.method}). Not inferred or interpreted."
        )
    if cls is EpistemicClass.EXTRACTED_DET:
        return (
            f"Found by exact pattern matching ({a.method}). Reproducible, the same "
            "document always produces the same result, with no AI involved."
        )
    if cls is EpistemicClass.EXTRACTED_MODEL:
        return (
            f"Proposed by an AI model ({a.method}) reading the document. Requires human "
            "approval before it can influence an answer."
        )
    if cls is EpistemicClass.INFERRED:
        return (
            f"Derived by rule {a.rule_id} from {len(a.premises)} supporting fact(s). Its "
            "confidence cannot exceed that of its weakest premise."
        )
    return (
        "A statistical suggestion based on the shape of the graph, not on any document. "
        "Never used to answer questions."
    )


@router.post("/tenants/{tenant}/reason")
async def run_reasoner(services: ServicesDep, principal: TenantDep) -> dict[str, Any]:
    """Fire the pack's rules over this tenant's approved facts.

    Conclusions are **staged, not written live**, so an inference passes through the same
    review gate as an extraction. A reasoner that wrote straight to the graph would be a code
    path that opts out of review, which is the one thing no code path may do.

    Rules come from whichever ontology pack the tenant runs, so this endpoint contains no
    domain knowledge: a legal tenant gets conflict and stale-authority checks, a healthcare
    tenant gets contraindication alerts, from the same engine.

    Premises are the tenant's *live* assertions rather than everything visible, because a
    conclusion may only rest on facts somebody stands behind. Idempotent: assertion ids are
    content-addressed, so re-running converges instead of duplicating.
    """
    require_admin(principal)
    ctx, _ = principal

    from src.reasoning.engine import Reasoner

    live = [r.assertion for r in services.review_queue.live_assertions(ctx)]
    report = Reasoner(services.ontology).run(ctx, live)

    staged: list[str] = []
    if report.inferences:
        try:
            staged = services.review_queue.stage(
                ctx, [i.assertion for i in report.inferences], job_id="reasoner"
            )
        except ScopeViolation as e:
            raise scope_violation_to_http(e) from e

    out = report.to_dict()
    out["staged"] = len(staged)
    out["note"] = (
        "Inferred facts are staged for review, not published. Each carries the facts it "
        "rests on, so a reviewer can follow it back to the documents underneath."
    )
    return out


class CorrectRequest(BaseModel):
    predicate: str | None = None
    subject_id: str | None = None
    object_id: str | None = None
    reason: str = Field(min_length=1, max_length=2000)
    """Mandatory. A correction without a stated reason is an unexplained override of a model,
    which is the one thing an audit trail cannot make sense of later."""


@router.post("/tenants/{tenant}/assertions/{assertion_id}/correct")
async def correct(
    services: ServicesDep,
    principal: TenantDep,
    assertion_id: str,
    body: Annotated[CorrectRequest, Body()],
) -> dict[str, Any]:
    """Record what a reviewer says instead, and close the model's version.

    The third option beside approve and reject. A model that spots two parties and misjudges what
    connects them has found something worth keeping, and neither accepting nor discarding it is
    the right answer.

    Never an edit: the correction is a new DECLARED assertion whose method names the reviewer, and
    the original is superseded rather than rewritten, so an as-of read still shows what the model
    said. Editing in place would leave the record claiming the model extracted something it did
    not.
    """
    require_reviewer(principal)
    ctx, _ = principal
    floor = services.settings_for(ctx.tenant_id).min_confidence_floor

    predicate = body.predicate
    allowed = services.ontology.allowed_for(predicate) if predicate else None
    if predicate and predicate in services.ontology.rule_conclusions:
        # Only a rule may conclude these, and a reviewer correcting a claim into one would create
        # a conflict flag with no premises -- exactly what the extractor is already barred from.
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            f"{predicate} is drawn by a rule from other facts, so it cannot be asserted directly. "
            "Correct the underlying relationships and the rule will draw it.",
        )

    try:
        corrected, original = services.review_queue.supersede(
            ctx,
            assertion_id,
            predicate=predicate,
            subject_id=body.subject_id,
            object_id=body.object_id,
            reason=body.reason,
            allowed_predicates=allowed,
        )
    except ScopeViolation as e:
        raise scope_violation_to_http(e) from e
    except AssertionNotFound as e:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(e)) from e
    except ReviewError as e:
        raise HTTPException(status.HTTP_409_CONFLICT, str(e)) from e
    except AssertionError_ as e:
        # An invariant refused the corrected claim -- most likely a predicate outside the closed
        # vocabulary. The message names which one, so it is passed through.
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(e)) from e

    _record_correction(services, ctx, original.assertion_id, corrected.assertion_id, body.reason)

    return {
        "corrected": _to_out(corrected, floor).model_dump(),
        "superseded": _to_out(original, floor).model_dump(),
        "note": (
            "The reviewer's version is live and declared by them, not by a model. The original "
            "is closed rather than deleted, so an as-of read before now still shows what the "
            "model proposed and why it was overridden."
        ),
    }


def _record_correction(
    services: Any, ctx: Any, original_id: str, corrected_id: str, reason: str
) -> None:
    """Log the override. Best-effort: the correction itself already succeeded."""
    audit = getattr(services, "graph_audit", None)
    if audit is None:
        return
    from src.graph_audit import SUPERSEDE, GraphEvent

    try:
        audit.append(
            GraphEvent(
                tenant_id=ctx.tenant_id,
                actor=ctx.user_id,
                action=SUPERSEDE,
                assertion_ids=(original_id, corrected_id),
                reason=reason,
                detail={"superseded": original_id, "corrected": corrected_id},
            )
        )
    except Exception as e:
        logger.warning("could not audit the correction of %s: %s", original_id, e)
