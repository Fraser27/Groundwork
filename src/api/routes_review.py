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

from src.api.deps import ServicesDep, TenantDep, require_reviewer, scope_violation_to_http
from src.documents.review import AssertionNotFound, ReviewError
from src.documents.storage import (
    DEFAULT_EXPIRY_SECONDS,
    DocumentNotFound,
    storage_from_config,
)
from src.graph.assertions import EpistemicClass, ReviewState
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

    doc = storage.describe(loc.document_id)
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
