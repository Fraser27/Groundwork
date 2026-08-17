"""Document ingest.

Two ways in, one pipeline:

    POST /tenants/{t}/documents        multipart file -> S3 -> transcribe -> below
    POST /tenants/{t}/documents/text   text already extracted -> below

    below:  chunk -> embed -> model-extract per chunk -> stage -> promote

Embedding runs before extraction and before the review gate, deliberately: verbatim
text is a quote rather than a claim, and a lawyer expects to search a document the
moment it lands. The gate protects the graph, not the search index.

Two steps are optional because both need Bedrock, and neither failing may lose the
work that succeeded. Without a vision client the bytes are still stored, so the
document is re-parseable the moment credentials exist; without extraction the document
is still chunked, embedded and searchable. Each is reported in the response rather than
raised, because the alternative is discarding the part that worked for the sake of the
part that could not.
"""

from __future__ import annotations

import hashlib
import logging
import secrets
from typing import Annotated, Any

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Body,
    Depends,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from pydantic import BaseModel, Field

from src.api.deps import Services, ServicesDep, TenantDep, get_services
from src.api.events import get_event_hub
from src.auth import AuthError
from src.documents.chunk import chunk_document
from src.documents.extractors.model import ModelExtractor
from src.documents.filing import (
    Filing,
    apply_filing,
    audit_filing,
    log_filing,
    resolve_filing,
)
from src.documents.ingest import DEFAULT_MEDIA_TYPE
from src.documents.job_store import latest_per_document
from src.documents.keys import parse_raw_key
from src.documents.models import IngestJob, JobState, document_id_for, sha256_hex
from src.documents.parse import ParsedDocument, parse_plain_text
from src.documents.runner import IngestBusy, IngestRunner
from src.documents.storage import (
    DEFAULT_EXPIRY_SECONDS,
    MAX_EXPIRY_SECONDS,
    DocumentMeta,
    DocumentNotFound,
    DocumentStorage,
    storage_from_config,
)
from src.graph.assertions import ReviewState
from src.graph.scope import AuthContext, ScopeViolation

logger = logging.getLogger(__name__)

router = APIRouter(tags=["documents"])

#: Whole-file upload cap. The file is read into memory to be hashed and rasterised, so
#: this bounds the task's RSS rather than expressing a view about document length —
#: `parse.MAX_PAGES` is the ceiling that costs money.
MAX_UPLOAD_BYTES = 200 * 1024 * 1024

#: Media types we can read without a vision model. A .txt upload needs no transcription
#: and must not be made to depend on Bedrock for one.
_PLAIN_TEXT_TYPES = frozenset({"text/plain", "text/markdown", "text/csv"})


class IngestTextRequest(BaseModel):
    filename: str = Field(min_length=1, max_length=512)
    text: str = Field(min_length=1)
    matter_id: str = Field(min_length=1, max_length=128)
    """Required, same as every other ingest route. An unfiled chunk is tenant-wide in vector
    search, so it stays readable by someone screened from the matter it belongs to."""
    run_model_extraction: bool = True
    """The only extraction path there is, so on by default. Off still chunks and
    embeds, which is what a machine with no Bedrock credentials can do."""


def _extraction_state(services: Services, tenant_id: str, requested: bool) -> str | None:
    """Why extraction will not run, or None to run it."""
    if not requested:
        return "not requested"
    if services.settings_for(tenant_id).block_model_extraction:
        return "disabled by tenant governance settings"
    return None


def _assert_matter_exists(services: Services, ctx: AuthContext, matter_id: str) -> None:
    """Refuse an upload naming a matter that does not exist.

    A matter must be created first. Accepting any string here is what let `NTL-2026-114` and
    `NTL-2026-0114` become two matters with nothing noticing, and a conflict check split across
    both returns half the rows while looking clean.
    """
    _assert_matter(ctx, matter_id)
    if services.graph is None:
        # Degraded rather than open: with no graph there are no records to check against, and
        # refusing every upload because the graph is down would be worse than accepting one whose
        # matter cannot be verified. The ingest path re-checks access regardless.
        logger.warning("cannot verify matter %s: no graph reachable", matter_id)
        return

    from src.matters import MatterStore

    try:
        exists = MatterStore(services.graph).exists(ctx, matter_id)
    except Exception as e:
        logger.warning("could not verify matter %s (%s), allowing the upload", matter_id, e)
        return
    if not exists:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            f"no matter {matter_id!r}. Create the matter first, so its documents and facts are "
            "attributable to it and a mistyped reference cannot become a second matter.",
        )


def _assert_matter(ctx: AuthContext, matter_id: str | None) -> None:
    if not matter_id:
        return
    try:
        ctx.assert_can_read_matter(matter_id)
    except ScopeViolation as e:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(e)) from e


def _require_storage(services: Services) -> DocumentStorage:
    storage = storage_from_config(services.config)
    if storage is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "no document store configured (DOCUMENT_BUCKET unset), an upload with "
            "nowhere immutable to land would produce citations that cannot be checked",
        )
    return storage


async def require_internal_caller(
    services: ServicesDep,
    x_lexgraph_internal: Annotated[str | None, Header()] = None,
) -> None:
    """Gate the notification endpoint on a shared secret.

    There is no user on this path, so there is no JWT to check. The endpoint is reachable
    only from inside the VPC, but "inside the VPC" is not an authorization decision — a
    compromised Lambda anywhere in the account should not be able to drive ingestion for
    an arbitrary tenant.

    Refuses when no secret is configured, rather than running open. An unset secret in a
    deployed environment is a misconfiguration, and the safe reading of it is "closed".
    """
    expected = services.config.internal_api_secret
    if not expected:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "internal ingest endpoint is not configured (INTERNAL_API_SECRET unset)",
        )
    if not x_lexgraph_internal or not secrets.compare_digest(x_lexgraph_internal, expected):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid internal caller")


def _require_runner(services: Services) -> IngestRunner:
    """Build a runner over the process-wide job store and ingest limiter.

    Constructed per call because the pipeline closure needs `services`, but the two pieces
    of shared state — the job store and the concurrency cap — come from `Services`, so
    the cap is genuinely global rather than per request.
    """
    storage = _require_storage(services)

    def pipeline(
        ctx: AuthContext,
        parsed: ParsedDocument,
        *,
        matter_id: str | None,
        run_model_extraction: bool,
        job_id: str,
    ) -> dict[str, Any]:
        return _run_pipeline(
            services,
            ctx,
            parsed,
            matter_id=matter_id,
            run_model_extraction=run_model_extraction,
            job_id=job_id,
        )

    return IngestRunner(
        storage,
        pipeline=pipeline,
        parser=services.parser,
        store=services.job_store,
        limiter=services.ingest_limiter,
        max_upload_bytes=MAX_UPLOAD_BYTES,
        plain_text_types=_PLAIN_TEXT_TYPES,
        parse_plain_text=parse_plain_text,
        on_event=get_event_hub().publish,
        graph_audit=services.graph_audit,
    )


def _filing_for(
    services: Services, ctx: AuthContext, document_id: str, requested: str | None
) -> Filing:
    """Which matter this ingest files under, and record it if that moved the document.

    Both synchronous routes require a matter, so adoption cannot fire here — what this is for
    is the other half: a re-upload naming a *different* matter is an access change effected
    through an upload, and it is audited exactly as a link would be. The async route resolves
    the same way inside `IngestRunner`, so the rule lives in `documents/filing.py` once.
    """
    filing = resolve_filing(services.job_store, ctx.tenant_id, document_id, requested)
    if filing.matter_id is not None:
        # An adopted matter has not been checked against this caller's grants yet: the earlier
        # filing was checked against whoever made it, which is not the same principal.
        _assert_matter(ctx, filing.matter_id)
    log_filing(filing, document_id, ctx.user_id)
    audit_filing(
        services.graph_audit,
        tenant_id=ctx.tenant_id,
        actor=ctx.user_id,
        document_id=document_id,
        filing=filing,
    )
    return filing


def _run_pipeline(
    services: Services,
    ctx: AuthContext,
    parsed: ParsedDocument,
    *,
    matter_id: str | None,
    run_model_extraction: bool,
    job_id: str,
) -> dict[str, Any]:
    """Chunk, embed, extract, stage and promote one parsed document."""
    chunks = chunk_document(
        parsed,
        tenant_id=ctx.tenant_id,
        matter_id=matter_id,
        # From the parse rather than from the request, so the file a locator names and
        # the file that was read cannot drift apart.
        filename=parsed.filename,
        target_chars=services.config.documents.chunk_chars,
        overlap_chars=services.config.documents.chunk_overlap_chars,
    )

    embedded = 0
    embed_error: str | None = None
    if services.embedder is not None:
        try:
            embedded = services.embedder.embed_and_store(ctx, chunks)
        except Exception as e:
            # A failed embedding must not lose the document: S3 and the graph are the
            # record, the vector index is derived and can be rebuilt.
            embed_error = str(e)
            logger.warning("embedding failed for %s: %s", parsed.document_id, e)

    assertions = []
    extraction = _extraction_state(services, ctx.tenant_id, run_model_extraction)
    if extraction is None:
        settings = services.settings_for(ctx.tenant_id)
        extractor = ModelExtractor(
            services.ontology,
            # The tenant's own model id, not the process default: a model change is a
            # settings change in Admin, not a redeploy.
            model_id=settings.extraction_model or services.config.models.extraction_model,
            region=services.config.models.region,
            settings=settings,
        )
        try:
            assertions = extractor.extract_document(chunks)
            extraction = "ran"
            # Logged because "ran and found nothing" and "never ran" were indistinguishable in
            # production: the result goes to a background task nobody reads, and a zero-yield
            # extraction produces no warning. The model id is included because a wrong or
            # unavailable model is the likeliest cause of a silent zero.
            logger.info(
                "extraction on %s: %d chunks -> %d claims (model %s)",
                parsed.document_id,
                len(chunks),
                len(assertions),
                extractor.model_id,
            )
        except Exception as e:
            # Broad on purpose: a missing credential surfaces from botocore rather than
            # as ModelExtractionFailed. Chunks are already embedded and searchable, so
            # degrading here costs graph edges and nothing else.
            extraction = f"unavailable: {e}"
            logger.warning("model extraction unavailable for %s: %s", parsed.document_id, e)

    try:
        staged = services.review_queue.stage(ctx, assertions, job_id=job_id)
    except ScopeViolation as e:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(e)) from e

    promoted = services.review_queue.promote(ctx, job_id=job_id)

    return {
        "page_count": parsed.page_count,
        "transcription": parsed.method,
        "chunks": len(chunks),
        "chunks_embedded": embedded,
        "embed_error": embed_error,
        "vector_search": (
            "enabled"
            if services.embedder is not None
            else "disabled (no VECTOR_ENDPOINT configured)"
        ),
        "extraction": extraction,
        "assertions_staged": len(staged),
        # The ids as well as the count, so the job record can name what it staged rather than
        # only how many, which is what makes a document's facts findable from its job.
        "staged_assertion_ids": list(staged),
        "assertions_live": len(promoted),
        "pending_review": services.review_queue.pending_count(ctx),
        "note": (
            "Claims whose quote was found verbatim in the document went live "
            "immediately, a search confirmed the text is there. Anything the model "
            "interpreted is waiting in the review queue."
        ),
    }


def _transcribe(
    services: Services, document_id: str, body: bytes, *, filename: str, media_type: str
) -> tuple[ParsedDocument | None, str | None]:
    """Read the bytes into text, or say why not.

    Returns `(None, reason)` rather than raising when no vision model is reachable. The
    bytes are already in S3 at this point, so a skipped transcription is a re-runnable
    step and not a lost upload.
    """
    if media_type in _PLAIN_TEXT_TYPES:
        # errors="replace" rather than strict: a mis-declared encoding should cost a few
        # mangled characters, not the document.
        text = body.decode("utf-8", errors="replace")
        return parse_plain_text(document_id, text, filename=filename), None

    if services.parser is None:
        return None, "skipped: no vision model client available (Bedrock unreachable)"

    try:
        parsed = services.parser.parse(document_id, body, filename=filename)
    except Exception as e:
        # Broad on purpose: a missing renderer raises ParseFailed, a missing credential
        # surfaces from botocore. Both mean the same thing to the caller.
        logger.warning("transcription unavailable for %s: %s", document_id, e)
        return None, f"skipped: {e}"
    return parsed, None


class PresignUploadRequest(BaseModel):
    filename: str = Field(min_length=1, max_length=512)
    media_type: str | None = None
    matter_id: str = Field(min_length=1, max_length=128)
    """Required, and must name a matter that already exists.

    Optional before, and every fact from those uploads carried no matter at all -- which made the
    Matters and Access pages empty, since both group facts by matter. Worse, it is silent: the
    upload succeeds, the pipeline runs, and the facts are simply unattributable afterwards.

    Checked against real records rather than accepted as a string, because otherwise a typo
    becomes a second matter that nothing queries -- and a conflict check that misses half its
    facts looks exactly like a clean conflict check."""


@router.post("/tenants/{tenant}/documents/presign")
async def presign_upload(
    services: ServicesDep,
    principal: TenantDep,
    body: Annotated[PresignUploadRequest, Body()],
) -> dict[str, Any]:
    """Mint a presigned POST so the browser uploads straight to S3.

    The file never crosses the API, which is the point: transcription of a 400-page
    bundle cannot fit inside CloudFront's 60s origin timeout, and the previous
    synchronous route returned 504 *after* writing S3 and the graph.

    The client is trusted with none of the things that matter. The key — and therefore
    the tenant prefix — is server-chosen, and the size cap is a signed policy condition
    that S3 itself enforces.

    The matter is required and verified here rather than at ingest, because by then the bytes are
    already in S3 and refusing costs the upload. Failing before the ticket is minted means the
    user is told while they can still act on it.
    """
    ctx, _ = principal
    _assert_matter_exists(services, ctx, body.matter_id)
    _assert_matter(ctx, body.matter_id)
    storage = _require_storage(services)

    try:
        ticket = storage.presign_upload(
            ctx,
            filename=body.filename,
            media_type=body.media_type,
            matter_id=body.matter_id,
            max_bytes=MAX_UPLOAD_BYTES,
        )
    except ScopeViolation as e:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(e)) from e

    return {
        "upload_url": ticket.url,
        "fields": ticket.fields,
        "key": ticket.key,
        "upload_id": ticket.upload_id,
        "expires_in": ticket.expires_in,
        "max_bytes": ticket.max_bytes,
        "note": (
            "POST the file to upload_url as multipart/form-data with these fields, the "
            "file last. Ingestion starts when the object lands; poll the job endpoint "
            "or subscribe to the events socket for progress."
        ),
    }


class IngestTriggerRequest(BaseModel):
    key: str = Field(min_length=1, max_length=1024)
    run_model_extraction: bool = True


@router.post("/internal/ingest", status_code=status.HTTP_202_ACCEPTED)
async def trigger_ingest(
    services: ServicesDep,
    background: BackgroundTasks,
    body: Annotated[IngestTriggerRequest, Body()],
    _: Annotated[None, Depends(require_internal_caller)],
) -> dict[str, Any]:
    """Start ingesting an object that has landed under `raw/`. Called by the S3
    notification Lambda, never by a browser.

    Returns 202 immediately and does the work in the background, because the caller is
    behind the same 60s ALB idle timeout as everything else — waiting here would just
    move the original 504 rather than fix it.

    Unauthenticated in the Cognito sense: there is no user. The tenant is taken from the
    object key, which only `presign_upload` can write, and the caller is checked against
    a shared secret.
    """
    runner = _require_runner(services)
    parsed = parse_raw_key(body.key)
    if parsed is None:
        # 200-with-ignored rather than 4xx: the notification would otherwise be retried
        # forever for a key that will never be ingestable.
        logger.info("ignoring notification for non-raw key %s", body.key)
        return {"status": "ignored", "reason": "not a raw upload key"}

    background.add_task(
        _run_ingest, runner, body.key, run_model_extraction=body.run_model_extraction
    )
    tenant_id, _upload_id, filename = parsed
    return {"status": "accepted", "tenant_id": tenant_id, "filename": filename}


def _run_ingest(runner: IngestRunner, key: str, *, run_model_extraction: bool) -> None:
    """Background entry point. Swallows nothing silently: a failed ingest is recorded on
    the job, and a refusal is logged so the retry is explicable."""
    try:
        job = runner.ingest_raw_key(key, run_model_extraction=run_model_extraction)
        # Logged because this is the only place the outcome is observable. The result goes to
        # a background task nobody reads, so an extraction that quietly found nothing looked
        # identical to one that was never attempted.
        logger.info(
            "ingest finished for %s: state=%s chunks=%d staged=%d",
            key,
            job.state.value,
            job.chunk_count,
            len(job.staged_assertion_ids),
        )
    except IngestBusy as e:
        # The object stays in S3 under a 7-day lifecycle, so a refusal costs a retry
        # rather than the document.
        logger.warning("ingest deferred for %s: %s", key, e)
    except Exception:
        logger.exception("ingest failed for %s", key)


@router.get("/tenants/{tenant}/documents/{document_id}/jobs")
async def document_jobs(
    services: ServicesDep,
    principal: TenantDep,
    document_id: str,
) -> dict[str, Any]:
    """Every ingest job for one document, newest last.

    The UI polls this every 30s and on demand. It is keyed by document rather than job
    because a document id is what the browser holds after an upload — the job id does
    not exist until the notification has been processed.
    """
    ctx, _ = principal
    jobs = services.job_store.jobs_for_document(ctx.tenant_id, document_id)
    return {
        "document_id": document_id,
        "jobs": [_job_summary(j) for j in jobs],
    }


@router.get("/tenants/{tenant}/jobs/{job_id}")
async def job_status(
    services: ServicesDep,
    principal: TenantDep,
    job_id: str,
) -> dict[str, Any]:
    ctx, _ = principal
    job = services.job_store.get_job(ctx.tenant_id, job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"no job {job_id!r}")
    return _job_summary(job)


def _job_summary(job: IngestJob) -> dict[str, Any]:
    return {
        "job_id": job.job_id,
        "document_id": job.document_id,
        "matter_id": job.matter_id,
        "state": job.state.value,
        "reason": job.reason,
        "is_failed": job.state.is_failed,
        "is_terminal": job.state.is_terminal,
        "retry_target": job.state.retry_target.value if job.state.retry_target else None,
        "chunk_count": job.chunk_count,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "history": [{"state": h.state.value, "at": h.at, "reason": h.reason} for h in job.history],
    }


def _assertion_out(a: Any) -> dict[str, Any]:
    """An assertion as the UI declares it.

    `source_locator` and `premises` are the two the UI dereferences without a guard, so a
    payload that omits them does not degrade, it white-screens the page.
    """
    return {
        "assertion_id": a.assertion_id,
        "tenant_id": a.tenant_id,
        "matter_id": a.matter_id,
        "subject_id": a.subject_id,
        "predicate": a.predicate,
        "object_id": a.object_id,
        "epistemic_class": a.epistemic_class.value,
        "method": a.method,
        "confidence": a.confidence,
        "source_locator": a.source_locator.to_dict(),
        "premises": list(a.premises),
        "rule_id": a.rule_id,
        "rule_version": a.rule_version,
        "valid_from": a.valid_from,
        "valid_until": a.valid_until,
        "recorded_at": a.recorded_at,
        "superseded_at": a.superseded_at,
        "review_state": a.review_state.value,
        "reviewed_by": a.reviewed_by,
        "reviewed_at": a.reviewed_at,
    }


def _screened(ctx: AuthContext, jobs: list[IngestJob], document_id: str) -> bool:
    """True if any attempt on this document names a matter the caller may not read."""
    for job in jobs:
        if job.document_id != document_id or not job.matter_id:
            continue
        try:
            ctx.assert_can_read_matter(job.matter_id)
        except ScopeViolation:
            return True
    return False


def _document_summary(
    job: IngestJob,
    *,
    meta: DocumentMeta | None,
    assertion_count: int = 0,
    pending_review_count: int = 0,
) -> dict[str, Any]:
    """What the documents table renders.

    Assembled from the job plus S3 metadata rather than stored as a row: a document record
    that can disagree with the job that produced it is worse than no record.
    """
    return {
        "document_id": job.document_id,
        "filename": meta.filename if meta else job.document_id,
        "matter_id": job.matter_id,
        "state": job.state.value,
        "uploaded_at": meta.uploaded_at if meta else job.created_at,
        "page_count": getattr(meta, "page_count", None) if meta else None,
        "size_bytes": getattr(meta, "size_bytes", None) if meta else None,
        "assertion_count": assertion_count,
        "pending_review_count": pending_review_count,
        "error": job.reason if job.state.is_failed else None,
    }


@router.get("/tenants/{tenant}/documents")
async def list_documents(
    services: ServicesDep,
    principal: TenantDep,
    matter_id: Annotated[str | None, Query()] = None,
) -> dict[str, Any]:
    """Documents ingested for this tenant, newest first.

    Derived from job state, because jobs are the only per-document record that is written
    on every path. Filtering by matter goes through the same scope check as a read, so a
    screened matter cannot be enumerated by passing its id here.
    """
    ctx, _ = principal
    if matter_id:
        _assert_matter(ctx, matter_id)

    all_jobs = services.job_store.jobs_for_tenant(ctx.tenant_id)

    # Matter scoping is applied to the *rows*, not just the filter: a document filed under
    # a matter this caller is screened from must not appear in an unfiltered list either.
    # Checked against every attempt's matter, not only the winning row's: attempts can
    # disagree if a refile updated some and failed on others, and the safe reading of a
    # disagreement is the more restrictive one.
    visible = []
    for job in latest_per_document(all_jobs):
        if matter_id and job.matter_id != matter_id:
            continue
        if _screened(ctx, all_jobs, job.document_id):
            continue
        visible.append(job)

    storage = storage_from_config(services.config)
    counts = _assertion_counts(services, ctx)
    documents = [
        _document_summary(
            job,
            meta=storage.describe(job.document_id, tenant_id=ctx.tenant_id) if storage else None,
            assertion_count=counts.get(job.document_id, (0, 0))[0],
            pending_review_count=counts.get(job.document_id, (0, 0))[1],
        )
        for job in visible
    ]
    return {"documents": documents, "total": len(documents)}


def _assertion_counts(services: Services, ctx: AuthContext) -> dict[str, tuple[int, int]]:
    """Per-document (total, pending) assertion counts, from what this caller may see.

    Built off `review_queue.visible`, so the numbers already respect the matter wall and a
    screened document contributes nothing rather than a count that hints at its size.
    """
    counts: dict[str, list[int]] = {}
    for record in services.review_queue.visible(ctx):
        doc_id = record.assertion.source_locator.document_id
        if not doc_id:
            continue
        entry = counts.setdefault(doc_id, [0, 0])
        entry[0] += 1
        # `review_state` on the assertion, which is derived from the epistemic class. There is
        # no separate needs-review flag, and there must not be: a fact that could be marked
        # reviewed independently of how it was reached could opt out of the gate.
        if record.assertion.review_state is ReviewState.PENDING:
            entry[1] += 1
    return {k: (v[0], v[1]) for k, v in counts.items()}


@router.get("/tenants/{tenant}/documents/{document_id}")
async def get_document(
    services: ServicesDep,
    principal: TenantDep,
    document_id: str,
) -> dict[str, Any]:
    """One document, with its ingest timeline and the facts read out of it."""
    ctx, _ = principal
    jobs = services.job_store.jobs_for_document(ctx.tenant_id, document_id)
    if not jobs:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"no document {document_id!r}")

    # Same rule as the list, so a document cannot read LIVE on one page and pending on the other.
    latest = latest_per_document(jobs)[0]
    if _screened(ctx, jobs, document_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"no document {document_id!r}")

    storage = storage_from_config(services.config)
    meta = storage.describe(document_id, tenant_id=ctx.tenant_id) if storage else None

    assertions = [
        r.assertion
        for r in services.review_queue.visible(ctx)
        if r.assertion.source_locator.document_id == document_id
    ]
    pending = sum(
        1
        for r in services.review_queue.visible(ctx)
        if r.assertion.source_locator.document_id == document_id
        and r.assertion.review_state is ReviewState.PENDING
    )

    summary = _document_summary(
        latest, meta=meta, assertion_count=len(assertions), pending_review_count=pending
    )
    # Every attempt, oldest first: a document that failed twice before succeeding should
    # show all three, because that history is what makes a stuck document diagnosable.
    timeline = [
        {"state": h.state.value, "at": h.at, "detail": h.reason}
        for job in sorted(jobs, key=lambda j: j.created_at)
        for h in job.history
    ]
    return {
        **summary,
        "s3_uri": f"s3://{storage.bucket}/{meta.key}" if storage and meta else "",
        "content_sha256": getattr(meta, "content_sha256", None) if meta else None,
        "timeline": timeline,
        "assertions": [_assertion_out(a) for a in assertions],
    }


@router.websocket("/tenants/{tenant}/ingest/events")
async def ingest_events(websocket: WebSocket, tenant: str, token: str = "") -> None:
    """Live ingest progress for one tenant.

    The token arrives as a query parameter because a browser cannot set headers on a
    WebSocket handshake. That is a real trade-off — query strings are more likely to end
    up in logs than an Authorization header — accepted because the alternative is a
    subprotocol dance for a channel that carries no document content, only job ids and
    states.

    **Per-process.** A client connected to one task will not see events published on
    another, so with more than one task this is best-effort. The 30s poll is the
    correctness guarantee; this only removes the wait. See `api/events.py`.
    """
    services = get_services()
    hub = get_event_hub()

    try:
        ctx, _ = services.authenticator.authenticate(token)
        services.authenticator.assert_tenant_matches(ctx, tenant)
    except (AuthError, ScopeViolation):
        # Closed before accept, so an unauthorised caller never gets a socket and cannot
        # tell a bad token from a wrong tenant.
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return

    await websocket.accept()
    sub = hub.subscribe(ctx.tenant_id)
    try:
        while True:
            event = await sub.queue.get()
            await websocket.send_json(event)
    except WebSocketDisconnect:
        pass
    finally:
        hub.unsubscribe(ctx.tenant_id, sub)


@router.post("/tenants/{tenant}/documents", status_code=status.HTTP_201_CREATED)
async def upload_document(
    services: ServicesDep,
    principal: TenantDep,
    file: Annotated[UploadFile, File()],
    matter_id: Annotated[str, Form(min_length=1)],
    run_model_extraction: Annotated[bool, Form()] = True,
) -> dict[str, Any]:
    """Store a file, transcribe it page by page, and run the pipeline over the text.

    `matter_id` is required, as it is on `presign`. It was optional here, and an unfiled document
    is not merely untidy: a chunk with no matter is deliberately tenant-wide in vector search, so
    it stays retrievable by someone screened from the matter the document belongs to.
    """
    ctx, _ = principal
    _assert_matter(ctx, matter_id)
    _assert_matter_exists(services, ctx, matter_id)
    storage = _require_storage(services)

    filename = file.filename or "document"
    body = await file.read()
    if not body:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "uploaded file is empty")
    if len(body) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            f"{len(body)} bytes exceeds the {MAX_UPLOAD_BYTES}-byte upload limit",
        )

    # A browser sends application/octet-stream for anything it does not recognise,
    # including .txt. Passing that through would send a text file to the vision model,
    # so the extension wins over a client's shrug.
    declared = file.content_type
    media_type = declared if declared and declared != DEFAULT_MEDIA_TYPE else None

    filing = _filing_for(
        services, ctx, document_id_for(ctx.tenant_id, sha256_hex(body)), matter_id
    )

    try:
        doc = apply_filing(
            storage.put_document(
                ctx,
                filename=filename,
                body=body,
                matter_id=filing.matter_id,
                media_type=media_type,
            ),
            filing,
        )
    except ScopeViolation as e:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(e)) from e

    parsed, skipped = _transcribe(
        services, doc.document_id, body, filename=doc.filename, media_type=doc.media_type
    )

    summary: dict[str, Any] = {
        "document_id": doc.document_id,
        "filename": doc.filename,
        "matter_id": doc.matter_id,
        "uploaded_at": doc.uploaded_at,
        "size_bytes": doc.byte_size,
        "content_sha256": doc.content_sha256,
        "s3_uri": doc.s3_uri,
    }

    if parsed is None:
        # PARSE_FAILED rather than a success state: nothing was read. Its retry target
        # is PARSING, which is exactly the right recovery — configure Bedrock, re-run.
        return {
            **summary,
            "state": JobState.PARSE_FAILED.value,
            "transcription": skipped,
            "error": skipped,
            "page_count": None,
            "chunks": 0,
            "assertion_count": 0,
            "pending_review_count": services.review_queue.pending_count(ctx),
            "note": (
                "The file is stored and its hash recorded, so nothing is lost, but no "
                "text was read from it, so it is not yet searchable or citable."
            ),
        }

    result = _run_pipeline(
        services,
        ctx,
        parsed,
        matter_id=doc.matter_id,
        run_model_extraction=run_model_extraction,
        job_id=doc.document_id,
    )
    return {
        **summary,
        **result,
        "state": (
            JobState.PENDING_REVIEW.value if result["pending_review"] else JobState.LIVE.value
        ),
        # Aliases matching the UI's DocumentSummary, which the list and detail views
        # already render. Cheaper than a second shape for one endpoint.
        "assertion_count": result["assertions_staged"],
        "pending_review_count": result["pending_review"],
        "error": result["embed_error"],
    }


@router.post("/tenants/{tenant}/documents/text")
async def ingest_text(
    services: ServicesDep,
    principal: TenantDep,
    body: Annotated[IngestTextRequest, Body()],
) -> dict[str, Any]:
    """Ingest text that has already been extracted elsewhere.

    Kept alongside the upload endpoint for tests and demos: it exercises everything
    below parsing without needing S3 or a vision model.
    """
    ctx, _ = principal
    _assert_matter(ctx, body.matter_id)

    # Content-addressed, so re-ingesting the same bytes is a no-op all the way down:
    # identical document id gives identical assertion ids.
    document_id = hashlib.sha256(body.text.encode()).hexdigest()[:16]
    parsed = parse_plain_text(document_id, body.text, filename=body.filename)
    filing = _filing_for(services, ctx, document_id, body.matter_id)

    result = _run_pipeline(
        services,
        ctx,
        parsed,
        matter_id=filing.matter_id,
        run_model_extraction=body.run_model_extraction,
        job_id=document_id,
    )
    return {
        "document_id": f"document:{document_id}",
        "filename": body.filename,
        "matter_id": filing.matter_id,
        **result,
    }


@router.get("/tenants/{tenant}/documents/{document_id}/download")
async def download_document(
    services: ServicesDep,
    principal: TenantDep,
    document_id: str,
    page: Annotated[int | None, Query(ge=1)] = None,
    expires_in: Annotated[int, Query(ge=1, le=MAX_EXPIRY_SECONDS)] = DEFAULT_EXPIRY_SECONDS,
) -> dict[str, Any]:
    """A short-lived link to the source file, for checking a citation.

    Ordering is the security property, not an implementation detail: the matter check
    runs *before* the URL is created, because a presigned URL carries no identity and
    is checked by nothing once it exists. Every failure path answers 404, matching
    `scope.py` — a 403 would confirm the document exists.
    """
    ctx, _ = principal

    storage = storage_from_config(services.config)
    if storage is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "no document store configured (DOCUMENT_BUCKET unset)",
        )

    doc = storage.describe(document_id, tenant_id=ctx.tenant_id)
    if doc is None or doc.tenant_id != ctx.tenant_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"no document {document_id!r}")

    if doc.matter_id is not None:
        try:
            ctx.assert_can_read_matter(doc.matter_id)
        except ScopeViolation as e:
            logger.warning("refused download of %s for %s: %s", document_id, ctx.user_id, e)
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"no document {document_id!r}") from e

    try:
        link = storage.presign_download(document_id, expires_in=expires_in, ctx=ctx, page=page)
    except (DocumentNotFound, ScopeViolation) as e:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"no document {document_id!r}") from e

    return {
        "document_id": document_id,
        "filename": link.filename,
        "media_type": doc.media_type,
        "page": link.page,
        "download_url": link.url,
        "expires_at": link.expires_at,
        "expires_in": link.expires_in,
        "note": (
            "This link expires and is never stored. Provenance records the document, "
            "page and quote; the link is minted fresh each time it is asked for."
        ),
    }
