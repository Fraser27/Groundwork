"""Read surfaces: dashboard, matters, ontology, graph neighbourhood.

The matters endpoint is the one with a subtlety worth reading: screened matters are
named, with a reason and a contact, in a list of their own. Never in `matters` — a
caller that iterates the visible list must not be able to reach a screened matter by
accident, which is why this is two lists rather than one with a flag.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Annotated, Any

from fastapi import APIRouter, Body, HTTPException, Query, status
from pydantic import BaseModel, Field

from src.admin_ops import ResetScope, replay, reset_derived
from src.api.deps import ServicesDep, TenantDep, require_admin, scope_violation_to_http
from src.discovery.catalog_store import CatalogTable
from src.discovery.glue_scanner import scan_catalog
from src.documents.models import JobState
from src.graph.assertions import EpistemicClass, ReviewState
from src.graph.scope import ScopeViolation
from src.ontology.loader import ONTOLOGY_DIR, load_ontology

logger = logging.getLogger(__name__)

router = APIRouter(tags=["catalog"])


def available_ontology_domains() -> list[str]:
    return sorted(p.stem for p in ONTOLOGY_DIR.glob("*.yaml"))


def _table_summary(table: CatalogTable) -> dict[str, Any]:
    return {
        "full_name": table.full_name,
        "name": table.name,
        "database": table.database,
        "source_id": table.source_id,
        "description": table.description,
        # Rows are never counted: they are not in the graph and counting them would mean
        # querying the warehouse to render a list page.
        "row_count": None,
        # A catalog scan reports what the catalog says. That is a declaration by a system
        # of record, not something a model inferred.
        "epistemic_class": EpistemicClass.DECLARED.value,
        "catalog_type": table.catalog_type,
        "column_count": len(table.columns),
    }


@router.get("/tenants/{tenant}/dashboard")
async def dashboard(services: ServicesDep, principal: TenantDep) -> dict[str, Any]:
    ctx, grants = principal
    settings = services.settings_for(ctx.tenant_id)
    records = services.review_queue.visible(ctx)

    by_class = Counter(r.assertion.epistemic_class.value for r in records if r.is_current)
    by_state = Counter(r.assertion.review_state.value for r in records if r.is_current)

    # Every field the UI reads has to be present, even when empty. The dashboard crashed on
    # `Object.entries(undefined)` because these four were declared in the TypeScript
    # interface but never sent: a type is a claim about runtime data, and the compiler cannot
    # check it against an API response.
    documents_by_state: dict[str, int] = {}
    try:
        for state in JobState:
            found = services.job_store.jobs_in_state(ctx.tenant_id, state)
            if found:
                documents_by_state[state.value] = len(found)
    except Exception as e:
        logger.debug("could not count documents by state: %s", e)

    metric_total = metric_approved = 0
    if services.graph is not None:
        try:
            from src.metrics.graph_store import GraphMetricStore

            store = GraphMetricStore(services.graph)
            metric_total = len(store.list_metrics(ctx.tenant_id))
            metric_approved = len(store.list_metrics(ctx.tenant_id, approved_only=True))
        except Exception as e:
            logger.debug("could not count metrics: %s", e)

    return {
        "tenant_id": ctx.tenant_id,
        "assertions_by_class": {c.value: by_class.get(c.value, 0) for c in EpistemicClass},
        "assertions_by_review_state": {s.value: by_state.get(s.value, 0) for s in ReviewState},
        "pending_review": by_state.get(ReviewState.PENDING.value, 0),
        "retracted": sum(1 for r in records if not r.is_current),
        "confidence_floor": settings.min_confidence_floor,
        "ontology_domain": settings.ontology_domain,
        "can_review": grants.can_review,
        "kill_switch_active": settings.block_ungoverned_queries,
        "documents_by_state": documents_by_state,
        "matters": len({r.assertion.matter_id for r in records if r.assertion.matter_id}),
        "metrics": {"total": metric_total, "approved": metric_approved},
        "recent_activity": [],
    }


@router.get("/tenants/{tenant}/matters")
async def list_matters(services: ServicesDep, principal: TenantDep) -> dict[str, Any]:
    """Matters the caller may see, and — named — those they are screened from.

    Naming them is the point. Reading "no conflicts found" when the truth is "none that
    you can see" is how an ethical wall causes the harm it exists to prevent, and a bare
    count does not tell a lawyer which client to go and ask about.
    """
    ctx, _ = principal
    records = services.review_queue.visible(ctx)
    counts: dict[str, int] = {}
    for r in records:
        mid = r.assertion.matter_id
        if mid:
            counts[mid] = counts.get(mid, 0) + 1

    # Records first, then anything only the data knows about. A matter now exists the moment
    # somebody creates it, which is the order real work happens in: a team is staffed and a screen
    # raised before the first document arrives. Grouping assertions alone could never express an
    # empty matter, so a matter could not be created before a document was filed under it.
    matters: list[dict[str, Any]] = []
    seen: set[str] = set()
    if services.graph is not None:
        from src.matters import MatterStore

        try:
            for record in MatterStore(services.graph).list(ctx):
                seen.add(record.matter_id)
                matters.append(
                    {**record.to_dict(), "assertion_count": counts.get(record.matter_id, 0)}
                )
        except Exception as e:
            # Degrade to the derived list rather than failing the page: a matter with facts is
            # still worth showing even if its record cannot be read.
            logger.warning("could not read matter records: %s", e)

    # Matters that facts refer to but no record names. Shown rather than hidden: these are either
    # pre-existing data or a reset that dropped the records, and silently omitting a matter that
    # holds facts would be worse than showing it without a name.
    for mid in sorted(set(counts) - seen):
        matters.append({"matter_id": mid, "name": mid, "assertion_count": counts[mid]})

    # A separate list, never a flag on a row in `matters`: no caller can then treat a
    # screened matter as readable by forgetting to check the flag. `visible()` has
    # already dropped them, so these come from the grant.
    withheld = ctx.withheld_matters()

    return {"matters": matters, "withheld": withheld, "withheld_count": len(withheld)}


@router.get("/ontology/{domain}")
async def get_ontology(domain: str) -> dict[str, Any]:
    """The vocabulary, including the help text the UI shows for each term."""
    try:
        onto = load_ontology(domain)
    except FileNotFoundError as e:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(e)) from e

    def _pred(p: Any) -> dict[str, Any]:
        return {
            "id": p.id,
            "label": p.label,
            "description": p.description,
            "help": p.help,
            "governing": p.governing,
            "domain": list(p.domain),
            "range": list(p.range),
            "symmetric": p.symmetric,
        }

    # Split rather than flat, and named to match the UI's `Ontology` type. The split
    # is the point: "which predicates are closed?" is the question an administrator
    # actually asks, and a flat list with a boolean makes them work it out.
    return {
        "domain": onto.domain,
        "version": onto.version,
        "entity_types": [
            {
                "id": e.id,
                "label": e.label,
                "description": e.description,
                "help": e.help,
            }
            for e in onto.entities.values()
        ],
        "governing_predicates": [_pred(p) for p in onto.predicates.values() if p.governing],
        "descriptive_predicates": [_pred(p) for p in onto.predicates.values() if not p.governing],
        "rules": [
            {
                "id": r.id,
                "version": r.version,
                "description": r.description,
                "help": r.help,
                "when": list(r.when),
                "then": r.then,
                "min_premise_class": r.min_premise_class,
                "method": r.method,
            }
            for r in onto.rules
        ],
    }


@router.get("/tenants/{tenant}/graph/neighbourhood")
async def neighbourhood(
    services: ServicesDep,
    principal: TenantDep,
    node_id: Annotated[str | None, Query()] = None,
    depth: Annotated[int, Query(ge=1, le=3)] = 2,
    limit: Annotated[int, Query(ge=1, le=2000)] = 400,
) -> dict[str, Any]:
    """Edges around a node, or an overview of the graph when no node is named.

    `node_id` is optional because the explorer opens before anything is selected -- there is no
    node to centre on yet. Requiring it meant the page's first request was a 422 and the graph
    never rendered at all, which read as "the graph is empty" rather than "you have not picked a
    starting point".

    The overview is capped: a firm's whole graph is not a diagram, and drawing ten thousand edges
    produces an unreadable hairball that also freezes the browser. Governing edges are kept
    first, because those are the ones a conflict check or a privilege wall reads.
    """
    ctx, _ = principal
    settings = services.settings_for(ctx.tenant_id)
    records = [r for r in services.review_queue.visible(ctx) if r.is_current]

    if node_id is None:
        return _graph_overview(services, ctx, records, settings, limit)

    frontier = {node_id}
    seen_nodes: set[str] = set()
    edges: list[dict[str, Any]] = []

    for _ in range(depth):
        next_frontier: set[str] = set()
        for r in records:
            a = r.assertion
            if a.subject_id in frontier or a.object_id in frontier:
                edges.append(
                    {
                        "assertion_id": a.assertion_id,
                        "source": a.subject_id,
                        "target": a.object_id,
                        "predicate": a.predicate,
                        "epistemic_class": a.epistemic_class.value,
                        "confidence": a.confidence,
                        "review_state": a.review_state.value,
                        "below_floor": a.confidence < settings.min_confidence_floor,
                        # Drawn more prominently: these are the edges a conflict check
                        # or a privilege wall actually reads.
                        "governing": services.ontology.is_governing(a.predicate),
                        "matter_id": a.matter_id,
                    }
                )
                next_frontier.update({a.subject_id, a.object_id})
        seen_nodes |= frontier
        frontier = next_frontier - seen_nodes
        if not frontier:
            break

    node_ids = seen_nodes | frontier
    unique_edges = {e["assertion_id"]: e for e in edges}

    # Entity ids are `kind:slug`, so the label and type come free. The renderer sizes
    # and colours nodes by `type` and needs a readable `label` — without them it drew
    # unlabelled dots and no edges, because `NODE_RADIUS[undefined]` is undefined.
    return {
        "nodes": [_node(n) for n in sorted(node_ids)],
        "edges": list(unique_edges.values()),
        "confidence_floor": settings.min_confidence_floor,
    }


def _graph_overview(
    services: Any,
    ctx: Any,
    records: list[Any],
    settings: Any,
    limit: int,
) -> dict[str, Any]:
    """The whole tenant graph, capped, for a first look with nothing selected.

    Governing edges first: if the cap bites, the edges worth keeping are the ones that drive a
    consequence, not the subject-matter tags. Sorted by confidence within that, so a truncated
    view shows the firmest facts rather than an arbitrary slice.
    """
    edges: list[dict[str, Any]] = []
    for r in records:
        a = r.assertion
        edges.append(
            {
                "assertion_id": a.assertion_id,
                "source": a.subject_id,
                "target": a.object_id,
                "predicate": a.predicate,
                "epistemic_class": a.epistemic_class.value,
                "confidence": a.confidence,
                "review_state": a.review_state.value,
                "below_floor": a.confidence < settings.min_confidence_floor,
                "governing": services.ontology.is_governing(a.predicate),
                "matter_id": a.matter_id,
            }
        )

    edges.sort(key=lambda e: (not e["governing"], -e["confidence"]))
    kept = edges[:limit]
    node_ids = {e["source"] for e in kept} | {e["target"] for e in kept}

    return {
        "nodes": [_node(n) for n in sorted(node_ids)],
        "edges": kept,
        "truncated": len(edges) > len(kept),
        "total_edges": len(edges),
        "confidence_floor": settings.min_confidence_floor,
    }


def _node(entity_id: str) -> dict[str, Any]:
    kind, _, rest = entity_id.partition(":")
    if not rest:
        kind, rest = "entity", entity_id
    return {
        "id": entity_id,
        "type": kind,
        "label": rest.replace("-", " ").replace("_", " "),
    }


# ── Structured catalog ───────────────────────────────────────────────────────
#
# Schemas, never rows. A scan records what tables exist and what shape they are; the
# rows stay in Athena and are queried in place. That is the whole reason the graph can
# hold "structured" and "unstructured" together without copying a warehouse into it.


@router.get("/tenants/{tenant}/tables")
async def list_tables(services: ServicesDep, principal: TenantDep) -> list[dict[str, Any]]:
    """Tables found by the last catalog scan. Empty until a source is scanned."""
    ctx, _ = principal
    return [_table_summary(t) for t in services.catalog.tables(ctx.tenant_id)]


@router.get("/tenants/{tenant}/tables/{full_name}")
async def get_table(services: ServicesDep, principal: TenantDep, full_name: str) -> dict[str, Any]:
    ctx, _ = principal
    table = services.catalog.table(ctx.tenant_id, full_name)
    if table is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"no table {full_name!r}")
    return {
        **_table_summary(table),
        "columns": [
            {
                "name": c.name,
                "data_type": c.data_type,
                "description": c.description,
                "is_partition": c.is_partition,
                "is_primary_key": c.is_primary_key,
            }
            for c in table.columns
        ],
        "method": f"catalog-scan:{table.source_id}",
        "scanned_at": table.scanned_at,
        "location": table.location,
    }


@router.get("/tenants/{tenant}/sources")
async def list_sources(services: ServicesDep, principal: TenantDep) -> list[dict[str, Any]]:
    """Configured sources and the outcome of their last scan.

    Configured sources appear before they are scanned, because an operator needs to see a
    source in order to press Scan on it.
    """
    ctx, _ = principal
    for source_id in services.config.structured.glue_databases or []:
        services.catalog.register_source(ctx.tenant_id, source_id, name=source_id)
    return [
        {
            "source_id": s.source_id,
            "name": s.name,
            "kind": s.kind,
            "database": s.database,
            "region": s.region,
            "table_count": s.table_count,
            "status": s.status,
            "last_scanned_at": s.last_scanned_at,
            "errors": s.errors,
        }
        for s in services.catalog.sources(ctx.tenant_id)
    ]


@router.get("/tenants/{tenant}/settings")
async def get_settings(services: ServicesDep, principal: TenantDep) -> dict[str, Any]:
    """What this tenant is configured to do, as the Admin page shows it.

    Assembled from governance settings plus process config rather than stored as a blob:
    a settings row that can drift from the values actually in force is worse than no
    settings page at all.
    """
    ctx, _ = principal
    settings = services.settings_for(ctx.tenant_id)
    models = services.config.models
    return {
        "tenant_id": ctx.tenant_id,
        "name": ctx.tenant_id,
        "ontology_domain": services.ontology.domain,
        "min_confidence": settings.min_confidence_floor,
        # Sent so the floor control can respect its own lower bound. The floor must stay above
        # the cap, and without this the UI offered a range that was mostly invalid: dragging to
        # 0.65 produced a rejection explaining an invariant the screen had never mentioned.
        "model_confidence_cap": settings.model_confidence_cap,
        "block_ungoverned_queries": settings.block_ungoverned_queries,
        "extraction_model": settings.extraction_model or models.extraction_model,
        "synthesis_model": models.synthesis_model,
        "embedding_model": services.config.vector.embedding_model,
        "available_models": [
            {"id": m, "label": m.split(".")[-1]}
            for m in sorted({models.extraction_model, models.synthesis_model})
        ],
        "available_domains": available_ontology_domains(),
    }


@router.get("/tenants/{tenant}/glue/databases")
async def glue_databases(services: ServicesDep, principal: TenantDep) -> dict[str, Any]:
    """What is in the Glue catalog, so an administrator can choose before scanning.

    Reads nothing into the graph. A firm's catalog usually holds databases belonging to other
    teams, and scanning everything makes the Tables page unusable and the graph misleading -- the
    point of a governed layer is that it holds what somebody chose to govern.

    Requires only `glue:GetDatabases` and `glue:GetTables`, both already granted, so this needs no
    IAM change.
    """
    require_admin(principal)

    try:
        import boto3

        glue = boto3.client("glue", region_name=services.config.models.region)
    except Exception as e:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, f"cannot reach Glue: {e}") from e

    from src.discovery.glue_scanner import list_databases

    try:
        found = list_databases(glue)
    except Exception as e:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"could not list databases: {e}") from e

    # Which are already in the graph, so the picker can show what a scan would replace rather
    # than making the user remember.
    scanned: set[str] = set()
    try:
        for table in services.catalog.tables(principal[0].tenant_id):
            if table.database:
                scanned.add(table.database)
    except Exception as e:
        logger.debug("could not read the catalog cache: %s", e)

    for entry in found["databases"]:
        entry["scanned"] = entry["name"] in scanned

    return {
        **found,
        "note": (
            "Nothing here is in the graph until you scan it. Scanning declares a database's "
            "tables as facts a system of record asserted, so choose the ones this firm actually "
            "works with."
        ),
    }


class ScanRequest(BaseModel):
    source_id: str = Field(default="glue-main", min_length=1, max_length=128)
    databases: list[str] = Field(default_factory=list)
    """Empty means discover every database the task role can see, which is the right
    default when onboarding a catalog nobody has enumerated yet."""


@router.post("/tenants/{tenant}/sources/scan")
async def scan_sources(
    services: ServicesDep,
    principal: TenantDep,
    body: Annotated[ScanRequest, Body()],
) -> dict[str, Any]:
    """Scan Glue and declare what it finds into the graph.

    Schemas only. No rows are read, so this is cheap and safe to re-run: the graph learns
    that a table exists and what shape it is, while the rows stay in the warehouse and are
    queried in place at answer time.

    The assertions are DECLARED (a system of record said so, no model involved), so they
    pass the review gate automatically. They still go through `stage`/`promote` rather than
    straight to the graph, because that is the only path that writes an audit trail.
    """
    require_admin(principal)
    ctx, _ = principal

    try:
        import boto3

        glue = boto3.client("glue", region_name=services.config.models.region)
    except Exception as e:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, f"cannot reach Glue: {e}") from e

    try:
        result = scan_catalog(
            glue,
            tenant_id=ctx.tenant_id,
            source_id=body.source_id,
            databases=body.databases or None,
        )
    except Exception as e:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, f"catalog scan failed: {e}") from e

    record = services.catalog.record_scan(
        ctx.tenant_id,
        source_id=body.source_id,
        result=result,
        region=services.config.models.region,
    )

    # The scan is useful even when the graph is unreachable: the catalog cache is what the
    # Tables page reads, so a failure here degrades to "listed but not queryable" rather
    # than losing the scan.
    staged: list[str] = []
    promoted: list[str] = []
    graph_error: str | None = None
    if result.assertions:
        try:
            job_id = f"scan-{body.source_id}"
            staged = services.review_queue.stage(ctx, result.assertions, job_id=job_id)
            promoted = services.review_queue.promote(ctx, job_id=job_id)
        except Exception as e:
            graph_error = str(e)
            logger.warning("catalog scan could not reach the graph: %s", e)

    return {
        "source_id": body.source_id,
        "tables_found": len(result.tables),
        "assertions_declared": len(result.assertions),
        "assertions_live": len(promoted),
        "assertions_staged": len(staged),
        "scan_errors": result.errors,
        "graph_error": graph_error,
        "status": record.status,
        "last_scanned_at": record.last_scanned_at,
        "note": (
            "Schemas only, no rows were read. Column types and descriptions are now "
            "citable as DECLARED facts, and metrics can be compiled against them."
        ),
    }


# ── Reset and replay ─────────────────────────────────────────────────────────
#
# These demonstrate the architecture's central claim rather than merely asserting it: if
# S3 and Glue are authoritative and everything else is derived, then the graph can be
# thrown away and rebuilt. If that is ever untrue, these are where it shows.


class ResetRequest(BaseModel):
    """What to remove. Every box is ticked by default except metrics."""

    graph: bool = True
    vectors: bool = True
    jobs: bool = True
    catalog: bool = True

    metrics: bool = False
    """Off by default, and the only option here that destroys something unrecoverable.
    Documents come back from S3 and schemas come back from Glue; a metric definition was
    authored in this app and has no upstream source to rebuild from."""

    confirm_metric_loss: bool = False
    """Required when `metrics` is true. A second, explicit acknowledgement, because the
    checkbox and the consequence are not obviously connected: "reset derived data" does not
    read like "delete work nobody can recover"."""


@router.post("/tenants/{tenant}/admin/reset")
async def reset_derived_route(
    services: ServicesDep,
    principal: TenantDep,
    body: Annotated[ResetRequest, Body()] = ResetRequest(),
) -> dict[str, Any]:
    """Drop selected derived data. S3 is never touched.

    Safe by default in the sense that matters: what it removes can be rebuilt, because the
    documents remain in S3 and the schemas remain in Glue. Scoped to the caller's own tenant,
    so a reset in one firm cannot empty another's graph.

    Metrics are the exception and are excluded unless asked for twice.
    """
    require_admin(principal)
    ctx, _ = principal

    if body.metrics and not body.confirm_metric_loss:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "Deleting metric definitions cannot be undone. They were authored here, not "
            "derived from S3 or Glue, so no replay reconstructs them. Set "
            "confirm_metric_loss to proceed.",
        )

    scope = ResetScope(
        graph=body.graph,
        vectors=body.vectors,
        jobs=body.jobs,
        catalog=body.catalog,
        metrics=body.metrics,
    )
    return reset_derived(services, ctx, scope).to_dict()


@router.post("/tenants/{tenant}/admin/replay")
async def replay_route(
    services: ServicesDep,
    principal: TenantDep,
    run_model_extraction: Annotated[bool, Query()] = True,
) -> dict[str, Any]:
    """Rebuild the graph and the search index from S3.

    Runs inline rather than in the background, because an operator pressing Replay wants the
    report. That does bound it by the 60s origin timeout, so a large corpus should be
    replayed by re-uploading rather than through this endpoint — noted here rather than
    hidden behind a spinner that will eventually 504.
    """
    require_admin(principal)
    ctx, _ = principal
    return replay(services, ctx, run_model_extraction=run_model_extraction).to_dict()


class WipeRequest(BaseModel):
    reason: str = Field(min_length=1, max_length=2000)
    """Mandatory. The reason is the point of the audit entry: a deletion nobody can explain
    later is indistinguishable from data loss."""

    drop_vectors: bool = True
    drop_jobs: bool = True


@router.post("/tenants/{tenant}/documents/{document_id}/wipe")
async def wipe_document_facts(
    services: ServicesDep,
    principal: TenantDep,
    document_id: str,
    body: Annotated[WipeRequest, Body()],
) -> dict[str, Any]:
    """Withdraw everything derived from one document, so it can be re-read cleanly.

    Soft: assertions are closed rather than deleted, so an as-of read before now still shows them
    and the Audit page records who withdrew them. The document itself stays in S3 -- it is the
    source of truth and the only thing here that cannot be rebuilt -- so a replay reconstructs
    the facts with whatever the extractor now produces.

    Inferences drawn from these facts are left standing, deliberately. They were true when drawn
    and their premises remain resolvable; retracting them would assert the firm never held a
    belief it did hold.
    """
    require_admin(principal)
    ctx, _ = principal

    from src.documents.wipe import wipe_document

    try:
        report = wipe_document(
            services,
            ctx,
            document_id,
            reason=body.reason,
            drop_vectors=body.drop_vectors,
            drop_jobs=body.drop_jobs,
        )
    except ScopeViolation as e:
        raise scope_violation_to_http(e) from e
    except ValueError as e:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(e)) from e
    return report.to_dict()


@router.post("/tenants/{tenant}/matters/{matter_id}/wipe")
async def wipe_matter_facts(
    services: ServicesDep,
    principal: TenantDep,
    matter_id: str,
    body: Annotated[WipeRequest, Body()],
) -> dict[str, Any]:
    """The same, for every document on one matter.

    Scoped through the caller's own grants, so a matter they are screened from is not wipeable by
    them: an ethical wall that holds for reads and not for deletions is not a wall.
    """
    require_admin(principal)
    ctx, _ = principal

    from src.documents.wipe import wipe_matter

    try:
        report = wipe_matter(
            services,
            ctx,
            matter_id,
            reason=body.reason,
            drop_vectors=body.drop_vectors,
            drop_jobs=body.drop_jobs,
        )
    except ScopeViolation as e:
        raise scope_violation_to_http(e) from e
    except ValueError as e:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(e)) from e
    return report.to_dict()


@router.get("/tenants/{tenant}/audit/graph")
async def graph_audit_log(
    services: ServicesDep,
    principal: TenantDep,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
) -> dict[str, Any]:
    """Who changed what the system believes, newest first.

    The trace back for a soft delete: a fact withdrawn from the current graph is still recorded
    here with who withdrew it and when, which is what makes the deletion auditable rather than a
    gap in the record.
    """
    require_admin(principal)
    ctx, _ = principal

    audit = services.graph_audit
    if audit is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "no graph audit log is configured")
    events = audit.events(ctx.tenant_id, limit=limit)
    return {
        "events": [e.to_dict() for e in events],
        "count": len(events),
        "note": (
            "Append-only. Nothing here is edited or removed, and neither are the facts it "
            "describes: they are closed rather than deleted, so an as-of read before an entry "
            "still reconstructs what the graph held."
        ),
    }


class CreateMatterRequest(BaseModel):
    matter_id: str = Field(min_length=1, max_length=128)
    """The firm's own reference, e.g. NTL-2026-0114. Not generated here: a matter already has a
    reference in the firm's systems, and inventing a second one guarantees they diverge."""

    name: str = Field(min_length=1, max_length=300)


@router.post("/tenants/{tenant}/matters", status_code=status.HTTP_201_CREATED)
async def create_matter(
    services: ServicesDep,
    principal: TenantDep,
    body: Annotated[CreateMatterRequest, Body()],
) -> dict[str, Any]:
    """Create a matter, which then exists before any document is filed under it.

    That ordering is the point. A team is staffed and an ethical screen is raised before the first
    document arrives, and neither is expressible against a matter that does not exist -- you
    cannot screen a lawyer from something the system has never heard of.

    Re-creating an existing reference renames it rather than failing: two people setting up the
    same matter should converge on one record.
    """
    require_admin(principal)
    ctx, _ = principal
    if services.graph is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "no graph is reachable, so a matter cannot be recorded",
        )

    from src.matters import MatterError, MatterStore

    try:
        matter = MatterStore(services.graph).create(ctx, body.matter_id, body.name)
    except MatterError as e:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(e)) from e
    return {**matter.to_dict(), "assertion_count": 0}


class LinkDocumentsRequest(BaseModel):
    document_ids: list[str] = Field(min_length=1)
    reason: str | None = None
    """Optional here, unlike a wipe. Filing a document is ordinary work rather than a
    withdrawal, so a reason is useful and not mandatory -- but it is kept when given."""


@router.post("/tenants/{tenant}/matters/{matter_id}/documents")
async def link_documents_to_matter(
    services: ServicesDep,
    principal: TenantDep,
    matter_id: str,
    body: Annotated[LinkDocumentsRequest, Body()],
) -> dict[str, Any]:
    """File several documents under a matter, or move them there.

    Audited, because this is an access change effected through a data operation: matter access is
    allowlist-primary, so a document moved into a matter somebody is not on becomes invisible to
    them, and moved out of a screened matter becomes visible.

    Assertion ids do not change. `matter_id` is deliberately absent from the hash that identifies
    a fact, so re-filing a document keeps every citation intact rather than forking it.
    """
    require_admin(principal)
    ctx, _ = principal
    if services.graph is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "no graph is reachable, so nothing can be linked"
        )

    from src.matters import MatterError, link_documents

    try:
        report = link_documents(services, ctx, matter_id, body.document_ids, reason=body.reason)
    except ScopeViolation as e:
        raise scope_violation_to_http(e) from e
    except MatterError as e:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(e)) from e
    return report.to_dict()
