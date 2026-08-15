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

from src.admin_ops import replay, reset_derived
from src.api.deps import ServicesDep, TenantDep, require_admin
from src.discovery.catalog_store import CatalogTable
from src.discovery.glue_scanner import scan_catalog
from src.graph.assertions import EpistemicClass, ReviewState
from src.metrics.compiler import compile_metric as compile_metric_to_sql
from src.ontology.loader import ONTOLOGY_DIR, load_ontology
from src.sample_data import load_sample_data

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


def _metric_out(metric: Any) -> dict[str, Any]:
    """A metric as the UI shows it.

    `status` and `version` are derived, not stored: metrics live in YAML at this stage, so
    reporting a stored version would be inventing state. A metric in the loaded pack is one
    the pack's author approved.
    """
    return {
        "metric_id": metric.metric_id,
        "name": metric.name,
        "definition": metric.definition,
        "expression": metric.expression,
        "source_table": metric.source_table,
        "grain": list(metric.grain),
        "time_grain_column": metric.time_grain_column or None,
        "time_grains": list(metric.time_grains),
        "aggregation": metric.aggregation,
        "parameters": [
            {
                "column": p.column,
                "operator": p.operator,
                "required": p.required,
                "description": p.description,
            }
            for p in metric.parameters
        ],
        "filters": list(metric.filters),
        "synonyms": list(metric.synonyms),
        "status": "approved",
        "version": 1,
        "owner": metric.owner or None,
    }


def _find_metric(services: Any, metric_id: str) -> Any:
    matcher = services.metric_matcher
    if matcher is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "no metric pack loaded")
    for metric in matcher.metrics:
        if metric.metric_id == metric_id:
            return metric
    raise HTTPException(status.HTTP_404_NOT_FOUND, f"no metric {metric_id!r}")


@router.get("/tenants/{tenant}/dashboard")
async def dashboard(services: ServicesDep, principal: TenantDep) -> dict[str, Any]:
    ctx, grants = principal
    settings = services.settings_for(ctx.tenant_id)
    records = services.review_queue.visible(ctx)

    by_class = Counter(r.assertion.epistemic_class.value for r in records if r.is_current)
    by_state = Counter(r.assertion.review_state.value for r in records if r.is_current)

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

    matter_ids = {r.assertion.matter_id for r in records if r.assertion.matter_id}
    matters = [
        {
            "matter_id": mid,
            "assertion_count": sum(1 for r in records if r.assertion.matter_id == mid),
        }
        for mid in sorted(matter_ids)
    ]

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
    node_id: Annotated[str, Query()],
    depth: Annotated[int, Query(ge=1, le=3)] = 2,
) -> dict[str, Any]:
    """Edges around a node, for the graph explorer.

    Built from the assertion store rather than Cypher while the graph writer is
    unfinished, so the shape is the same once it is swapped for a scoped read.
    """
    ctx, _ = principal
    settings = services.settings_for(ctx.tenant_id)
    records = [r for r in services.review_queue.visible(ctx) if r.is_current]

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


@router.get("/tenants/{tenant}/metrics")
async def list_metrics(services: ServicesDep, principal: TenantDep) -> list[dict[str, Any]]:
    """Governed metrics. Empty when no pack is loaded, which disables tier 1."""
    _ctx, _ = principal
    matcher = services.metric_matcher
    if matcher is None:
        return []
    return [_metric_out(m) for m in matcher.metrics]


@router.get("/tenants/{tenant}/metrics/{metric_id}")
async def get_metric(services: ServicesDep, principal: TenantDep, metric_id: str) -> dict[str, Any]:
    _ctx, _ = principal
    metric = _find_metric(services, metric_id)
    return _metric_out(metric)


@router.post("/tenants/{tenant}/metrics/{metric_id}/compile")
async def compile_metric(
    services: ServicesDep, principal: TenantDep, metric_id: str
) -> dict[str, Any]:
    """Compile a metric to SQL without running it.

    The point of a governed metric is that a human can read exactly what it will do
    before it touches the warehouse, so this is a first-class endpoint rather than a
    debug affordance.
    """
    _ctx, _ = principal
    metric = _find_metric(services, metric_id)
    matcher = services.metric_matcher
    if matcher is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "no metric pack loaded")
    try:
        result = compile_metric_to_sql(metric, matcher.catalog)
    except Exception as e:
        # A metric that cannot compile is a definition problem an author can fix, so the
        # compiler's own message is the useful part of the response.
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(e)) from e
    return {
        "metric_id": metric.metric_id,
        "sql": result.sql,
        "source_table": result.source_table,
        "is_valid": result.is_valid,
        "errors": result.errors,
        "warnings": result.warnings,
        "note": (
            "Compiled from the metric definition with no model involved, so this SQL is "
            "the same every time for the same definition."
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

    Schemas only. No rows are read, so this is cheap and safe to re-run — the graph learns
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
            "Schemas only — no rows were read. Column types and descriptions are now "
            "citable as DECLARED facts, and metrics can be compiled against them."
        ),
    }


# ── Reset and replay ─────────────────────────────────────────────────────────
#
# These demonstrate the architecture's central claim rather than merely asserting it: if
# S3 and Glue are authoritative and everything else is derived, then the graph can be
# thrown away and rebuilt. If that is ever untrue, these are where it shows.


@router.post("/tenants/{tenant}/admin/sample-data")
async def load_sample_data_route(
    services: ServicesDep,
    principal: TenantDep,
    run_model_extraction: Annotated[bool, Query()] = True,
) -> dict[str, Any]:
    """Load the shipped legal documents through the real ingest pipeline.

    Not a graph seed. The documents are uploaded and ingested exactly as an operator's
    upload would be, so every assertion cites a page and a verbatim span that the
    Provenance page can resolve. A seeded graph would look identical until somebody
    clicked a citation.
    """
    require_admin(principal)
    ctx, _ = principal
    report = load_sample_data(services, ctx, run_model_extraction=run_model_extraction)
    if report.errors and report.documents_loaded == 0:
        raise HTTPException(status.HTTP_502_BAD_GATEWAY, "; ".join(report.errors[:3]))
    return report.to_dict()


@router.post("/tenants/{tenant}/admin/reset")
async def reset_derived_route(services: ServicesDep, principal: TenantDep) -> dict[str, Any]:
    """Drop the graph, the vector index, job state and the catalog cache. S3 is untouched.

    Safe in the sense that matters: nothing irreplaceable goes, because the documents remain
    in S3 and Replay reconstructs what was removed. Scoped to the caller's own tenant, so a
    reset in one firm cannot empty another's graph.
    """
    require_admin(principal)
    ctx, _ = principal
    return reset_derived(services, ctx).to_dict()


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
