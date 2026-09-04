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

from fastapi import APIRouter, BackgroundTasks, Body, HTTPException, Query, status
from pydantic import BaseModel, Field

from src.admin_ops import ResetScope, replay, reset_derived, sweep_undeclared_endpoints
from src.api.deps import (
    ServicesDep,
    TenantDep,
    build_router_indexer,
    require_admin,
    require_reviewer,
    scope_violation_to_http,
)
from src.constants import SELECTABLE_MODELS
from src.discovery.catalog_store import CatalogTable
from src.discovery.enrichment import (
    DESCRIBED_AS,
    description_node,
    is_catalog_claim,
)
from src.discovery.enrichment_run import (
    MAX_TABLES_PER_RUN,
    STATE_RUNNING,
    pending_for_table,
    run_enrichment,
    subject_ids_for,
)
from src.discovery.glue_scanner import (
    CATALOG_KINDS,
    parse_catalog_node_id,
    scan_catalog,
    table_node_id,
)
from src.graph.assertions import (
    DESCRIPTIVE_CONFIDENCE,
    EpistemicClass,
    ReviewState,
    SourceLocator,
    build_assertion,
)
from src.graph.scope import ScopeViolation
from src.ontology.loader import available_domains, load_ontology
from src.query_audit import MAX_SCAN

logger = logging.getLogger(__name__)

router = APIRouter(tags=["catalog"])


def available_ontology_domains() -> list[str]:
    return available_domains()


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

    # Documents, not job rows. `jobs_in_state` counted attempts, so one document uploaded four
    # times added four to the totals and appeared under several states at once -- the panel both
    # inflated and contradicted itself. One query rather than one per state, too.
    #
    # Present even when empty, like every field here: the dashboard once crashed on
    # `Object.entries(undefined)` because a field was declared in the TypeScript interface and
    # never sent, and a type is a claim about runtime data that the compiler cannot check.
    documents_by_state: dict[str, int] = {}
    try:
        from src.documents.job_store import documents_by_state as count_by_state

        documents_by_state = count_by_state(services.job_store.jobs_for_tenant(ctx.tenant_id))
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
        # Excludes catalog descriptions, so the sidebar badge matches what the review queue shows.
        # A badge reading 39 against a queue holding 0 sends somebody looking for work that is not
        # there, and the work that is there is on the Tables page.
        "pending_review": sum(
            1
            for r in records
            if r.is_current
            and r.assertion.review_state is ReviewState.PENDING
            and not is_catalog_claim(r.assertion)
        ),
        "catalog_pending": sum(
            1
            for r in records
            if r.is_current
            and r.assertion.review_state is ReviewState.PENDING
            and is_catalog_claim(r.assertion)
        ),
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
            "transitive": p.transitive,
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
                # `slug` is the id prefix and `layer` is which half of the graph it belongs to.
                # Both are sent so the UI can group nodes without hardcoding prefixes -- a pack
                # adding an entity kind should not require a UI change.
                "slug": e.slug,
                "layer": e.layer,
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


#: What the explorer calls the group for kinds no pack declares. `Ontology.layer_of` reports
#: "unknown" for those, so the two names are reconciled here rather than in the UI.
UNDECLARED_LAYER = "__undeclared__"

#: The `layer:` a pack declares on its Source, Table and Column kinds. Named because the cap
#: orders this one layer differently, and a typo would silently restore the starved default view.
CATALOG_LAYER = "catalog"


@router.get("/tenants/{tenant}/graph/neighbourhood")
async def neighbourhood(
    services: ServicesDep,
    principal: TenantDep,
    node_id: Annotated[str | None, Query()] = None,
    depth: Annotated[int, Query(ge=1, le=3)] = 2,
    limit: Annotated[int, Query(ge=1, le=2000)] = 400,
    layer: Annotated[str | None, Query(max_length=64)] = None,
) -> dict[str, Any]:
    """Edges around a node, or an overview of the graph when no node is named.

    `node_id` is optional because the explorer opens before anything is selected -- there is no
    node to centre on yet. Requiring it meant the page's first request was a 422 and the graph
    never rendered at all, which read as "the graph is empty" rather than "you have not picked a
    starting point".

    The overview is capped: a firm's whole graph is not a diagram, and drawing ten thousand edges
    produces an unreadable hairball that also freezes the browser. Governing edges are kept
    first, because those are the ones a conflict check or a privilege wall reads.

    `layer` applies that cap *within* one half of the graph instead of across the whole of it.
    Catalog edges are all descriptive, so a whole-graph cap sorted governing-first truncated a
    firm's schema away before any client-side filter could see it, and asking for the catalog then
    showed an empty or arbitrary slice of it. Only the overview takes it: a neighbourhood is
    already narrowed by the node it centres on.
    """
    ctx, _ = principal
    settings = services.settings_for(ctx.tenant_id)
    records = [r for r in services.review_queue.visible(ctx) if r.is_current]

    if node_id is None:
        return _graph_overview(services, ctx, records, settings, limit, layer)

    frontier = {node_id}
    seen_nodes: set[str] = set()
    edges: list[dict[str, Any]] = []
    onto = services.ontology_for(ctx.tenant_id)

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
                        "governing": onto.is_governing(a.predicate),
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
    layer: str | None = None,
) -> dict[str, Any]:
    """The whole tenant graph, capped, for a first look with nothing selected.

    Governing edges first: if the cap bites, the edges worth keeping are the ones that drive a
    consequence, not the subject-matter tags. Sorted by confidence within that, so a truncated
    view shows the firmest facts rather than an arbitrary slice.

    `layer` narrows before the cap, and `truncated`/`total_edges` then describe the narrowed set.
    Reporting the whole graph's total against a filtered slice would tell a reader their catalog
    view is complete when most of it was cut.

    Inside the catalog layer the sort is by depth instead, because there every edge is
    non-governing at 1.0 and the ranking above cannot tell `HAS_TABLE` from `HAS_COLUMN` -- on a
    wide schema the cap filled with column edges and the database and table level, which is the
    default view, came back empty.
    """
    edges: list[dict[str, Any]] = []
    onto = services.ontology_for(ctx.tenant_id)
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
                "governing": onto.is_governing(a.predicate),
                "matter_id": a.matter_id,
            }
        )

    if layer:
        # Either endpoint, not both, matching the explorer: the edge joining a table to a matter
        # is what makes this one graph rather than two, so it belongs to both views.
        edges = [
            e
            for e in edges
            if layer in (_layer_of(onto, e["source"]), _layer_of(onto, e["target"]))
        ]

    if layer == CATALOG_LAYER:
        edges.sort(key=lambda e: (_spine_depth(e), not e["governing"], -e["confidence"]))
    else:
        edges.sort(key=lambda e: (not e["governing"], -e["confidence"]))
    kept = edges[:limit]
    node_ids = {e["source"] for e in kept} | {e["target"] for e in kept}

    return {
        "nodes": [_node(n) for n in sorted(node_ids)],
        "edges": kept,
        "truncated": len(edges) > len(kept),
        "total_edges": len(edges),
        "layer": layer,
        "confidence_floor": settings.min_confidence_floor,
    }


def _layer_of(onto: Any, entity_id: str) -> str:
    """The layer this id's kind is declared under, named as the explorer names it."""
    found = onto.layer_of(entity_id)
    return UNDECLARED_LAYER if found == "unknown" else found


#: Anything off the spine, which is one step past its deepest rung.
_OFF_SPINE = len(CATALOG_KINDS)


def _spine_depth(edge: dict[str, Any]) -> int:
    """How far from the root of the catalog spine this edge reaches.

    Derived from the endpoints rather than from a list of predicates, so a new catalog edge is
    ordered without an edit here: `CATALOG_KINDS` is the spine in order, source then table then
    column, and an id it does not name is a leaf hanging off it -- a description, a synonym, a
    topic, a metric. The deeper end decides, which is what puts `HAS_COLUMN` after `HAS_TABLE` and
    `MEASURES` last however wide the schema is.
    """
    return max(_node_depth(edge["source"]), _node_depth(edge["target"]))


def _node_depth(entity_id: str) -> int:
    ref = parse_catalog_node_id(entity_id)
    return CATALOG_KINDS.index(ref.kind) if ref is not None else _OFF_SPINE


def _node(entity_id: str) -> dict[str, Any]:
    kind, _, rest = entity_id.partition(":")
    if not rest:
        kind, rest = "entity", entity_id
    node: dict[str, Any] = {
        "id": entity_id,
        "type": kind,
        "label": rest.replace("-", " ").replace("_", " "),
    }
    ref = parse_catalog_node_id(entity_id)
    if ref is not None:
        # `demo glue:anycorp.returns` is not a table name. Fixed here rather than in the UI,
        # which owns no id format, and the database is sent as its own field for the same reason.
        node["label"] = ref.label
        if ref.database:
            node["database"] = ref.database
    return node


# ── Structured catalog ───────────────────────────────────────────────────────
#
# Schemas, never rows. A scan records what tables exist and what shape they are; the
# rows stay in Athena and are queried in place. That is the whole reason the graph can
# hold "structured" and "unstructured" together without copying a warehouse into it.

#: What may be said about an empty catalog. Three states, because two of them look identical from
#: the cache alone and asserting the wrong one is the bug `catalog_status` exists to fix.
CATALOG_SCANNED = "scanned"
CATALOG_NEVER_SCANNED = "never_scanned"
CATALOG_UNKNOWN = "unknown"


@router.get("/tenants/{tenant}/tables")
async def list_tables(services: ServicesDep, principal: TenantDep) -> list[dict[str, Any]]:
    """Tables found by the last catalog scan. Empty until a source is scanned.

    Read through the enriched catalog, so the description shown here is the one the SQL generator
    was given. Two different answers to "what does this column mean" would make the page useless
    for judging a generated query.
    """
    ctx, _ = principal
    return [_table_summary(t) for t in services.enriched_catalog().tables(ctx.tenant_id)]


@router.get("/tenants/{tenant}/tables/{full_name}")
async def get_table(services: ServicesDep, principal: TenantDep, full_name: str) -> dict[str, Any]:
    ctx, _ = principal
    try:
        table, sources = services.enriched_catalog().with_sources(ctx.tenant_id, full_name)
    except KeyError:
        raise HTTPException(  # noqa: B904
            status.HTTP_404_NOT_FOUND, f"no table {full_name!r}"
        ) from None

    pending = _pending_descriptions(services, ctx, full_name)
    detail = {
        **_table_summary(table),
        "description_source": sources.get(""),
        "pending_description": pending.get(""),
        "pending_enrichment": len(pending),
        "synonyms": _approved_synonyms(services, ctx, table),
        "topics": _approved_topics(services, ctx, full_name),
        "columns": [
            {
                "name": c.name,
                "data_type": c.data_type,
                "description": c.description,
                "description_source": sources.get(c.name, ""),
                "pending_description": pending.get(c.name),
                "is_partition": c.is_partition,
                "is_primary_key": c.is_primary_key,
            }
            for c in table.columns
        ],
        "method": f"catalog-scan:{table.source_id}",
        "scanned_at": table.scanned_at,
        "location": table.location,
    }

    # Present and empty when the lineage was readable and nothing measures this table, absent when
    # it could not be read. The page claims "no approved metric reads this table" on the first and
    # says nothing on the second, so an empty list must never stand in for an unanswered question.
    metrics = _metrics_measuring(services, ctx.tenant_id, full_name)
    if metrics is not None:
        detail["metrics"] = metrics
    return detail


def _metrics_measuring(
    services: Any, tenant_id: str, full_name: str
) -> list[dict[str, str]] | None:
    """Approved metrics whose SQL reads this table, or None if the lineage was not readable.

    Approved only, which the graph read cannot say on its own: `MetricDefinition` carries no status,
    so the ids are intersected with the approved list. A draft counted here would make the page's
    claim about approved coverage false.
    """
    if services.graph is None:
        return None
    try:
        from src.metrics.graph_store import GraphMetricStore

        store = GraphMetricStore(services.graph)
        approved = {m.metric_id for m in store.list_metrics(tenant_id, approved_only=True)}
        measuring = store.metrics_measuring(tenant_id, full_name)
    except Exception as e:  # noqa: BLE001
        logger.debug("could not read metric lineage for %s: %s", full_name, e)
        return None
    return [
        {"metric_id": m.metric_id, "name": m.name, "definition": m.definition}
        for m in measuring
        if m.metric_id in approved
    ]


def _approved_synonyms(services: Any, ctx: Any, table: CatalogTable) -> list[str]:
    """Other names a reviewer signed off for this table.

    Keyed by graph id on the way in and by nothing on the way out: the id is built here with
    `table_node_id` rather than parsed, the same rule `catalog_overlay` states.
    """
    store = services.catalog_graph_store()
    if store is None:
        return []
    try:
        by_subject = store.approved_synonyms(ctx)
    except Exception as e:  # noqa: BLE001
        logger.debug("could not read synonyms for %s: %s", table.full_name, e)
        return []
    return list(by_subject.get(table_node_id(table.source_id, table.full_name), ()))


def _approved_topics(services: Any, ctx: Any, full_name: str) -> list[str]:
    """Subject matter a reviewer signed off for this table.

    One scoped traversal, not a scan of every live assertion the firm holds. `CONCERNS_TOPIC` is
    shared with document extraction, so `APPROVED_TOPICS` anchors the subject on `(s:Table)`:
    without that every topic a filing mentions would appear on a table page.
    """
    store = services.catalog_graph_store()
    if store is None:
        return []
    try:
        by_table = store.approved_topics(ctx)
    except Exception as e:  # noqa: BLE001
        logger.debug("could not read topics for %s: %s", full_name, e)
        return []
    return list(by_table.get(full_name, ()))


def _pending_descriptions(
    services: ServicesDep, ctx: Any, full_name: str
) -> dict[str, dict[str, Any]]:
    """Unreviewed descriptions for this table, keyed by column name (empty for the table).

    Shown so the review gate is visible on the page it matters. Without this a proposal exists,
    does nothing, and there is nowhere to approve it from.
    """
    try:
        items = services.review_queue.list_pending(ctx, limit=2000)
    except Exception as e:  # noqa: BLE001
        logger.debug("could not read pending descriptions for %s: %s", full_name, e)
        return {}

    out: dict[str, dict[str, Any]] = {}
    for item in items:
        if item.predicate != DESCRIBED_AS or item.table != full_name:
            continue
        out[item.column or ""] = {
            "assertion_id": item.assertion_id,
            "object_id": item.object_id,
            "method": item.method,
            "confidence": item.confidence,
        }
    return out


@router.get("/tenants/{tenant}/sources")
async def list_sources(services: ServicesDep, principal: TenantDep) -> list[dict[str, Any]]:
    """Configured sources and the outcome of their last scan.

    Configured sources appear before they are scanned, because an operator needs to see a
    source in order to press Scan on it.
    """
    ctx, _ = principal
    catalog = services.catalog_reader()
    for source_id in services.config.structured.glue_databases or []:
        catalog.register_source(ctx.tenant_id, source_id, name=source_id)
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
        for s in catalog.sources(ctx.tenant_id)
    ]


@router.get("/tenants/{tenant}/catalog/status")
async def catalog_status(services: ServicesDep, principal: TenantDep) -> dict[str, Any]:
    """Whether this firm's catalog is empty because nothing was scanned, or because it could not
    be read.

    The Tables page inferred "no catalogue scan has been run" from an empty list, which cannot tell
    those apart: the store is process-local, so a redeploy produced the same empty list over a graph
    holding every table. That is the same failure `HELP.reasonerStates` names for the conflict
    checker, where "no conflict found" and "no conflict could be looked for" must never render
    alike.

    A separate endpoint rather than a field on `/tables`, which returns a bare array the UI is
    already shipped against.
    """
    ctx, _ = principal
    confirmed = services.hydrate_catalog(ctx.tenant_id)
    catalog = services.catalog_reader()
    tables = len(catalog.tables(ctx.tenant_id))
    sources = len(catalog.sources(ctx.tenant_id))

    if tables:
        state, note = CATALOG_SCANNED, "A catalog scan has been recorded for this firm."
    elif confirmed:
        state, note = (
            CATALOG_NEVER_SCANNED,
            (
                "Nothing has been scanned yet. The durable copy holds no table for this firm, so "
                "there is nothing to recover: register a source and run a scan."
            ),
        )
    else:
        state, note = (
            CATALOG_UNKNOWN,
            (
                "The catalog could not be read. The graph did not answer, so an empty list here "
                "is not a claim that nothing has been scanned. Check the graph, then reload."
            ),
        )
    return {"state": state, "note": note, "tables": tables, "sources": sources}


@router.get("/tenants/{tenant}/settings")
async def get_settings(services: ServicesDep, principal: TenantDep) -> dict[str, Any]:
    """What this tenant is configured to do, as the Admin page shows it.

    Assembled from governance settings plus process config rather than stored as a blob:
    a settings row that can drift from the values actually in force is worse than no
    settings page at all.

    **This projection is what the Admin page renders after a save**, because `updateSettings`
    patches governance and then re-reads here rather than trusting the patch response. So a field
    missing from this dict does not merely fail to display -- it silently reverts the control the
    user just moved, and the save looks broken while the value sits correctly in DynamoDB. Every
    setting the page can change has to be projected here.
    """
    ctx, _ = principal
    settings = services.settings_for(ctx.tenant_id)
    models = services.config.models
    return {
        "tenant_id": ctx.tenant_id,
        "name": ctx.tenant_id,
        # The tenant's own setting, not `services.ontology.domain` -- that is the pack loaded at
        # boot, process-wide, so switching this tenant to healthcare still reported legal. The
        # entity vocabulary depends on which pack is live, which makes a wrong answer here worse
        # than cosmetic.
        "ontology_domain": settings.ontology_domain or services.ontology.domain,
        "min_confidence": settings.min_confidence_floor,
        # Sent so the floor control can respect its own lower bound. The floor must stay above
        # the cap, and without this the UI offered a range that was mostly invalid: dragging to
        # 0.65 produced a rejection explaining an invariant the screen had never mentioned.
        "model_confidence_cap": settings.model_confidence_cap,
        "block_ungoverned_queries": settings.block_ungoverned_queries,
        # The four router controls. Absent until now, so the Admin toggle read `undefined`, showed
        # off however the tenant was configured, and every attempt to turn it on was reverted by
        # the re-read that followed the save.
        "router_enabled": settings.router_enabled,
        "router_min_similarity": settings.router_min_similarity,
        "router_margin": settings.router_margin,
        "router_metric_boost": settings.router_metric_boost,
        "allowed_tiers": sorted(settings.allowed_tiers),
        # Derived, and sent anyway. Tier 2 and tier 3 are one search in opposite directions and a
        # tenant gets one of them, so "which direction" is what an administrator actually chose --
        # leaving the page to infer it from a list of integers is how a setting gets misread.
        "retrieval_direction": settings.retrieval_direction,
        "extraction_model": settings.extraction_model or models.extraction_model,
        "synthesis_model": models.synthesis_model,
        "query_model": settings.query_model,
        "retrieval_agent_model": settings.retrieval_agent_model,
        "enrichment_model": settings.enrichment_model,
        "embedding_model": services.config.vector.embedding_model,
        "available_models": _selectable_models(settings, models),
        "available_domains": available_ontology_domains(),
        # What this tenant's pack calls the unit work is organised by: Matter for law, Encounter
        # for care, Facility for lending. Sent so the nav, page titles and filter labels follow the
        # pack instead of hardcoding "Matter". The scoping key is still `matter_id` -- renaming that
        # would touch Cedar, a Cognito group and a Neptune constraint to change a caption.
        "unit_label": _unit_label(services, ctx.tenant_id),
        # Questions worth asking of this pack's data. Hardcoded in the UI until now, and one of them
        # asked whether acting for a client called Calder created a conflict -- so under any pack but
        # legal, the one affordance telling a new reader what to ask returned nothing at all.
        "example_questions": list(services.ontology_for(ctx.tenant_id).example_questions),
    }


def _selectable_models(settings: Any, models: Any) -> list[dict[str, str]]:
    """The models Admin may choose between, plus whatever this tenant already runs.

    The union matters. `available_models` used to be derived from the two configured models, so
    the dropdown could only ever offer what was already set -- there was no way to pick a cheaper
    one. But a model set by environment and absent from the curated list must still appear, or
    opening the page and saving would silently move the tenant onto a different model than the one
    that has been running.
    """
    known = {model_id for model_id, _, _ in SELECTABLE_MODELS}
    out = [
        {"id": model_id, "label": label, "note": note}
        for model_id, label, note in SELECTABLE_MODELS
    ]
    configured = {
        settings.extraction_model,
        settings.query_model,
        settings.retrieval_agent_model,
        settings.enrichment_model,
        models.extraction_model,
        models.synthesis_model,
    }
    for model_id in sorted(m for m in configured if m and m not in known):
        out.append(
            {
                "id": model_id,
                "label": model_id.split(".")[-1],
                "note": "Configured for this deployment.",
            }
        )
    return out


def _unit_label(services: Any, tenant_id: str) -> dict[str, str]:
    """Singular and plural wording for the organising unit, with a fallback.

    Falls back to Matter rather than to the entity id: a pack that declares no organising unit
    would otherwise put a bare kind name in the navigation, and the default pack is legal.
    """
    unit = services.ontology_for(tenant_id).organising_unit
    if unit is None:
        return {"singular": "Matter", "plural": "Matters"}
    return {"singular": unit.label, "plural": unit.label_plural or unit.label + "s"}


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
        for table in services.catalog_reader().tables(principal[0].tenant_id):
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
    nodes_written = 0
    graph_error: str | None = None

    # Nodes before assertions, because an edge references them. These were built and discarded
    # for the life of the project: `LINK_METRIC_TO_TABLE` matches `(t:Table {full_name})` and so
    # never linked, and `enrich_tables` had nowhere to hang a description.
    catalog_store = services.catalog_graph_store()
    if catalog_store is None:
        # Degrading is fine, reporting it as a clean scan is not. With no graph the stores below
        # are all in-process, so this returned "6 tables, 91 assertions live, graph_error: null"
        # and 200 OK, which is indistinguishable from a persisted scan. Two of those were run, the
        # container was replaced, and the firewall then refused every table the metric named --
        # by which point nothing pointed at the scan.
        graph_error = (
            "the graph was unreachable, so this scan reached only this process's cache and is "
            "lost when the container is replaced. Nothing it found is queryable. Scan again once "
            "the graph is up."
        )
        logger.warning("catalog scan for %s did not reach the graph, cache only", ctx.tenant_id)
    elif result.nodes:
        try:
            nodes_written = catalog_store.persist(result.nodes)
        except Exception as e:
            graph_error = str(e)
            logger.warning("catalog scan could not write its nodes: %s", e)

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
        "nodes_written": nodes_written,
        "scan_errors": result.errors,
        "graph_error": graph_error,
        "status": record.status,
        "last_scanned_at": record.last_scanned_at,
        "note": (
            "Schemas only, no rows were read. Column types and descriptions are now "
            "citable as DECLARED facts, and metrics can be compiled against them."
            if graph_error is None
            # The old note claimed compilable facts either way, and that is the sentence a model
            # summarising this response repeated back as success.
            else "Schemas only, no rows were read. This scan did not reach the graph, so the "
            "counts below describe this process only and nothing it found is queryable yet."
        ),
    }


class EnrichRequest(BaseModel):
    source_id: str = Field(default="glue-main", max_length=128)
    tables: list[str] = Field(default_factory=list, max_length=200)
    """Empty enriches every catalogued table, up to the run cap."""


@router.post("/tenants/{tenant}/sources/enrich", status_code=status.HTTP_202_ACCEPTED)
async def enrich_catalog(
    services: ServicesDep,
    principal: TenantDep,
    background: BackgroundTasks,
    body: Annotated[EnrichRequest, Body()],
) -> dict[str, Any]:
    """Ask a model to describe these tables and columns. Nothing goes live.

    Glue says a column is `mtr_stat_cd varchar(2)`. It does not say that is a matter status, and a
    model is good at that guess. A guess is exactly what it stays: every description is staged as
    EXTRACTED_MODEL and waits for a human, because the descriptions reach the model that writes
    SQL and an unreviewed one would steer a query nobody checked.

    202 with a background task, because a Bedrock call per table is far too slow to hold a request
    open. Each table is staged as it completes, so a container replaced mid-run loses the tables
    not yet reached rather than the whole run.

    No per-request model override: the administrator's setting is the control, and since the model
    id is the assertion's version a one-off model would quietly fork assertion identity.
    """
    require_admin(principal)
    ctx, _ = principal

    running = services.enrichment_runs.get(ctx.tenant_id)
    if running is not None and running.state == STATE_RUNNING:
        # Refusing costs a retry. An unbounded queue turns a bulk action into memory pressure and a
        # thundering herd at Bedrock, which is `IngestLimiter`'s reasoning.
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "an enrichment run is already in progress for this firm; wait for it to finish",
        )

    settings = services.settings_for(ctx.tenant_id)
    if not settings.enrichment_model:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "no enrichment model is configured, so there is nothing to propose descriptions with",
        )

    catalog = services.enriched_catalog()
    known = {t.full_name for t in catalog.tables(ctx.tenant_id)}
    unknown = [t for t in body.tables if t not in known]
    if unknown:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"not in this firm's catalog: {sorted(unknown)}. Scan the source first.",
        )

    queued = len(body.tables) if body.tables else len(known)
    background.add_task(
        run_enrichment,
        services,
        ctx,
        source_id=body.source_id,
        only=tuple(body.tables),
    )
    return {
        "status": "accepted",
        "tables_queued": min(queued, MAX_TABLES_PER_RUN),
        "max_tables_per_run": MAX_TABLES_PER_RUN,
        "model": settings.enrichment_model,
        "note": (
            "Descriptions are proposed, never applied. Poll the status endpoint for progress, "
            "then approve what is right on each table."
        ),
    }


@router.get("/tenants/{tenant}/sources/enrich/status")
async def enrichment_status(services: ServicesDep, principal: TenantDep) -> dict[str, Any]:
    """How the current or last enrichment run is going."""
    require_admin(principal)
    ctx, _ = principal
    run = services.enrichment_runs.get(ctx.tenant_id)
    if run is None:
        return {"state": "none", "note": "No enrichment run has been started on this container."}
    return run.to_dict()


class DescriptionRequest(BaseModel):
    column: str | None = Field(default=None, max_length=256)
    """None describes the table itself."""

    text: str = Field(default="", max_length=2000)
    """Empty retracts the current description rather than storing an empty one."""


@router.patch("/tenants/{tenant}/tables/{full_name:path}/description")
async def set_description(
    services: ServicesDep,
    principal: TenantDep,
    full_name: str,
    background: BackgroundTasks,
    body: Annotated[DescriptionRequest, Body()],
) -> dict[str, Any]:
    """Write a description a person typed. Live immediately.

    DECLARED, so `_derive_review_state` makes it AUTO_ASSERTED and it needs no review: a person
    asserting something is the thing review exists to obtain. Nothing here opts out of the gate;
    the assertion contract derives the state from the class.

    Confidence is DESCRIPTIVE_CONFIDENCE and not REVIEWER_CONFIDENCE, which is the subtle part.
    A reviewer's 0.98 on a description would outrank a governing fact a partner personally approved
    at 0.9, and that inversion is exactly what `DESCRIPTIVE_CONFIDENCE` exists to prevent.

    A model's competing proposal is left in the queue. Precedence at read time is the single
    definition of which description is used, and rejecting here as well would be a second answer to
    one question in a second place.
    """
    require_admin(principal)
    ctx, _ = principal

    catalog = services.enriched_catalog()
    table = catalog.table(ctx.tenant_id, full_name)
    if table is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"no catalogued table {full_name!r}")
    if body.column and body.column not in {c.name for c in table.columns}:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"{full_name} has no column {body.column!r}")

    # The request is checked before the infrastructure. A bad request is a bad request whether or
    # not the graph happens to be up, and answering 503 to it tells the caller to retry something
    # that will never succeed.
    text = body.text.strip()
    if not text:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "an empty description is not stored. Reject the proposal instead, or say what the "
            "column means.",
        )

    store = services.catalog_graph_store()
    if store is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "no graph is reachable, so nothing can be stored"
        )

    source_id = table.source_id or "glue-main"
    subject_id = subject_ids_for(source_id, full_name, body.column)

    node = description_node(ctx.tenant_id, text)
    try:
        store.persist([node])
    except Exception as e:
        # Refused rather than staged: the edge would point at a node with no text, and the
        # description would be unreadable with no way to tell it had been lost.
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY, f"the description text could not be stored: {e}"
        ) from e

    onto = services.ontology_for(ctx.tenant_id)
    assertion = build_assertion(
        tenant_id=ctx.tenant_id,
        subject_id=subject_id,
        predicate=DESCRIBED_AS,
        object_id=node.node_id,
        epistemic_class=EpistemicClass.DECLARED,
        method=f"admin:{ctx.user_id}",
        confidence=DESCRIPTIVE_CONFIDENCE,
        source_locator=SourceLocator(
            source_id=source_id, table=full_name, column=body.column or None
        ),
        allowed_predicates=onto.allowed_for(DESCRIBED_AS),
        endpoint_kinds=onto.endpoint_kinds(DESCRIBED_AS),
    )
    job_id = f"describe-{full_name}"
    services.review_queue.stage(ctx, [assertion], job_id=job_id)
    live = services.review_queue.promote(ctx, job_id=job_id)
    background.add_task(reindex_tables, services, ctx)

    return {
        "subject_id": subject_id,
        "text": text,
        "assertion_id": assertion.assertion_id,
        "live": assertion.assertion_id in live,
        "source": "human",
        "note": (
            "Recorded as a declaration, so it is live at once and outranks a model's proposal "
            "for the same column."
        ),
    }


@router.post("/tenants/{tenant}/tables/{full_name:path}/enrichment/approve")
async def approve_table_enrichment(
    services: ServicesDep,
    principal: TenantDep,
    full_name: str,
    background: BackgroundTasks,
) -> dict[str, Any]:
    """Approve every pending description, synonym and topic for one table.

    A reviewer act, so `require_reviewer` rather than `require_admin`, matching
    `routes_review.approve_many`.

    Reported per id rather than all-or-nothing: a reviewer clearing sixty columns must not lose
    fifty-nine decisions to one cascade.

    No inference pass afterwards, unlike `routes_review`. No rule in any shipped pack matches on a
    descriptive catalog predicate, so a full pass over the tenant's facts per table would be pure
    waste. That changes the day a pack writes a rule over one of these.
    """
    require_reviewer(principal)
    ctx, _ = principal

    ids = pending_for_table(services, ctx, full_name)
    approved: list[str] = []
    failed: dict[str, str] = {}
    for assertion_id in ids:
        try:
            services.review_queue.approve(ctx, assertion_id, note=f"catalog enrichment {full_name}")
            approved.append(assertion_id)
        except Exception as e:  # noqa: BLE001
            failed[assertion_id] = str(e)

    promoted = services.review_queue.promote(ctx) if approved else []
    # After the response. An approval that 500s because Bedrock is slow to embed would be worse
    # than a routing index that is a moment stale, and the approval itself is already durable.
    if approved:
        background.add_task(reindex_tables, services, ctx)
    return {
        "table": full_name,
        "pending": len(ids),
        "approved": len(approved),
        "live": len(promoted),
        "failed": failed,
        "note": (
            "Approved descriptions now reach the model that writes SQL for ungoverned questions. "
            "The words just approved reach tier selection once the index refresh finishes, a "
            "moment behind this response."
        ),
    }


def reindex_tables(services: Any, ctx: Any) -> None:
    """Re-describe the table layer after a description or a synonym changed.

    Without this, approving a description changes nothing a question can reach until somebody
    presses Rebuild: the routing index holds the words that were embedded, not the words that are
    approved now.

    Tables only. A catalog approval cannot change a metric definition or a fact, and the entity
    layer is the expensive one.
    """
    indexer = build_router_indexer(services)
    if indexer is None:
        return
    try:
        report = indexer.rebuild(ctx, metrics=False, tables=True, entities=False)
    except Exception as e:  # noqa: BLE001
        logger.warning("table layer not reindexed for %s: %s", ctx.tenant_id, e)
        return
    if report.errors:
        logger.warning("table layer reindexed with errors for %s: %s", ctx.tenant_id, report.errors)


class RouterRebuildRequest(BaseModel):
    """Which layers to rebuild. All three by default, which is what an operator means by
    Rebuild; the flags exist so a catalog scan can refresh only tables."""

    metrics: bool = True
    tables: bool = True
    entities: bool = True


@router.post("/tenants/{tenant}/admin/router/rebuild")
async def rebuild_router_index(
    services: ServicesDep,
    principal: TenantDep,
    body: Annotated[RouterRebuildRequest | None, Body()] = None,
) -> dict[str, Any]:
    """Re-describe this tenant's metrics, tables and entities for the tier router.

    Nothing here changes a metric definition, a table or a fact — it rebuilds the descriptions
    tier selection is matched against, so it is safe to re-run and converges: a description's id
    is derived from its item.

    Approved metrics only. A draft that could route a question would send it to tier 1 and then
    fail to match, which is worse than routing it honestly elsewhere.

    Inline rather than in the background, because the operator pressing Rebuild wants the counts.
    That bounds it by the origin timeout, which is why the entity layer is capped.
    """
    require_admin(principal)
    ctx, _ = principal
    body = body or RouterRebuildRequest()

    indexer = build_router_indexer(services)
    if indexer is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "no routing index configured (VECTOR_ENDPOINT unset), so tier selection stays "
            "keyword-based and there is nothing to rebuild",
        )

    report = indexer.rebuild(ctx, metrics=body.metrics, tables=body.tables, entities=body.entities)
    return report.to_dict()


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
    routing: bool = True

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
        routing=body.routing,
        metrics=body.metrics,
    )
    return reset_derived(services, ctx, scope).to_dict()


class EndpointSweepRequest(BaseModel):
    dry_run: bool = True
    """Defaults to a preview. This withdraws facts a firm may have relied on, so seeing the list
    is the default and writing is the request — the same posture as `retract` and a merge."""


@router.post("/tenants/{tenant}/admin/sweep-endpoints")
async def sweep_endpoints_route(
    services: ServicesDep,
    principal: TenantDep,
    # No default body, unlike the reset route: this one writes only when asked, so `dry_run` has
    # to be stated. An omitted body defaulting to a preview would be safe but silently ambiguous.
    body: Annotated[EndpointSweepRequest, Body()],
) -> dict[str, Any]:
    """Withdraw live facts whose endpoints their pack does not declare.

    `build_assertion` refuses these on write now, but that is prospective: the graph already holds
    facts written before the check existed. One was a `POTENTIAL_CONFLICT` stored `Party -> Party`
    against its declared `Matter -> Party`, and since that predicate declares `blocks: both` it
    withheld the firm's own client's file.

    Retracts rather than deletes, so an as-of read before now still shows what the graph asserted
    while advice rested on it.
    """
    require_admin(principal)
    ctx, _ = principal
    return sweep_undeclared_endpoints(services, ctx, dry_run=body.dry_run).to_dict()


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
    emails = _actor_emails(services, ctx.tenant_id)
    return {
        "events": [{**e.to_dict(), "actor_email": emails.get(e.actor)} for e in events],
        "count": len(events),
        "note": (
            "Append-only. Nothing here is edited or removed, and neither are the facts it "
            "describes: they are closed rather than deleted, so an as-of read before an entry "
            "still reconstructs what the graph held."
        ),
    }


def _actor_emails(services: Any, tenant_id: str) -> dict[str, str]:
    """Cognito sub to email, for the audit surfaces.

    Resolved at read time rather than stored on the event: an email can change and a sub cannot,
    so the sub stays the recorded identity. But "84289448-90d1-70f2-..." tells a reader nothing
    about who acted, which is most of what an audit trail is for.

    Empty on failure and the caller falls back to the sub. A less readable trail beats a page that
    will not load.
    """
    directory = getattr(services, "tenant_directory", None)
    if directory is None or not hasattr(directory, "users_for_tenant"):
        return {}
    try:
        return {u.sub: u.email for u in directory.users_for_tenant(tenant_id) if u.email}
    except Exception as e:  # noqa: BLE001
        logger.debug("could not resolve actor emails: %s", e)
        return {}


@router.get("/tenants/{tenant}/audit/questions")
async def question_audit_log(
    services: ServicesDep,
    principal: TenantDep,
    limit: Annotated[int, Query(ge=1, le=500)] = 100,
    assertion_id: Annotated[str | None, Query()] = None,
) -> dict[str, Any]:
    """What was asked, who asked it, which tier answered, and on what basis. Newest first.

    `assertion_id` inverts it: which questions rested on one fact. That is the question asked
    after a fact turns out to be wrong, and it is why the assertion ids are stored rather than
    only counted. Bounded by `scanned` — there is no index on a citation, so this is exact within
    the window it read and makes no claim beyond it.
    """
    require_admin(principal)
    ctx, _ = principal

    audit = services.query_audit
    if audit is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "no query audit log is configured")

    emails = _actor_emails(services, ctx.tenant_id)

    if assertion_id:
        window = min(max(limit, MAX_SCAN), 500)
        scanned = audit.questions(ctx.tenant_id, limit=window)
        events = [e for e in scanned if e.uses(assertion_id)]
        return {
            "questions": [{**e.to_dict(), "actor_email": emails.get(e.actor)} for e in events],
            "count": len(events),
            "scanned": len(scanned),
            "assertion_id": assertion_id,
            "note": (
                f"Questions among the last {len(scanned)} that used this fact. Exact within that "
                "window and no further: one question cites many facts, so there is no index to "
                "read instead."
            ),
        }

    events = audit.questions(ctx.tenant_id, limit=limit)
    return {
        "questions": [{**e.to_dict(), "actor_email": emails.get(e.actor)} for e in events],
        "count": len(events),
        "scanned": len(events),
        "assertion_id": None,
        "note": (
            "Append-only. Every answered question is recorded with the tier that answered and the "
            "facts it used, so which advice rested on a given fact stays answerable. Refused "
            "questions are not here -- they produced no answer and appear in Governance."
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
