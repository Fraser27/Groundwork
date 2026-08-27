"""Authoring governed metrics.

A data engineer defines a metric here and it becomes a tier-1 answer: a question matching
it compiles to Athena SQL with no model in the path. That is a lot of authority to hand to
a form, so three things are enforced rather than encouraged.

**A metric must compile before it can be saved.** An uncompilable definition is not a draft,
it is a broken one, and storing it means a question can match a metric that then fails at
answer time. `POST .../preview` exists so an author sees the SQL first.

**A new metric is a draft.** Authoring does not put it into service; approving does, and
approval is a separate act by someone holding the role. `list_metrics(approved_only=True)`
is what tier 1 reads.

**Every write snapshots the previous definition.** Handled by `GraphMetricStore`, and it is
why editing a live metric is safe: what it meant when it produced yesterday's answer is
still recoverable.

Admin-gated throughout. A metric is a statement about what a number means, and letting any
authenticated user redefine "fees billed" is the same class of mistake as letting them
assign themselves to a matter.
"""

from __future__ import annotations

import logging
from typing import Annotated, Any

from fastapi import APIRouter, Body, HTTPException, Query, status
from pydantic import BaseModel, Field, ValidationError, field_validator

from src.api.deps import (
    Services,
    ServicesDep,
    TenantDep,
    load_example_pack,
    require_admin,
)
from src.metrics.compiler import compile_metric
from src.metrics.graph_store import (
    STATUS_APPROVED,
    STATUS_DEPRECATED,
    STATUS_DRAFT,
    VALID_STATUSES,
    GraphMetricStore,
)
from src.metrics.models import (
    MetricDefinition,
    MetricJoin,
    MetricParameter,
    MetricRegistry,
    StaticCatalog,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["metrics"])


def _require_store(services: Services) -> GraphMetricStore:
    """Metric authoring needs the graph. Degrading is not an option here the way it is for
    retrieval: writing a definition somewhere it will not persist is worse than refusing."""
    if services.graph is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "metric authoring needs the graph, which is currently unreachable. Existing "
            "metrics keep answering questions; new definitions cannot be saved yet.",
        )
    return GraphMetricStore(services.graph)


def _catalog(services: Services, tenant_id: str) -> Any:
    """The schema the compiler validates against.

    Built from the catalog scan when there is one. An empty catalog is permissive rather
    than obstructive: `TableSchema` treats no columns as *unknown*, so a metric can be
    authored against a table nobody has scanned yet instead of being blocked on it.

    Through `catalog_reader()` rather than `services.catalog`: the raw store starts empty in
    every process, so reading it directly turns the column checks off in a cold container while
    the Tables page still looks populated.
    """
    tables = {}
    try:
        from src.metrics.models import TableSchema

        for table in services.catalog_reader().tables(tenant_id):
            tables[table.full_name] = TableSchema(
                full_name=table.full_name,
                columns={c.name: c.data_type for c in table.columns},
                primary_keys=frozenset(c.name for c in table.columns if c.is_primary_key),
            )
    except Exception as e:
        logger.debug("could not build a schema catalog: %s", e)
    return StaticCatalog(tables=tables)


def _registry(services: Services, tenant_id: str, candidate: MetricDefinition) -> MetricRegistry:
    """The metrics a derived definition may compose.

    Every stored metric, not only the approved ones. An author writing a ratio has usually
    just written both halves and neither is approved yet, and requiring an approved base would
    make composition impossible in one sitting. Approval gates *serving*, not authoring:
    `build_metric_matcher` still reads `approved_only=True`, so a draft base composed here
    still cannot answer a question.

    The submitted definition replaces its stored version, so a metric naming itself as a base
    resolves to itself and the compiler refuses it for composing a derived metric.
    """
    stored: list[MetricDefinition] | None = None
    if services.graph is not None:
        try:
            stored = GraphMetricStore(services.graph).list_metrics(tenant_id)
        except Exception as e:
            logger.warning("could not read metrics to resolve base metrics: %s", e)
    if stored is None:
        matcher = services.metric_matcher
        stored = list(matcher.metrics) if matcher is not None else []
    return MetricRegistry.from_list([*stored, candidate])


class MetricParameterIn(BaseModel):
    column: str = Field(min_length=1, max_length=128)
    operator: str = "="
    required: bool = False
    description: str = ""


class MetricJoinIn(BaseModel):
    table: str = Field(min_length=1, max_length=256)
    source_column: str = Field(min_length=1, max_length=128)
    target_column: str = Field(min_length=1, max_length=128)
    join_type: str = "INNER"


class MetricIn(BaseModel):
    """A metric definition as an author submits it."""

    metric_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.-]+$")
    name: str = Field(min_length=1, max_length=128)
    expression: str = Field(min_length=1, max_length=2000)
    source_table: str = Field(default="", max_length=256)
    definition: str = ""
    type: str = "simple"
    synonyms: list[str] = Field(default_factory=list)
    grain: list[str] = Field(default_factory=list)
    filters: list[str] = Field(default_factory=list)
    time_grains: list[str] = Field(default_factory=list)
    time_grain_column: str = ""
    aggregation: str = "additive"
    value_type: str = Field(default="number", max_length=32)
    unit: str = Field(default="", max_length=32)
    format: str = Field(default="", max_length=64)
    parameters: list[MetricParameterIn] = Field(default_factory=list)
    joins: list[MetricJoinIn] = Field(default_factory=list)
    base_metrics: list[str] = Field(default_factory=list)
    entity_columns: dict[str, str] = Field(default_factory=dict)
    owner: str = ""

    @field_validator(
        "definition", "source_table", "time_grain_column", "unit", "format", "owner", mode="before"
    )
    @classmethod
    def _null_is_empty(cls, v: Any) -> Any:
        """`_out` renders an unset field as null, so the API's own response body has to be a
        valid request body: fetch-then-save is exactly what the edit form does."""
        return "" if v is None else v

    def to_definition(self) -> MetricDefinition:
        return MetricDefinition(
            metric_id=self.metric_id,
            name=self.name,
            expression=self.expression,
            source_table=self.source_table,
            definition=self.definition,
            type=self.type,
            synonyms=list(self.synonyms),
            grain=list(self.grain),
            filters=list(self.filters),
            time_grains=list(self.time_grains),
            time_grain_column=self.time_grain_column,
            aggregation=self.aggregation,
            value_type=self.value_type,
            unit=self.unit,
            format=self.format,
            parameters=[MetricParameter(**p.model_dump()) for p in self.parameters],
            joins=[MetricJoin(**j.model_dump()) for j in self.joins],
            base_metrics=list(self.base_metrics),
            entity_columns=dict(self.entity_columns),
            owner=self.owner,
        )


def _reason(e: Exception) -> str:
    """Pydantic renders a type tag and a docs URL per error, which is noise in a toast."""
    if isinstance(e, ValidationError):
        return "; ".join(
            (".".join(str(p) for p in err["loc"]) + ": " if err["loc"] else "") + err["msg"]
            for err in e.errors()
        )
    return str(e)


def _compile_or_422(
    services: Services, tenant_id: str, metric: MetricIn | MetricDefinition
) -> tuple[MetricDefinition, Any]:
    """Model validation runs inside the same try as the compile.

    A missing source_table, a name that is not an identifier and an unparseable expression are
    all the same thing to an author -- a typo in the definition -- so all three have to read as
    422. Building the `MetricDefinition` outside made the first two an opaque 500.

    The registry is built only for a derived metric: a simple one resolves nothing through it,
    and building it costs a graph read on every preview keystroke.
    """
    try:
        definition = metric.to_definition() if isinstance(metric, MetricIn) else metric
        registry = (
            _registry(services, tenant_id, definition) if definition.type == "derived" else None
        )
        result = compile_metric(definition, _catalog(services, tenant_id), registry=registry)
    except Exception as e:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            f"this definition does not compile: {_reason(e)}",
        ) from e
    if not result.is_valid:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "this definition does not compile: " + "; ".join(result.errors),
        )
    return definition, result


def _out(metric: MetricDefinition, *, status_value: str = "", version: int = 0) -> dict[str, Any]:
    return {
        "metric_id": metric.metric_id,
        "name": metric.name,
        "definition": metric.definition,
        "expression": metric.expression,
        "source_table": metric.source_table,
        "type": metric.type,
        "synonyms": list(metric.synonyms),
        "grain": list(metric.grain),
        "filters": list(metric.filters),
        "time_grains": list(metric.time_grains),
        "time_grain_column": metric.time_grain_column or None,
        "aggregation": metric.aggregation,
        "value_type": metric.value_type,
        "unit": metric.unit,
        "format": metric.format,
        "base_metrics": list(metric.base_metrics),
        "joins": [
            {
                "table": j.table,
                "source_column": j.source_column,
                "target_column": j.target_column,
                "join_type": j.join_type,
            }
            for j in metric.joins
        ],
        "parameters": [
            {
                "column": p.column,
                "operator": p.operator,
                "required": p.required,
                "description": p.description,
            }
            for p in metric.parameters
        ],
        "entity_columns": dict(metric.entity_columns),
        "owner": metric.owner or None,
        "status": status_value or None,
        "version": version or None,
    }


@router.post("/tenants/{tenant}/metrics/preview")
async def preview_metric(
    services: ServicesDep,
    principal: TenantDep,
    body: Annotated[MetricIn, Body()],
) -> dict[str, Any]:
    """Compile a definition without saving it.

    The reviewability that makes a governed metric governed: an author sees exactly the SQL
    their definition produces before anyone can be answered with it.
    """
    require_admin(principal)
    ctx, _ = principal
    _definition, result = _compile_or_422(services, ctx.tenant_id, body)
    return {
        "metric_id": body.metric_id,
        "sql": result.sql,
        "source_table": result.source_table,
        "warnings": result.warnings,
        "note": (
            "Compiled from the definition with no model involved. Saving stores the "
            "definition as a draft; approving is what lets it answer questions."
        ),
    }


@router.post("/tenants/{tenant}/metrics", status_code=status.HTTP_201_CREATED)
async def create_metric(
    services: ServicesDep,
    principal: TenantDep,
    body: Annotated[MetricIn, Body()],
) -> dict[str, Any]:
    """Save a metric as a draft.

    Compiled first: storing a definition that cannot compile means a question can match a
    metric that then fails at answer time, which is worse than having no metric.
    """
    require_admin(principal)
    ctx, _ = principal
    store = _require_store(services)

    definition, result = _compile_or_422(services, ctx.tenant_id, body)
    saved = store.save_metric(ctx.tenant_id, definition, updated_by=ctx.user_id)

    return {
        **_out(definition, status_value=STATUS_DRAFT, version=int(saved.get("version") or 1)),
        "sql": result.sql,
        # Sent on the save as well as the preview: fan-out inflation and a unit mismatch are
        # ways the figure can be wrong while the SQL is valid, so an author who skipped the
        # preview still has to be told.
        "warnings": result.warnings,
        "note": (
            "Saved as a draft. It will not answer questions until it is approved, and "
            "approving is a separate action so authoring alone cannot put it into service."
        ),
    }


@router.put("/tenants/{tenant}/metrics/{metric_id}")
async def update_metric(
    services: ServicesDep,
    principal: TenantDep,
    metric_id: str,
    body: Annotated[MetricIn, Body()],
) -> dict[str, Any]:
    """Replace a definition, snapshotting the previous one.

    Editing a live metric is safe because of that snapshot: what it meant when it produced
    an earlier answer stays recoverable. The status is preserved, so correcting an approved
    metric does not silently take it out of service.
    """
    require_admin(principal)
    ctx, _ = principal
    store = _require_store(services)

    if body.metric_id != metric_id:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"metric_id in the body ({body.metric_id!r}) does not match the path ({metric_id!r})",
        )
    existing_status = store.status_of(ctx.tenant_id, metric_id)
    if existing_status is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"no metric {metric_id!r}")

    definition, result = _compile_or_422(services, ctx.tenant_id, body)
    saved = store.save_metric(
        ctx.tenant_id, definition, updated_by=ctx.user_id, status=existing_status
    )

    return {
        **_out(definition, status_value=existing_status, version=int(saved.get("version") or 1)),
        "sql": result.sql,
        "warnings": result.warnings,
        "note": "The previous definition was snapshotted and can be restored.",
    }


class StatusIn(BaseModel):
    status: str = Field(pattern="^(draft|approved|deprecated)$")


@router.post("/tenants/{tenant}/metrics/{metric_id}/status")
async def set_metric_status(
    services: ServicesDep,
    principal: TenantDep,
    metric_id: str,
    body: Annotated[StatusIn, Body()],
) -> dict[str, Any]:
    """Approve or deprecate a metric.

    The moment a definition starts or stops answering questions, so it is versioned like
    any other change: "when was this approved, and by whom" has to be answerable.
    """
    require_admin(principal)
    ctx, _ = principal
    store = _require_store(services)

    if body.status not in VALID_STATUSES:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"unknown status {body.status!r}")
    try:
        store.set_status(ctx.tenant_id, metric_id, body.status, updated_by=ctx.user_id)
    except LookupError as e:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(e)) from e

    serving = (
        "It can now answer questions."
        if body.status == STATUS_APPROVED
        else (
            "It will no longer answer questions."
            if body.status == STATUS_DEPRECATED
            else "It is back to a draft and will not answer questions."
        )
    )
    return {"metric_id": metric_id, "status": body.status, "note": serving}


@router.get("/tenants/{tenant}/metrics/{metric_id}/versions")
async def list_metric_versions(
    services: ServicesDep, principal: TenantDep, metric_id: str
) -> dict[str, Any]:
    """The definition's history, newest first. Capped at the ten most recent."""
    require_admin(principal)
    ctx, _ = principal
    store = _require_store(services)
    return {"metric_id": metric_id, "versions": store.list_versions(ctx.tenant_id, metric_id)}


@router.get("/tenants/{tenant}/metrics/{metric_id}/versions/{version}")
async def get_metric_version(
    services: ServicesDep, principal: TenantDep, metric_id: str, version: int
) -> dict[str, Any]:
    require_admin(principal)
    ctx, _ = principal
    store = _require_store(services)
    old = store.get_version(ctx.tenant_id, metric_id, version)
    if old is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"no version {version} of {metric_id!r}")
    return _out(old, version=version)


@router.post("/tenants/{tenant}/metrics/{metric_id}/restore/{version}")
async def restore_metric_version(
    services: ServicesDep, principal: TenantDep, metric_id: str, version: int
) -> dict[str, Any]:
    """Bring an old definition back as the current one.

    A forward write, not a rewind: the current definition is snapshotted first and the old
    one is saved as a new version, so the fact that the intervening definition once
    answered a question survives. It returns as a draft, because restoring is not the same
    as approving.
    """
    require_admin(principal)
    ctx, _ = principal
    store = _require_store(services)
    try:
        saved = store.restore_version(ctx.tenant_id, metric_id, version, updated_by=ctx.user_id)
    except LookupError as e:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(e)) from e
    return {
        "metric_id": metric_id,
        "restored_from": version,
        "version": saved.get("version"),
        "status": STATUS_DRAFT,
        "note": (
            "Restored as a draft. Approve it to put it back into service. The definition "
            "it replaced was snapshotted, so this is reversible."
        ),
    }


@router.delete("/tenants/{tenant}/metrics/{metric_id}")
async def delete_metric(
    services: ServicesDep, principal: TenantDep, metric_id: str
) -> dict[str, Any]:
    """Delete a metric and its version history.

    Deprecating is almost always the better move: it stops the metric answering while
    keeping the record of what it meant. Deletion is here for a metric created by mistake.
    """
    require_admin(principal)
    ctx, _ = principal
    store = _require_store(services)
    if store.status_of(ctx.tenant_id, metric_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"no metric {metric_id!r}")
    store.delete_metric(ctx.tenant_id, metric_id)
    return {
        "metric_id": metric_id,
        "deleted": True,
        "note": (
            "The definition and its version history are gone. Deprecating instead would "
            "have stopped it answering while keeping the record."
        ),
    }


@router.post("/tenants/{tenant}/metrics/seed")
async def seed_metrics(
    services: ServicesDep,
    principal: TenantDep,
    approve: Annotated[bool, Query()] = False,
) -> dict[str, Any]:
    """Load the YAML pack into the graph.

    Metrics authored in the UI are never overwritten by this: a deploy or a re-seed must not
    silently replace somebody's definition.
    """
    require_admin(principal)
    ctx, _ = principal
    store = _require_store(services)

    # Read from disk rather than from `services.metric_matcher`. That attribute is a test
    # seam and is None in a running system, because the matcher is built per request from
    # the tenant's approved metrics now that definitions live in the graph.
    # This tenant's pack, not a global one. A metric names real tables, so the examples worth
    # offering depend on the vocabulary the tenant chose -- and the retail pack's examples point
    # at tables a legal deployment does not have, in both directions.
    domain = services.ontology_for(ctx.tenant_id).domain
    pack = load_example_pack(domain)
    if not pack:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            f"no example metrics ship for the {domain} pack, so there is nothing to seed. "
            "Author one on the Metrics page instead: a metric names tables in your own catalog, "
            "which is why there is nothing domain-neutral to offer here.",
        )

    counts = store.seed_from_pack(
        ctx.tenant_id,
        pack,
        updated_by=ctx.user_id,
        status=STATUS_APPROVED if approve else STATUS_DRAFT,
    )
    return {
        **counts,
        "approved": approve,
        "note": (
            f"Seeded the {domain} example pack as "
            + ("approved" if approve else "drafts")
            + ". These are examples for a fictional company and name specific tables, so check "
            "each one against your own catalog before approving it. Metrics you authored in the "
            "app were left alone."
        ),
    }


# ── Reads ────────────────────────────────────────────────────────────────────
#
# Not admin-gated: knowing which metrics exist and what they mean is the point of a
# semantic layer. Only *changing* a definition is privileged.


@router.get("/tenants/{tenant}/metrics")
async def list_metrics(
    services: ServicesDep,
    principal: TenantDep,
    approved_only: Annotated[bool, Query()] = False,
) -> list[dict[str, Any]]:
    """Every metric this tenant has defined.

    Reads the graph when it is reachable and falls back to the YAML pack when it is not, so
    the Metrics page keeps working in degraded mode rather than looking empty. `approved_only`
    is what tier 1 uses.
    """
    ctx, _ = principal

    if services.graph is not None:
        try:
            store = GraphMetricStore(services.graph)
            # Whatever the graph says, including nothing. Falling back to the pack when the
            # graph answers "no approved metrics" would report six approved metrics to a
            # tenant whose metrics are all drafts, which defeats the status gate entirely.
            # The pack is a fallback for an *unreachable* graph, not for an empty answer.
            metrics = store.list_metrics(ctx.tenant_id, approved_only=approved_only)
            return [
                _out(m, status_value=store.status_of(ctx.tenant_id, m.metric_id) or "")
                for m in metrics
            ]
        except Exception as e:
            logger.warning("could not read metrics from the graph: %s", e)

    # The graph is unreachable. Report the example pack, but marked as drafts and flagged
    # as examples: these are demo metrics about a fictional firm, and presenting them as
    # this tenant's approved metrics would be a lie the Metrics page tells on every load.
    matcher = services.metric_matcher
    if matcher is None:
        return []
    return [{**_out(m, status_value=STATUS_DRAFT), "is_example": True} for m in matcher.metrics]


@router.get("/tenants/{tenant}/metrics/{metric_id}")
async def get_metric(services: ServicesDep, principal: TenantDep, metric_id: str) -> dict[str, Any]:
    ctx, _ = principal
    metric, status_value = _find(services, ctx.tenant_id, metric_id)
    return _out(metric, status_value=status_value)


@router.post("/tenants/{tenant}/metrics/{metric_id}/compile")
async def compile_existing_metric(
    services: ServicesDep, principal: TenantDep, metric_id: str
) -> dict[str, Any]:
    """Compile a stored metric to SQL without running it.

    A first-class endpoint rather than a debug affordance: the point of a governed metric is
    that a human can read exactly what it will do before it touches the warehouse.
    """
    ctx, _ = principal
    metric, _status = _find(services, ctx.tenant_id, metric_id)
    _definition, result = _compile_or_422(services, ctx.tenant_id, metric)
    return {
        "metric_id": metric.metric_id,
        "sql": result.sql,
        "source_table": result.source_table,
        "warnings": result.warnings,
        "note": (
            "Compiled from the metric definition with no model involved, so this SQL is the "
            "same every time for the same definition."
        ),
    }


def _find(services: Services, tenant_id: str, metric_id: str) -> tuple[MetricDefinition, str]:
    """A metric from the graph, or from the pack when the graph is unreachable."""
    if services.graph is not None:
        try:
            store = GraphMetricStore(services.graph)
            metric = store.get_metric(tenant_id, metric_id)
            if metric is not None:
                return metric, store.status_of(tenant_id, metric_id) or ""
        except Exception as e:
            logger.warning("could not read metric %s from the graph: %s", metric_id, e)

    matcher = services.metric_matcher
    if matcher is not None:
        for metric in matcher.metrics:
            if metric.metric_id == metric_id:
                return metric, STATUS_DRAFT
    raise HTTPException(status.HTTP_404_NOT_FOUND, f"no metric {metric_id!r}")
