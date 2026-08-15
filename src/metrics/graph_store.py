"""Governed metrics in the graph, with version history.

The metric pack in YAML becomes a **seed** rather than the system of record. That buys
runtime authoring — a data engineer can define a governed metric in the UI — without giving
up the property that makes "governed" mean anything: every write snapshots the previous
definition, so what a metric meant at the moment it answered a question is recoverable.

Determinism is untouched. `src/metrics/compiler.py` still compiles a definition to SQL with
no model in the path; this module only changes where the definition is read from. A metric
loaded from the graph and the same metric loaded from YAML compile identically.

Two rules that are not negotiable here:

**Reads and writes are tenant-scoped.** Rosetta's equivalent is single-tenant and matches on
`metric_id` alone; doing that here would let one firm read another's definitions.

**Only `approved` metrics serve tier 1.** A draft is authorable and compilable but must not
answer a question, or "governed" degrades to "someone was working on it".
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

from src.graph import metric_queries as q
from src.metrics.models import MetricDefinition, MetricJoin, MetricParameter

logger = logging.getLogger(__name__)

#: A metric that has not been approved cannot answer a question.
STATUS_DRAFT = "draft"
STATUS_APPROVED = "approved"
STATUS_DEPRECATED = "deprecated"

VALID_STATUSES = frozenset({STATUS_DRAFT, STATUS_APPROVED, STATUS_DEPRECATED})

#: Where a definition came from. Seeded metrics are overwritten by a re-seed; authored ones
#: never are, because clobbering an author's work on deploy would be indefensible.
SOURCE_YAML = "yaml"
SOURCE_AUTHORED = "authored"


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _to_params(
    metric: MetricDefinition, *, tenant_id: str, updated_by: str, status: str, source: str
) -> dict[str, Any]:
    """Flatten a definition for Cypher.

    Nested structures are stored as JSON strings because Neptune property values are
    scalars or lists of scalars — a list of join objects has nowhere to live otherwise.
    Storing them as JSON keeps a snapshot restorable, which a lossy flattening would not.
    """
    return {
        "tenant_id": tenant_id,
        "metric_id": metric.metric_id,
        "name": metric.name,
        "definition": metric.definition,
        "expression": metric.expression,
        "type": metric.type,
        "source_table": metric.source_table,
        "synonyms": list(metric.synonyms),
        "grain": list(metric.grain),
        "filters": list(metric.filters),
        "time_grains": list(metric.time_grains),
        "time_grain_column": metric.time_grain_column,
        "aggregation": metric.aggregation,
        "status": status,
        "owner": metric.owner,
        "updated_by": updated_by,
        "updated_at": _now(),
        "joins_json": json.dumps([j.model_dump() for j in metric.joins]),
        "parameters_json": json.dumps([p.model_dump() for p in metric.parameters]),
        "base_metrics": list(metric.base_metrics),
        "entity_columns_json": json.dumps(getattr(metric, "entity_columns", {}) or {}),
        "source": source,
    }


def _from_row(row: dict[str, Any]) -> MetricDefinition:
    """Rebuild a definition from a graph row.

    Tolerant of missing fields on purpose: a metric seeded before a field existed should keep
    working rather than failing validation, which is the difference between an additive schema
    change and a migration.
    """
    joins = [MetricJoin(**j) for j in json.loads(row.get("joins_json") or "[]")]
    parameters = [MetricParameter(**p) for p in json.loads(row.get("parameters_json") or "[]")]
    return MetricDefinition(
        metric_id=row["metric_id"],
        name=row["name"],
        definition=row.get("definition") or "",
        expression=row["expression"],
        type=row.get("type") or "simple",
        source_table=row.get("source_table") or "",
        synonyms=list(row.get("synonyms") or []),
        grain=list(row.get("grain") or []),
        filters=list(row.get("filters") or []),
        time_grains=list(row.get("time_grains") or []),
        time_grain_column=row.get("time_grain_column") or "",
        aggregation=row.get("aggregation") or "additive",
        owner=row.get("owner") or "",
        joins=joins,
        parameters=parameters,
        base_metrics=list(row.get("base_metrics") or []),
    )


class GraphMetricStore:
    """Metric definitions in the graph. Requires a `GraphClient`."""

    def __init__(self, graph: Any) -> None:
        self.graph = graph

    # ── Reads ────────────────────────────────────────────────────────────────

    def list_metrics(
        self, tenant_id: str, *, approved_only: bool = False
    ) -> list[MetricDefinition]:
        cypher = q.LIST_APPROVED_METRICS if approved_only else q.LIST_METRICS
        rows = self.graph.query(cypher, {"tenant_id": tenant_id})
        return [_from_row(r) for r in rows]

    def get_metric(self, tenant_id: str, metric_id: str) -> MetricDefinition | None:
        rows = self.graph.query(q.GET_METRIC, {"tenant_id": tenant_id, "metric_id": metric_id})
        return _from_row(rows[0]) if rows else None

    def status_of(self, tenant_id: str, metric_id: str) -> str | None:
        rows = self.graph.query(q.GET_METRIC, {"tenant_id": tenant_id, "metric_id": metric_id})
        return rows[0].get("status") if rows else None

    def list_versions(self, tenant_id: str, metric_id: str) -> list[dict[str, Any]]:
        return self.graph.query(
            q.LIST_METRIC_VERSIONS, {"tenant_id": tenant_id, "metric_id": metric_id}
        )

    def get_version(self, tenant_id: str, metric_id: str, version: int) -> MetricDefinition | None:
        rows = self.graph.query(
            q.GET_METRIC_VERSION,
            {"tenant_id": tenant_id, "metric_id": metric_id, "version": version},
        )
        return _from_row(rows[0]) if rows else None

    def metrics_measuring(self, tenant_id: str, full_name: str) -> list[MetricDefinition]:
        """Which metrics read a table. The lineage question asked before altering a column."""
        rows = self.graph.query(
            q.METRICS_MEASURING_TABLE, {"tenant_id": tenant_id, "full_name": full_name}
        )
        return [_from_row(r) for r in rows]

    # ── Writes ───────────────────────────────────────────────────────────────

    def save_metric(
        self,
        tenant_id: str,
        metric: MetricDefinition,
        *,
        updated_by: str,
        status: str = STATUS_DRAFT,
        source: str = SOURCE_AUTHORED,
    ) -> dict[str, Any]:
        """Snapshot the current definition, then write the new one.

        Snapshot first, unconditionally. Doing it after the write would capture the new
        definition as its own history, which is worse than no history because it looks like
        a record and is not one.
        """
        if status not in VALID_STATUSES:
            raise ValueError(f"status must be one of {sorted(VALID_STATUSES)}, got {status!r}")

        self.graph.write(
            q.SNAPSHOT_METRIC_VERSION,
            {"tenant_id": tenant_id, "metric_id": metric.metric_id, "snapshot_at": _now()},
        )
        rows = self.graph.query(
            q.UPSERT_METRIC,
            _to_params(
                metric, tenant_id=tenant_id, updated_by=updated_by, status=status, source=source
            ),
        )

        if metric.source_table:
            # Best-effort: the table node exists only after a catalog scan, and a metric
            # authored before its table is scanned is still a valid metric.
            try:
                self.graph.write(
                    q.LINK_METRIC_TO_TABLE,
                    {
                        "tenant_id": tenant_id,
                        "metric_id": metric.metric_id,
                        "full_name": metric.source_table,
                    },
                )
            except Exception as e:
                logger.debug(
                    "could not link %s to %s: %s", metric.metric_id, metric.source_table, e
                )

        result = rows[0] if rows else {"metric_id": metric.metric_id, "version": 1}
        logger.info(
            "saved metric %s v%s for %s (%s)",
            metric.metric_id,
            result.get("version"),
            tenant_id,
            status,
        )
        return result

    def set_status(self, tenant_id: str, metric_id: str, status: str, *, updated_by: str) -> None:
        """Approve or deprecate. Snapshots first, because a status change is a governance
        event — "when was this approved" has to be answerable."""
        if status not in VALID_STATUSES:
            raise ValueError(f"status must be one of {sorted(VALID_STATUSES)}, got {status!r}")
        metric = self.get_metric(tenant_id, metric_id)
        if metric is None:
            raise LookupError(f"no metric {metric_id!r}")
        current_source = self.graph.query(
            q.GET_METRIC, {"tenant_id": tenant_id, "metric_id": metric_id}
        )
        source = (
            current_source[0].get("source") if current_source else SOURCE_AUTHORED
        ) or SOURCE_AUTHORED
        self.save_metric(tenant_id, metric, updated_by=updated_by, status=status, source=source)

    def delete_metric(self, tenant_id: str, metric_id: str) -> None:
        self.graph.write(q.DELETE_METRIC, {"tenant_id": tenant_id, "metric_id": metric_id})
        logger.info("deleted metric %s and its versions for %s", metric_id, tenant_id)

    def restore_version(
        self, tenant_id: str, metric_id: str, version: int, *, updated_by: str
    ) -> dict[str, Any]:
        """Bring a historical definition back as the current one.

        A restore is a forward write, not a rewind: it snapshots what is there now and then
        saves the old definition as a new version. Rewinding would erase the fact that the
        intervening definition ever answered a question.
        """
        old = self.get_version(tenant_id, metric_id, version)
        if old is None:
            raise LookupError(f"no version {version} of metric {metric_id!r}")
        return self.save_metric(
            tenant_id, old, updated_by=updated_by, status=STATUS_DRAFT, source=SOURCE_AUTHORED
        )

    # ── Seeding ──────────────────────────────────────────────────────────────

    def seed_from_pack(
        self, tenant_id: str, metrics: list[MetricDefinition], *, updated_by: str = "system"
    ) -> dict[str, int]:
        """Load a YAML pack into the graph, without clobbering authored work.

        A metric already marked `authored` is skipped: a deploy must not silently replace a
        definition someone wrote in the UI. Re-seeding an unchanged pack is otherwise a no-op
        beyond a version bump.
        """
        counts = {"created": 0, "skipped": 0}
        for metric in metrics:
            rows = self.graph.query(
                q.GET_METRIC, {"tenant_id": tenant_id, "metric_id": metric.metric_id}
            )
            if rows and (rows[0].get("source") or SOURCE_YAML) == SOURCE_AUTHORED:
                counts["skipped"] += 1
                continue
            self.save_metric(
                tenant_id,
                metric,
                updated_by=updated_by,
                # Seeded metrics arrive approved: they are the reviewed pack from the
                # repository, and shipping them as drafts would leave tier 1 dead on a
                # fresh deployment.
                status=STATUS_APPROVED,
                source=SOURCE_YAML,
            )
            counts["created"] += 1

        logger.info(
            "seeded %d metrics for %s (%d authored metrics left alone)",
            counts["created"],
            tenant_id,
            counts["skipped"],
        )
        return counts
