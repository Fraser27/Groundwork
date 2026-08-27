"""Cypher for governed metrics and their version history.

Ported from `rosetta-sdl`, with one change that runs through every statement: **every query
is tenant-scoped**. Rosetta is single-tenant, so `MATCH (m:Metric {metric_id: $id})` is safe
there and would be a cross-tenant read here. Two firms may both define `fees_billed` and they
are different metrics.

Why the graph rather than YAML, given that the compiler is deterministic either way: a
snapshot is what lets you prove *what a definition meant when an answer was produced*. YAML in
git gives you that for a repository, but not for a metric an author edited in the UI at
11:40 on a Tuesday. `HAS_VERSION` makes the audit question answerable from the same store the
answer came from.

The retention cap is deliberate and lossy. Ten snapshots is enough to see how a definition
drifted and to restore a recent one; unbounded history on a metric someone edits twenty times
in an afternoon is storage with no reader. If a firm needs the full trail, that is an export,
not a graph.

All Cypher lives here rather than in `src/metrics/` because `src/graph/` owns query
construction — see the working agreement. Nothing outside this package builds Cypher.
"""

from __future__ import annotations

#: How many historical versions to keep per metric.
VERSION_RETENTION = 10

#: Fields copied onto a snapshot. Complete enough to restore the definition — a snapshot you
#: cannot restore from is a changelog, not a version.
_SNAPSHOT_FIELDS = """
    tenant_id: m.tenant_id,
    metric_id: m.metric_id,
    version: COALESCE(m.version, 1),
    name: m.name,
    definition: m.definition,
    expression: m.expression,
    type: m.type,
    source_table: m.source_table,
    synonyms_json: m.synonyms_json,
    grain_json: m.grain_json,
    filters_json: m.filters_json,
    time_grains_json: m.time_grains_json,
    time_grain_column: m.time_grain_column,
    aggregation: m.aggregation,
    value_type: m.value_type,
    unit: m.unit,
    format: m.format,
    status: m.status,
    owner: m.owner,
    updated_by: m.updated_by,
    updated_at: m.updated_at,
    joins_json: m.joins_json,
    parameters_json: m.parameters_json,
    base_metrics_json: m.base_metrics_json,
    entity_columns_json: m.entity_columns_json,
    source: m.source
"""

#: Snapshot the current definition, then prune beyond the retention cap.
#:
#: `WHERE m.name IS NOT NULL` guards against snapshotting a metric that does not exist yet:
#: MERGE-then-snapshot on a first write would otherwise create a version full of nulls that
#: a restore would happily apply.
SNAPSHOT_METRIC_VERSION = f"""
MATCH (m:Metric {{tenant_id: $tenant_id, metric_id: $metric_id}})
WHERE m.name IS NOT NULL
CREATE (mv:MetricVersion {{
{_SNAPSHOT_FIELDS},
    snapshot_at: $snapshot_at
}})
CREATE (m)-[:HAS_VERSION {{tenant_id: $tenant_id}}]->(mv)
WITH m
MATCH (m)-[:HAS_VERSION]->(old:MetricVersion)
WITH old ORDER BY old.version DESC
SKIP {VERSION_RETENTION}
DETACH DELETE old
"""

#: Create or update a metric. `version` increments on every write, which is what a snapshot
#: keys on — so an author cannot overwrite a definition without leaving the previous one
#: recoverable.
UPSERT_METRIC = """
MERGE (m:Metric {tenant_id: $tenant_id, metric_id: $metric_id})
ON CREATE SET m.version = 1, m.created_at = $updated_at, m.created_by = $updated_by
ON MATCH SET m.version = COALESCE(m.version, 1) + 1
SET m.name = $name,
    m.definition = $definition,
    m.expression = $expression,
    m.type = $type,
    m.source_table = $source_table,
    m.synonyms_json = $synonyms_json,
    m.grain_json = $grain_json,
    m.filters_json = $filters_json,
    m.time_grains_json = $time_grains_json,
    m.time_grain_column = $time_grain_column,
    m.aggregation = $aggregation,
    m.value_type = $value_type,
    m.unit = $unit,
    m.format = $format,
    m.status = $status,
    m.owner = $owner,
    m.updated_by = $updated_by,
    m.updated_at = $updated_at,
    m.joins_json = $joins_json,
    m.parameters_json = $parameters_json,
    m.base_metrics_json = $base_metrics_json,
    m.entity_columns_json = $entity_columns_json,
    m.source = $source
RETURN m.metric_id AS metric_id, m.version AS version, m.status AS status
"""

_METRIC_RETURN = """
RETURN m.metric_id AS metric_id, m.name AS name, m.definition AS definition,
       m.expression AS expression, m.type AS type, m.source_table AS source_table,
       m.synonyms_json AS synonyms_json, m.grain_json AS grain_json, m.filters_json AS filters_json,
       m.time_grains_json AS time_grains_json, m.time_grain_column AS time_grain_column,
       m.aggregation AS aggregation, m.value_type AS value_type, m.unit AS unit,
       m.format AS format, m.status AS status, m.owner AS owner,
       m.version AS version, m.updated_by AS updated_by, m.updated_at AS updated_at,
       m.joins_json AS joins_json, m.parameters_json AS parameters_json,
       m.base_metrics_json AS base_metrics_json, m.entity_columns_json AS entity_columns_json,
       m.source AS source
"""

LIST_METRICS = f"""
MATCH (m:Metric {{tenant_id: $tenant_id}})
{_METRIC_RETURN}
ORDER BY m.name
"""

#: What tier 1 is allowed to serve.
#:
#: `COALESCE(m.status, 'approved')` treats a metric with no status as approved, because a
#: YAML-seeded pack predates the field and refusing those would silently disable tier 1.
LIST_APPROVED_METRICS = f"""
MATCH (m:Metric {{tenant_id: $tenant_id}})
WHERE COALESCE(m.status, 'approved') = 'approved'
{_METRIC_RETURN}
ORDER BY m.name
"""

GET_METRIC = f"""
MATCH (m:Metric {{tenant_id: $tenant_id, metric_id: $metric_id}})
{_METRIC_RETURN}
"""

LIST_METRIC_VERSIONS = """
MATCH (:Metric {tenant_id: $tenant_id, metric_id: $metric_id})-[:HAS_VERSION]->(mv:MetricVersion)
RETURN mv.version AS version, mv.name AS name, mv.status AS status,
       mv.updated_by AS updated_by, mv.updated_at AS updated_at,
       mv.snapshot_at AS snapshot_at, mv.expression AS expression
ORDER BY mv.version DESC
"""

GET_METRIC_VERSION = """
MATCH (:Metric {tenant_id: $tenant_id, metric_id: $metric_id})
      -[:HAS_VERSION]->(mv:MetricVersion {version: $version})
RETURN mv.metric_id AS metric_id, mv.name AS name, mv.definition AS definition,
       mv.expression AS expression, mv.type AS type, mv.source_table AS source_table,
       mv.synonyms_json AS synonyms_json, mv.grain_json AS grain_json, mv.filters_json AS filters_json,
       mv.time_grains_json AS time_grains_json, mv.time_grain_column AS time_grain_column,
       mv.aggregation AS aggregation, mv.value_type AS value_type, mv.unit AS unit,
       mv.format AS format, mv.status AS status, mv.owner AS owner,
       mv.version AS version, mv.updated_by AS updated_by, mv.updated_at AS updated_at,
       mv.joins_json AS joins_json, mv.parameters_json AS parameters_json,
       mv.base_metrics_json AS base_metrics_json, mv.entity_columns_json AS entity_columns_json,
       mv.source AS source
"""

#: Deleting a metric takes its snapshots with it. Orphaned MetricVersion nodes would be
#: history for a metric nobody can name, which is unreadable rather than merely untidy.
DELETE_METRIC = """
MATCH (m:Metric {tenant_id: $tenant_id, metric_id: $metric_id})
OPTIONAL MATCH (m)-[:HAS_VERSION]->(mv:MetricVersion)
DETACH DELETE m, mv
"""

#: Link a metric to the table it measures, so a catalog scan and a metric definition meet in
#: the graph. This is what makes "which metrics read this column" answerable, and it is the
#: edge the planner follows to ground a metric result in document facts.
LINK_METRIC_TO_TABLE = """
MATCH (m:Metric {tenant_id: $tenant_id, metric_id: $metric_id})
MATCH (t:Table {tenant_id: $tenant_id, full_name: $full_name})
MERGE (m)-[r:MEASURES]->(t)
SET r.tenant_id = $tenant_id
"""

#: Which metrics read a given table. The lineage question a data engineer asks before
#: changing a column.
METRICS_MEASURING_TABLE = f"""
MATCH (m:Metric {{tenant_id: $tenant_id}})-[:MEASURES]->(t:Table {{full_name: $full_name}})
{_METRIC_RETURN}
ORDER BY m.name
"""
