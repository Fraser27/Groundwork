"""Cypher for the catalog nodes a scan finds, and the descriptions attached to them.

These nodes were being built and thrown away. `scan_catalog` and `enrich_tables` both return a
`nodes` list of `CatalogNode`, and nothing ever wrote it: a live tenant held 43 `:Entity` nodes and
**zero** `:Table` or `:Column`. Two consequences, both silent.

`LINK_METRIC_TO_TABLE` matches `(t:Table {full_name})`, so metric lineage never linked and said so
only at debug level. And a `DESCRIBED_AS` edge pointed at an `:Entity` whose id was
`description:<digest>` while the description **text** lived only on the discarded node, so the
description was unrecoverable without re-running the model.

Two constraints shape every statement here.

**`GraphClient.write_batch` prepends `UNWIND $batch AS item`.** So parameters are `item.x`, never
`$x`. A statement written with `$tenant_id` binds nothing and fails silently on an empty batch.

**A node must MERGE on the key `assertion_queries.upsert_edge` uses**, which is
`(:Entity {tenant_id, entity_id})`, and gain its label with `SET n:Label`. Merging on a separate
`(:Column {node_id})` would build a parallel node set that the edges do not point at, which is the
bug this module exists to fix rather than a second copy of it.

All Cypher lives in `src/graph/` per the working agreement, and every statement is tenant-scoped.
"""

from __future__ import annotations

from src.graph.assertion_queries import safe_type

#: Nodes per transaction. Same reasoning as `schema.DELETE_BATCH_SIZE`: an unbounded write builds
#: one transaction large enough to exhaust Neptune's memory, and a wide table alone is hundreds of
#: column nodes.
UPSERT_BATCH_SIZE = 500


def upsert_node(label: str) -> str:
    """Create or update one catalog node, keyed the way an assertion edge expects.

    The label is interpolated because Cypher cannot parameterise one, so it goes through
    `safe_type` first -- the same validation a relationship type gets, reused rather than
    re-implemented so the two cannot disagree about what a safe identifier is.

    `n += item.props` rather than named SET lines: the props differ per label, and spelling out six
    near-identical statements would be six places for a field to go missing.
    """
    lbl = safe_type(label)
    return f"""
MERGE (n:Entity {{tenant_id: item.tenant_id, entity_id: item.node_id}})
SET n:{lbl}, n += item.props
"""


#: Description text for a tenant's tables and columns, scoped.
#:
#: `{scope}` is mandatory: `read_scoped` refuses a template without it. Built with `edge_scope`
#: defaults left alone, which is what makes this an *approved*-only read -- `include_pending=False`
#: filters to `SIGNED_OFF_STATES`, and the confidence floor is `>=` so a descriptive claim sitting
#: exactly on 0.8 after approval still passes.
#:
#: Returns the trust fields as well as the text, because the overlay's precedence rule needs them:
#: a person's description must beat a model's, and `epistemic_class` is the only thing that says
#: which is which.
APPROVED_DESCRIPTIONS = """
MATCH (s:Entity)-[r:DESCRIBED_AS]->(d:Description)
WHERE {scope}
RETURN s.entity_id AS subject_id,
       d.text AS text,
       r.epistemic_class AS epistemic_class,
       r.confidence AS confidence,
       r.method AS method
"""

#: Synonyms for a tenant's tables. Same scoping as the descriptions, and the same reason: an
#: unapproved synonym must not silently widen what a question matches.
APPROVED_SYNONYMS = """
MATCH (s:Entity)-[r:HAS_SYNONYM]->(t:Synonym)
WHERE {scope}
RETURN s.entity_id AS subject_id, t.name AS name
"""

#: Topics for a tenant's tables, keyed by the table's own name. Same scoping as the synonyms.
#:
#: `(s:Table)` is load-bearing rather than decoration. `CONCERNS_TOPIC` is the pack's general
#: subject-matter tag and document extraction writes it too, so matching the predicate alone would
#: list every filing's subject matter as a property of a table. The label is the `:Topic` node's own
#: `name`, never the slug in its id: ids are built, not parsed.
APPROVED_TOPICS = """
MATCH (s:Table)-[r:CONCERNS_TOPIC]->(t:Topic)
WHERE {scope}
RETURN s.full_name AS full_name, t.name AS name
"""

#: The tenant's configured sources, for rebuilding the catalog cache after a restart.
#:
#: Scoped with `node_scope`, not `edge_scope`: a source that has been registered but never scanned
#: has no `HAS_TABLE` edge, and scoping on one would hide exactly the source an operator needs to
#: see in order to press Scan.
SOURCES_FOR_TENANT = """
MATCH (s:DataSource)
WHERE {scope}
RETURN s.source_id AS source_id,
       s.type AS type,
       s.last_scanned_at AS last_scanned_at
ORDER BY s.source_id
"""

#: A tenant's tables, reached through the `HAS_TABLE` edge that declared each one.
#:
#: Through the edge rather than by matching `(t:Table {tenant_id})`: the edge is the assertion, so a
#: table only appears here if something declared it, and the source it belongs to comes from the
#: traversal instead of from a property that could disagree with it.
#:
#: `edge_scope` defaults are left alone, and unlike `APPROVED_DESCRIPTIONS` that is not an
#: approved-only read: a catalog edge is DECLARED at confidence 1.0 and so lands AUTO_ASSERTED,
#: which is in `SIGNED_OFF_STATES` and clears the floor. Passing `include_pending` here would only
#: widen this to states a catalog edge never occupies.
#:
#: `recorded_at` is on the `:Assertion` node rather than the edge, and it is the only record of when
#: the scan ran, so a source node written before `last_scanned_at` existed can still date itself.
TABLES_FOR_TENANT = """
MATCH (s:DataSource)-[r:HAS_TABLE]->(t:Table)
WHERE {scope}
OPTIONAL MATCH (a:Assertion)
WHERE a.tenant_id = r.tenant_id AND a.assertion_id = r.assertion_id
RETURN s.source_id AS source_id,
       t.full_name AS full_name,
       t.name AS name,
       t.database AS database,
       t.description AS description,
       t.catalog_type AS catalog_type,
       t.location AS location,
       a.recorded_at AS scanned_at
ORDER BY t.full_name
"""

#: A tenant's columns, with the parent table's `full_name` so rows can be grouped.
#:
#: The parent comes from the `HAS_COLUMN` edge, not from `c.table`. Same reason as above, and here
#: it matters more: the string property is a convenience copy, while the edge is what a provenance
#: read can be pointed at.
COLUMNS_FOR_TENANT = """
MATCH (t:Table)-[r:HAS_COLUMN]->(c:Column)
WHERE {scope}
RETURN t.full_name AS full_name,
       c.name AS name,
       c.data_type AS data_type,
       c.description AS description,
       c.is_partition AS is_partition,
       c.is_primary_key AS is_primary_key
ORDER BY t.full_name, c.name
"""

#: Every statement in this module, so the shape tests can be parametrised over it. A statement
#: added without a tenant filter then fails a test rather than being noticed in review.
ALL_CATALOG_QUERIES = {
    "APPROVED_DESCRIPTIONS": APPROVED_DESCRIPTIONS,
    "APPROVED_SYNONYMS": APPROVED_SYNONYMS,
    "APPROVED_TOPICS": APPROVED_TOPICS,
    "SOURCES_FOR_TENANT": SOURCES_FOR_TENANT,
    "TABLES_FOR_TENANT": TABLES_FOR_TENANT,
    "COLUMNS_FOR_TENANT": COLUMNS_FOR_TENANT,
}
