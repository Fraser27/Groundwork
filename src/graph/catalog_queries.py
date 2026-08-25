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

#: Every statement in this module, so the shape tests can be parametrised over it. A statement
#: added without a tenant filter then fails a test rather than being noticed in review.
ALL_CATALOG_QUERIES = {
    "APPROVED_DESCRIPTIONS": APPROVED_DESCRIPTIONS,
    "APPROVED_SYNONYMS": APPROVED_SYNONYMS,
}
