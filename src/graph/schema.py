"""Graph constraints and indexes.

Two rules shape this file.

**Neptune has no schema DDL.** No constraints, no `CREATE INDEX`, no full-text
indexes — it indexes everything automatically. So every statement here is
Neo4j-only, applied in local development and skipped in deployment. That is a real
divergence, not a shim: a uniqueness constraint that catches a duplicate locally
will not catch it in Neptune, so uniqueness has to be enforced by content-addressed
ids (`Assertion.assertion_id`) rather than by the database.

**The index that matters is the scoped-traversal one.** Every read goes through
`scope.edge_scope`, which filters on `tenant_id`, `epistemic_class`, `confidence`,
`review_state` and `superseded_at` together. Without an index leading on
`tenant_id`, that is a full relationship scan on every request.
"""

from __future__ import annotations

import logging

from src.graph.client import GraphClient

logger = logging.getLogger(__name__)

#: Content-addressed ids mean these are belt-and-braces locally rather than the
#: only line of defence — which is what makes their absence in Neptune survivable.
CONSTRAINTS = [
    ("CREATE CONSTRAINT assertion_unique IF NOT EXISTS "
    "FOR (a:Assertion) REQUIRE a.assertion_id IS UNIQUE"),
    ("CREATE CONSTRAINT entity_unique IF NOT EXISTS "
    "FOR (e:Entity) REQUIRE (e.tenant_id, e.entity_id) IS UNIQUE"),
    ("CREATE CONSTRAINT document_unique IF NOT EXISTS "
    "FOR (d:Document) REQUIRE (d.tenant_id, d.document_id) IS UNIQUE"),
    ("CREATE CONSTRAINT matter_unique IF NOT EXISTS "
    "FOR (m:Matter) REQUIRE (m.tenant_id, m.matter_id) IS UNIQUE"),
    ("CREATE CONSTRAINT table_unique IF NOT EXISTS "
    "FOR (t:Table) REQUIRE (t.tenant_id, t.full_name) IS UNIQUE"),
    ("CREATE CONSTRAINT metric_unique IF NOT EXISTS "
    "FOR (m:Metric) REQUIRE (m.tenant_id, m.metric_id) IS UNIQUE"),
]

INDEXES = [
    # The hot path. Leads on tenant_id because that predicate is present on every
    # single read — see scope.edge_scope.
    ("CREATE INDEX assertion_scope IF NOT EXISTS "
    "FOR (a:Assertion) ON (a.tenant_id, a.epistemic_class, a.review_state)"),
    # The review queue: pending assertions for one tenant, least confident first.
    ("CREATE INDEX assertion_review IF NOT EXISTS "
    "FOR (a:Assertion) ON (a.tenant_id, a.review_state, a.confidence)"),
    # Bitemporal reads (`as_of`) and the "is this current?" check.
    ("CREATE INDEX assertion_temporal IF NOT EXISTS "
    "FOR (a:Assertion) ON (a.tenant_id, a.superseded_at)"),
    # Cascading retraction walks premises in reverse: given a retracted assertion,
    # find everything that rests on it.
    ("CREATE INDEX assertion_subject IF NOT EXISTS "
    "FOR (a:Assertion) ON (a.tenant_id, a.subject_id)"),
    ("CREATE INDEX assertion_object IF NOT EXISTS "
    "FOR (a:Assertion) ON (a.tenant_id, a.object_id)"),
    # Matter-scoped listing, and the ethical-wall filter.
    ("CREATE INDEX assertion_matter IF NOT EXISTS "
    "FOR (a:Assertion) ON (a.tenant_id, a.matter_id)"),
    "CREATE INDEX entity_lookup IF NOT EXISTS FOR (e:Entity) ON (e.tenant_id, e.kind)",
    ("CREATE INDEX document_matter IF NOT EXISTS "
    "FOR (d:Document) ON (d.tenant_id, d.matter_id)"),
]

#: Neo4j-only, and not merely because Neptune lacks the syntax: Neptune has no
#: full-text search at all. Anything that depends on these degrades to vector
#: search in deployment, so nothing may *require* them.
FULLTEXT_INDEXES = [
    (
        "entity_search",
        ("CREATE FULLTEXT INDEX entity_search IF NOT EXISTS "
        "FOR (e:Entity) ON EACH [e.name, e.aliases_text]"),
    ),
    (
        "document_search",
        ("CREATE FULLTEXT INDEX document_search IF NOT EXISTS "
        "FOR (d:Document) ON EACH [d.title, d.filename]"),
    ),
    (
        "table_search",
        ("CREATE FULLTEXT INDEX table_search IF NOT EXISTS "
        "FOR (t:Table) ON EACH [t.name, t.full_name, t.description]"),
    ),
    (
        "metric_search",
        ("CREATE FULLTEXT INDEX metric_search IF NOT EXISTS "
        "FOR (m:Metric) ON EACH [m.name, m.definition, m.synonyms_text]"),
    ),
]


def init_schema(graph: GraphClient, *, is_neptune: bool = False) -> dict[str, int]:
    """Apply constraints and indexes. Idempotent.

    Skipped entirely against Neptune, which auto-indexes and rejects DDL. Failures
    are logged rather than raised: a missing index is a performance problem, while
    refusing to start is an outage.
    """
    if is_neptune:
        logger.info("Neptune: skipping DDL (auto-indexed, no constraint support)")
        return {"constraints": 0, "indexes": 0, "fulltext": 0}

    counts = {"constraints": 0, "indexes": 0, "fulltext": 0}

    for kind, statements in (("constraints", CONSTRAINTS), ("indexes", INDEXES)):
        for stmt in statements:
            try:
                graph.write(stmt)
                counts[kind] += 1
            except Exception as e:
                logger.warning("Schema statement failed (%s): %s", kind, e)

    for name, stmt in FULLTEXT_INDEXES:
        try:
            graph.write(stmt)
            counts["fulltext"] += 1
        except Exception as e:
            logger.warning("Full-text index %s failed: %s", name, e)

    logger.info(
        "Schema ready: %d constraints, %d indexes, %d full-text",
        counts["constraints"],
        counts["indexes"],
        counts["fulltext"],
    )
    return counts
