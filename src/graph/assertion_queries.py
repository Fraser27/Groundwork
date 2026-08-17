"""Cypher for persisting assertions, and the hybrid reification it implements.

Every statement is tenant-scoped. All Cypher lives in `src/graph/` per the working
agreement, so `src/documents/` calls these rather than building query strings.

**The storage shape**, which the README specifies and this file is the implementation of:

    (:Entity)-[:PREDICATE {assertion_id, epistemic_class, confidence, ...}]->(:Entity)
         +
    (:Assertion {method, source_locator, valid_from, ...})-[:PREMISE]->(:Assertion)

The edge carries only what a filtered traversal needs, so "walk the edges I trust" is one
hop and `scope.edge_scope` can do its five filters without touching a second node. Full
provenance and the premise DAG live on the `:Assertion` node, read only when a user asks
*why*. Pure reification would triple the hops on every read; edge properties alone could not
express a proof tree.

Two consequences worth stating because they look like duplication and are not:

- Trust-filter fields (`tenant_id`, `epistemic_class`, `confidence`, `review_state`,
  `matter_id`, `superseded_at`) appear on **both** the edge and the node. The edge copy is
  what makes a scoped traversal cheap; the node copy is what makes the audit read complete.
- The predicate is the *relationship type*, which cannot be parameterised in Cypher. It is
  interpolated, so it must be validated against the closed vocabulary first — which
  `build_assertion` already did before anything reaches here. `_TYPE_SAFE` is the second
  belt: a type that is not an identifier never reaches the database.
"""

from __future__ import annotations

import re

#: A relationship type must be a bare identifier. Interpolation is unavoidable for a
#: relationship type, so anything that could close a bracket or start a new clause is
#: refused rather than escaped.
_TYPE_SAFE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


class UnsafeRelationshipType(ValueError):
    """A predicate that cannot be interpolated into Cypher.

    Should be unreachable: `build_assertion` validates against the pack's closed vocabulary
    first. Kept because "unreachable" and "unchecked" must not become the same thing on the
    one code path that builds a query string.
    """


def safe_type(predicate: str) -> str:
    if not _TYPE_SAFE.match(predicate or ""):
        raise UnsafeRelationshipType(f"{predicate!r} is not a valid relationship type")
    return predicate


#: Properties carried on the `:Assertion` node. The full record: everything needed to defend
#: or retract the claim without consulting anything else.
_NODE_PROPS = """
    a.tenant_id = $tenant_id,
    a.subject_id = $subject_id,
    a.predicate = $predicate,
    a.object_id = $object_id,
    a.epistemic_class = $epistemic_class,
    a.method = $method,
    a.confidence = $confidence,
    a.raw_confidence = $raw_confidence,
    a.matter_id = $matter_id,
    a.review_state = $review_state,
    a.reviewed_by = $reviewed_by,
    a.reviewed_at = $reviewed_at,
    a.rule_id = $rule_id,
    a.rule_version = $rule_version,
    a.valid_from = $valid_from,
    a.valid_until = $valid_until,
    a.recorded_at = $recorded_at,
    a.superseded_at = $superseded_at,
    a.lifecycle = $lifecycle,
    a.job_id = $job_id,
    a.review_note = $review_note,
    a.retracted_reason = $retracted_reason,
    a.retracted_by = $retracted_by,
    a.corrects = $corrects,
    a.source_locator_json = $source_locator_json,
    a.premises_json = $premises_json,
    a.document_id = $document_id,
    a.filename = $filename,
    a.page = $page,
    a.quote = $quote
"""

#: Upsert the provenance node. MERGE on `assertion_id` because it is content-addressed:
#: re-ingesting the same document produces the same id, so a second run converges instead of
#: duplicating. Scoped by tenant in the MERGE key, not just the properties, so two firms that
#: somehow computed the same id could never collide on one node.
UPSERT_ASSERTION_NODE = f"""
MERGE (a:Assertion {{tenant_id: $tenant_id, assertion_id: $assertion_id}})
SET {_NODE_PROPS}
"""

#: Link an inference to the assertions it rests on.
#:
#: Written as a separate statement because a premise may be staged in the same batch, so the
#: node it points at need not exist when the inference is written. MATCH on both ends means a
#: premise that never arrives leaves no edge rather than a dangling one — and
#: `build_assertion` has already refused an INFERRED assertion with no premises, so an empty
#: result here means a genuinely missing premise, not a rule without any.
LINK_PREMISES = """
MATCH (a:Assertion {tenant_id: $tenant_id, assertion_id: $assertion_id})
MATCH (p:Assertion {tenant_id: $tenant_id, assertion_id: $premise_id})
MERGE (a)-[:PREMISE {tenant_id: $tenant_id}]->(p)
"""


def upsert_edge(predicate: str) -> str:
    """The traversable edge, with the trust fields `edge_scope` filters on.

    `assertion_id` is in the MERGE key rather than only in SET. Two different assertions can
    legitimately claim the same subject-predicate-object — a model reading it from one
    document and a rule inferring it are distinct claims with distinct provenance, and
    collapsing them would silently discard one audit trail.
    """
    rel = safe_type(predicate)
    return f"""
MERGE (s:Entity {{tenant_id: $tenant_id, entity_id: $subject_id}})
MERGE (o:Entity {{tenant_id: $tenant_id, entity_id: $object_id}})
MERGE (s)-[r:{rel} {{tenant_id: $tenant_id, assertion_id: $assertion_id}}]->(o)
SET r.epistemic_class = $epistemic_class,
    r.confidence = $confidence,
    r.matter_id = $matter_id,
    r.review_state = $review_state,
    r.valid_from = $valid_from,
    r.valid_until = $valid_until,
    r.superseded_at = $superseded_at,
    r.method = $method
"""


#: Live assertions for a tenant, scoped. Reads the node rather than the edges because the
#: caller wants whole assertions; a traversal that walks edges uses `edge_scope` directly.
#:
#: `{scope}` is mandatory: `GraphClient.read_scoped` refuses a template without it, which is
#: what makes an unscoped read unexpressible.
LIST_ASSERTIONS = """
MATCH (a:Assertion)
WHERE {scope}
RETURN a
ORDER BY a.recorded_at DESC
LIMIT $limit
"""

#: Distinct entity ids and the predicates they participate in, for the router's index.
#:
#: Read off the `:Assertion` node rather than by walking edges, so `{scope}` binds to `a` exactly
#: as it does in `LIST_ASSERTIONS` and one scope builder covers both. Walking `(s)-[r]->(o)` would
#: need a second scope on the relationship and the predicate would come from `type(r)`, which is
#: the same value stored here.
#:
#: Subject and object are unioned so an entity that only ever appears as an object still gets a
#: routable description. Otherwise a court named in twenty filings would be absent from the index
#: and a question about it would route away from the graph entirely.
LIST_ENTITY_IDS = """
MATCH (a:Assertion)
WHERE {scope}
UNWIND [a.subject_id, a.object_id] AS entity_id
WITH entity_id, a.predicate AS predicate
WHERE entity_id IS NOT NULL
RETURN entity_id, collect(DISTINCT predicate) AS predicates
ORDER BY entity_id
LIMIT $limit
"""

#: One assertion by id, for the provenance read.
GET_ASSERTION = """
MATCH (a:Assertion)
WHERE {scope} AND a.assertion_id = $assertion_id
RETURN a
"""

#: Assertions that name this one as a premise. Drives the retraction cascade: withdraw a fact
#: and anything inferred from it must be withdrawn too, or a conclusion outlives its reason.
DEPENDENTS_OF = """
MATCH (a:Assertion)-[:PREMISE]->(p:Assertion {tenant_id: $scope_tenant, assertion_id: $assertion_id})
WHERE {scope}
RETURN a
"""

#: Mark superseded rather than delete, on both node and edge. Assertions are never edited or
#: removed: a correction records a new fact and closes the old one, so the audit trail stays
#: intact and an `as_of` read still reconstructs what the file showed at the time.
SUPERSEDE_ASSERTION = """
MATCH (a:Assertion {tenant_id: $tenant_id, assertion_id: $assertion_id})
SET a.superseded_at = $at,
    a.review_state = COALESCE($review_state, a.review_state),
    a.lifecycle = COALESCE($lifecycle, a.lifecycle)
WITH a
MATCH ()-[r {tenant_id: $tenant_id, assertion_id: $assertion_id}]->()
SET r.superseded_at = $at
"""

#: Update review state on both copies. The edge copy is what `edge_scope` filters on, so an
#: approval that updated only the node would leave the fact invisible to every traversal.
SET_REVIEW_STATE = """
MATCH (a:Assertion {tenant_id: $tenant_id, assertion_id: $assertion_id})
SET a.review_state = $review_state,
    a.reviewed_by = $reviewed_by,
    a.reviewed_at = $reviewed_at,
    a.lifecycle = $lifecycle
WITH a
OPTIONAL MATCH ()-[r {tenant_id: $tenant_id, assertion_id: $assertion_id}]->()
SET r.review_state = $review_state
"""

#: Drop every assertion for a tenant. Used by the reset surface, which is only defensible
#: because S3 keeps the documents and a replay rebuilds all of this.
DELETE_TENANT_ASSERTIONS = """
MATCH (a:Assertion {tenant_id: $tenant_id})
DETACH DELETE a
"""

DELETE_TENANT_EDGES = """
MATCH (s:Entity {tenant_id: $tenant_id})-[r]->()
DELETE r
"""

DELETE_TENANT_ENTITIES = """
MATCH (e:Entity {tenant_id: $tenant_id})
DETACH DELETE e
"""
