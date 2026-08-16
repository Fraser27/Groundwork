"""Cypher for matters as records rather than as a side effect of grouping.

Until now a matter was not stored: `GET /matters` grouped assertions by `matter_id` and reported
the distinct values. That has two consequences which look small and are not.

**An empty matter cannot exist.** So a matter cannot be created before a document is filed under
it, which is backwards: staffing a team and raising an ethical screen both happen *before* the
first document arrives, and you cannot screen a lawyer from a matter that does not exist yet.

**A typo becomes a matter.** `NTL-2026-0114` and `NTL-2026-114` are two matters if two uploads
disagree, and nothing notices, because the list is whatever the data happens to contain.

Stored in the graph rather than DynamoDB, deliberately, and with a known consequence: a reset
drops derived graph data and would take these with it. That is acceptable because matter metadata
also lives in the Glue catalog in structured form, so the graph stays a derived index that can be
rebuilt rather than becoming a system of record. If that stops being true, these belong in
DynamoDB beside the grants.

All Cypher lives in `src/graph/` per the working agreement, and every statement is tenant-scoped.
"""

from __future__ import annotations

#: Create or update a matter. MERGE on (tenant, matter_id) so re-creating an existing matter is an
#: update rather than a duplicate -- two people setting up the same matter should converge.
#:
#: `created_at` is set only on first write, via ON CREATE, so an update does not rewrite history.
UPSERT_MATTER = """
MERGE (m:Matter {tenant_id: $tenant_id, matter_id: $matter_id})
ON CREATE SET m.created_at = $at, m.created_by = $actor
SET m.name = $name,
    m.updated_at = $at,
    m.updated_by = $actor
RETURN m
"""

#: Every matter for a tenant. Not scoped by the matter wall here: the caller applies that, because
#: an administrator managing a screen has to be able to see the matter the screen applies to.
LIST_MATTERS = """
MATCH (m:Matter {tenant_id: $tenant_id})
RETURN m
ORDER BY m.matter_id
"""

#: One matter, for the existence check an upload depends on.
GET_MATTER = """
MATCH (m:Matter {tenant_id: $tenant_id, matter_id: $matter_id})
RETURN m
"""

#: Move a document's facts to a different matter, on **both** copies.
#:
#: Both, because the trust fields are duplicated by design: the edge copy is what `edge_scope`
#: filters a traversal on, the node copy is what the audit read returns. Updating only the node
#: would leave a fact visible to the old matter's team and invisible to the new one -- a silent
#: access change, which is the worst kind.
#:
#: `matter_id` is deliberately absent from the assertion hash (`_compute_id`), so this is a
#: property update and not a new assertion. Re-filing a document does not fork its facts.
#:
#: Returns the ids rather than a count, because the audit event records *which* facts changed
#: hands, and a count alone cannot answer "did this move the fact I am asking about". Bounded by
#: one document's assertions, and `graph_audit.MAX_STORED_IDS` clips the stored list.
#:
#: OPTIONAL on the edge match: an assertion whose edge is missing still had its matter changed, and
#: a plain MATCH drops it from the result, so the audit would under-report a move that happened.
RELINK_DOCUMENT_ASSERTIONS = """
MATCH (a:Assertion {tenant_id: $tenant_id})
WHERE a.document_id = $document_id
SET a.matter_id = $matter_id
WITH collect(a.assertion_id) AS ids
UNWIND ids AS aid
OPTIONAL MATCH ()-[r {tenant_id: $tenant_id, assertion_id: aid}]->()
SET r.matter_id = $matter_id
RETURN collect(DISTINCT aid) AS assertion_ids
"""

#: How many facts a document contributed, so a link can report what it moved.
COUNT_DOCUMENT_ASSERTIONS = """
MATCH (a:Assertion {tenant_id: $tenant_id})
WHERE a.document_id = $document_id
RETURN count(a) AS n
"""

#: Delete a matter record. Its facts are untouched: they keep their `matter_id`, so deleting the
#: record loses the name and nothing else. Withdrawing the facts is `wipe_matter`, which is a
#: separate and louder act.
DELETE_MATTER = """
MATCH (m:Matter {tenant_id: $tenant_id, matter_id: $matter_id})
DELETE m
"""
