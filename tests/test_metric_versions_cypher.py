"""Shape checks on the metric Cypher.

Borrowed wholesale from rosetta-sdl's `test_metric_versions.py`, and worth the same
justification: full behaviour needs a live graph, but the properties below are single tokens
in a query string that a careless edit removes silently. A dropped `tenant_id` filter does not
fail a test that only asserts "a metric came back" — it returns another firm's metric.

The tenant checks are additions rather than ports. Rosetta is single-tenant and matches on
`metric_id` alone, which is exactly the mistake this guards against here.
"""

from __future__ import annotations

import re

from src.graph import metric_queries as q

#: Every statement that touches a metric. If one is added without a tenant filter, the
#: parametrised test below fails rather than the omission being noticed in review.
ALL_METRIC_QUERIES = {
    "SNAPSHOT_METRIC_VERSION": q.SNAPSHOT_METRIC_VERSION,
    "UPSERT_METRIC": q.UPSERT_METRIC,
    "LIST_METRICS": q.LIST_METRICS,
    "LIST_APPROVED_METRICS": q.LIST_APPROVED_METRICS,
    "GET_METRIC": q.GET_METRIC,
    "LIST_METRIC_VERSIONS": q.LIST_METRIC_VERSIONS,
    "GET_METRIC_VERSION": q.GET_METRIC_VERSION,
    "DELETE_METRIC": q.DELETE_METRIC,
    "LINK_METRIC_TO_TABLE": q.LINK_METRIC_TO_TABLE,
    "METRICS_MEASURING_TABLE": q.METRICS_MEASURING_TABLE,
}


class TestTenantScoping:
    """The property that separates two law firms' metric definitions."""

    def test_every_metric_query_filters_on_tenant(self):
        missing = [
            name for name, cypher in ALL_METRIC_QUERIES.items() if "$tenant_id" not in cypher
        ]
        assert missing == [], f"queries with no tenant filter: {missing}"

    def test_no_metric_query_matches_on_metric_id_alone(self):
        """`MATCH (m:Metric {metric_id: $metric_id})` is the rosetta form and is a
        cross-tenant read here."""
        for name, cypher in ALL_METRIC_QUERIES.items():
            for match in re.finditer(r"\(\s*\w*\s*:Metric\s*\{([^}]*)\}", cypher):
                assert "tenant_id" in match.group(1), f"{name} matches :Metric without a tenant"


class TestSnapshot:
    def test_it_creates_a_versioned_node_and_links_it(self):
        assert "CREATE (mv:MetricVersion" in q.SNAPSHOT_METRIC_VERSION
        assert "CREATE (m)-[:HAS_VERSION" in q.SNAPSHOT_METRIC_VERSION

    def test_it_carries_the_fields_needed_to_restore(self):
        """A snapshot you cannot restore from is a changelog, not a version."""
        for field in (
            "expression",
            "joins_json",
            "parameters_json",
            "grain",
            "filters",
            "time_grains",
            "aggregation",
        ):
            assert field in q.SNAPSHOT_METRIC_VERSION, f"{field} missing from snapshot"

    def test_it_prunes_beyond_the_retention_cap(self):
        assert "ORDER BY old.version DESC" in q.SNAPSHOT_METRIC_VERSION
        assert f"SKIP {q.VERSION_RETENTION}" in q.SNAPSHOT_METRIC_VERSION
        assert "DETACH DELETE old" in q.SNAPSHOT_METRIC_VERSION

    def test_it_keeps_the_newest_versions(self):
        """DESC before SKIP. Ascending would delete the recent history and keep the ancient."""
        cypher = q.SNAPSHOT_METRIC_VERSION
        assert cypher.index("ORDER BY old.version DESC") < cypher.index("SKIP")

    def test_it_refuses_to_snapshot_a_metric_that_does_not_exist(self):
        """Otherwise a first write records a version full of nulls, restorable to nonsense."""
        assert "WHERE m.name IS NOT NULL" in q.SNAPSHOT_METRIC_VERSION


class TestUpsert:
    def test_it_increments_the_version_on_edit(self):
        assert "ON MATCH SET m.version = COALESCE(m.version, 1) + 1" in q.UPSERT_METRIC

    def test_a_new_metric_starts_at_version_one(self):
        assert "ON CREATE SET m.version = 1" in q.UPSERT_METRIC

    def test_it_records_who_changed_it(self):
        """A governance event with no actor is not an audit trail."""
        assert "m.updated_by = $updated_by" in q.UPSERT_METRIC
        assert "m.updated_at = $updated_at" in q.UPSERT_METRIC


class TestStatusGate:
    def test_tier_one_serves_approved_metrics_only(self):
        assert "COALESCE(m.status, 'approved') = 'approved'" in q.LIST_APPROVED_METRICS

    def test_a_seeded_metric_without_status_still_serves(self):
        """COALESCE, not equality: a YAML pack predating the field must not silently disable
        tier 1."""
        assert "COALESCE(m.status, 'approved')" in q.LIST_APPROVED_METRICS

    def test_the_unfiltered_list_does_not_gate_on_status(self):
        """An author has to be able to see their own drafts."""
        assert "= 'approved'" not in q.LIST_METRICS


class TestVersionReads:
    def test_versions_are_listed_newest_first(self):
        assert "ORDER BY mv.version DESC" in q.LIST_METRIC_VERSIONS

    def test_a_specific_version_is_addressable(self):
        assert "MetricVersion {version: $version}" in q.GET_METRIC_VERSION


class TestDeletion:
    def test_deleting_a_metric_cascades_to_its_versions(self):
        assert "HAS_VERSION" in q.DELETE_METRIC
        assert "MetricVersion" in q.DELETE_METRIC
        assert "DETACH DELETE m, mv" in q.DELETE_METRIC

    def test_the_version_match_is_optional(self):
        """A metric with no snapshots yet must still delete."""
        assert "OPTIONAL MATCH" in q.DELETE_METRIC


class TestLineage:
    def test_a_metric_links_to_the_table_it_measures(self):
        assert "MERGE (m)-[r:MEASURES]->(t)" in q.LINK_METRIC_TO_TABLE

    def test_the_link_carries_a_tenant_for_scoped_reads(self):
        """`scope.py` filters edges on tenant, so an unlabelled edge is invisible to it."""
        assert "r.tenant_id = $tenant_id" in q.LINK_METRIC_TO_TABLE

    def test_the_linked_table_is_tenant_scoped_too(self):
        """Linking to another firm's table node would cross the boundary in one hop."""
        assert "(t:Table {tenant_id: $tenant_id" in q.LINK_METRIC_TO_TABLE


class TestNeptuneCompatibility:
    """Neptune rejects a list-valued property with "Property value must be a simple literal".
    Neo4j accepts one, so this is invisible to a fake graph and to the whole suite: the Cypher
    reads identically either way, and only a real write against Neptune fails.

    That is exactly what happened. These assert the encoding directly, because the alternative
    is finding out on a deploy.
    """

    #: Every metric field that holds more than one value. All must be JSON-encoded.
    COLLECTION_FIELDS = (
        "synonyms",
        "grain",
        "filters",
        "time_grains",
        "base_metrics",
        "parameters",
        "joins",
        "entity_columns",
    )

    def test_no_collection_is_written_as_a_bare_property(self):
        for field in self.COLLECTION_FIELDS:
            assert f"m.{field} = ${field}," not in q.UPSERT_METRIC, (
                f"{field} is written as a bare property; Neptune will refuse it"
            )

    def test_every_collection_is_written_json_encoded(self):
        for field in self.COLLECTION_FIELDS:
            assert f"m.{field}_json = ${field}_json" in q.UPSERT_METRIC, (
                f"{field} must be stored as {field}_json"
            )

    def test_snapshots_carry_the_encoded_form(self):
        """A snapshot copying a bare list would fail on the same Neptune restriction."""
        for field in self.COLLECTION_FIELDS:
            assert f"{field}_json: m.{field}_json" in q.SNAPSHOT_METRIC_VERSION

    def test_reads_project_the_encoded_form(self):
        for field in self.COLLECTION_FIELDS:
            assert f"m.{field}_json AS {field}_json" in q.GET_METRIC
            assert f"mv.{field}_json AS {field}_json" in q.GET_METRIC_VERSION
