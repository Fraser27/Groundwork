"""Governed metrics in the graph, and their version history.

Two properties carry the governance weight, and both are silent when broken:

**A write must snapshot the previous definition first.** Otherwise "what did this metric mean
when it produced that answer" is unanswerable, which is the whole reason definitions moved
into the graph rather than staying in YAML.

**Only approved metrics serve tier 1.** A draft that can answer a question turns "governed"
into "someone was working on it".

The graph is faked. These test the store's logic and the shape of the Cypher, not Neptune —
`tests/test_metric_versions_cypher.py` guards the query text.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.metrics.graph_store import (
    SOURCE_AUTHORED,
    STATUS_APPROVED,
    STATUS_DEPRECATED,
    STATUS_DRAFT,
    GraphMetricStore,
)
from src.metrics.models import MetricDefinition, MetricParameter

TENANT = "demo-firm"
OTHER = "other-firm"


def a_metric(metric_id: str = "m_001", **over: Any) -> MetricDefinition:
    kw: dict[str, Any] = {
        "metric_id": metric_id,
        "name": "fees_billed",
        "definition": "Value invoiced to clients.",
        "expression": "SUM(invoice_amount)",
        "source_table": "legal_ops.invoices",
        "synonyms": ["billings"],
        "grain": ["invoice_date"],
        "filters": ["invoice_status = 'ISSUED'"],
        "time_grains": ["month", "quarter"],
        "time_grain_column": "invoice_date",
        "aggregation": "additive",
        "parameters": [MetricParameter(column="practice_group", operator="=", required=False)],
    }
    kw.update(over)
    return MetricDefinition(**kw)


class FakeGraph:
    """A graph that implements just enough of the metric Cypher to test the store."""

    def __init__(self) -> None:
        self.metrics: dict[tuple[str, str], dict[str, Any]] = {}
        self.versions: list[dict[str, Any]] = []
        self.linked: list[dict[str, Any]] = []
        self.deleted: list[tuple[str, str]] = []
        self.order: list[str] = []

    def write(self, cypher: str, params: dict[str, Any] | None = None) -> None:
        p = params or {}
        if "CREATE (mv:MetricVersion" in cypher:
            self.order.append("snapshot")
            key = (p["tenant_id"], p["metric_id"])
            if key in self.metrics:
                self.versions.append(dict(self.metrics[key]))
        elif "MERGE (m)-[r:MEASURES]" in cypher:
            self.linked.append(p)
        elif "DETACH DELETE m, mv" in cypher:
            key = (p["tenant_id"], p["metric_id"])
            self.metrics.pop(key, None)
            self.versions = [v for v in self.versions if v.get("metric_id") != p["metric_id"]]
            self.deleted.append(key)

    def query(self, cypher: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        p = params or {}
        key = (p.get("tenant_id"), p.get("metric_id"))

        if "MERGE (m:Metric" in cypher:
            self.order.append("upsert")
            prior = self.metrics.get(key)
            row = dict(p)
            row["version"] = (prior.get("version", 1) + 1) if prior else 1
            self.metrics[key] = row
            return [
                {"metric_id": row["metric_id"], "version": row["version"], "status": row["status"]}
            ]

        if "ORDER BY mv.version DESC" in cypher:
            return sorted(
                (
                    {
                        "version": v["version"],
                        "name": v["name"],
                        "status": v["status"],
                        "expression": v["expression"],
                    }
                    for v in self.versions
                    if v.get("metric_id") == p.get("metric_id")
                ),
                key=lambda r: -r["version"],
            )

        if "MetricVersion {version: $version}" in cypher:
            for v in self.versions:
                if v.get("metric_id") == p.get("metric_id") and v["version"] == p.get("version"):
                    return [v]
            return []

        if "WHERE COALESCE(m.status, 'approved') = 'approved'" in cypher:
            return [
                m
                for (t, _), m in self.metrics.items()
                if t == p.get("tenant_id") and (m.get("status") or "approved") == "approved"
            ]

        if "ORDER BY m.name" in cypher and "MEASURES" in cypher:
            return [
                m
                for (t, _), m in self.metrics.items()
                if t == p.get("tenant_id") and m.get("source_table") == p.get("full_name")
            ]

        if "ORDER BY m.name" in cypher:
            return [m for (t, _), m in self.metrics.items() if t == p.get("tenant_id")]

        if "RETURN m.metric_id" in cypher:
            row = self.metrics.get(key)
            return [row] if row else []

        return []


@pytest.fixture
def graph() -> FakeGraph:
    return FakeGraph()


@pytest.fixture
def store(graph: FakeGraph) -> GraphMetricStore:
    return GraphMetricStore(graph)


class TestRoundTrip:
    def test_a_saved_metric_reads_back_intact(self, store):
        """A definition that does not survive the round trip compiles to different SQL, which
        would break determinism without any error."""
        store.save_metric(TENANT, a_metric(), updated_by="alice")
        got = store.get_metric(TENANT, "m_001")
        assert got is not None
        assert got.expression == "SUM(invoice_amount)"
        assert got.source_table == "legal_ops.invoices"
        assert got.time_grains == ["month", "quarter"]
        assert got.filters == ["invoice_status = 'ISSUED'"]

    def test_parameters_survive_the_json_flattening(self, store):
        """Parameters are the closed set of columns a caller may filter on, so losing them
        silently widens what a governed metric permits."""
        store.save_metric(TENANT, a_metric(), updated_by="alice")
        got = store.get_metric(TENANT, "m_001")
        assert [p.column for p in got.parameters] == ["practice_group"]

    def test_an_unknown_metric_is_none(self, store):
        assert store.get_metric(TENANT, "nope") is None


class TestTenantScoping:
    def test_one_tenants_metric_is_invisible_to_another(self, store):
        """Rosetta matches on metric_id alone because it is single-tenant. Doing that here
        would be a cross-tenant read of a firm's definitions."""
        store.save_metric(TENANT, a_metric(), updated_by="alice")
        assert store.get_metric(OTHER, "m_001") is None
        assert store.list_metrics(OTHER) == []

    def test_two_tenants_may_define_the_same_metric_id(self, store):
        store.save_metric(TENANT, a_metric(name="ours"), updated_by="alice")
        store.save_metric(OTHER, a_metric(name="theirs"), updated_by="bob")
        assert store.get_metric(TENANT, "m_001").name == "ours"
        assert store.get_metric(OTHER, "m_001").name == "theirs"


class TestVersioning:
    def test_the_snapshot_happens_before_the_write(self, graph, store):
        """Snapshotting after the write would capture the new definition as its own history —
        worse than no history, because it looks like a record and is not one."""
        store.save_metric(TENANT, a_metric(), updated_by="alice")
        graph.order.clear()
        store.save_metric(TENANT, a_metric(expression="SUM(net)"), updated_by="bob")
        assert graph.order == ["snapshot", "upsert"]

    def test_a_first_write_snapshots_nothing(self, graph, store):
        """There is no previous definition to record, and a version full of nulls would be
        restorable to nonsense."""
        store.save_metric(TENANT, a_metric(), updated_by="alice")
        assert graph.versions == []

    def test_an_edit_preserves_the_previous_expression(self, store):
        store.save_metric(TENANT, a_metric(expression="SUM(gross)"), updated_by="alice")
        store.save_metric(TENANT, a_metric(expression="SUM(net)"), updated_by="bob")

        versions = store.list_versions(TENANT, "m_001")
        assert len(versions) == 1
        assert versions[0]["expression"] == "SUM(gross)"
        assert store.get_metric(TENANT, "m_001").expression == "SUM(net)"

    def test_version_increments_on_each_write(self, store):
        first = store.save_metric(TENANT, a_metric(), updated_by="alice")
        second = store.save_metric(TENANT, a_metric(), updated_by="alice")
        assert first["version"] == 1
        assert second["version"] == 2

    def test_a_restore_moves_forward_rather_than_rewinding(self, store):
        """Rewinding would erase the fact that the intervening definition ever answered a
        question."""
        store.save_metric(TENANT, a_metric(expression="SUM(gross)"), updated_by="alice")
        store.save_metric(TENANT, a_metric(expression="SUM(net)"), updated_by="bob")

        restored = store.restore_version(TENANT, "m_001", 1, updated_by="carol")
        assert restored["version"] == 3
        assert store.get_metric(TENANT, "m_001").expression == "SUM(gross)"

    def test_restoring_a_missing_version_is_refused(self, store):
        store.save_metric(TENANT, a_metric(), updated_by="alice")
        with pytest.raises(LookupError):
            store.restore_version(TENANT, "m_001", 99, updated_by="alice")


class TestStatus:
    def test_a_new_metric_is_a_draft_by_default(self, store):
        """Authoring a metric must not silently put it into service."""
        store.save_metric(TENANT, a_metric(), updated_by="alice")
        assert store.status_of(TENANT, "m_001") == STATUS_DRAFT

    def test_only_approved_metrics_are_listed_for_tier_one(self, store):
        store.save_metric(TENANT, a_metric("m_draft"), updated_by="alice", status=STATUS_DRAFT)
        store.save_metric(TENANT, a_metric("m_live"), updated_by="alice", status=STATUS_APPROVED)

        approved = {m.metric_id for m in store.list_metrics(TENANT, approved_only=True)}
        assert approved == {"m_live"}

    def test_a_deprecated_metric_stops_serving(self, store):
        store.save_metric(TENANT, a_metric(), updated_by="alice", status=STATUS_APPROVED)
        store.set_status(TENANT, "m_001", STATUS_DEPRECATED, updated_by="bob")
        assert store.list_metrics(TENANT, approved_only=True) == []

    def test_a_status_change_is_itself_versioned(self, store):
        """ "When was this approved" has to be answerable, so a status change snapshots."""
        store.save_metric(TENANT, a_metric(), updated_by="alice")
        store.set_status(TENANT, "m_001", STATUS_APPROVED, updated_by="bob")
        assert len(store.list_versions(TENANT, "m_001")) == 1

    def test_an_invalid_status_is_refused(self, store):
        with pytest.raises(ValueError, match="status must be one of"):
            store.save_metric(TENANT, a_metric(), updated_by="alice", status="live-ish")


class TestSeeding:
    def test_a_pack_seeds_as_approved(self, store):
        """A reviewed pack from the repository arriving as drafts would leave tier 1 dead on a
        fresh deployment."""
        store.seed_from_pack(TENANT, [a_metric("m_1"), a_metric("m_2")])
        assert len(store.list_metrics(TENANT, approved_only=True)) == 2

    def test_seeding_does_not_clobber_authored_work(self, store):
        """A deploy must not silently replace a definition somebody wrote in the UI."""
        store.save_metric(
            TENANT,
            a_metric(expression="SUM(hand_written)"),
            updated_by="alice",
            source=SOURCE_AUTHORED,
        )
        counts = store.seed_from_pack(TENANT, [a_metric(expression="SUM(from_yaml)")])

        assert counts == {"created": 0, "skipped": 1}
        assert store.get_metric(TENANT, "m_001").expression == "SUM(hand_written)"

    def test_reseeding_a_yaml_metric_is_allowed(self, store):
        """The pack is the source for seeded metrics, so a corrected pack must land."""
        store.seed_from_pack(TENANT, [a_metric(expression="SUM(old)")])
        store.seed_from_pack(TENANT, [a_metric(expression="SUM(corrected)")])
        assert store.get_metric(TENANT, "m_001").expression == "SUM(corrected)"


class TestLineage:
    def test_a_metric_links_to_the_table_it_measures(self, graph, store):
        """Makes "which metrics read this column" answerable before someone alters it."""
        store.save_metric(TENANT, a_metric(), updated_by="alice")
        assert graph.linked[0]["full_name"] == "legal_ops.invoices"

    def test_metrics_measuring_a_table_are_findable(self, store):
        store.save_metric(TENANT, a_metric("m_1"), updated_by="alice")
        store.save_metric(
            TENANT, a_metric("m_2", source_table="legal_ops.other"), updated_by="alice"
        )
        found = {m.metric_id for m in store.metrics_measuring(TENANT, "legal_ops.invoices")}
        assert found == {"m_1"}


class TestDeletion:
    def test_deleting_a_metric_takes_its_versions(self, graph, store):
        """Orphaned snapshots would be history for a metric nobody can name."""
        store.save_metric(TENANT, a_metric(), updated_by="alice")
        store.save_metric(TENANT, a_metric(expression="SUM(v2)"), updated_by="alice")
        assert len(graph.versions) == 1

        store.delete_metric(TENANT, "m_001")
        assert store.get_metric(TENANT, "m_001") is None
        assert graph.versions == []
