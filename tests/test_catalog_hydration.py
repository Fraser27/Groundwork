"""The catalog cache reloads itself from the graph.

`CatalogStore` is process-local and `record_scan` was its only writer, so the Tables page said "No
catalogue scan has been run" after any redeploy, in a second Fargate task, and permanently in the
MCP sidecar, while the graph held every `:Table` and `:Column` the scan wrote. These tests pin the
read back, and the two states that must stay distinguishable: never scanned, and scanned by a
process that no longer exists.

The graph rows are derived from a real `scan_catalog` result rather than hand-written, walking the
same edges the Cypher walks. A prop renamed on either side breaks this rather than quietly
hydrating a table with no columns.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from src.discovery.catalog_hydrate import build_sources, build_tables, hydrate, hydrate_once
from src.discovery.catalog_store import CatalogStore
from src.discovery.glue_scanner import HAS_COLUMN, HAS_TABLE, ScanResult, scan_catalog
from src.discovery.graph_store import CatalogGraphStore
from src.graph import catalog_queries as q
from src.graph.assertions import EpistemicClass, ReviewState
from src.graph.scope import AuthContext, TrustFilter

TENANT = "firm-acme"
OTHER_TENANT = "firm-borden"
SOURCE = "glue-prod"

INVOICES = {
    "Name": "invoices",
    "Description": "Issued client invoices",
    "UpdateTime": datetime(2026, 3, 1, tzinfo=UTC),
    "Parameters": {"primary_key": "invoice_id"},
    "StorageDescriptor": {
        "Location": "s3://firm-lake/invoices/",
        "Columns": [
            {"Name": "invoice_id", "Type": "string", "Comment": "Invoice reference"},
            {"Name": "invoice_amount", "Type": "decimal(18,2)"},
        ],
    },
    "PartitionKeys": [{"Name": "month", "Type": "string"}],
}

MATTERS = {
    "Name": "matters",
    "StorageDescriptor": {"Columns": [{"Name": "matter_id", "Type": "string"}]},
}


class FakeGlue:
    def __init__(self, tables_by_db: dict[str, list[dict]]):
        self._tables = tables_by_db

    def get_paginator(self, operation: str):
        if operation == "get_databases":
            return _Paginator(lambda: [{"DatabaseList": [{"Name": db} for db in self._tables]}])
        return _Paginator(
            lambda DatabaseName: [{"TableList": self._tables.get(DatabaseName, [])}]
        )


class _Paginator:
    def __init__(self, pages):
        self._pages = pages

    def paginate(self, **kwargs):
        return self._pages(**kwargs)


def ctx(tenant: str = TENANT) -> AuthContext:
    return AuthContext(user_id="probe@firm.example", tenant_id=tenant)


def scan(tenant: str = TENANT, source_id: str = SOURCE) -> ScanResult:
    glue = FakeGlue({"fin": [INVOICES], "legal": [MATTERS]})
    return scan_catalog(glue, tenant_id=tenant, source_id=source_id)


def rows_from_scan(result: ScanResult, *, scanned_at: str | None = None) -> dict[str, list[dict]]:
    """What the three read queries would return for this scan.

    Built by walking `HAS_TABLE` and `HAS_COLUMN` exactly as the Cypher does, so a column reaches
    its table through the edge and not through the `table` string property on the node.
    """
    nodes = {n.node_id: n for n in result.nodes}
    tables = {nid: n for nid, n in nodes.items() if "Table" in n.labels}

    table_rows: list[dict[str, Any]] = []
    column_rows: list[dict[str, Any]] = []
    for assertion in result.assertions:
        if assertion.predicate == HAS_TABLE:
            props = nodes[assertion.object_id].props
            table_rows.append(
                {
                    "source_id": nodes[assertion.subject_id].props["source_id"],
                    "full_name": props["full_name"],
                    "name": props["name"],
                    "database": props["database"],
                    "description": props["description"],
                    "catalog_type": props["catalog_type"],
                    "location": props["location"],
                    "scanned_at": scanned_at if scanned_at is not None else assertion.recorded_at,
                }
            )
        elif assertion.predicate == HAS_COLUMN:
            props = nodes[assertion.object_id].props
            column_rows.append(
                {
                    "full_name": tables[assertion.subject_id].props["full_name"],
                    "name": props["name"],
                    "data_type": props["data_type"],
                    "description": props["description"],
                    "is_partition": props["is_partition"],
                    "is_primary_key": props["is_primary_key"],
                }
            )

    source_rows = [
        {
            "source_id": n.props["source_id"],
            "type": n.props["type"],
            "last_scanned_at": n.props.get("last_scanned_at"),
        }
        for n in result.nodes
        if "DataSource" in n.labels
    ]
    table_rows.sort(key=lambda r: r["full_name"])
    column_rows.sort(key=lambda r: (r["full_name"], r["name"]))
    return {"tables": table_rows, "columns": column_rows, "sources": source_rows}


class FakeGraph:
    """Replays canned rows per tenant, and refuses a read whose scope does not fit its pattern.

    The tenant filter is honoured rather than ignored: that is what makes the isolation test mean
    something, since an unscoped query would hand back the other firm's tables.
    """

    def __init__(self, rows_by_tenant: dict[str, dict[str, list[dict]]]) -> None:
        self.rows = rows_by_tenant
        self.reads: list[str] = []
        self.scopes: dict[str, Any] = {}

    def read_scoped(self, template: str, scope: Any, params: Any = None) -> list[dict[str, Any]]:
        assert "{scope}" in template, "a scoped read must carry the token"
        var = scope.where.split(".")[0].lstrip("(")
        assert f"({var}:" in template or f"[{var}:" in template, (
            f"scope is bound to {var!r}, which this pattern never matches"
        )
        tenant = scope.params["scope_tenant"]
        for name, key in (("TABLES", "tables"), ("COLUMNS", "columns"), ("SOURCES", "sources")):
            if template == getattr(q, f"{name}_FOR_TENANT"):
                self.reads.append(key)
                self.scopes[key] = scope
                return list(self.rows.get(tenant, {}).get(key, []))
        raise AssertionError("unexpected query")


def graph_store(*results: tuple[str, ScanResult], scanned_at: str | None = None) -> Any:
    return CatalogGraphStore(
        FakeGraph({tenant: rows_from_scan(r, scanned_at=scanned_at) for tenant, r in results})
    )


class TestALostCacheReloads:
    def test_an_empty_store_gets_its_tables_back(self):
        store = CatalogStore()
        assert store.tables(TENANT) == []

        loaded = hydrate(store, graph_store((TENANT, scan())), ctx())

        assert loaded == 2
        assert [t.full_name for t in store.tables(TENANT)] == ["fin.invoices", "legal.matters"]

    def test_the_columns_come_back_with_the_table(self):
        store = CatalogStore()
        hydrate(store, graph_store((TENANT, scan())), ctx())

        invoices = store.table(TENANT, "fin.invoices")
        assert invoices is not None
        assert [c.name for c in invoices.columns] == ["invoice_amount", "invoice_id", "month"]

    def test_the_partition_and_primary_key_flags_survive(self):
        """Both are read off the column node. Losing them makes the compiler's fan-out detection
        and the time-grain guard silently permissive."""
        store = CatalogStore()
        hydrate(store, graph_store((TENANT, scan())), ctx())

        columns = {c.name: c for c in store.table(TENANT, "fin.invoices").columns}
        assert columns["month"].is_partition is True
        assert columns["invoice_id"].is_primary_key is True
        assert columns["invoice_amount"].is_partition is False
        assert columns["invoice_amount"].is_primary_key is False

    def test_the_descriptive_props_survive(self):
        store = CatalogStore()
        hydrate(store, graph_store((TENANT, scan())), ctx())

        invoices = store.table(TENANT, "fin.invoices")
        assert invoices.description == "Issued client invoices"
        assert invoices.location == "s3://firm-lake/invoices/"
        assert invoices.database == "fin"
        assert invoices.source_id == SOURCE

    def test_a_hydrated_table_matches_what_the_scan_would_have_recorded(self):
        """The round trip, since the two halves are written independently: whatever a scan puts in
        the cache, a reload from the graph must reproduce."""
        scanned = CatalogStore()
        result = scan()
        scanned.record_scan(TENANT, source_id=SOURCE, result=result)
        reloaded = CatalogStore()
        hydrate(reloaded, graph_store((TENANT, result)), ctx())

        for a, b in zip(scanned.tables(TENANT), reloaded.tables(TENANT), strict=True):
            assert a.full_name == b.full_name
            assert a.description == b.description
            assert sorted(c.name for c in a.columns) == sorted(c.name for c in b.columns)

    def test_the_source_comes_back_scanned_rather_than_not_scanned(self):
        store = CatalogStore()
        hydrate(store, graph_store((TENANT, scan())), ctx())

        source = store.sources(TENANT)[0]
        assert source.source_id == SOURCE
        assert source.table_count == 2
        assert source.status == "CONNECTED"


class TestItHappensOnce:
    def test_a_second_call_reads_nothing(self):
        store = CatalogStore()
        gs = graph_store((TENANT, scan()))

        assert hydrate_once(store, gs, ctx()) == 2
        reads_after_first = len(gs.graph.reads)
        assert hydrate_once(store, gs, ctx()) == 0
        assert len(gs.graph.reads) == reads_after_first

    def test_a_scan_leaves_nothing_to_load(self):
        """`record_scan` just filled the cache, so a hydrate afterwards is a graph read for
        nothing on the one page load that already paid for a scan."""
        store = CatalogStore()
        store.record_scan(TENANT, source_id=SOURCE, result=scan())
        gs = graph_store((TENANT, scan()))

        assert hydrate_once(store, gs, ctx()) == 0
        assert gs.graph.reads == []

    def test_a_reset_makes_it_load_again(self):
        """Leaving the flag set would make a reset permanent: the tenant would read as empty
        forever, whatever the graph then held."""
        store = CatalogStore()
        gs = graph_store((TENANT, scan()))
        hydrate_once(store, gs, ctx())
        store.clear(TENANT)

        assert store.is_hydrated(TENANT) is False
        assert hydrate_once(store, gs, ctx()) == 2

    def test_a_failed_read_stays_retryable(self):
        """A Neptune outage must not be recorded as a complete load, or the cache stays empty
        until the container is replaced."""

        class Broken:
            def table_rows(self, ctx):
                raise RuntimeError("neptune is unreachable")

        store = CatalogStore()
        with pytest.raises(RuntimeError):
            hydrate_once(store, Broken(), ctx())
        assert store.is_hydrated(TENANT) is False


class TestNeverScannedIsNotALostCache:
    def test_a_tenant_with_no_catalog_hydrates_to_nothing(self):
        store = CatalogStore()
        assert hydrate(store, graph_store((TENANT, scan())), ctx(OTHER_TENANT)) == 0
        assert store.tables(OTHER_TENANT) == []
        assert store.sources(OTHER_TENANT) == []

    def test_and_stops_being_asked(self):
        store = CatalogStore()
        gs = graph_store((TENANT, scan()))
        hydrate_once(store, gs, ctx(OTHER_TENANT))
        before = len(gs.graph.reads)

        assert hydrate_once(store, gs, ctx(OTHER_TENANT)) == 0
        assert len(gs.graph.reads) == before

    def test_the_two_states_report_differently(self):
        """The distinction the Tables page turns on: no source at all means nobody has scanned,
        while a source with a timestamp means the cache was lost and has been rebuilt."""
        store = CatalogStore()
        gs = graph_store((TENANT, scan()))
        hydrate(store, gs, ctx(OTHER_TENANT))
        hydrate(store, gs, ctx())

        assert store.sources(OTHER_TENANT) == []
        assert store.sources(TENANT)[0].last_scanned_at is not None


class TestTenantIsolation:
    def test_one_tenants_tables_never_reach_another(self):
        store = CatalogStore()
        gs = graph_store((TENANT, scan()), (OTHER_TENANT, scan(OTHER_TENANT, "glue-borden")))

        hydrate(store, gs, ctx())

        assert store.tables(OTHER_TENANT) == []
        assert [s.source_id for s in store.sources(TENANT)] == [SOURCE]

    def test_each_tenant_hydrates_only_its_own(self):
        store = CatalogStore()
        gs = graph_store((TENANT, scan()), (OTHER_TENANT, scan(OTHER_TENANT, "glue-borden")))

        hydrate(store, gs, ctx())
        hydrate(store, gs, ctx(OTHER_TENANT))

        assert [s.source_id for s in store.sources(OTHER_TENANT)] == ["glue-borden"]
        assert store.is_hydrated(TENANT) and store.is_hydrated(OTHER_TENANT)


class TestHydrateDoesNotClobberAScan:
    def test_a_live_scans_table_wins(self):
        """The graph can be behind a scan running in this process, and `hydrate` is not a
        `record_scan`: it fills gaps rather than replacing a source wholesale."""
        store = CatalogStore()
        store.record_scan(TENANT, source_id=SOURCE, result=scan())
        scanned_at = store.table(TENANT, "fin.invoices").scanned_at

        store.hydrate(TENANT, _stale_tables(), [])

        assert store.table(TENANT, "fin.invoices").scanned_at == scanned_at
        assert store.table(TENANT, "fin.invoices").description == "Issued client invoices"

    def test_a_scanned_source_keeps_its_timestamp(self):
        store = CatalogStore()
        record = store.record_scan(TENANT, source_id=SOURCE, result=scan())
        stale = build_sources(
            [{"source_id": SOURCE, "type": "glue", "last_scanned_at": "1999-01-01T00:00:00+00:00"}],
            _stale_tables(),
        )

        store.hydrate(TENANT, [], stale)

        assert store.sources(TENANT)[0].last_scanned_at == record.last_scanned_at

    def test_a_registered_but_unscanned_source_takes_the_graphs_timestamp(self):
        """`register_source` makes a source visible with no timestamp. That is exactly the record
        the graph can improve on."""
        store = CatalogStore()
        store.register_source(TENANT, SOURCE)
        hydrate(store, graph_store((TENANT, scan())), ctx())

        assert store.sources(TENANT)[0].status == "CONNECTED"


class TestTheScanTimestamp:
    def test_the_source_node_carries_one(self):
        """Without it a rehydrated source has no scan date at all, and `SourceRecord.status`
        reads NOT_SCANNED over a fully populated catalog."""
        node = next(n for n in scan().nodes if "DataSource" in n.labels)
        assert datetime.fromisoformat(node.props["last_scanned_at"]).tzinfo is not None

    def test_an_older_source_node_falls_back_to_its_newest_table(self):
        """Nodes written before `last_scanned_at` existed have no property to read, and the
        HAS_TABLE assertion is the only other record of when the scan ran."""
        rows = rows_from_scan(scan(), scanned_at="2026-05-05T00:00:00+00:00")
        sources = build_sources(
            [{"source_id": SOURCE, "type": "glue", "last_scanned_at": None}],
            build_tables(rows["tables"], rows["columns"]),
        )
        assert sources[0].last_scanned_at == "2026-05-05T00:00:00+00:00"

    def test_with_nothing_to_fall_back_to_it_stays_empty(self):
        """An invented timestamp would report a scan that may never have run."""
        sources = build_sources(
            [{"source_id": SOURCE, "type": "glue", "last_scanned_at": None}], []
        )
        assert sources[0].last_scanned_at is None
        assert sources[0].status == "NOT_SCANNED"


class TestTheDeclaredEdgesPassTheDefaults:
    """Why these reads use `edge_scope` defaults, unlike `APPROVED_DESCRIPTIONS`.

    If a catalog edge ever stopped clearing the trust filter, hydration would return zero rows and
    look exactly like a tenant that has never been scanned.
    """

    @pytest.mark.parametrize("predicate", [HAS_TABLE, HAS_COLUMN])
    def test_a_catalog_assertion_is_readable_by_default(self, predicate):
        trust = TrustFilter.for_context(ctx())
        assertion = next(a for a in scan().assertions if a.predicate == predicate)

        assert assertion.epistemic_class is EpistemicClass.DECLARED
        assert assertion.review_state is ReviewState.AUTO_ASSERTED
        assert trust.matches(assertion)

    @pytest.mark.parametrize("read", ["tables", "columns"])
    def test_the_reads_do_not_widen_to_pending(self, read):
        """`include_pending` would admit states a catalog edge never occupies, so passing it here
        would only weaken the filter for no gain."""
        store = CatalogStore()
        gs = graph_store((TENANT, scan()))
        hydrate(store, gs, ctx())

        assert set(gs.graph.scopes[read].params["scope_states"]) == {"AUTO_ASSERTED", "APPROVED"}
        assert gs.graph.scopes[read].params["scope_min_conf"] == 0.8

    def test_the_source_read_carries_no_trust_filter(self):
        """A node has no epistemic class to filter on, and a source registered but never scanned
        has no edge either. Scoping it on one would hide it."""
        store = CatalogStore()
        gs = graph_store((TENANT, scan()))
        hydrate(store, gs, ctx())

        assert "scope_states" not in gs.graph.scopes["sources"].params


def _stale_tables() -> list[Any]:
    rows = rows_from_scan(scan(), scanned_at="1999-01-01T00:00:00+00:00")
    return build_tables(rows["tables"], rows["columns"])


class ApiGraph(FakeGraph):
    """`FakeGraph` plus the description read, which the routes make and `hydrate` does not.

    Every read path goes through the enriched catalog, so a fake that refuses this query would
    exercise the overlay's degrade path instead of the hydration it is here to test.
    """

    def read_scoped(self, template: str, scope: Any, params: Any = None) -> list[dict[str, Any]]:
        if template == q.APPROVED_DESCRIPTIONS:
            return []
        return super().read_scoped(template, scope, params)


class BrokenGraph:
    def read_scoped(self, template: str, scope: Any, params: Any = None) -> list[dict[str, Any]]:
        raise RuntimeError("neptune is unreachable")


class TestTheApiServesAProcessThatNeverScanned:
    """The bug end to end.

    `record_scan` is the cache's only writer and the API was its only reader, so every process
    except the one that ran the scan served an empty catalog: a redeploy, a second task, and the
    MCP sidecar permanently. Nothing here calls `record_scan`.
    """

    def api(self, graph: Any) -> tuple[Any, Any]:
        from fastapi.testclient import TestClient

        from src.api.app import create_app
        from src.api.deps import get_services
        from src.config import AuthConfig, GraphConfig, GroundworkConfig

        cfg = GroundworkConfig(
            environment="local",
            auth=AuthConfig(dev_bypass_tenant=TENANT),
            graph=GraphConfig(uri="bolt://127.0.0.1:1", user="none", password="none"),
        )
        cfg.structured.athena_results_bucket = "results"
        cfg.validate()
        app = create_app(cfg)
        services = get_services()
        # Attached without the lifespan, which is the sidecar's situation as much as a redeploy's:
        # a graph is reachable and this process was never told what the scan found.
        services.graph = graph
        return TestClient(app), services

    def rows(self, tenant: str = TENANT) -> Any:
        return ApiGraph({tenant: rows_from_scan(scan(tenant))})

    def test_the_tables_page_is_served_from_the_graph(self):
        client, services = self.api(self.rows())
        assert services.catalog.tables(TENANT) == []

        body = client.get(f"/api/tenants/{TENANT}/tables").json()

        assert sorted(t["full_name"] for t in body) == ["fin.invoices", "legal.matters"]
        assert [t["column_count"] for t in body if t["full_name"] == "fin.invoices"] == [3]

    def test_one_table_comes_back_with_its_columns(self):
        client, _ = self.api(self.rows())
        body = client.get(f"/api/tenants/{TENANT}/tables/fin.invoices").json()

        assert [c["name"] for c in body["columns"]] == ["invoice_amount", "invoice_id", "month"]

    def test_the_source_reads_as_scanned_rather_than_never_scanned(self):
        client, _ = self.api(self.rows())
        body = client.get(f"/api/tenants/{TENANT}/sources").json()

        assert [s["source_id"] for s in body] == [SOURCE]
        assert body[0]["last_scanned_at"]
        assert body[0]["table_count"] == 2

    def test_the_firewall_allowlist_is_rebuilt_too(self):
        """The tier 3 lane reported zero catalogued tables for the same reason the page did, and
        there the consequence is a refusal rather than an empty list."""
        from src.api.deps import build_athena_executor

        _, services = self.api(self.rows())
        executor = build_athena_executor(services, TENANT)

        assert executor._firewall.validate("SELECT count(*) FROM fin.invoices").allowed

    def test_the_resolver_gets_the_same_catalog(self):
        """How the MCP sidecar reaches the catalog: it has no tables route and no lifespan, only
        `build_resolver` and `build_planner`."""
        _, services = self.api(self.rows())
        resolver = services.build_resolver(TENANT)

        assert [t.full_name for t in resolver._catalog.tables(TENANT)] == [
            "fin.invoices",
            "legal.matters",
        ]


class TestTheThreeStates:
    """Nothing was scanned, and the catalog could not be read, are different claims.

    The page inferred both from the same empty list. Same rule as `HELP.reasonerStates`, where "no
    conflict found" and "no conflict could be looked for" must never render alike.
    """

    def status(self, graph: Any) -> dict[str, Any]:
        client, _ = TestTheApiServesAProcessThatNeverScanned().api(graph)
        return client.get(f"/api/tenants/{TENANT}/catalog/status").json()

    def test_a_lost_cache_reports_scanned(self):
        body = self.status(ApiGraph({TENANT: rows_from_scan(scan())}))
        assert body["state"] == "scanned"
        assert body["tables"] == 2

    def test_a_graph_with_no_tables_reports_never_scanned(self):
        body = self.status(ApiGraph({}))
        assert body["state"] == "never_scanned"
        assert body["tables"] == 0

    def test_an_unreachable_graph_reports_neither(self):
        body = self.status(BrokenGraph())
        assert body["state"] == "unknown"
        assert body["tables"] == 0

    def test_no_graph_at_all_reports_neither_either(self):
        """Vector search off is a supported deployment; no graph is not. Either way the honest
        answer is that nothing here can say whether a scan has run."""
        client, services = TestTheApiServesAProcessThatNeverScanned().api(None)
        assert client.get(f"/api/tenants/{TENANT}/catalog/status").json()["state"] == "unknown"
        assert services.catalog_confirmed[TENANT] is False

    def test_the_three_read_differently(self):
        notes = {
            self.status(g)["note"]
            for g in (ApiGraph({TENANT: rows_from_scan(scan())}), ApiGraph({}), BrokenGraph())
        }
        assert len(notes) == 3
        assert not any("—" in n for n in notes)

    def test_an_unreachable_graph_is_retried(self):
        """A tenant marked hydrated on a failed read would report `unknown` until the task is
        replaced, which is `mark_hydrated`'s own rule: an outage leaves the cache retryable."""
        client, services = TestTheApiServesAProcessThatNeverScanned().api(BrokenGraph())
        assert client.get(f"/api/tenants/{TENANT}/catalog/status").json()["state"] == "unknown"

        services.graph = ApiGraph({TENANT: rows_from_scan(scan())})

        assert client.get(f"/api/tenants/{TENANT}/catalog/status").json()["state"] == "scanned"
