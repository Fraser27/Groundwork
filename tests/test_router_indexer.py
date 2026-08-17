"""What gets described, and what deliberately does not.

Three properties decide whether this index is trustworthy, and each corresponds to a failure that
is silent without a test.

**An unapproved metric is not indexed.** A draft that routes a question sends it to tier 1, which
then finds no approved metric and declines — the decision was already made on the strength of work
in progress.

**An entity whose kind the pack does not declare produces no record.** Parsing an id prefix is the
bug the closed vocabulary was added to prevent, and a routing description would hide the drift
behind a label that looks fine.

**A layer is deleted before it is rewritten.** An upsert leaves behind a deprecated metric or a
table dropped from Glue, and a routing hit on something that no longer exists sends the question
to a tier that finds nothing.

The graph read goes through `read_scoped`, which refuses a template without a `{scope}` token, so
the fake client asserts the token was there.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import pytest

from src.discovery.catalog_store import CatalogColumn, CatalogTable
from src.graph.scope import AuthContext
from src.metrics.models import MetricDefinition
from src.ontology.loader import load_ontology
from src.query.router_index import KIND_ENTITY, KIND_METRIC, KIND_TABLE
from src.query.router_indexer import (
    RouterIndexer,
    RoutingRebuildReport,
    entity_text,
    metric_text,
    table_text,
)

TENANT = "demo-firm"
INDEX = "tenant-demo-firm-routing"


@pytest.fixture
def ctx() -> AuthContext:
    return AuthContext(user_id="alice", tenant_id=TENANT)


@dataclass
class FakeIndex:
    """Records what was written and what was deleted, in order."""

    writes: list[tuple[str, list[Any]]] = None  # type: ignore[assignment]
    deletes: list[tuple[str, str]] = None  # type: ignore[assignment]
    tenant_deletes: list[tuple[str, str]] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.writes = []
        self.deletes = []
        self.tenant_deletes = []

    def upsert(self, index: str, records: list[Any]) -> int:
        self.writes.append((index, list(records)))
        return len(records)

    def delete_kind(self, index: str, kind: str) -> int:
        self.deletes.append((index, kind))
        return 0

    def delete_tenant(self, index: str, tenant_id: str) -> int:
        self.tenant_deletes.append((index, tenant_id))
        return 7

    def records(self, kind: str) -> list[Any]:
        return [r for _, batch in self.writes for r in batch if r.kind == kind]


class FakeEmbedder:
    """Deterministic, and records what it was asked to embed."""

    model_id = "amazon.titan-embed-text-v2:0"

    def __init__(self) -> None:
        self.texts: list[str] = []

    def embed_text(self, text: str) -> list[float]:
        self.texts.append(text)
        return [float(len(text)), 0.5]


class FakeGraph:
    """A `read_scoped` that refuses an unscoped template, like the real client."""

    def __init__(self, rows: list[dict[str, Any]] | None = None) -> None:
        self.rows = rows or []
        self.queries: list[str] = []
        self.params: list[dict[str, Any]] = []

    def read_scoped(self, template: str, scope: Any, params: dict[str, Any] | None = None):
        if "{scope}" not in template:
            raise ValueError("scoped reads must contain a {scope} placeholder")
        self.queries.append(template.replace("{scope}", scope.where))
        self.params.append(dict(params or {}))
        # LIMIT is in the Cypher, so honouring it here keeps the fake from over-reporting.
        return self.rows[: (params or {}).get("limit", len(self.rows))]


class FakeMetricStore:
    def __init__(self, approved: list[MetricDefinition], drafts: list[MetricDefinition]) -> None:
        self.approved = approved
        self.drafts = drafts
        self.calls: list[bool] = []

    def list_metrics(self, tenant_id: str, *, approved_only: bool = False):
        self.calls.append(approved_only)
        return list(self.approved) if approved_only else [*self.approved, *self.drafts]


class FakeCatalog:
    def __init__(self, tables: list[CatalogTable]) -> None:
        self._tables = tables

    def tables(self, tenant_id: str) -> list[CatalogTable]:
        return list(self._tables)


def metric(metric_id: str = "fees_billed", **over: Any) -> MetricDefinition:
    fields: dict[str, Any] = {
        "metric_id": metric_id,
        "name": "fees_billed",
        "synonyms": ["invoiced", "billings"],
        "definition": "Value invoiced to clients in a period.",
        "expression": "SUM(billed_value)",
        "source_table": "legal_ops.invoices",
    }
    fields.update(over)
    return MetricDefinition(**fields)


def table(full_name: str = "legal_ops.invoices") -> CatalogTable:
    database, _, name = full_name.partition(".")
    return CatalogTable(
        full_name=full_name,
        name=name,
        database=database,
        source_id="glue-main",
        description="One row per issued invoice.",
        columns=(
            CatalogColumn(name="invoice_id", data_type="string"),
            CatalogColumn(name="billed_value", data_type="double"),
        ),
    )


def indexer(
    *,
    index: FakeIndex | None = None,
    embedder: FakeEmbedder | None = None,
    graph: FakeGraph | None = None,
    metric_store: FakeMetricStore | None = None,
    catalog: FakeCatalog | None = None,
) -> RouterIndexer:
    return RouterIndexer(
        index or FakeIndex(),
        embedder=embedder or FakeEmbedder(),
        ontology=load_ontology("legal"),
        catalog=catalog,
        graph=graph,
        metric_store=metric_store,
    )


class TestMetrics:
    def test_only_approved_metrics_are_indexed(self):
        """A draft that routed a question would send it to tier 1, which then finds no approved
        metric and declines -- after the routing decision had already been made."""
        store = FakeMetricStore(approved=[metric("fees_billed")], drafts=[metric("wip_value")])
        index = FakeIndex()
        written = indexer(index=index, metric_store=store).reindex_metrics(
            AuthContext(user_id="a", tenant_id=TENANT)
        )

        assert written == 1
        assert store.calls == [True]
        assert [r.item_id for r in index.records(KIND_METRIC)] == ["fees_billed"]

    def test_the_embedded_text_is_name_synonyms_and_definition(self, ctx):
        """The reason this exists: "invoiced" is a synonym, so "what did we invoice last quarter"
        can reach `fees_billed` without anyone maintaining a keyword list."""
        embedder = FakeEmbedder()
        indexer(
            embedder=embedder, metric_store=FakeMetricStore([metric()], [])
        ).reindex_metrics(ctx)

        embedded = embedder.texts[0]
        assert "fees_billed" in embedded
        assert "invoiced" in embedded
        assert "Value invoiced to clients in a period." in embedded

    def test_the_expression_is_not_embedded(self, ctx):
        """`SUM(billed_value)` is how the answer is computed, not what it means, and nobody
        phrases a question in SQL."""
        embedder = FakeEmbedder()
        indexer(
            embedder=embedder, metric_store=FakeMetricStore([metric()], [])
        ).reindex_metrics(ctx)

        assert "SUM(" not in embedder.texts[0]

    def test_the_detail_carries_the_expression_and_source_table_for_the_trace(self, ctx):
        index = FakeIndex()
        indexer(index=index, metric_store=FakeMetricStore([metric()], [])).reindex_metrics(ctx)

        detail = index.records(KIND_METRIC)[0].detail
        assert detail == {
            "expression": "SUM(billed_value)",
            "source_table": "legal_ops.invoices",
        }

    def test_the_vector_id_is_derived_so_a_rebuild_converges(self, ctx):
        index = FakeIndex()
        idx = indexer(index=index, metric_store=FakeMetricStore([metric()], []))
        idx.reindex_metrics(ctx)
        idx.reindex_metrics(ctx)

        ids = {r.vector_id for r in index.records(KIND_METRIC)}
        assert ids == {f"{TENANT}:metric:fees_billed"}

    def test_no_metric_store_indexes_nothing_rather_than_failing(self, ctx):
        assert indexer().reindex_metrics(ctx) == 0


class TestTables:
    def test_the_embedded_text_is_name_database_description_and_columns(self, ctx):
        """Column names carry most of the signal for a warehouse table whose description is
        empty, which is most of them."""
        embedder = FakeEmbedder()
        indexer(embedder=embedder, catalog=FakeCatalog([table()])).reindex_tables(ctx)

        embedded = embedder.texts[0]
        assert "invoices in legal_ops" in embedded
        assert "One row per issued invoice." in embedded
        assert "billed_value" in embedded

    def test_the_detail_carries_the_column_list(self, ctx):
        index = FakeIndex()
        indexer(index=index, catalog=FakeCatalog([table()])).reindex_tables(ctx)

        assert index.records(KIND_TABLE)[0].detail == {
            "columns": ["invoice_id", "billed_value"]
        }

    def test_the_item_id_is_the_full_name(self, ctx):
        """Two firms may both have `warehouse.matters`, and within a tenant the full name is what
        a metric's `source_table` refers to."""
        index = FakeIndex()
        indexer(index=index, catalog=FakeCatalog([table()])).reindex_tables(ctx)

        assert index.records(KIND_TABLE)[0].item_id == "legal_ops.invoices"

    def test_no_catalog_indexes_nothing(self, ctx):
        assert indexer().reindex_tables(ctx) == 0


class TestEntities:
    def _rows(self) -> list[dict[str, Any]]:
        return [
            {"entity_id": "party:meridian-holdings", "predicates": ["REPRESENTS", "ADVERSE_TO"]},
            {"entity_id": "matter:m-001", "predicates": ["RELATES_TO"]},
        ]

    def test_an_undeclared_entity_kind_produces_no_record(self, ctx):
        """The vocabulary is the authority on what an id's prefix means. `vessel:mv-aurelia` and
        `ship:mv-aurelia` are two nodes and a traversal finds one, so an undeclared kind is drift
        to surface rather than something to file under a guess."""
        graph = FakeGraph([{"entity_id": "vessel:mv-aurelia", "predicates": ["REPRESENTS"]}])
        index = FakeIndex()
        written, skipped = indexer(index=index, graph=graph).reindex_entities(ctx)

        assert (written, skipped) == (0, 1)
        assert index.records(KIND_ENTITY) == []

    def test_an_unprefixed_id_is_skipped_too(self, ctx):
        """An id with no prefix claims no kind, and guessing one defeats the point of having a
        vocabulary."""
        graph = FakeGraph([{"entity_id": "meridian-holdings", "predicates": []}])
        written, skipped = indexer(graph=graph).reindex_entities(ctx)
        assert (written, skipped) == (0, 1)

    def test_the_kind_comes_from_the_ontology_not_the_prefix(self, ctx):
        """`entity_kind_of` decides, so a declared `Party` and a stray `parties:` do not both
        become "party" by string manipulation here."""
        graph = FakeGraph(self._rows())
        embedder = FakeEmbedder()
        indexer(embedder=embedder, graph=graph).reindex_entities(ctx)

        assert any("Party" in t for t in embedder.texts)
        assert any("Matter" in t for t in embedder.texts)

    def test_the_embedded_text_carries_the_predicate_labels(self, ctx):
        """What distinguishes two similarly named entities: a party that represents reads
        differently from one that is adverse."""
        graph = FakeGraph(self._rows())
        embedder = FakeEmbedder()
        indexer(embedder=embedder, graph=graph).reindex_entities(ctx)

        party = next(t for t in embedder.texts if "meridian" in t)
        assert "Represents" in party or "represents" in party.lower()

    def test_the_detail_carries_the_ontology_layer(self, ctx):
        """A reader auditing a conflict check does not want a firm's Glue columns on screen, and
        the layer is how the router's trace can say which half of the graph a hit came from."""
        graph = FakeGraph(self._rows())
        index = FakeIndex()
        indexer(index=index, graph=graph).reindex_entities(ctx)

        assert index.records(KIND_ENTITY)[0].detail == {"layer": "domain"}

    def test_the_label_is_readable_without_the_prefix(self, ctx):
        index = FakeIndex()
        indexer(index=index, graph=FakeGraph(self._rows())).reindex_entities(ctx)

        assert index.records(KIND_ENTITY)[0].label == "meridian holdings"

    def test_the_graph_read_is_scoped(self, ctx):
        """`read_scoped` refuses a template without a `{scope}` token, which is what makes an
        unscoped read unexpressible. The tenant filter has to survive into the query text."""
        graph = FakeGraph(self._rows())
        indexer(graph=graph).reindex_entities(ctx)

        assert graph.queries
        assert "a.tenant_id = $scope_tenant" in graph.queries[0]

    def test_the_matter_wall_reaches_the_graph_read(self):
        """A screened matter's entities must not become routable descriptions. A routing hit is a
        subject name, so the screen has to bite before the index is built, not after."""
        graph = FakeGraph(self._rows())
        ctx = AuthContext(
            user_id="alice",
            tenant_id=TENANT,
            matter_allowlist=frozenset({"M-1"}),
            matter_denylist=frozenset({"M-2"}),
        )
        indexer(graph=graph).reindex_entities(ctx)

        cypher = graph.queries[0]
        assert "$scope_matters" in cypher
        assert "NOT a.matter_id IN $scope_denied" in cypher

    def test_the_entity_cap_is_pushed_into_the_query(self, ctx):
        """Entities are the one unbounded layer, and each record is a Bedrock call. Capped in the
        Cypher rather than after the read, so a firm with a million parties does not pull them all
        back over the wire before the slice happens."""
        rows = [{"entity_id": f"party:p-{i}", "predicates": []} for i in range(10)]
        graph = FakeGraph(rows)
        idx = indexer(graph=graph)
        idx.entity_limit = 3

        written, _ = idx.reindex_entities(ctx)
        assert written == 3
        assert graph.params[0]["limit"] == 3

    def test_no_graph_indexes_nothing(self, ctx):
        assert indexer().reindex_entities(ctx) == (0, 0)


class TestALayerIsReplacedNotAccumulated:
    def test_the_kind_is_deleted_before_it_is_written(self, ctx):
        """A metric deprecated since the last rebuild must stop routing. Upserting over the layer
        would leave its description in place, and a hit on it sends the question to a tier that
        will not answer."""
        index = FakeIndex()
        indexer(index=index, metric_store=FakeMetricStore([metric()], [])).reindex_metrics(ctx)

        assert index.deletes == [(INDEX, KIND_METRIC)]

    def test_rebuilding_one_layer_does_not_delete_the_others(self, ctx):
        index = FakeIndex()
        indexer(index=index, catalog=FakeCatalog([table()])).reindex_tables(ctx)

        assert index.deletes == [(INDEX, KIND_TABLE)]

    def test_an_emptied_layer_is_still_cleared(self, ctx):
        """Every approved metric deprecated means zero records and no bulk write, but the old
        descriptions must still go -- otherwise a tenant with no approved metrics keeps routing to
        tier 1 forever."""
        index = FakeIndex()
        assert indexer(index=index, metric_store=FakeMetricStore([], [metric()])).reindex_metrics(
            ctx
        ) == 0
        assert index.deletes == [(INDEX, KIND_METRIC)]

    def test_the_index_name_is_the_tenants(self, ctx):
        index = FakeIndex()
        indexer(index=index, metric_store=FakeMetricStore([metric()], [])).reindex_metrics(ctx)

        assert index.writes[0][0] == INDEX


class TestRebuild:
    def test_it_reports_a_count_per_layer(self, ctx):
        report = indexer(
            metric_store=FakeMetricStore([metric()], []),
            catalog=FakeCatalog([table(), table("legal_ops.matters")]),
            graph=FakeGraph([{"entity_id": "party:meridian", "predicates": []}]),
        ).rebuild(ctx)

        assert (report.metrics_indexed, report.tables_indexed, report.entities_indexed) == (1, 2, 1)
        assert report.total_indexed == 4
        assert report.errors == []

    def test_one_broken_layer_does_not_lose_the_others(self, ctx):
        """A graph that is down must not leave the router with no index at all -- that degrades to
        running every tier on every question."""

        class Broken(FakeGraph):
            def read_scoped(self, template: str, scope: Any, params: Any = None):
                raise RuntimeError("Neptune unreachable")

        report = indexer(
            metric_store=FakeMetricStore([metric()], []),
            catalog=FakeCatalog([table()]),
            graph=Broken(),
        ).rebuild(ctx)

        assert report.metrics_indexed == 1
        assert report.tables_indexed == 1
        assert report.entities_indexed == 0
        assert any("entities" in e for e in report.errors)

    def test_skipped_entities_are_reported_rather_than_swallowed(self, ctx):
        """A rising count here is vocabulary drift, which is exactly what the closed kind list
        exists to make visible."""
        graph = FakeGraph(
            [
                {"entity_id": "party:meridian", "predicates": []},
                {"entity_id": "vessel:mv-aurelia", "predicates": []},
            ]
        )
        report = indexer(graph=graph).rebuild(ctx)

        assert report.entities_indexed == 1
        assert report.entities_skipped == 1
        assert report.to_dict()["entities_skipped"] == 1

    def test_the_report_says_nothing_was_changed_but_descriptions(self, ctx):
        note = indexer().rebuild(ctx).to_dict()["note"]
        assert "draft" in note


class TestDropTenant:
    def test_it_clears_the_tenants_routing_index(self):
        """Takes a bare tenant id: a reset runs from an admin route, not the requesting user's
        scope, so the index name is derived the same way rather than from a `ctx`."""
        index = FakeIndex()
        assert indexer(index=index).drop_tenant(TENANT) == 7
        assert index.tenant_deletes == [(INDEX, TENANT)]


class TestReset:
    def test_routing_is_dropped_by_default(self):
        """The index is derived from the metrics, tables and entities a reset removes. Leaving it
        would route questions toward layers whose contents just went."""
        from src.admin_ops import ResetScope

        assert ResetScope().routing is True

    def test_a_reset_clears_the_routing_index(self, ctx):
        from types import SimpleNamespace

        from src.admin_ops import reset_derived

        index = FakeIndex()
        services = SimpleNamespace(
            review_queue=SimpleNamespace(
                store=SimpleNamespace(
                    all_for_tenant=lambda t: [], drop_tenant=lambda t: 0
                )
            ),
            graph=None,
            embedder=None,
            job_store=SimpleNamespace(drop_tenant=lambda t: 0),
            catalog=SimpleNamespace(tables=lambda t: [], clear=lambda t: None),
            router_indexer=indexer(index=index),
        )
        report = reset_derived(services, ctx)

        assert report.routing_dropped == 7
        assert index.tenant_deletes == [(INDEX, TENANT)]
        assert report.to_dict()["routing_dropped"] == 7

    def test_a_reset_without_a_router_is_not_an_error(self, ctx):
        """Vector search off means no routing index, and a reset must still work."""
        from types import SimpleNamespace

        from src.admin_ops import reset_derived

        services = SimpleNamespace(
            review_queue=SimpleNamespace(
                store=SimpleNamespace(all_for_tenant=lambda t: [], drop_tenant=lambda t: 0)
            ),
            graph=None,
            embedder=None,
            job_store=SimpleNamespace(drop_tenant=lambda t: 0),
            catalog=SimpleNamespace(tables=lambda t: [], clear=lambda t: None),
            router_indexer=None,
        )
        report = reset_derived(services, ctx)

        assert report.routing_dropped == 0
        assert report.errors == []


class TestOverHttp:
    """The admin route. Gated, and honest when there is nothing to rebuild."""

    @contextmanager
    def _client(self, *, router_indexer: Any = None):
        from fastapi.testclient import TestClient

        from src.api.app import create_app
        from src.api.deps import get_services
        from src.config import AuthConfig, GraphConfig, LexGraphConfig

        cfg = LexGraphConfig(
            environment="local",
            auth=AuthConfig(dev_bypass_tenant=TENANT),
            graph=GraphConfig(uri="bolt://127.0.0.1:1", user="none", password="none"),
        )
        cfg.validate()
        with TestClient(create_app(cfg)) as client:
            get_services().router_indexer = router_indexer
            yield client

    def test_a_rebuild_reports_a_count_per_layer(self):
        idx = indexer(metric_store=FakeMetricStore([metric()], []), catalog=FakeCatalog([table()]))
        with self._client(router_indexer=idx) as client:
            body = client.post(f"/api/tenants/{TENANT}/admin/router/rebuild", json={}).json()

        assert body["metrics_indexed"] == 1
        assert body["tables_indexed"] == 1

    def test_a_partial_rebuild_touches_only_the_layers_asked_for(self):
        index = FakeIndex()
        idx = indexer(
            index=index,
            metric_store=FakeMetricStore([metric()], []),
            catalog=FakeCatalog([table()]),
        )
        with self._client(router_indexer=idx) as client:
            body = client.post(
                f"/api/tenants/{TENANT}/admin/router/rebuild",
                json={"metrics": False, "entities": False},
            ).json()

        assert body["tables_indexed"] == 1
        assert body["metrics_indexed"] == 0
        assert index.deletes == [(INDEX, KIND_TABLE)]

    def test_no_routing_index_answers_503_rather_than_pretending(self):
        """Vector search off means there is nothing to rebuild, and a 200 with zeroes would read
        as "your router is built and found nothing", which is the opposite of the truth."""
        with self._client(router_indexer=None) as client:
            r = client.post(f"/api/tenants/{TENANT}/admin/router/rebuild", json={})

        assert r.status_code == 503
        assert "VECTOR_ENDPOINT" in r.json()["detail"]


class TestTheTextBuilders:
    """Pure functions, so the choice of what to embed is reviewable without a fake anything."""

    def test_a_metric_with_no_synonyms_still_produces_text(self):
        assert metric_text(metric(synonyms=[], definition="")) == "fees_billed"

    def test_a_table_with_no_description_still_names_its_columns(self):
        bare = CatalogTable(
            full_name="legal_ops.invoices",
            name="invoices",
            database="legal_ops",
            source_id="glue-main",
            columns=(CatalogColumn(name="billed_value", data_type="double"),),
        )
        text = table_text(bare)
        assert "invoices in legal_ops" in text
        assert "billed_value" in text

    def test_an_entity_with_no_predicates_is_still_describable(self):
        """A newly declared subject with one unreviewed fact should be routable, not absent."""
        assert entity_text("meridian holdings", "Party", []) == "meridian holdings. Party"


class TestTheReport:
    def test_an_empty_rebuild_totals_zero(self):
        assert RoutingRebuildReport().total_indexed == 0
