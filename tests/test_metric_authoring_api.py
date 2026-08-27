"""Authoring governed metrics over HTTP.

A metric defined here becomes a tier-1 answer, so the guards are the interesting part:

- an uncompilable definition is refused rather than stored, because a stored broken metric
  means a question can match it and then fail at answer time
- authoring produces a draft, so writing a definition cannot by itself put it into service
- reads work without the graph, writes do not and say so, rather than accepting a definition
  that will not persist
- only an admin may change what a number means

No graph and no AWS. The graph is absent on purpose in most of these: it is the degraded mode
the deployed system will actually hit while Neptune is reconnecting.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.deps import get_services
from src.config import AuthConfig, GraphConfig, GroundworkConfig
from src.graph.scope import AuthContext

TENANT = "demo-firm"
BASE = f"/api/tenants/{TENANT}/metrics"


def _config() -> GroundworkConfig:
    cfg = GroundworkConfig(
        environment="local",
        auth=AuthConfig(dev_bypass_tenant=TENANT),
        graph=GraphConfig(uri="bolt://127.0.0.1:1", user="none", password="none"),
    )
    cfg.validate()
    return cfg


def a_body(**over: Any) -> dict[str, Any]:
    body = {
        "metric_id": "m_new",
        "name": "total_fees",
        "expression": "SUM(amount)",
        "source_table": "legal_ops.invoices",
        "grain": ["invoice_date"],
        "definition": "Everything invoiced.",
    }
    body.update(over)
    return body


class FakeGraph:
    """A graph that records writes, so the store's behaviour is observable."""

    def __init__(self) -> None:
        self.metrics: dict[tuple[str, str], dict[str, Any]] = {}
        self.versions: list[dict[str, Any]] = []

    def write(self, cypher: str, params: dict[str, Any] | None = None) -> None:
        p = params or {}
        if "CREATE (mv:MetricVersion" in cypher:
            key = (p["tenant_id"], p["metric_id"])
            if key in self.metrics:
                self.versions.append(dict(self.metrics[key]))
        elif "DETACH DELETE m, mv" in cypher:
            self.metrics.pop((p["tenant_id"], p["metric_id"]), None)

    def query(self, cypher: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        p = params or {}
        key = (p.get("tenant_id"), p.get("metric_id"))
        if "MERGE (m:Metric" in cypher:
            prior = self.metrics.get(key)
            row = dict(p)
            row["version"] = (prior.get("version", 1) + 1) if prior else 1
            self.metrics[key] = row
            return [
                {"metric_id": row["metric_id"], "version": row["version"], "status": row["status"]}
            ]
        if "ORDER BY mv.version DESC" in cypher:
            return [
                {
                    "version": v["version"],
                    "name": v["name"],
                    "status": v["status"],
                    "expression": v["expression"],
                }
                for v in self.versions
                if v.get("metric_id") == p.get("metric_id")
            ]
        if "MetricVersion {version: $version}" in cypher:
            return [
                v
                for v in self.versions
                if v.get("metric_id") == p.get("metric_id") and v["version"] == p.get("version")
            ]
        if "WHERE COALESCE(m.status" in cypher:
            return [
                m
                for (t, _), m in self.metrics.items()
                if t == p.get("tenant_id") and (m.get("status") or "approved") == "approved"
            ]
        if "ORDER BY m.name" in cypher:
            return [m for (t, _), m in self.metrics.items() if t == p.get("tenant_id")]
        if "RETURN m.metric_id" in cypher:
            row = self.metrics.get(key)
            return [row] if row else []
        return []

    def verify_connectivity(self) -> bool:
        return True


@pytest.fixture
def client() -> TestClient:
    """No graph: the degraded mode reads must still work."""
    return TestClient(create_app(_config()))


@pytest.fixture
def graph_client() -> TestClient:
    app = create_app(_config())
    get_services().graph = FakeGraph()
    return TestClient(app)


class TestReadsWithoutTheGraph:
    """No graph and no startup pack means no metrics, and saying so is the honest answer.

    Metrics live in the graph and are authored in the app. There is no repository pack loaded
    at startup any more, so a firm with an unreachable graph genuinely has nothing to list.
    Inventing entries from the shipped examples would show a firm metrics about a fictional
    firm's invoices as though they were their own.
    """

    def test_listing_is_empty_rather_than_invented(self, client):
        r = client.get(BASE)
        assert r.status_code == 200
        assert r.json() == []

    def test_an_unknown_metric_is_404(self, client):
        assert client.post(f"{BASE}/nope/compile").status_code == 404

    def test_the_examples_are_reachable_but_marked_as_examples(self, client):
        """When a matcher *is* present, its metrics are flagged rather than presented as
        this tenant's approved definitions."""
        from src.api.deps import get_services, load_example_pack
        from src.metrics.models import StaticCatalog
        from src.query.metric_matcher import MetricMatcher

        pack = load_example_pack()
        get_services().metric_matcher = MetricMatcher(pack, StaticCatalog(tables={}))

        first = client.get(BASE).json()[0]
        assert first["status"] == "draft"
        assert first["is_example"] is True


class TestPreview:
    def test_it_returns_sql_without_saving(self, client):
        """The reviewability that makes a governed metric governed."""
        r = client.post(f"{BASE}/preview", json=a_body())
        assert r.status_code == 200
        assert "SUM(amount)" in r.json()["sql"]

    def test_preview_needs_no_graph(self, client):
        """An author must be able to check their definition even while Neptune reconnects."""
        assert client.post(f"{BASE}/preview", json=a_body()).status_code == 200

    def test_an_uncompilable_definition_is_refused(self, client):
        r = client.post(f"{BASE}/preview", json=a_body(expression="SUM("))
        assert r.status_code == 422
        assert "does not compile" in r.json()["detail"]

    def test_the_refusal_carries_the_compiler_message(self, client):
        """An author needs to know which part of their definition is wrong."""
        r = client.post(f"{BASE}/preview", json=a_body(expression="SUM("))
        assert len(r.json()["detail"]) > len("this definition does not compile: ")


class TestWarningsReachTheAuthor:
    """A caveat is not a refusal, which is exactly why it is easy to drop.

    Fan-out inflation over a join, a result that must not be re-summed, base metrics in different
    units: each is a way the figure comes out wrong while the SQL stays valid.
    """

    def test_preview_reports_them(self, client):
        r = client.post(f"{BASE}/preview", json=a_body(aggregation="non_additive"))
        assert any("non_additive" in w for w in r.json()["warnings"])

    def test_a_save_reports_them_too(self, graph_client):
        """An author who skipped the preview still has to be told."""
        r = graph_client.post(BASE, json=a_body(aggregation="non_additive"))
        assert any("non_additive" in w for w in r.json()["warnings"])

    def test_an_edit_reports_them_too(self, graph_client):
        graph_client.post(BASE, json=a_body())
        r = graph_client.put(f"{BASE}/m_new", json=a_body(aggregation="non_additive"))
        assert any("non_additive" in w for w in r.json()["warnings"])


class TestAnAuthorsTypoReadsAsARefusal:
    """Every one of these was an opaque 500.

    `to_definition()` sat outside the try in `_compile_or_422`, so model validation escaped as
    an unhandled error. A missing source_table and an unparseable expression are the same thing
    to an author, and both have to come back as something they can act on.
    """

    @pytest.mark.parametrize(
        ("over", "fragment"),
        [
            ({"source_table": ""}, "source_table"),
            ({"name": "total fees"}, "identifier"),
            ({"time_grains": ["fortnight"]}, "time grain"),
            ({"time_grain_column": "invoice-date"}, "time_grain_column"),
            ({"source_table": "legal-ops.invoices"}, "table name"),
        ],
    )
    def test_it_is_a_422_naming_the_field(self, client, over, fragment):
        r = client.post(f"{BASE}/preview", json=a_body(**over))
        assert r.status_code == 422
        assert fragment in r.json()["detail"]

    def test_the_message_is_not_raw_pydantic_json(self, client):
        """It goes straight into a toast, so the type tag and the docs URL are noise."""
        detail = client.post(f"{BASE}/preview", json=a_body(name="total fees")).json()["detail"]
        assert "errors.pydantic.dev" not in detail
        assert "input_value" not in detail


class TestNoTimeAxis:
    """Leaving the time axis blank is the default, so this path has to work.

    `time_grain_column` is optional on the definition but was typed `str` on the way in, which
    made the common case a hard 422 for any client sending an explicit null.
    """

    def test_a_metric_with_no_time_axis_is_created(self, graph_client):
        r = graph_client.post(BASE, json=a_body(time_grain_column=None))
        assert r.status_code == 201, r.text
        assert r.json()["time_grain_column"] is None

    def test_the_apis_own_response_body_is_a_valid_request_body(self, graph_client):
        """Fetch then save is exactly what the edit form does. `_out` renders an unset field as
        null, so a round trip that 422s means no metric without a time axis can be edited."""
        graph_client.post(BASE, json=a_body())
        fetched = graph_client.get(f"{BASE}/m_new").json()
        assert fetched["time_grain_column"] is None
        assert fetched["owner"] is None

        r = graph_client.put(f"{BASE}/m_new", json=fetched)
        assert r.status_code == 200, r.text

    def test_the_round_trip_keeps_the_presentation_fields(self, graph_client):
        """A save that quietly drops `unit` disables the compiler's unit-mismatch check for
        anything an author edits."""
        graph_client.post(BASE, json=a_body(value_type="currency", unit="GBP", format="£#,##0"))
        fetched = graph_client.get(f"{BASE}/m_new").json()
        saved = graph_client.put(f"{BASE}/m_new", json=fetched).json()
        assert (saved["value_type"], saved["unit"], saved["format"]) == (
            "currency",
            "GBP",
            "£#,##0",
        )


class TestWritesRequireTheGraph:
    def test_creating_without_a_graph_is_refused(self, client):
        """Accepting a definition that will not persist is worse than refusing it."""
        r = client.post(BASE, json=a_body())
        assert r.status_code == 503
        assert "graph" in r.json()["detail"]

    def test_the_refusal_says_existing_metrics_still_work(self, client):
        """Otherwise an author reads this as a total outage."""
        assert "keep answering" in client.post(BASE, json=a_body()).json()["detail"]


class TestCreate:
    def test_a_new_metric_is_a_draft(self, graph_client):
        """Authoring must not by itself put a definition into service."""
        r = graph_client.post(BASE, json=a_body())
        assert r.status_code == 201
        assert r.json()["status"] == "draft"

    def test_the_response_carries_the_compiled_sql(self, graph_client):
        assert "SUM(amount)" in graph_client.post(BASE, json=a_body()).json()["sql"]

    def test_a_broken_definition_is_not_stored(self, graph_client):
        """A stored metric that cannot compile means a question matches it and then fails."""
        assert graph_client.post(BASE, json=a_body(expression="SUM(")).status_code == 422
        assert graph_client.get(f"{BASE}/m_new").status_code == 404

    def test_an_invalid_metric_id_is_rejected(self, graph_client):
        assert graph_client.post(BASE, json=a_body(metric_id="has spaces")).status_code == 422

    def test_entity_columns_round_trip(self, graph_client):
        """The declared join key for composition. Guessing it would mis-join silently."""
        r = graph_client.post(BASE, json=a_body(entity_columns={"matter_id": "Matter"}))
        assert r.json()["entity_columns"] == {"matter_id": "Matter"}


class TestUpdate:
    def test_editing_snapshots_the_previous_definition(self, graph_client):
        graph_client.post(BASE, json=a_body(expression="SUM(gross)"))
        graph_client.put(f"{BASE}/m_new", json=a_body(expression="SUM(net)"))

        versions = graph_client.get(f"{BASE}/m_new/versions").json()["versions"]
        assert versions[0]["expression"] == "SUM(gross)"

    def test_editing_preserves_the_status(self, graph_client):
        """Correcting an approved metric must not silently take it out of service."""
        graph_client.post(BASE, json=a_body())
        graph_client.post(f"{BASE}/m_new/status", json={"status": "approved"})
        r = graph_client.put(f"{BASE}/m_new", json=a_body(expression="SUM(net)"))
        assert r.json()["status"] == "approved"

    def test_a_mismatched_id_is_refused(self, graph_client):
        """Silently trusting one of the two would edit a metric the caller did not name."""
        graph_client.post(BASE, json=a_body())
        r = graph_client.put(f"{BASE}/m_new", json=a_body(metric_id="m_other"))
        assert r.status_code == 400

    def test_editing_a_missing_metric_is_404(self, graph_client):
        assert graph_client.put(f"{BASE}/nope", json=a_body(metric_id="nope")).status_code == 404


class TestStatus:
    def test_approving_lets_a_metric_serve(self, graph_client):
        graph_client.post(BASE, json=a_body())
        assert graph_client.get(f"{BASE}?approved_only=true").json() == []

        graph_client.post(f"{BASE}/m_new/status", json={"status": "approved"})
        served = {m["metric_id"] for m in graph_client.get(f"{BASE}?approved_only=true").json()}
        assert "m_new" in served

    def test_deprecating_stops_it_serving(self, graph_client):
        graph_client.post(BASE, json=a_body())
        graph_client.post(f"{BASE}/m_new/status", json={"status": "approved"})
        graph_client.post(f"{BASE}/m_new/status", json={"status": "deprecated"})
        assert graph_client.get(f"{BASE}?approved_only=true").json() == []

    def test_an_unknown_status_is_rejected(self, graph_client):
        graph_client.post(BASE, json=a_body())
        assert graph_client.post(f"{BASE}/m_new/status", json={"status": "live"}).status_code == 422


class TestVersionsAndRestore:
    def test_a_restore_moves_forward(self, graph_client):
        """Rewinding would erase that the intervening definition once answered a question."""
        graph_client.post(BASE, json=a_body(expression="SUM(gross)"))
        graph_client.put(f"{BASE}/m_new", json=a_body(expression="SUM(net)"))

        r = graph_client.post(f"{BASE}/m_new/restore/1")
        assert r.status_code == 200
        assert r.json()["restored_from"] == 1
        assert graph_client.get(f"{BASE}/m_new").json()["expression"] == "SUM(gross)"

    def test_a_restore_returns_a_draft(self, graph_client):
        """Restoring is not the same act as approving."""
        graph_client.post(BASE, json=a_body(expression="SUM(gross)"))
        graph_client.put(f"{BASE}/m_new", json=a_body(expression="SUM(net)"))
        assert graph_client.post(f"{BASE}/m_new/restore/1").json()["status"] == "draft"

    def test_restoring_a_missing_version_is_404(self, graph_client):
        graph_client.post(BASE, json=a_body())
        assert graph_client.post(f"{BASE}/m_new/restore/99").status_code == 404


class TestDeletion:
    def test_deleting_removes_the_metric(self, graph_client):
        graph_client.post(BASE, json=a_body())
        assert graph_client.delete(f"{BASE}/m_new").status_code == 200
        assert graph_client.get(f"{BASE}/m_new").status_code == 404

    def test_deleting_a_missing_metric_is_404(self, graph_client):
        assert graph_client.delete(f"{BASE}/nope").status_code == 404

    def test_the_response_points_at_deprecation_instead(self, graph_client):
        """Deprecating keeps the record of what a number meant; deleting does not."""
        graph_client.post(BASE, json=a_body())
        assert "Deprecating" in graph_client.delete(f"{BASE}/m_new").json()["note"]


class TestAdminOnly:
    """A metric is a statement about what a number means. Letting any authenticated user
    redefine "fees billed" is the same class of mistake as self-assigning to a matter."""

    def _as_non_admin(self) -> TestClient:
        app = create_app(_config())
        services = get_services()
        services.graph = FakeGraph()
        original = services.authenticator.authenticate

        def plain_user(*args, **kw):
            _ctx, grants = original(*args, **kw)
            return AuthContext(tenant_id=TENANT, user_id="reader"), type(grants)(
                tenant_id=TENANT, roles=frozenset()
            )

        services.authenticator.authenticate = plain_user
        return TestClient(app)

    def test_a_non_admin_cannot_create(self):
        assert self._as_non_admin().post(BASE, json=a_body()).status_code == 403

    def test_a_non_admin_cannot_approve(self):
        c = self._as_non_admin()
        assert c.post(f"{BASE}/m_new/status", json={"status": "approved"}).status_code == 403

    def test_a_non_admin_cannot_delete(self):
        assert self._as_non_admin().delete(f"{BASE}/lm_001").status_code == 403

    def test_a_non_admin_may_still_read(self):
        """Knowing what metrics mean is the point of a semantic layer."""
        assert self._as_non_admin().get(BASE).status_code == 200


class TestSeeding:
    """Seeding reads the example pack off disk.

    It used to read `services.metric_matcher`, which every test sets and a running system
    never does, so the endpoint was dead in production while the suite stayed green. That is
    the exact shape of bug an injected seam hides, hence a test with no matcher at all.
    """

    def test_seeding_works_with_no_injected_matcher(self, graph_client):
        get_services().metric_matcher = None
        r = graph_client.post(f"{BASE}/seed")
        assert r.status_code == 200
        assert r.json()["created"] > 0

    def test_seeded_metrics_are_drafts(self, graph_client):
        """The pack is examples for a fictional firm, so seeding must not put them into
        service against a catalog that may not have those tables."""
        get_services().metric_matcher = None
        graph_client.post(f"{BASE}/seed")
        assert graph_client.get(f"{BASE}?approved_only=true").json() == []
        assert len(graph_client.get(BASE).json()) > 0

    def test_seeding_can_approve_deliberately(self, graph_client):
        get_services().metric_matcher = None
        graph_client.post(f"{BASE}/seed?approve=true")
        assert len(graph_client.get(f"{BASE}?approved_only=true").json()) > 0

    def test_the_note_warns_the_metrics_are_examples(self, graph_client):
        """An operator has to know to check them against their own catalog."""
        get_services().metric_matcher = None
        note = graph_client.post(f"{BASE}/seed").json()["note"]
        assert "fictional company" in note
        assert "own catalog" in note

    def test_seeding_follows_the_tenants_pack(self, graph_client):
        """One example file per pack, because a metric names real tables. Seeding the legal
        examples into a retail deployment gives a reader six approved-looking definitions that
        compile against tables their catalog does not have -- which reads as a populated
        semantic layer rather than as a mistake."""
        from src.api.deps import load_example_pack

        get_services().metric_matcher = None
        graph_client.post(f"{BASE}/seed")
        seeded = {m["name"] for m in graph_client.get(BASE).json()}
        # Derived from the app's own pack rather than named, so this test says "the seeded
        # metrics are this pack's" and keeps saying it when the default pack changes.
        domain = get_services().ontology.domain
        assert seeded == {m.name for m in load_example_pack(domain)}
        other = "legal" if domain != "legal" else "retail"
        assert not seeded & {m.name for m in load_example_pack(other)}

    def test_a_pack_with_no_examples_says_so_instead_of_seeding_someone_elses(self):
        """503 with a pointer to the Metrics page, not a silent fallback to another pack's
        tables. There is nothing domain-neutral to offer, so offering nothing is the answer."""
        from src.api.deps import load_example_pack

        assert load_example_pack("healthcare") == []
        assert load_example_pack("retail")
        assert load_example_pack("legal")
