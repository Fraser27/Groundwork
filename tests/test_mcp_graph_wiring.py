"""The MCP server has to connect the graph itself.

This is a regression test for a silent read failure. `build_services` always constructs an
`InMemoryAssertionStore`, and only the REST app's lifespan hook swapped in the graph-backed
one. The MCP server has no lifespan hook, so it served every tool from an empty dict: a tenant
holding 56 assertions got `{"assertions": [], "total": 0}` from `search_assertions`, with no
error anywhere, and an agent reading that concluded the facts did not exist and said so.

Two processes serve this container. Anything that only one of them does to its container is a
place where the same question gets two answers.
"""

from __future__ import annotations

from typing import Any

from src.api.deps import build_services, connect_graph, reconnect_if_due
from src.config import AuthConfig, GraphConfig, GroundworkConfig
from src.documents.review import InMemoryAssertionStore


def _config() -> GroundworkConfig:
    cfg = GroundworkConfig(
        environment="local",
        auth=AuthConfig(dev_bypass_tenant="demo-firm"),
        # Unreachable on purpose: the point is what happens either way, and a test must not
        # need a graph to assert that a connection was attempted.
        graph=GraphConfig(uri="bolt://127.0.0.1:1", user="none", password="none"),
    )
    cfg.validate()
    return cfg


class _FakeGraph:
    def verify_connectivity(self) -> bool:
        return True

    def query(self, *_: Any, **__: Any) -> list[Any]:
        return []

    def write(self, *_: Any, **__: Any) -> None:
        return None

    def close(self) -> None:
        return None


class TestConnectingTheGraphIsSharedNotCopied:
    def test_an_unreachable_graph_degrades_rather_than_raising(self):
        """`/health` has to be able to report a graph that is down, which it cannot do if the
        process failed to start."""
        services = build_services(_config())
        assert connect_graph(services) is False
        assert services.graph is None

    def test_a_reachable_graph_puts_the_store_on_it(self, monkeypatch):
        """The swap that was missing. Without it a read returns nothing and a write is lost on
        the next deploy, both silently."""
        services = build_services(_config())
        assert isinstance(services.review_queue.store, InMemoryAssertionStore)

        monkeypatch.setattr("src.graph.client.GraphClient", lambda *a, **k: _FakeGraph())
        monkeypatch.setattr("src.graph.schema.init_schema", lambda *a, **k: None)

        assert connect_graph(services) is True
        assert services.graph is not None
        assert type(services.review_queue.store).__name__ == "GraphAssertionStore"

    def test_connecting_twice_leaves_the_first_connection_alone(self, monkeypatch):
        """Both processes call this, and the REST app calls it from a lifespan that can run
        more than once in tests. Reconnecting would drop a live client on the floor."""
        services = build_services(_config())
        monkeypatch.setattr("src.graph.client.GraphClient", lambda *a, **k: _FakeGraph())
        monkeypatch.setattr("src.graph.schema.init_schema", lambda *a, **k: None)

        connect_graph(services)
        first = services.graph
        assert connect_graph(services) is True
        assert services.graph is first


class TestAFailedConnectIsNotTerminal:
    """A task that started while Neptune was still provisioning stayed degraded for its whole
    life. `POST /metrics` returned 503 for hours after the cluster went healthy, because
    `connect_graph` ran once and nothing tried again. Only a redeploy cleared it.
    """

    def test_the_graph_heals_without_a_restart(self, monkeypatch):
        monkeypatch.setattr("src.api.deps._graph_retry_after", 0.0)
        monkeypatch.setattr("src.api.deps.GRAPH_RECONNECT_COOLDOWN_SECONDS", 0.0)
        services = build_services(_config())
        assert connect_graph(services) is False

        monkeypatch.setattr("src.graph.client.GraphClient", lambda *a, **k: _FakeGraph())
        monkeypatch.setattr("src.graph.schema.init_schema", lambda *a, **k: None)

        assert reconnect_if_due(services) is True
        assert type(services.review_queue.store).__name__ == "GraphAssertionStore"

    def test_a_down_graph_is_not_probed_once_per_request(self, monkeypatch):
        """The reason for the cooldown. `verify_connectivity` against a dead endpoint costs a
        connect timeout, and paying it on every request turns a degraded API into a hung one."""
        monkeypatch.setattr("src.api.deps._graph_retry_after", 0.0)
        attempts: list[str] = []

        def counting_client(*_: Any, **__: Any) -> Any:
            attempts.append("built")
            raise OSError("unreachable")

        monkeypatch.setattr("src.graph.client.GraphClient", counting_client)
        services = build_services(_config())

        for _ in range(20):
            assert reconnect_if_due(services) is False
        assert attempts == ["built"]

    def test_an_explicit_connect_is_never_refused_by_the_cooldown(self, monkeypatch):
        """`connect_graph` means now: a lifespan hook or an operator asking must not be told to
        wait because a request happened to fail a moment earlier."""
        monkeypatch.setattr("src.api.deps._graph_retry_after", 0.0)
        services = build_services(_config())
        assert reconnect_if_due(services) is False

        monkeypatch.setattr("src.graph.client.GraphClient", lambda *a, **k: _FakeGraph())
        monkeypatch.setattr("src.graph.schema.init_schema", lambda *a, **k: None)

        assert reconnect_if_due(services) is False
        assert connect_graph(services) is True

    def test_a_request_is_what_notices(self, monkeypatch):
        """Nothing polls in the background, so the retry has to hang off the path both processes
        already take."""
        monkeypatch.setattr("src.api.deps._graph_retry_after", 0.0)
        monkeypatch.setattr("src.api.deps.GRAPH_RECONNECT_COOLDOWN_SECONDS", 0.0)
        from src.api.deps import get_services, set_services

        services = build_services(_config())
        set_services(services)
        assert connect_graph(services) is False

        monkeypatch.setattr("src.graph.client.GraphClient", lambda *a, **k: _FakeGraph())
        monkeypatch.setattr("src.graph.schema.init_schema", lambda *a, **k: None)

        assert get_services().graph is not None

    def test_no_graph_leaves_the_catalog_retryable(self):
        """Marking it hydrated was safe only while a reconnect was impossible. It would now pin an
        empty catalog across the reconnect, and the column allowlist is built from that."""
        services = build_services(_config())
        assert services.hydrate_catalog("demo-firm") is False
        assert services.catalog.is_hydrated("demo-firm") is False


class TestBothProcessesConnect:
    def test_the_mcp_app_connects_on_construction(self, monkeypatch):
        """The bug. The MCP server has no lifespan hook, so if `create_app` does not do this
        nothing does, and every assertion-reading tool answers "nothing found"."""
        connected: list[str] = []

        def spy(services: Any) -> bool:
            connected.append("called")
            return False

        monkeypatch.setattr("src.mcp.server.connect_graph", spy)
        from src.mcp.server import create_app

        create_app(_config())
        assert connected == ["called"]

    def test_the_rest_app_connects_in_its_lifespan(self, monkeypatch):
        from fastapi.testclient import TestClient

        connected: list[str] = []
        monkeypatch.setattr(
            "src.api.app.connect_graph", lambda services: connected.append("called") or False
        )
        from src.api.app import create_app

        with TestClient(create_app(_config())):
            pass
        assert connected == ["called"]
