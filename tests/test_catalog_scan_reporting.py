"""A scan that did not reach the graph has to say so.

The response could not distinguish a persisted scan from one that reached only this process. With
no graph, `catalog_graph_store()` returns None and every store below it is in-memory, so the route
returned `tables_found: 6, assertions_live: 91, graph_error: null` and 200 OK, plus a note saying
metrics could be compiled against what it found.

That is what a live deployment did twice. The container was replaced, hydration reported `0 tables,
0 sources`, and the firewall then refused every table a governed metric named -- reported to the
user as "unauthorized tables", which reads like an IAM problem and is not one. Nothing in the scan
response had pointed at the cause.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.deps import get_services
from src.config import AuthConfig, GraphConfig, GroundworkConfig
from src.discovery.glue_scanner import ScanResult, scan_catalog

TENANT = "demo-firm"
SCAN = f"/api/tenants/{TENANT}/sources/scan"

ORDERS = {
    "Name": "orders",
    "StorageDescriptor": {
        "Location": "s3://lake/iceberg_db.db/orders/",
        "Columns": [
            {"Name": "order_id", "Type": "string"},
            {"Name": "status", "Type": "string"},
        ],
    },
}

RETURNS = {
    "Name": "returns",
    "StorageDescriptor": {
        "Location": "s3://lake/iceberg_db.db/returns/",
        "Columns": [{"Name": "order_id", "Type": "string"}],
    },
}


def _config() -> GroundworkConfig:
    cfg = GroundworkConfig(
        environment="local",
        auth=AuthConfig(dev_bypass_tenant=TENANT),
        graph=GraphConfig(uri="bolt://127.0.0.1:1", user="none", password="none"),
    )
    cfg.validate()
    return cfg


class _FakeGlue:
    def get_paginator(self, operation: str):
        if operation == "get_databases":
            return _Pages(lambda: [{"DatabaseList": [{"Name": "iceberg_db"}]}])
        return _Pages(lambda DatabaseName: [{"TableList": [ORDERS, RETURNS]}])


class _Pages:
    def __init__(self, pages):
        self._pages = pages

    def paginate(self, **kwargs):
        return self._pages(**kwargs)


def _found() -> ScanResult:
    """Through the real scanner, so `nodes` and `assertions` are the shapes the route writes."""
    return scan_catalog(_FakeGlue(), tenant_id=TENANT, source_id="glue-main")


@pytest.fixture
def client(monkeypatch) -> TestClient:
    app = create_app(_config())
    monkeypatch.setattr("src.api.routes_catalog.scan_catalog", lambda *a, **k: _found())
    monkeypatch.setattr("boto3.client", lambda *a, **k: object())
    return TestClient(app)


class TestAScanWithNoGraphIsNotReportedAsClean:
    def test_it_names_the_graph_as_the_reason(self, client):
        body = client.post(SCAN, json={"source_id": "glue-main"}).json()
        assert body["graph_error"], "a cache-only scan reported no error at all"
        assert "graph" in body["graph_error"]

    def test_the_note_does_not_promise_queryable_tables(self, client):
        """The note is the sentence a model summarising this response repeats back, and the old one
        said metrics could be compiled against the result either way."""
        body = client.post(SCAN, json={"source_id": "glue-main"}).json()
        assert "not reach the graph" in body["note"]
        assert "can be compiled" not in body["note"]

    def test_nothing_was_written_to_the_graph(self, client):
        assert client.post(SCAN, json={"source_id": "glue-main"}).json()["nodes_written"] == 0

    def test_the_scan_is_still_kept(self, client):
        """Degrading is the documented intent: the Tables page reads the cache, so a graph outage
        must not throw the scan away. Only the silent part was wrong."""
        assert client.post(SCAN, json={"source_id": "glue-main"}).json()["tables_found"] == 2
        assert get_services().catalog.tables(TENANT)


class TestAPersistedScanIsStillReportedAsClean:
    def test_no_graph_error_and_the_original_note(self, monkeypatch):
        """The regression guard on the other side: the message must not fire whenever a scan finds
        nothing to write, only when there is nowhere to write it."""
        app = create_app(_config())
        monkeypatch.setattr("src.api.routes_catalog.scan_catalog", lambda *a, **k: _found())
        monkeypatch.setattr("boto3.client", lambda *a, **k: object())

        class _Store:
            def persist(self, nodes: list[Any]) -> int:
                return len(nodes)

        monkeypatch.setattr(get_services(), "catalog_graph_store", lambda: _Store())
        body = TestClient(app).post(SCAN, json={"source_id": "glue-main"}).json()

        assert body["graph_error"] is None
        assert "can be compiled" in body["note"]
