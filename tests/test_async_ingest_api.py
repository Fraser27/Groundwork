"""HTTP boundary for the async ingest endpoints.

The internal trigger endpoint is the interesting one. It has no user and no JWT — the
tenant comes out of an S3 key — so its only protection is a shared secret and the fact
that only the API can mint the keys it accepts. Both are tested here, because "reachable
only from inside the VPC" is not an authorization decision.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.config import AuthConfig, DocumentConfig, GraphConfig, LexGraphConfig
from src.documents.keys import PROCESSED_PREFIX, raw_key
from src.documents.storage import DocumentStorage, set_document_storage

TENANT = "dev-tenant"
BUCKET = "lexgraph-docs"
SECRET = "test-internal-secret"


class FakeS3:
    def __init__(self) -> None:
        self.posts: list[dict[str, Any]] = []

    def generate_presigned_post(self, Bucket: str, Key: str, **kw: Any) -> dict[str, Any]:
        self.posts.append({"Bucket": Bucket, "Key": Key, **kw})
        return {"url": f"https://{Bucket}.s3.amazonaws.com/", "fields": {"key": Key}}

    def head_object(self, **kw: Any) -> dict[str, Any]:
        raise RuntimeError("404 Not Found")


def _config(**over: Any) -> LexGraphConfig:
    cfg = LexGraphConfig(
        environment="local",
        auth=AuthConfig(dev_bypass_tenant=TENANT),
        graph=GraphConfig(uri="bolt://127.0.0.1:1", user="none", password="none"),
        documents=DocumentConfig(bucket=BUCKET),
        internal_api_secret=SECRET,
    )
    for k, v in over.items():
        setattr(cfg, k, v)
    cfg.validate()
    return cfg


@pytest.fixture
def s3() -> FakeS3:
    return FakeS3()


@pytest.fixture
def client(s3: FakeS3):
    set_document_storage(DocumentStorage(BUCKET, s3=s3))
    app = create_app(_config())
    yield TestClient(app)
    set_document_storage(None)


class TestPresignEndpoint:
    def test_returns_a_ticket_with_server_chosen_key(self, client):
        res = client.post(f"/api/tenants/{TENANT}/documents/presign", json={"filename": "a.pdf"})
        assert res.status_code == 200
        body = res.json()
        assert body["key"].startswith(f"raw/{TENANT}/")
        assert "upload_url" in body and "fields" in body

    def test_the_cap_is_reported_so_the_ui_can_precheck(self, client):
        body = client.post(
            f"/api/tenants/{TENANT}/documents/presign", json={"filename": "a.pdf"}
        ).json()
        assert body["max_bytes"] > 0

    def test_a_filename_is_required(self, client):
        res = client.post(f"/api/tenants/{TENANT}/documents/presign", json={})
        assert res.status_code == 422

    def test_another_tenant_is_404(self, client):
        res = client.post("/api/tenants/other-firm/documents/presign", json={"filename": "a.pdf"})
        assert res.status_code == 404


class TestInternalTriggerAuth:
    def test_no_secret_is_refused(self, client):
        res = client.post("/api/internal/ingest", json={"key": raw_key(TENANT, "u1", "a.pdf")})
        assert res.status_code == 401

    def test_a_wrong_secret_is_refused(self, client):
        res = client.post(
            "/api/internal/ingest",
            json={"key": raw_key(TENANT, "u1", "a.pdf")},
            headers={"X-Lexgraph-Internal": "not-it"},
        )
        assert res.status_code == 401

    def test_the_endpoint_is_closed_when_unconfigured(self, s3):
        """An unset secret in a deployed environment is a misconfiguration, and the safe
        reading of it is 'closed' rather than 'open'."""
        set_document_storage(DocumentStorage(BUCKET, s3=s3))
        try:
            unconfigured = TestClient(create_app(_config(internal_api_secret="")))
            res = unconfigured.post(
                "/api/internal/ingest",
                json={"key": raw_key(TENANT, "u1", "a.pdf")},
                headers={"X-Lexgraph-Internal": "anything"},
            )
            assert res.status_code == 503
        finally:
            set_document_storage(None)

    def test_a_valid_secret_is_accepted(self, client):
        res = client.post(
            "/api/internal/ingest",
            json={"key": raw_key(TENANT, "u1", "a.pdf")},
            headers={"X-Lexgraph-Internal": SECRET},
        )
        assert res.status_code == 202
        assert res.json()["status"] == "accepted"

    def test_the_tenant_comes_from_the_key(self, client):
        """Not from a body field: a caller must not be able to name the tenant it is
        ingesting for."""
        res = client.post(
            "/api/internal/ingest",
            json={"key": raw_key("firm-other", "u1", "a.pdf")},
            headers={"X-Lexgraph-Internal": SECRET},
        )
        assert res.json()["tenant_id"] == "firm-other"


class TestInternalTriggerKeyHandling:
    def test_a_processed_key_is_ignored_not_retried(self, client):
        """4xx would make S3 redeliver forever for a key that can never be ingested."""
        res = client.post(
            "/api/internal/ingest",
            json={"key": f"{PROCESSED_PREFIX}{TENANT}/abc/a.pdf"},
            headers={"X-Lexgraph-Internal": SECRET},
        )
        assert res.status_code == 202
        assert res.json()["status"] == "ignored"

    def test_a_traversing_key_is_ignored(self, client):
        res = client.post(
            "/api/internal/ingest",
            json={"key": "raw/../other-firm/u1/a.pdf"},
            headers={"X-Lexgraph-Internal": SECRET},
        )
        assert res.json()["status"] == "ignored"

    def test_the_endpoint_is_not_tenant_scoped_in_its_path(self, client):
        """It deliberately sits outside /tenants/{t}: there is no authenticated tenant on
        this path, and a tenant in the URL would invite trusting it."""
        assert (
            client.post(
                "/api/internal/ingest",
                json={"key": raw_key(TENANT, "u1", "a.pdf")},
                headers={"X-Lexgraph-Internal": SECRET},
            ).status_code
            == 202
        )


class TestJobEndpoints:
    def test_an_unknown_job_is_404(self, client):
        assert client.get(f"/api/tenants/{TENANT}/jobs/job-nope").status_code == 404

    def test_jobs_for_an_unknown_document_is_an_empty_list(self, client):
        """Not a 404: "no jobs yet" is the normal state immediately after an upload, and
        the UI polls this before the notification has been processed."""
        res = client.get(f"/api/tenants/{TENANT}/documents/doc-nope/jobs")
        assert res.status_code == 200
        assert res.json()["jobs"] == []

    def test_another_tenants_jobs_are_404(self, client):
        assert client.get("/api/tenants/other-firm/jobs/job-1").status_code == 404
