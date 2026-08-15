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


class TestTheDocumentListExists:
    """`GET /documents` and `GET /documents/{id}` had no implementation at all.

    The UI called both. FastAPI matched the paths against the POST routes and answered 405,
    which the Documents and Matters pages surfaced as "could not load" — and because
    `updateSettings` pointed at a PUT that also did not exist, saving any governance setting
    failed the same way. Missing routes are worth a test precisely because nothing else
    notices: a typed client compiles happily against an endpoint that was never written.
    """

    def _job(self, tenant: str, doc_id: str, matter_id: str | None = None):
        from src.documents.models import IngestJob, JobState

        job = IngestJob(
            job_id=f"job-{doc_id}",
            document_id=doc_id,
            tenant_id=tenant,
            matter_id=matter_id,
            state=JobState.LIVE,
        )
        return job

    def test_listing_documents_is_not_405(self, client):
        res = client.get(f"/api/tenants/{TENANT}/documents")
        assert res.status_code == 200
        assert "documents" in res.json()

    def test_an_ingested_document_is_listed(self, client):
        from src.api.deps import get_services

        get_services().job_store.put_job(self._job(TENANT, "doc-1"))
        body = client.get(f"/api/tenants/{TENANT}/documents").json()
        assert [d["document_id"] for d in body["documents"]] == ["doc-1"]
        assert body["documents"][0]["state"] == "LIVE"

    def test_only_the_latest_job_per_document_is_shown(self, client):
        """A document re-ingested after a failure has several jobs. The list is a statement
        about where each document stands now, not a log of attempts."""
        from src.api.deps import get_services
        from src.documents.models import JobState

        store = get_services().job_store
        old = self._job(TENANT, "doc-1")
        old.state = JobState.PARSE_FAILED
        old.created_at = "2020-01-01T00:00:00Z"
        store.put_job(old)
        new = self._job(TENANT, "doc-1")
        new.job_id = "job-newer"
        new.created_at = "2030-01-01T00:00:00Z"
        store.put_job(new)

        docs = client.get(f"/api/tenants/{TENANT}/documents").json()["documents"]
        assert len(docs) == 1
        assert docs[0]["state"] == "LIVE"

    def test_another_tenants_documents_are_not_listed(self, client):
        from src.api.deps import get_services

        get_services().job_store.put_job(self._job("other-firm", "theirs"))
        body = client.get(f"/api/tenants/{TENANT}/documents").json()
        assert body["documents"] == []

    def test_a_missing_document_is_404_not_405(self, client):
        res = client.get(f"/api/tenants/{TENANT}/documents/nope")
        assert res.status_code == 404

    def test_the_detail_carries_a_timeline(self, client):
        from src.api.deps import get_services

        get_services().job_store.put_job(self._job(TENANT, "doc-1"))
        body = client.get(f"/api/tenants/{TENANT}/documents/doc-1").json()
        assert body["document_id"] == "doc-1"
        assert isinstance(body["timeline"], list)
        assert isinstance(body["assertions"], list)


class TestGovernanceIsWritable:
    def test_the_ungoverned_switch_can_be_turned_on(self, client):
        """The Admin toggle wrote to PUT /settings, which does not exist. This is the route
        that actually holds the setting."""
        res = client.patch(
            f"/api/tenants/{TENANT}/governance", json={"block_ungoverned_queries": True}
        )
        assert res.status_code == 200
        assert res.json()["settings"]["block_ungoverned_queries"] is True

        shown = client.get(f"/api/tenants/{TENANT}/settings").json()
        assert shown["block_ungoverned_queries"] is True

    def test_there_is_no_put_settings(self, client):
        """Pinned so the UI is not tempted back. GET /settings is a read-only projection
        that joins governance onto process config; writes belong on governance."""
        res = client.put(f"/api/tenants/{TENANT}/settings", json={"min_confidence": 0.9})
        assert res.status_code == 405
