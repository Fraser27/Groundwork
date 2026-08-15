"""Provenance over HTTP: what a PDF viewer gets, and who is allowed to get a link.

The security case is the download endpoint. A presigned URL answers to nobody once it
exists, so the authorization check has to happen *before* one is created — a test that
only asserts the status code would pass against an implementation that mints the URL
and then throws it away, which is a leak if anything ever logs it.

boto3 is stubbed. No AWS.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.deps import get_services
from src.config import AuthConfig, DocumentConfig, GraphConfig, LexGraphConfig
from src.documents.chunk import chunk_document
from src.documents.parse import ParseFailed, assemble
from src.documents.storage import DocumentStorage, set_document_storage
from src.graph.assertions import EpistemicClass, SourceLocator, build_assertion
from src.graph.scope import AuthContext
from tests.test_storage import PDF, FakeS3

TENANT = "dev-tenant"
BUCKET = "lexgraph-docs"
QUOTE = "The court declined to follow Brown."


class _FakeParser:
    """Stands in for `VisionParser`. No Bedrock, no PyMuPDF, no rendering."""

    def __init__(self, pages: list[str], *, fail: str | None = None) -> None:
        self.pages = pages
        self.fail = fail
        self.seen_filename: str | None = None

    def parse(self, document_id: str, data: bytes, *, filename: str | None = None):
        self.seen_filename = filename
        if self.fail:
            raise ParseFailed(self.fail)
        return assemble(document_id, self.pages, method="vision:fake@v1", filename=filename)


def _config(bucket: str = BUCKET) -> LexGraphConfig:
    cfg = LexGraphConfig(
        environment="local",
        auth=AuthConfig(dev_bypass_tenant=TENANT),
        graph=GraphConfig(uri="bolt://127.0.0.1:1", user="none", password="none"),
        documents=DocumentConfig(bucket=bucket, kms_key_id="alias/lexgraph-docs"),
    )
    cfg.validate()
    return cfg


@pytest.fixture
def s3() -> FakeS3:
    return FakeS3()


@pytest.fixture
def storage(s3: FakeS3) -> DocumentStorage:
    return DocumentStorage(BUCKET, kms_key_id="alias/lexgraph-docs", s3=s3)


@pytest.fixture
def client(storage: DocumentStorage) -> TestClient:
    set_document_storage(storage)
    app = create_app(_config())
    yield TestClient(app)
    set_document_storage(None)


@pytest.fixture
def unconfigured_client() -> TestClient:
    """No DOCUMENT_BUCKET: provenance must degrade rather than 500."""
    set_document_storage(None)
    return TestClient(create_app(_config(bucket="")))


@pytest.fixture
def ctx() -> AuthContext:
    return AuthContext(user_id="dev@localhost", tenant_id=TENANT)


def _store(storage: DocumentStorage, ctx: AuthContext, *, matter_id: str | None = None):
    return storage.put_document(
        ctx, filename="motion-to-dismiss.pdf", body=PDF, matter_id=matter_id
    )


def _stage(document_id: str, *, matter_id: str | None = "M-1", page: int = 4) -> str:
    services = get_services()
    ctx = AuthContext(user_id="dev@localhost", tenant_id=TENANT)
    a = build_assertion(
        tenant_id=TENANT,
        subject_id=f"document:{document_id}",
        predicate="CONCERNS_TOPIC",
        object_id="Topic-Antitrust",
        epistemic_class=EpistemicClass.EXTRACTED_MODEL,
        method="llm:claude-sonnet-5",
        confidence=0.7,
        source_locator=SourceLocator(
            document_id=document_id,
            filename="motion-to-dismiss.pdf",
            page=page,
            chunk_id=f"{document_id}:p{page}:100-160",
            quote=QUOTE,
        ),
        matter_id=matter_id,
    )
    services.review_queue.stage(ctx, [a])
    return a.assertion_id


def _provenance(client: TestClient, assertion_id: str) -> dict:
    r = client.get(f"/api/tenants/{TENANT}/assertions/{assertion_id}/provenance")
    assert r.status_code == 200, r.text
    return r.json()


class TestProvenanceCarriesWhatAViewerNeeds:
    def test_includes_filename_page_and_quote(self, client, storage, ctx):
        doc = _store(storage, ctx)
        body = _provenance(client, _stage(doc.document_id))["document"]
        assert body["filename"] == "motion-to-dismiss.pdf"
        assert body["page"] == 4
        assert body["quote"] == QUOTE

    def test_includes_chunk_id(self, client, storage, ctx):
        doc = _store(storage, ctx)
        body = _provenance(client, _stage(doc.document_id))["document"]
        assert body["chunk_id"].startswith(doc.document_id)

    def test_mints_a_download_url_with_an_expiry(self, client, storage, ctx):
        doc = _store(storage, ctx)
        body = _provenance(client, _stage(doc.document_id))["document"]
        assert "X-Amz-Signature" in body["download_url"]
        assert body["expires_at"]

    def test_url_seeks_to_the_cited_page(self, client, storage, ctx):
        doc = _store(storage, ctx)
        body = _provenance(client, _stage(doc.document_id, page=9))["document"]
        assert body["download_url"].endswith("#page=9")

    def test_url_is_fresh_on_every_request(self, client, storage, ctx, s3):
        """Minted per request, which is what makes storing it unnecessary."""
        doc = _store(storage, ctx)
        aid = _stage(doc.document_id)
        _provenance(client, aid)
        _provenance(client, aid)
        assert len(s3.presigns) == 2

    def test_premise_tree_and_explanation_survive(self, client, storage, ctx):
        """The existing behaviour is the substance; the link is the convenience."""
        doc = _store(storage, ctx)
        body = _provenance(client, _stage(doc.document_id))
        assert "premises" in body
        assert "ai model" in body["explanation"].lower()
        assert body["is_current"] is True

    def test_no_url_is_stored_on_the_assertion(self, client, storage, ctx):
        doc = _store(storage, ctx)
        source = _provenance(client, _stage(doc.document_id))["assertion"]["source_locator"]
        assert not any("url" in k for k in source)
        assert not any("expires" in k for k in source)


class TestDegradesWithoutStorage:
    def test_provenance_still_200s(self, unconfigured_client, storage, ctx):
        doc = _store(storage, ctx)
        body = _provenance(unconfigured_client, _stage(doc.document_id))
        assert body["document"]["download_url"] is None

    def test_says_why_there_is_no_link(self, unconfigured_client, storage, ctx):
        doc = _store(storage, ctx)
        body = _provenance(unconfigured_client, _stage(doc.document_id))
        assert "DOCUMENT_BUCKET" in body["document"]["link_unavailable"]

    def test_page_and_quote_are_still_there(self, unconfigured_client, storage, ctx):
        """A citation with no link is still checkable by hand, which is the whole point."""
        doc = _store(storage, ctx)
        body = _provenance(unconfigured_client, _stage(doc.document_id))["document"]
        assert body["page"] == 4
        assert body["quote"] == QUOTE

    def test_unuploaded_document_degrades_rather_than_erroring(self, client):
        """An assertion from text ingest has no S3 object behind it."""
        body = _provenance(client, _stage("doc-never-uploaded"))
        assert body["document"]["download_url"] is None
        assert body["document"]["link_unavailable"]

    def test_signing_failure_does_not_break_the_explanation(
        self, client, storage, ctx, monkeypatch
    ):
        doc = _store(storage, ctx)
        aid = _stage(doc.document_id)

        def boom(*_a, **_k):
            raise RuntimeError("KMS unavailable")

        monkeypatch.setattr(storage, "presign_download", boom)
        body = _provenance(client, aid)
        assert body["document"]["download_url"] is None
        assert body["explanation"]


class TestDownloadEndpointAuthorization:
    def test_owner_gets_a_url(self, client, storage, ctx):
        doc = _store(storage, ctx, matter_id="M-1")
        r = client.get(f"/api/tenants/{TENANT}/documents/{doc.document_id}/download")
        assert r.status_code == 200
        assert "X-Amz-Signature" in r.json()["download_url"]

    def test_walled_caller_gets_404_and_no_url_is_minted(self, storage, s3, ctx):
        """The assertion that matters: nothing is signed before the check passes.

        A 404 alone would also be returned by an implementation that presigns first and
        discards the result — which has already leaked into any log or trace.
        """
        doc = _store(storage, ctx, matter_id="M-WALL")
        set_document_storage(storage)
        app = create_app(_config())

        services = get_services()
        original = services.authenticator.authenticate

        def walled(*args, **kwargs):
            _, grants = original(*args, **kwargs)
            return (
                AuthContext(
                    user_id="associate@firm",
                    tenant_id=TENANT,
                    matter_denylist=frozenset({"M-WALL"}),
                ),
                grants,
            )

        services.authenticator.authenticate = walled
        try:
            s3.presigns.clear()
            r = TestClient(app).get(f"/api/tenants/{TENANT}/documents/{doc.document_id}/download")
            assert r.status_code == 404
            assert s3.presigns == []
        finally:
            services.authenticator.authenticate = original
            set_document_storage(None)

    def test_allowlist_excluding_the_matter_gets_404(self, storage, s3, ctx):
        doc = _store(storage, ctx, matter_id="M-OTHER")
        set_document_storage(storage)
        app = create_app(_config())

        services = get_services()
        original = services.authenticator.authenticate

        def narrow(*args, **kwargs):
            _, grants = original(*args, **kwargs)
            return (
                AuthContext(
                    user_id="associate@firm",
                    tenant_id=TENANT,
                    matter_allowlist=frozenset({"M-MINE"}),
                ),
                grants,
            )

        services.authenticator.authenticate = narrow
        try:
            s3.presigns.clear()
            r = TestClient(app).get(f"/api/tenants/{TENANT}/documents/{doc.document_id}/download")
            assert r.status_code == 404
            assert s3.presigns == []
        finally:
            services.authenticator.authenticate = original
            set_document_storage(None)

    def test_provenance_withholds_the_link_for_a_walled_document(self, storage, s3, ctx):
        doc = _store(storage, ctx, matter_id="M-WALL")
        set_document_storage(storage)
        app = create_app(_config())
        aid = _stage(doc.document_id, matter_id=None)

        services = get_services()
        original = services.authenticator.authenticate

        def walled(*args, **kwargs):
            _, grants = original(*args, **kwargs)
            return (
                AuthContext(
                    user_id="associate@firm",
                    tenant_id=TENANT,
                    matter_denylist=frozenset({"M-WALL"}),
                ),
                grants,
            )

        services.authenticator.authenticate = walled
        try:
            s3.presigns.clear()
            body = _provenance(TestClient(app), aid)
            assert body["document"]["download_url"] is None
            assert s3.presigns == []
        finally:
            services.authenticator.authenticate = original
            set_document_storage(None)

    def test_other_tenant_path_is_404(self, client, storage, ctx):
        doc = _store(storage, ctx)
        r = client.get(f"/api/tenants/other-firm/documents/{doc.document_id}/download")
        assert r.status_code == 404

    def test_unknown_document_is_404(self, client):
        r = client.get(f"/api/tenants/{TENANT}/documents/doc-nope/download")
        assert r.status_code == 404

    def test_unconfigured_storage_is_503_not_500(self, unconfigured_client):
        r = unconfigured_client.get(f"/api/tenants/{TENANT}/documents/doc-1/download")
        assert r.status_code == 503


class TestDownloadEndpointResponse:
    def test_expiry_is_short_by_default(self, client, storage, ctx):
        doc = _store(storage, ctx)
        body = client.get(f"/api/tenants/{TENANT}/documents/{doc.document_id}/download").json()
        assert body["expires_in"] == 900
        assert body["expires_at"]

    def test_page_parameter_seeks(self, client, storage, ctx):
        doc = _store(storage, ctx)
        body = client.get(
            f"/api/tenants/{TENANT}/documents/{doc.document_id}/download",
            params={"page": 12},
        ).json()
        assert body["download_url"].endswith("#page=12")

    def test_over_long_expiry_is_rejected_by_validation(self, client, storage, ctx, s3):
        doc = _store(storage, ctx)
        s3.presigns.clear()
        r = client.get(
            f"/api/tenants/{TENANT}/documents/{doc.document_id}/download",
            params={"expires_in": 86400},
        )
        assert r.status_code == 422
        assert s3.presigns == []

    def test_response_carries_filename_and_media_type(self, client, storage, ctx):
        doc = _store(storage, ctx)
        body = client.get(f"/api/tenants/{TENANT}/documents/{doc.document_id}/download").json()
        assert body["filename"] == "motion-to-dismiss.pdf"
        assert body["media_type"] == "application/pdf"


def _upload(client: TestClient, *, name: str = "motion.pdf", body: bytes = PDF, **data):
    return client.post(
        f"/api/tenants/{TENANT}/documents",
        files={"file": (name, body, "application/pdf")},
        data={"run_model_extraction": "false", **data},
    )


class TestUploadEndpoint:
    """Store first, then transcribe.

    The ordering is the point: S3 is the only source of truth, so the bytes must land
    before anything that can fail gets a chance to. A transcription failure then costs a
    re-run rather than the document.
    """

    def test_upload_stores_the_bytes(self, client, s3):
        r = _upload(client)
        assert r.status_code == 201, r.text
        assert PDF in s3.objects[next(iter(s3.objects))]["Body"]

    def test_response_identifies_the_stored_document(self, client):
        body = _upload(client).json()
        assert body["filename"] == "motion.pdf"
        assert body["content_sha256"]
        assert body["s3_uri"].startswith(f"s3://{BUCKET}/")

    def test_the_stored_document_is_immediately_downloadable(self, client):
        """The upload and download paths must agree on the document id."""
        document_id = _upload(client).json()["document_id"]
        r = client.get(f"/api/tenants/{TENANT}/documents/{document_id}/download")
        assert r.status_code == 200

    def test_reuploading_the_same_bytes_is_idempotent(self, client):
        first = _upload(client).json()["document_id"]
        second = _upload(client).json()["document_id"]
        assert first == second

    def test_an_empty_file_is_refused(self, client):
        assert _upload(client, body=b"").status_code == 400

    def test_a_walled_matter_is_refused_before_anything_is_stored(self, client, s3):
        """404 rather than 403, matching scope.py, and nothing reaches S3."""
        services = get_services()
        original = services.authenticator.authenticate

        def walled(*a, **kw):
            _, grants = original(*a, **kw)
            return (
                AuthContext(
                    user_id="associate@firm",
                    tenant_id=TENANT,
                    matter_denylist=frozenset({"M-9"}),
                ),
                grants,
            )

        services.authenticator.authenticate = walled
        try:
            r = _upload(client, matter_id="M-9")
        finally:
            services.authenticator.authenticate = original
        assert r.status_code == 404
        assert s3.puts == []


class TestUploadDegradesWithoutAVisionModel:
    """No Bedrock must cost transcription, never the upload.

    A 500 here would lose the one irreplaceable thing in the pipeline for the sake of a
    step that can be re-run the moment credentials exist.
    """

    def test_upload_still_succeeds(self, client):
        get_services().parser = None
        assert _upload(client).status_code == 201

    def test_the_bytes_are_still_stored(self, client, s3):
        get_services().parser = None
        _upload(client)
        assert s3.puts, "the document must land in S3 even with nothing to read it"

    def test_the_response_says_transcription_was_skipped(self, client):
        get_services().parser = None
        body = _upload(client).json()
        assert "skipped" in body["transcription"]
        assert body["page_count"] is None

    def test_the_state_is_parse_failed_so_a_retry_targets_parsing(self, client):
        get_services().parser = None
        assert _upload(client).json()["state"] == "PARSE_FAILED"


class TestUploadTranscribesAndChunks:
    def test_a_transcribed_upload_is_chunked_and_citable(self, client):
        get_services().parser = _FakeParser(["Held for the plaintiff.", "Costs reserved."])
        body = _upload(client).json()
        assert body["page_count"] == 2
        assert body["chunks"] == 2
        assert body["transcription"] == "vision:fake@v1"

    def test_the_filename_reaches_the_locator_a_reviewer_reads(self):
        """`SourceLocator.filename` is what tells a reviewer which PDF to open.

        Asserted on the locator rather than on the parser's argument: the whole point is
        that the name survives parse -> chunk -> locator, and each hop can drop it.
        """
        parsed = assemble(
            "doc-1", ["Held for the plaintiff."], method="vision:fake@v1", filename="skeleton.pdf"
        )
        chunks = chunk_document(parsed, tenant_id=TENANT, filename=parsed.filename)
        assert [c.to_locator().filename for c in chunks] == ["skeleton.pdf"]

    def test_the_upload_hands_the_parser_the_stored_filename(self, client):
        parser = _FakeParser(["Held for the plaintiff."])
        get_services().parser = parser
        _upload(client, name="skeleton.pdf")
        assert parser.seen_filename == "skeleton.pdf"

    @pytest.mark.parametrize("declared", ["text/plain", "application/octet-stream"])
    def test_a_plain_text_upload_needs_no_vision_model(self, client, declared):
        """A .txt file must not be made to depend on Bedrock to be read.

        Parametrised over the content type because a browser sends
        application/octet-stream for anything it does not recognise, and trusting that
        would route a text file to the vision model.
        """
        get_services().parser = None
        r = client.post(
            f"/api/tenants/{TENANT}/documents",
            files={"file": ("notes.txt", b"Held for the plaintiff.", declared)},
            data={"run_model_extraction": "false"},
        )
        assert r.status_code == 201
        assert r.json()["transcription"] == "text:inline@v1"

    def test_a_transcription_failure_stores_the_document_anyway(self, client, s3):
        get_services().parser = _FakeParser([], fail="transcription failed on page 3")
        body = _upload(client).json()
        assert body["state"] == "PARSE_FAILED"
        assert "page 3" in body["error"]
        assert s3.puts
