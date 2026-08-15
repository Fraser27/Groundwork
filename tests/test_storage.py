"""Document storage, and the rules about presigned URLs.

Two of these are the point of the module rather than coverage of it: a URL is never
persisted anywhere, and one is never minted for a caller who cannot read the matter.
Both are silent failures if they regress — a stored URL looks like working provenance
until it expires, and an over-broad URL looks like nothing at all.

boto3 is stubbed throughout. No AWS credentials, no network.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.config import DocumentConfig, LexGraphConfig
from src.documents.keys import PROCESSED_PREFIX, RAW_PREFIX, parse_raw_key, raw_key
from src.documents.storage import (
    MAX_EXPIRY_SECONDS,
    DocumentNotFound,
    DocumentStorage,
    document_key,
    safe_filename,
    set_document_storage,
    storage_from_config,
)
from src.graph.assertions import SourceLocator
from src.graph.scope import AuthContext, ScopeViolation

TENANT = "dev-tenant"
BUCKET = "lexgraph-docs"
PDF = b"%PDF-1.7 fake bytes"


class FakeS3:
    """Records calls so a test can assert what was and was not sent to AWS."""

    def __init__(self, *, existing: set[str] | None = None) -> None:
        self.objects: dict[str, dict[str, Any]] = {}
        self.existing = existing or set()
        self.puts: list[dict[str, Any]] = []
        self.deletes: list[dict[str, Any]] = []
        self.presigns: list[dict[str, Any]] = []
        self.lists: list[dict[str, Any]] = []

    def put_object(self, **kwargs: Any) -> dict[str, Any]:
        self.puts.append(kwargs)
        self.objects[kwargs["Key"]] = kwargs
        self.existing.add(kwargs["Key"])
        return {"VersionId": f"v{len(self.puts)}"}

    def head_object(self, **kwargs: Any) -> dict[str, Any]:
        if kwargs["Key"] not in self.existing:
            raise RuntimeError("404 Not Found")
        return {"VersionId": "v1"}

    def delete_object(self, **kwargs: Any) -> dict[str, Any]:
        self.deletes.append(kwargs)
        self.existing.discard(kwargs["Key"])
        return {"DeleteMarker": True}

    def generate_presigned_url(self, ClientMethod: str, **kwargs: Any) -> str:
        self.presigns.append({"ClientMethod": ClientMethod, **kwargs})
        key = kwargs["Params"]["Key"]
        return f"https://{BUCKET}.s3.amazonaws.com/{key}?X-Amz-Signature=deadbeef"

    def list_objects_v2(self, **kwargs: Any) -> dict[str, Any]:
        self.lists.append(kwargs)
        prefix = kwargs.get("Prefix", "")
        keys = sorted(k for k in self.existing if k.startswith(prefix))
        return {"Contents": [{"Key": k, "Size": len(PDF)} for k in keys]}


@pytest.fixture
def s3() -> FakeS3:
    return FakeS3()


@pytest.fixture
def storage(s3: FakeS3) -> DocumentStorage:
    return DocumentStorage(BUCKET, kms_key_id="alias/lexgraph-docs", s3=s3)


@pytest.fixture
def ctx() -> AuthContext:
    return AuthContext(user_id="lawyer@firm", tenant_id=TENANT)


@pytest.fixture(autouse=True)
def _clear_override():
    yield
    set_document_storage(None)


class TestKeyDerivation:
    def test_key_is_tenant_prefixed(self):
        """The prefix is what an IAM boundary pins to, so it leads the key."""
        key = document_key(TENANT, "a" * 64, "motion.pdf")
        assert key.startswith(f"{PROCESSED_PREFIX}{TENANT}/")

    def test_key_is_content_addressed(self):
        digest = "b" * 64
        assert (
            document_key(TENANT, digest, "motion.pdf")
            == f"{PROCESSED_PREFIX}{TENANT}/{digest}/motion.pdf"
        )

    def test_processed_and_raw_prefixes_are_disjoint(self):
        """The notification filters on raw/. Overlap would make it observe its own output."""
        assert not document_key(TENANT, "f" * 64, "a.pdf").startswith(RAW_PREFIX)
        assert not raw_key(TENANT, "upload-1", "a.pdf").startswith(PROCESSED_PREFIX)

    def test_same_bytes_same_key_across_filenames_differ(self):
        digest = "c" * 64
        a = document_key(TENANT, digest, "motion.pdf")
        b = document_key(TENANT, digest, "motion.pdf")
        assert a == b

    def test_different_tenants_never_share_a_key(self):
        digest = "d" * 64
        assert document_key("firm-a", digest, "x.pdf") != document_key("firm-b", digest, "x.pdf")

    def test_traversal_cannot_escape_the_tenant_prefix(self):
        """An unsanitised filename would place the object outside the tenant prefix."""
        key = document_key(TENANT, "e" * 64, "../../other-firm/secret.pdf")
        assert ".." not in key
        assert key.startswith(f"{PROCESSED_PREFIX}{TENANT}/")
        assert key.count("/") == 3

    def test_filename_is_reduced_not_rejected(self):
        assert safe_filename("Motion to Dismiss (final).pdf") == "Motion_to_Dismiss_final_.pdf"

    def test_empty_filename_still_yields_a_key(self):
        assert safe_filename("///") == "document"


class TestRawKey:
    """The S3 notification hands over a key and nothing else, so the tenant is read
    back out of the key. Anything `parse_raw_key` accepts becomes a tenant id."""

    def test_round_trips(self):
        key = raw_key(TENANT, "upload-1", "motion.pdf")
        assert parse_raw_key(key) == (TENANT, "upload-1", "motion.pdf")

    def test_two_uploads_of_one_filename_do_not_collide(self):
        assert raw_key(TENANT, "a", "x.pdf") != raw_key(TENANT, "b", "x.pdf")

    def test_a_processed_key_is_not_a_raw_key(self):
        """Otherwise the pipeline would re-ingest its own output."""
        assert parse_raw_key(document_key(TENANT, "a" * 64, "x.pdf")) is None

    def test_another_prefix_is_ignored_rather_than_raising(self):
        assert parse_raw_key("some/other/thing.pdf") is None

    def test_traversal_in_the_tenant_segment_is_refused(self):
        """`..` as a tenant would read another firm's prefix, so this must not parse."""
        assert parse_raw_key(f"{RAW_PREFIX}../other-firm/u1/x.pdf") is None

    def test_a_deeper_key_is_refused(self):
        assert parse_raw_key(f"{RAW_PREFIX}{TENANT}/u1/nested/x.pdf") is None

    def test_a_shallower_key_is_refused(self):
        assert parse_raw_key(f"{RAW_PREFIX}{TENANT}/x.pdf") is None

    def test_an_empty_segment_is_refused(self):
        assert parse_raw_key(f"{RAW_PREFIX}{TENANT}//x.pdf") is None

    def test_a_prefix_only_key_is_refused(self):
        assert parse_raw_key(RAW_PREFIX) is None


class TestPutDocument:
    def test_uploads_with_sse_kms_when_configured(self, storage: DocumentStorage, ctx, s3):
        storage.put_document(ctx, filename="motion.pdf", body=PDF)
        put = s3.puts[0]
        assert put["ServerSideEncryption"] == "aws:kms"
        assert put["SSEKMSKeyId"] == "alias/lexgraph-docs"

    def test_omits_encryption_headers_when_no_key_configured(self, s3, ctx):
        """Naming AES256 would silently downgrade a KMS bucket, so nothing is sent."""
        storage = DocumentStorage(BUCKET, s3=s3)
        storage.put_document(ctx, filename="motion.pdf", body=PDF)
        assert "ServerSideEncryption" not in s3.puts[0]

    def test_records_media_type(self, storage: DocumentStorage, ctx):
        doc = storage.put_document(ctx, filename="motion.pdf", body=PDF)
        assert doc.media_type == "application/pdf"

    def test_document_id_is_stable_for_identical_bytes(self, storage: DocumentStorage, ctx):
        first = storage.put_document(ctx, filename="motion.pdf", body=PDF)
        second = storage.put_document(ctx, filename="motion.pdf", body=PDF)
        assert first.document_id == second.document_id

    def test_existing_object_is_not_re_put(self, storage: DocumentStorage, ctx, s3):
        """Object lock forbids deleting the extra version a re-put would create."""
        storage.put_document(ctx, filename="motion.pdf", body=PDF)
        storage.put_document(ctx, filename="motion.pdf", body=PDF)
        assert len(s3.puts) == 1

    def test_walled_matter_refuses_the_upload(self, storage: DocumentStorage, s3):
        walled = AuthContext(
            user_id="associate@firm", tenant_id=TENANT, matter_denylist=frozenset({"M-WALL"})
        )
        with pytest.raises(ScopeViolation):
            storage.put_document(walled, filename="x.pdf", body=PDF, matter_id="M-WALL")
        assert s3.puts == []


class TestPresign:
    def test_mints_a_url_for_a_stored_document(self, storage: DocumentStorage, ctx):
        doc = storage.put_document(ctx, filename="motion.pdf", body=PDF)
        link = storage.presign_download(doc.document_id, ctx=ctx)
        assert "X-Amz-Signature" in link.url
        assert link.expires_at

    def test_default_expiry_is_short(self, storage: DocumentStorage, ctx):
        doc = storage.put_document(ctx, filename="motion.pdf", body=PDF)
        assert storage.presign_download(doc.document_id, ctx=ctx).expires_in == 900

    def test_expiry_is_capped(self, storage: DocumentStorage, ctx):
        """Beyond an hour the URL stops being a click-through and becomes a copy."""
        doc = storage.put_document(ctx, filename="motion.pdf", body=PDF)
        with pytest.raises(ValueError, match="unrevocable"):
            storage.presign_download(doc.document_id, expires_in=MAX_EXPIRY_SECONDS + 1, ctx=ctx)

    def test_zero_expiry_refused(self, storage: DocumentStorage, ctx):
        doc = storage.put_document(ctx, filename="motion.pdf", body=PDF)
        with pytest.raises(ValueError):
            storage.presign_download(doc.document_id, expires_in=0, ctx=ctx)

    def test_page_fragment_is_appended_not_signed(self, storage: DocumentStorage, ctx, s3):
        """A fragment never reaches S3, so it cannot invalidate the signature."""
        doc = storage.put_document(ctx, filename="motion.pdf", body=PDF)
        link = storage.presign_download(doc.document_id, ctx=ctx, page=7)
        assert link.url.endswith("#page=7")
        assert "#page" not in str(s3.presigns[0]["Params"])

    def test_opens_inline_for_the_viewer(self, storage: DocumentStorage, ctx, s3):
        doc = storage.put_document(ctx, filename="motion.pdf", body=PDF)
        storage.presign_download(doc.document_id, ctx=ctx)
        assert s3.presigns[0]["Params"]["ResponseContentDisposition"].startswith("inline")

    def test_unknown_document_raises_rather_than_signing(self, storage: DocumentStorage, ctx, s3):
        with pytest.raises(DocumentNotFound):
            storage.presign_download("doc-nope", ctx=ctx)
        assert s3.presigns == []

    def test_other_tenant_cannot_presign(self, storage: DocumentStorage, ctx, s3):
        doc = storage.put_document(ctx, filename="motion.pdf", body=PDF)
        intruder = AuthContext(user_id="rival@other", tenant_id="other-firm")
        with pytest.raises(ScopeViolation):
            storage.presign_download(doc.document_id, ctx=intruder)
        assert s3.presigns == []


class TestUrlIsNeverPersisted:
    def test_source_locator_has_no_url_field(self):
        """The load-bearing assertion of the whole design.

        A presigned URL on an assertion is an expiring credential in the audit trail:
        it stops resolving, and until it does it is a bearer token in a database.
        """
        loc = SourceLocator(
            document_id="doc-1", filename="motion.pdf", page=3, quote="the court held"
        )
        keys = set(loc.to_dict())
        assert not any("url" in k for k in keys)
        assert not any("presign" in k for k in keys)
        assert not any("expires" in k for k in keys)

    def test_locator_carries_the_id_the_url_is_derived_from(self):
        loc = SourceLocator(document_id="doc-1", page=3, quote="the court held")
        assert loc.to_dict()["document_id"] == "doc-1"

    def test_download_link_is_not_a_stored_shape(self, storage: DocumentStorage, ctx):
        """The link is a return value only; nothing writes it back to the index."""
        doc = storage.put_document(ctx, filename="motion.pdf", body=PDF)
        storage.presign_download(doc.document_id, ctx=ctx)
        stored = storage.describe(doc.document_id).model_dump()
        assert not any("url" in k for k in stored)


class TestExistsAndDelete:
    def test_exists_is_true_after_put(self, storage: DocumentStorage, ctx):
        doc = storage.put_document(ctx, filename="motion.pdf", body=PDF)
        assert storage.document_exists(doc.document_id) is True

    def test_exists_is_false_for_unknown_id(self, storage: DocumentStorage):
        assert storage.document_exists("doc-nope") is False

    def test_delete_removes_object_and_index_entry(self, storage: DocumentStorage, ctx, s3):
        doc = storage.put_document(ctx, filename="motion.pdf", body=PDF)
        assert storage.delete_document(doc.document_id) is True
        assert s3.deletes[0]["Key"] == doc.key
        assert storage.describe(doc.document_id) is None

    def test_delete_of_unknown_document_is_false_not_an_error(self, storage: DocumentStorage):
        assert storage.delete_document("doc-nope") is False

    def test_deleted_document_cannot_be_presigned(self, storage: DocumentStorage, ctx, s3):
        doc = storage.put_document(ctx, filename="motion.pdf", body=PDF)
        storage.delete_document(doc.document_id)
        with pytest.raises(DocumentNotFound):
            storage.presign_download(doc.document_id, ctx=ctx)
        assert s3.presigns == []


class TestConfigWiring:
    def test_no_bucket_means_no_storage(self):
        cfg = LexGraphConfig(documents=DocumentConfig(bucket=""))
        assert storage_from_config(cfg) is None

    def test_bucket_yields_storage_without_touching_aws(self):
        """boto3 is reached lazily, so constructing this needs no credentials."""
        cfg = LexGraphConfig(documents=DocumentConfig(bucket=BUCKET, kms_key_id="alias/k"))
        got = storage_from_config(cfg)
        assert got is not None
        assert got.bucket == BUCKET
        assert got.kms_key_id == "alias/k"

    def test_override_wins(self, storage: DocumentStorage):
        set_document_storage(storage)
        cfg = LexGraphConfig(documents=DocumentConfig(bucket=""))
        assert storage_from_config(cfg) is storage


class TestTheSigningVersion:
    """A KMS bucket refuses a SigV2 presigned POST, and boto3's default produces one.

    `boto3.client("s3")` reports `signature_version` as `s3v4`, but
    `generate_presigned_post` builds its policy from the *default* config and emits
    `AWSAccessKeyId`/`signature` unless the version is set explicitly. S3 then rejects the
    upload with "Requests specifying Server Side Encryption with AWS KMS managed keys
    require AWS Signature Version 4" -- a 400 that only ever appears in the browser, since
    nothing server-side signs or sends the form.
    """

    def test_the_client_is_built_with_sigv4(self):
        storage = DocumentStorage("bucket", kms_key_id="alias/x")
        assert storage.s3.meta.config.signature_version == "s3v4"

    def test_a_presigned_post_carries_v4_fields(self, ctx):
        """Asserted on the wire format rather than on the config, because the config was
        already right and the emitted policy was still v2. No fake S3 here on purpose: a
        stub cannot show which policy botocore builds."""
        storage = DocumentStorage("bucket", kms_key_id="alias/x")
        ticket = storage.presign_upload(ctx, filename="a.pdf", max_bytes=1024)

        assert ticket.fields["x-amz-algorithm"] == "AWS4-HMAC-SHA256"
        assert "x-amz-credential" in ticket.fields
        assert "AWSAccessKeyId" not in ticket.fields


class TestAnIndexMissFallsBackToTheBucket:
    """The index is per-process, so a deploy empties it while S3 keeps every byte.

    Before this, a document uploaded before the last deploy reported "The file is no longer
    in storage" in the UI, with the object sitting untouched in the bucket. S3 is the source
    of truth, so a cache miss must not read as absence.
    """

    def test_a_document_missing_from_the_index_is_found_in_s3(self, s3, ctx):
        writer = DocumentStorage(BUCKET, s3=s3)
        doc = writer.put_document(ctx, filename="motion.pdf", body=PDF)

        # A fresh process: same bucket, same objects, empty index.
        restarted = DocumentStorage(BUCKET, s3=s3)
        assert restarted.describe(doc.document_id) is None

        found = restarted.describe(doc.document_id, tenant_id=TENANT)
        assert found is not None
        assert found.document_id == doc.document_id
        assert found.key == doc.key
        assert found.filename == "motion.pdf"

    def test_the_listing_is_scoped_to_one_tenant(self, s3, ctx):
        """A miss must not walk another firm's prefix, whatever id is passed."""
        restarted = DocumentStorage(BUCKET, s3=s3)
        restarted.describe("doc-whatever", tenant_id=TENANT)
        assert all(c["Prefix"] == f"processed/{TENANT}/" for c in s3.lists)

    def test_the_result_is_cached_so_a_burst_costs_one_listing(self, s3, ctx):
        writer = DocumentStorage(BUCKET, s3=s3)
        doc = writer.put_document(ctx, filename="motion.pdf", body=PDF)

        restarted = DocumentStorage(BUCKET, s3=s3)
        restarted.describe(doc.document_id, tenant_id=TENANT)
        before = len(s3.lists)
        restarted.describe(doc.document_id, tenant_id=TENANT)
        assert len(s3.lists) == before

    def test_an_unknown_document_is_still_absent(self, s3):
        """The fallback must not invent a document. A wrong id stays a miss."""
        restarted = DocumentStorage(BUCKET, s3=s3)
        assert restarted.describe("doc-nonexistent", tenant_id=TENANT) is None

    def test_no_tenant_means_no_listing(self, s3):
        """Callers that cannot name a tenant get the old behaviour rather than a scan of
        the bucket."""
        restarted = DocumentStorage(BUCKET, s3=s3)
        assert restarted.describe("doc-x") is None
        assert s3.lists == []
