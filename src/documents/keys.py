"""S3 key layout for documents. One definition, imported by both storage and ingest.

This exists as its own module because `document_key` was previously defined twice —
in `storage.py` and `ingest.py` — and the two had already drifted into producing
different keys for the same document. A key is a contract between the uploader, the
S3 notification filter, the IAM prefix boundary and every stored `SourceLocator`, so
there can only be one of it.

Two prefixes, and the split is load-bearing:

`raw/` is where a presigned upload lands. The client cannot know the digest the server
will compute, so this key is *not* content-addressed.

`processed/` is where the document lives once hashed, content-addressed so a
double-submit is idempotent. A separate prefix also stops the notification observing
its own output and re-triggering itself.
"""

from __future__ import annotations

import re

#: Must match `RAW_PREFIX` in cdk/lib/config.ts, which declares the S3 notification
#: filter. If the two disagree, uploads succeed and nothing ever processes them.
RAW_PREFIX = "raw/"

PROCESSED_PREFIX = "processed/"

_UNSAFE_FILENAME = re.compile(r"[^A-Za-z0-9._-]+")
_MAX_FILENAME_CHARS = 120


def safe_filename(filename: str) -> str:
    """Reduce a filename to something that cannot escape its key prefix.

    Directory separators and `..` are stripped rather than escaped. S3 keys are opaque
    strings, but an unsanitised `../../other-tenant/x.pdf` would place the object
    outside the tenant prefix that the IAM boundary relies on.
    """
    base = filename.replace("\\", "/").rsplit("/", 1)[-1]
    cleaned = _UNSAFE_FILENAME.sub("_", base).strip("._")
    return cleaned[:_MAX_FILENAME_CHARS] or "document"


def tenant_prefixes(tenant_id: str) -> tuple[str, ...]:
    """Every key prefix one tenant's bytes can live under.

    Built from the same constants the writers use, so a third prefix added later cannot be
    missed by a sweep that hardcoded two. The trailing slash matters: without it `raw/demo`
    would also match `raw/demo-clinic/`.
    """
    return (f"{RAW_PREFIX}{tenant_id}/", f"{PROCESSED_PREFIX}{tenant_id}/")


def document_key(tenant_id: str, content_sha256: str, filename: str) -> str:
    """Content-addressed key under a tenant prefix.

    Matter is deliberately absent: matter assignment changes, and a document should not
    need copying when it does.
    """
    return f"{PROCESSED_PREFIX}{tenant_id}/{content_sha256}/{safe_filename(filename)}"


def raw_key(tenant_id: str, upload_id: str, filename: str) -> str:
    """Where a presigned upload lands, before anything has read the bytes.

    `upload_id` keeps two people uploading the same filename from overwriting each
    other while both are still in flight.
    """
    return f"{RAW_PREFIX}{tenant_id}/{upload_id}/{safe_filename(filename)}"


def parse_raw_key(key: str) -> tuple[str, str, str] | None:
    """Recover (tenant_id, upload_id, filename) from a raw key, or None if it is not one.

    The S3 notification hands us a key and nothing else, so the tenant has to come from
    the key itself. Returning None rather than raising is deliberate: an event for
    another prefix is not an error, it is simply not ours.
    """
    if not key.startswith(RAW_PREFIX):
        return None
    parts = key[len(RAW_PREFIX) :].split("/")
    if len(parts) != 3 or not all(parts):
        return None
    tenant_id, upload_id, filename = parts
    # The prefix is the tenant boundary the IAM policy leans on, so a traversal segment
    # here would be a cross-tenant read. Reject rather than sanitise: a key we did not
    # mint is not one we should guess the intent of.
    if tenant_id != safe_filename(tenant_id) or upload_id != safe_filename(upload_id):
        return None
    if filename != safe_filename(filename):
        return None
    return tenant_id, upload_id, filename


def parse_document_key(key: str) -> tuple[str, str, str] | None:
    """Recover (tenant_id, content_sha256, filename) from a processed key, or None.

    The counterpart to `document_key`. It exists because the key is the only durable record
    of where a document lives: `document_id` is a hash of tenant and content, so it cannot be
    reversed, but it *can* be recomputed from what this returns.
    """
    if not key.startswith(PROCESSED_PREFIX):
        return None
    parts = key[len(PROCESSED_PREFIX) :].split("/")
    if len(parts) != 3 or not all(parts):
        return None
    tenant_id, content_sha256, filename = parts
    # Same reasoning as parse_raw_key: the prefix is the tenant boundary IAM relies on, so a
    # traversal segment would be a cross-tenant read. Reject rather than sanitise.
    if tenant_id != safe_filename(tenant_id) or content_sha256 != safe_filename(content_sha256):
        return None
    if filename != safe_filename(filename):
        return None
    return tenant_id, content_sha256, filename
