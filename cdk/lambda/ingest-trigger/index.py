"""Turns an S3 upload notification into one call to the API, then gets out of the way.

Deliberately thin. It does not read the document, hash it, or decide anything about it —
all of that lives in `src/documents/runner.py`, in the same image as the API, so there is
one implementation of ingestion rather than one here and one there.

It also does not wait. The API answers 202 and does the work in a background task, so
this returns in milliseconds. Blocking would put the whole ingest under the ALB's 60s
idle timeout, which is the exact failure the async rewrite exists to remove.

Retries are S3's: a failure here raises, and the notification is redelivered. That is
why the API's ingest is idempotent on content — a redelivered event costs a re-hash, not
a duplicate document.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request

logger = logging.getLogger()
logger.setLevel(logging.INFO)

API_BASE_URL = os.environ["API_BASE_URL"].rstrip("/")
INTERNAL_SECRET = os.environ["INTERNAL_API_SECRET"]
RAW_PREFIX = os.environ.get("RAW_PREFIX", "raw/")

#: Long enough to survive a cold API task, short enough to fail before Lambda's own
#: timeout so the error says "timed out calling the API" rather than being killed.
REQUEST_TIMEOUT_SECONDS = 10


def _trigger(key: str) -> int:
    request = urllib.request.Request(
        f"{API_BASE_URL}/api/internal/ingest",
        data=json.dumps({"key": key}).encode(),
        headers={
            "Content-Type": "application/json",
            "X-Lexgraph-Internal": INTERNAL_SECRET,
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
        return response.status


def handler(event: dict, context: object) -> dict:
    """One S3 event may carry several records, and each is independent."""
    triggered = 0
    skipped = 0

    for record in event.get("Records", []):
        raw_key = record.get("s3", {}).get("object", {}).get("key", "")
        # S3 form-encodes the key in the event, so a filename with a space arrives as
        # `a+b.pdf`. Unquoting with `unquote_plus` is required or the API is handed a key
        # that does not exist.
        key = urllib.parse.unquote_plus(raw_key)

        if not key.startswith(RAW_PREFIX):
            # The notification filter should prevent this; if it is ever misconfigured,
            # skipping beats handing the API a processed key and re-ingesting our own
            # output in a loop.
            logger.warning("skipping %s: outside %s", key, RAW_PREFIX)
            skipped += 1
            continue

        try:
            status = _trigger(key)
            logger.info("triggered ingest for %s (%s)", key, status)
            triggered += 1
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")[:500]
            if e.code in (400, 422):
                # A key the API will never accept. Raising would retry it until the
                # event expired, so this is logged loudly and dropped.
                logger.error("API rejected %s: %s %s", key, e.code, body)
                skipped += 1
                continue
            logger.error("API error for %s: %s %s", key, e.code, body)
            raise
        except Exception:
            # Raise so S3 redelivers. Ingestion is idempotent on content, so a duplicate
            # delivery costs a re-hash rather than a duplicate document.
            logger.exception("could not trigger ingest for %s", key)
            raise

    return {"triggered": triggered, "skipped": skipped}
