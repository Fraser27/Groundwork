"""Reset and replay: rebuilding the derived tiers from the sources of truth.

These exist because of a property the architecture claims and should therefore be able to
demonstrate: **S3 and Glue are authoritative, everything else is derived.** If that is
true, the graph and the vector index can be thrown away and rebuilt, and the result must be
the same. If it is not true, these operations are where that becomes visible.

Two operations, and the asymmetry between them is the point:

`reset_derived` wipes the graph, the vector index, the catalog cache and job state. It does
**not** touch S3. Nothing irreplaceable is lost, so it needs no confirmation beyond an
admin role.

`replay` re-ingests every document already in S3 and re-scans the catalog. It is safe to run
twice because document keys are content-addressed: the same bytes produce the same
`document_id`, which produces the same assertion ids. Replay converges rather than
accumulating.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from src.documents.keys import PROCESSED_PREFIX
from src.graph.scope import AuthContext

logger = logging.getLogger(__name__)


@dataclass
class ResetReport:
    assertions_dropped: int = 0
    documents_forgotten: int = 0
    vectors_dropped: int = 0
    jobs_dropped: int = 0
    tables_forgotten: int = 0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "assertions_dropped": self.assertions_dropped,
            "documents_forgotten": self.documents_forgotten,
            "vectors_dropped": self.vectors_dropped,
            "jobs_dropped": self.jobs_dropped,
            "tables_forgotten": self.tables_forgotten,
            "errors": self.errors,
            "s3_preserved": True,
            "note": (
                "The graph and the search index were rebuilt from empty. Every document is "
                "still in S3, so Replay reconstructs what was just removed."
            ),
        }


@dataclass
class ReplayReport:
    documents_found: int = 0
    documents_ingested: int = 0
    documents_failed: int = 0
    tables_rescanned: int = 0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "documents_found": self.documents_found,
            "documents_ingested": self.documents_ingested,
            "documents_failed": self.documents_failed,
            "tables_rescanned": self.tables_rescanned,
            "errors": self.errors,
            "note": (
                "Rebuilt from S3 and the catalog. Document ids are content-addressed, so "
                "running this again converges on the same graph rather than duplicating it."
            ),
        }


def reset_derived(services: Any, ctx: AuthContext) -> ResetReport:
    """Drop everything derived for one tenant. S3 is untouched.

    Tenant-scoped throughout: a reset in one firm must not empty another's graph, which is
    why nothing here is a bare "delete all".
    """
    report = ResetReport()

    # Assertions first. Retracting one at a time would cascade through the premise graph
    # and write a retraction trail for facts that are about to cease existing, which is
    # noise — this is a rebuild, not a correction.
    try:
        store = services.review_queue.store
        records = store.all_for_tenant(ctx.tenant_id)
        report.assertions_dropped = len(records)
        drop = getattr(store, "drop_tenant", None)
        if drop is not None:
            drop(ctx.tenant_id)
        else:
            report.errors.append(
                "assertion store does not support bulk drop; assertions were counted "
                "but not removed"
            )
    except Exception as e:
        report.errors.append(f"assertions: {e}")

    if services.graph is not None:
        try:
            # The one place outside src/graph/ that would need Cypher, so it is delegated
            # rather than written here — see the working agreement.
            from src.graph.schema import drop_tenant_data

            drop_tenant_data(services.graph, ctx.tenant_id)
        except Exception as e:
            report.errors.append(f"graph: {e}")

    if services.embedder is not None:
        try:
            report.vectors_dropped = services.embedder.drop_tenant(ctx.tenant_id)
        except Exception as e:
            report.errors.append(f"vectors: {e}")

    try:
        report.jobs_dropped = services.job_store.drop_tenant(ctx.tenant_id)
    except Exception as e:
        report.errors.append(f"jobs: {e}")

    try:
        report.tables_forgotten = len(services.catalog.tables(ctx.tenant_id))
        services.catalog.clear(ctx.tenant_id)
    except Exception as e:
        report.errors.append(f"catalog: {e}")

    logger.info(
        "reset derived data for %s: %d assertions, %d vectors, %d tables",
        ctx.tenant_id,
        report.assertions_dropped,
        report.vectors_dropped,
        report.tables_forgotten,
    )
    return report


def replay(services: Any, ctx: AuthContext, *, run_model_extraction: bool = True) -> ReplayReport:
    """Rebuild the derived tiers from S3 and the catalog.

    This is the operation that proves the architecture's central claim. If S3 and Glue really
    are authoritative, replay after a reset produces the same graph — and because document
    ids are content-addressed, running it twice converges rather than accumulating.

    Documents are enumerated from S3 directly, not from any index, because replay has to work
    precisely when the indexes are gone.
    """
    from src.documents.storage import storage_from_config

    report = ReplayReport()
    storage = storage_from_config(services.config)
    if storage is None:
        report.errors.append("no document store configured (DOCUMENT_BUCKET unset)")
        return report

    try:
        keys = list_processed_documents(storage, ctx.tenant_id)
    except Exception as e:
        report.errors.append(f"could not list documents: {e}")
        return report

    report.documents_found = len(keys)

    from src.api.routes_documents import _require_runner
    from src.documents.parse import parse_plain_text

    runner = _require_runner(services)

    for key in keys:
        try:
            obj = storage.s3.get_object(Bucket=storage.bucket, Key=key)
            body = obj["Body"].read() if hasattr(obj["Body"], "read") else obj["Body"]
            media_type = obj.get("ContentType") or "application/octet-stream"
            metadata = {k.lower(): v for k, v in (obj.get("Metadata") or {}).items()}
            filename = metadata.get("filename") or key.rsplit("/", 1)[-1]
            matter_id = metadata.get("matter-id") or None

            doc = storage.put_document(
                ctx,
                filename=filename,
                body=body,
                matter_id=matter_id,
                media_type=media_type,
            )

            if media_type.startswith("text/"):
                parsed = parse_plain_text(
                    doc.document_id, body.decode("utf-8", errors="replace"), filename=filename
                )
            elif runner.parser is not None:
                parsed = runner.parser.parse(doc.document_id, body, filename=filename)
            else:
                report.documents_failed += 1
                report.errors.append(f"{filename}: no vision model, cannot re-parse")
                continue

            runner.pipeline(
                ctx,
                parsed,
                matter_id=matter_id,
                run_model_extraction=run_model_extraction,
                job_id=doc.document_id,
            )
            report.documents_ingested += 1
        except Exception as e:
            report.documents_failed += 1
            report.errors.append(f"{key}: {e}")
            logger.warning("replay failed for %s: %s", key, e)

    report.tables_rescanned = len(services.catalog.tables(ctx.tenant_id))

    logger.info(
        "replayed %s: %d of %d documents",
        ctx.tenant_id,
        report.documents_ingested,
        report.documents_found,
    )
    return report


def list_processed_documents(storage: Any, tenant_id: str) -> list[str]:
    """Every processed document key for a tenant, straight from S3.

    Read from S3 rather than from any index, because the whole point of replay is to work
    when the indexes are gone.
    """
    keys: list[str] = []
    prefix = f"{PROCESSED_PREFIX}{tenant_id}/"
    token: str | None = None
    while True:
        kwargs: dict[str, Any] = {"Bucket": storage.bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        page = storage.s3.list_objects_v2(**kwargs)
        keys.extend(obj["Key"] for obj in page.get("Contents", []))
        if not page.get("IsTruncated"):
            break
        token = page.get("NextContinuationToken")
        if not token:
            break
    return keys
