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
class ResetScope:
    """What a reset is allowed to remove.

    Everything defaults to on except `metrics`, and that asymmetry is the point. Documents
    come back from S3 and schemas come back from Glue, so removing them is reversible.
    A metric definition is authored work with no upstream source: nothing can rebuild it, so
    including it in a "rebuild from source" operation is destruction wearing a rebuild's
    clothes.
    """

    graph: bool = True
    vectors: bool = True
    jobs: bool = True
    catalog: bool = True
    routing: bool = True
    """The tier router's index. Derived from the metrics, tables and entities being dropped
    here, so leaving it would route questions toward layers whose contents just went."""

    metrics: bool = False

    @property
    def destroys_unrecoverable_work(self) -> bool:
        return self.metrics


@dataclass
class ResetReport:
    assertions_dropped: int = 0
    documents_forgotten: int = 0
    vectors_dropped: int = 0
    jobs_dropped: int = 0
    tables_forgotten: int = 0
    routing_dropped: int = 0
    routing_reindexed: int = 0
    """Routing descriptions rebuilt for the metrics that survived. A preserved metric whose
    description was dropped is still approved and still answers a question that reaches tier 1
    -- but the router no longer knows it exists, so fewer questions reach tier 1 at all. The
    reset would silently narrow governance while reporting the metric as preserved."""

    metrics_dropped: int = 0
    metrics_preserved: int = 0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        rebuildable = (
            "Every document is still in S3 and every schema is still in Glue, so Replay and "
            "a catalog scan reconstruct what was just removed."
        )
        lost = (
            f"{self.metrics_dropped} metric definitions were deleted and CANNOT be "
            "reconstructed: they were authored here, not derived from S3 or Glue. "
        )
        return {
            "assertions_dropped": self.assertions_dropped,
            "documents_forgotten": self.documents_forgotten,
            "vectors_dropped": self.vectors_dropped,
            "jobs_dropped": self.jobs_dropped,
            "tables_forgotten": self.tables_forgotten,
            "routing_dropped": self.routing_dropped,
            "routing_reindexed": self.routing_reindexed,
            "metrics_dropped": self.metrics_dropped,
            "metrics_preserved": self.metrics_preserved,
            "errors": self.errors,
            "s3_preserved": True,
            "note": (lost + rebuildable) if self.metrics_dropped else rebuildable,
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


@dataclass
class EndpointSweepReport:
    """Live facts whose endpoints their pack does not declare, and what was done about them."""

    checked: int = 0
    offenders: list[dict[str, Any]] = field(default_factory=list)
    retracted: list[str] = field(default_factory=list)
    cascaded: list[str] = field(default_factory=list)
    dry_run: bool = True
    errors: list[str] = field(default_factory=list)

    @property
    def blocking(self) -> int:
        """Offenders that veto an answer. These are the ones doing active harm."""
        return sum(1 for o in self.offenders if o.get("blocks"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "checked": self.checked,
            "offenders": self.offenders,
            "blocking": self.blocking,
            "retracted": self.retracted,
            "cascaded": self.cascaded,
            "dry_run": self.dry_run,
            "errors": self.errors,
            "note": (
                "Nothing was written. Each fact listed here uses a predicate against endpoint "
                "kinds its pack does not declare."
                if self.dry_run
                else "Retracted, not deleted: `superseded_at` is set, so an as-of read before now "
                "still shows what the graph asserted while advice rested on it."
            ),
        }


def sweep_undeclared_endpoints(
    services: Any, ctx: AuthContext, *, dry_run: bool = True
) -> EndpointSweepReport:
    """Find live facts that violate their predicate's declared domain or range, and withdraw them.

    `build_assertion` refuses these at write time now, but that is prospective: the graph already
    holds facts written before the check existed. One of them was a `POTENTIAL_CONFLICT` stored
    `Party -> Party` against its declared `Matter -> Party`, and because that predicate declares
    `blocks: both` it withheld the firm's own client's file — a question about that client's own
    counsel returned nothing.

    Deliberately **not** a read-time filter in `blocking_facts`. Skipping an invalid veto at read
    time would make the wall silently ignore a recorded refusal, which is verbatim the failure
    `BlockCheckUnavailable` exists to prevent, and it would put a second definition of validity
    somewhere it can drift from the first.

    Deliberately **not** reset-and-replay either. That works, but it costs a full re-extraction and
    discards every review decision in the tenant rather than the handful that are wrong.

    `dry_run` defaults to True, following `retract` and the merge endpoint: this withdraws facts a
    firm may have relied on, so seeing the list is the default and writing is the request.
    """
    report = EndpointSweepReport(dry_run=dry_run)
    ontology = getattr(services, "ontology", None)
    queue = getattr(services, "review_queue", None)
    if ontology is None or queue is None:
        report.errors.append("no ontology pack or review queue is wired, so nothing can be checked")
        return report

    from src.documents.retract import retract

    for record in queue.live_assertions(ctx):
        a = record.assertion
        report.checked += 1
        kinds = ontology.endpoint_kinds(a.predicate)
        if kinds is None:
            continue
        subject_kind = ontology.entity_kind_of(a.subject_id)
        object_kind = ontology.entity_kind_of(a.object_id)
        if subject_kind is None or object_kind is None:
            # No prefix to judge. `build_assertion` allows these too -- see `_reject_unnormalised`.
            continue
        if kinds.admits(subject_kind, object_kind):
            continue

        report.offenders.append(
            {
                "assertion_id": a.assertion_id,
                "predicate": a.predicate,
                "subject_id": a.subject_id,
                "object_id": a.object_id,
                "written": f"{subject_kind} -> {object_kind}",
                "declared": f"{sorted(kinds.subject)} -> {sorted(kinds.object)}",
                # Named because it decides urgency: an offender that vetoes answers is withholding
                # evidence right now, while one that merely informs is a correctness problem.
                "blocks": bool(ontology.blocked_endpoints(a.predicate)),
                "review_state": a.review_state.value,
            }
        )
        if dry_run:
            continue
        try:
            result = retract(
                queue,
                ctx,
                a.assertion_id,
                reason=(
                    f"{a.predicate} is declared {sorted(kinds.subject)} -> "
                    f"{sorted(kinds.object)} and was written {subject_kind} -> {object_kind}"
                ),
            )
        except Exception as e:  # noqa: BLE001
            # One bad retraction must not abandon the rest: the remaining offenders are still
            # withholding evidence.
            report.errors.append(f"{a.assertion_id}: {e}")
            continue
        report.retracted.append(a.assertion_id)
        report.cascaded.extend(result.cascaded)

    logger.info(
        "endpoint sweep for %s: %d checked, %d invalid (%d blocking), %d retracted",
        ctx.tenant_id,
        report.checked,
        len(report.offenders),
        report.blocking,
        len(report.retracted),
    )
    return report


def reset_derived(services: Any, ctx: AuthContext, scope: ResetScope | None = None) -> ResetReport:
    """Drop the derived tiers for one tenant. S3 is untouched.

    `scope` selects what goes. Metrics are excluded by default because they are the one thing
    in the graph with no upstream source to rebuild from.

    Tenant-scoped throughout: a reset in one firm must not empty another's graph, which is
    why nothing here is a bare "delete all".
    """
    scope = scope or ResetScope()
    report = ResetReport()

    # Read the metrics before anything is dropped, so they can be restored afterwards. The
    # graph wipe cannot distinguish a Metric node from an assertion, so preserving them means
    # holding them in memory across the delete rather than filtering the delete itself.
    preserved: list[Any] = []
    _count_metrics_before_reset = _count_metrics(services, ctx) if scope.metrics else 0
    if not scope.metrics and services.graph is not None:
        try:
            from src.metrics.graph_store import GraphMetricStore

            store = GraphMetricStore(services.graph)
            preserved = [
                (m, store.status_of(ctx.tenant_id, m.metric_id) or "draft")
                for m in store.list_metrics(ctx.tenant_id)
            ]
        except Exception as e:
            report.errors.append(f"could not read metrics to preserve them: {e}")

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

    if scope.graph and services.graph is not None:
        try:
            # The one place outside src/graph/ that would need Cypher, so it is delegated
            # rather than written here, see the working agreement.
            from src.graph.schema import drop_tenant_data

            drop_tenant_data(services.graph, ctx.tenant_id)
        except Exception as e:
            report.errors.append(f"graph: {e}")

    # Written back after the wipe. The graph delete cannot distinguish a Metric node from an
    # assertion, so preserving them means holding them across the delete rather than
    # filtering it. A restored definition keeps its status, so an approved metric is still
    # answering the moment the reset finishes.
    if preserved and services.graph is not None:
        try:
            from src.metrics.graph_store import GraphMetricStore

            metric_store = GraphMetricStore(services.graph)
            for metric, metric_status in preserved:
                metric_store.save_metric(
                    ctx.tenant_id, metric, updated_by="reset", status=metric_status
                )
            report.metrics_preserved = len(preserved)
        except Exception as e:
            report.errors.append(f"could not restore preserved metrics: {e}")
    elif scope.metrics:
        report.metrics_dropped = _count_metrics_before_reset

    if scope.vectors and services.embedder is not None:
        try:
            report.vectors_dropped = services.embedder.drop_tenant(ctx.tenant_id)
        except Exception as e:
            report.errors.append(f"vectors: {e}")

    if scope.routing and getattr(services, "router_indexer", None) is not None:
        try:
            report.routing_dropped = services.router_indexer.drop_tenant(ctx.tenant_id)
        except Exception as e:
            report.errors.append(f"routing index: {e}")
        # Rebuilt for whatever survived, in the same operation. Entities and tables are gone and
        # come back with Replay and a scan, but a preserved metric is here *now* -- and a metric
        # that is approved with no routing description is one the router cannot steer toward, so
        # the reset would narrow governance while reporting the metric as preserved.
        _reindex_surviving_metrics(services, ctx, report)

    if scope.jobs:
        try:
            report.jobs_dropped = services.job_store.drop_tenant(ctx.tenant_id)
        except Exception as e:
            report.errors.append(f"jobs: {e}")

    if scope.catalog:
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


def _reindex_surviving_metrics(services: Any, ctx: AuthContext, report: ResetReport) -> None:
    """Re-describe the metrics that outlived the reset, so the router can still steer to them.

    Reported rather than raised: the reset has already happened, and failing here would leave
    the caller unsure what was removed. A metric with no description still answers when a
    question reaches tier 1 -- it is the reaching that degrades, which is why this is loud.
    """
    indexer = getattr(services, "router_indexer", None)
    if indexer is None or services.graph is None:
        return

    from src.metrics.graph_store import GraphMetricStore

    # The indexer is built before the graph connects, so its metric store may be unset -- the
    # same reason `build_router_indexer` attaches one per request.
    if getattr(indexer, "metric_store", None) is None:
        indexer.metric_store = GraphMetricStore(services.graph)
    if getattr(indexer, "graph", None) is None:
        indexer.graph = services.graph

    try:
        report.routing_reindexed = indexer.reindex_metrics(ctx)
    except Exception as e:  # noqa: BLE001
        report.errors.append(
            f"metrics survived but the router cannot steer to them until Rebuild is run: {e}"
        )
        logger.warning("routing reindex after reset failed for %s: %s", ctx.tenant_id, e)


def _count_metrics(services: Any, ctx: AuthContext) -> int:
    if services.graph is None:
        return 0
    try:
        from src.metrics.graph_store import GraphMetricStore

        return len(GraphMetricStore(services.graph).list_metrics(ctx.tenant_id))
    except Exception:
        return 0


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
