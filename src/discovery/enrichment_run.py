"""One enrichment run: a Bedrock call per table, staged as it goes.

**Staged per table rather than once at the end.** A background task dies with its container, and
this is the same posture `documents/runner.py` takes: state is written before the next unit of work
is attempted, so a death mid-run loses only the tables not yet reached. `existing_descriptions` then
suppresses re-guessing on the next run, which makes re-running the retry mechanism.

**Nodes before assertions, and no staging if the nodes failed.** The description text lives only on
the `:Description` node. An edge staged without it points at an empty node and the overlay returns
a blank description for good, with no way back except paying for the model again.

Status is held in process, like `Services.blocked_queries`. It dies with the container, which is the
same limit `BackgroundTasks` already has and is better than a durable record stuck at RUNNING with
no worker behind it. The status endpoint says so.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from src.discovery.enrichment import enrich_tables
from src.discovery.glue_scanner import column_node_id, table_node_id
from src.graph.scope import AuthContext
from src.metrics.models import TableSchema

logger = logging.getLogger(__name__)

#: Tables per run. Each is one Bedrock call and roughly 1 + N column + up to 14 term assertions, so
#: an uncapped run over a large catalog would push a tenant past `GraphAssertionStore.DEFAULT_LIMIT`
#: and silently truncate the review queue it is meant to fill.
MAX_TABLES_PER_RUN = 40

STATE_RUNNING = "running"
STATE_DONE = "done"
STATE_FAILED = "failed"


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class EnrichmentRun:
    """What one run did, readable while it is still going."""

    tenant_id: str
    source_id: str
    tables_total: int
    state: str = STATE_RUNNING
    tables_done: int = 0
    staged: int = 0
    nodes_written: int = 0
    errors: list[str] = field(default_factory=list)
    started_at: str = field(default_factory=_now)
    finished_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "source_id": self.source_id,
            "tables_total": self.tables_total,
            "tables_done": self.tables_done,
            "staged": self.staged,
            "nodes_written": self.nodes_written,
            "errors": self.errors,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "note": (
                "Descriptions are proposed, not applied. Each one waits for review, and only an "
                "approved description reaches the query planner. Progress is held in memory, so a "
                "deploy mid-run loses the progress rather than the work already staged."
            ),
        }


def existing_descriptions(tables: Sequence[Any], source_id: str) -> dict[str, str]:
    """What is already described, keyed as `enrich_tables` expects.

    Built from the **overlaid** tables, so an approved model description or a person's edit is not
    re-guessed. That is what makes "a model never overwrites a human's text" hold across runs
    rather than only within one.
    """
    have: dict[str, str] = {}
    for table in tables:
        if table.description:
            have[table.full_name] = table.description
        for column in table.columns:
            if column.description:
                have[f"{table.full_name}.{column.name}"] = column.description
    return have


def schemas_of(tables: Sequence[Any]) -> list[TableSchema]:
    """`CatalogTable` as the compiler's narrower view, which is what `enrich_tables` takes."""
    return [
        TableSchema(
            full_name=t.full_name,
            columns={c.name: c.data_type for c in t.columns},
            primary_keys=frozenset(c.name for c in t.columns if c.is_primary_key),
        )
        for t in tables
    ]


def run_enrichment(
    services: Any,
    ctx: AuthContext,
    *,
    source_id: str,
    only: Sequence[str] = (),
    bedrock_client: Any = None,
) -> EnrichmentRun:
    """Propose descriptions for a tenant's tables. Nothing goes live.

    `only` limits the run to named tables, which is how a person enriches one table from its own
    page without paying for the whole catalog.
    """
    settings = services.settings_for(ctx.tenant_id)
    model_id = settings.enrichment_model
    catalog = services.enriched_catalog()
    tables = [t for t in catalog.tables(ctx.tenant_id) if not only or t.full_name in set(only)]
    tables = tables[:MAX_TABLES_PER_RUN]

    run = EnrichmentRun(tenant_id=ctx.tenant_id, source_id=source_id, tables_total=len(tables))
    services.enrichment_runs[ctx.tenant_id] = run

    if not model_id:
        run.state = STATE_FAILED
        run.errors.append("no enrichment model is configured for this tenant")
        run.finished_at = _now()
        return run

    client = bedrock_client if bedrock_client is not None else _bedrock(services)
    if client is None:
        run.state = STATE_FAILED
        run.errors.append("Bedrock is unreachable, so nothing could be proposed")
        run.finished_at = _now()
        return run

    store = services.catalog_graph_store()
    have = existing_descriptions(tables, source_id)

    for table, schema in zip(tables, schemas_of(tables), strict=True):
        try:
            result = enrich_tables(
                client,
                [schema],
                tenant_id=ctx.tenant_id,
                source_id=table.source_id or source_id,
                model_id=model_id,
                domain=settings.ontology_domain or services.ontology.domain,
                existing_descriptions=have,
            )
            run.errors.extend(result.errors)
            if result.assertions:
                # Nodes first, and abandon the table if they do not land. See the module header.
                if store is not None:
                    run.nodes_written += store.persist(result.nodes)
                job_id = f"enrich-{table.full_name}"
                run.staged += len(
                    services.review_queue.stage(ctx, result.assertions, job_id=job_id)
                )
        except Exception as e:  # noqa: BLE001
            # Per table, so one failure does not abandon the rest of the run.
            run.errors.append(f"{table.full_name}: {e}")
            logger.warning("enrichment failed for %s: %s", table.full_name, e)
        run.tables_done += 1

    run.state = STATE_DONE
    run.finished_at = _now()
    logger.info(
        "enrichment for %s staged %d claims over %d tables, %d errors",
        ctx.tenant_id,
        run.staged,
        run.tables_done,
        len(run.errors),
    )
    return run


def _bedrock(services: Any) -> Any | None:
    try:
        import boto3

        return boto3.client("bedrock-runtime", region_name=services.config.models.region)
    except Exception as e:  # noqa: BLE001
        logger.warning("could not build a Bedrock client for enrichment: %s", e)
        return None


def pending_for_table(services: Any, ctx: AuthContext, full_name: str) -> list[str]:
    """Assertion ids of this table's unreviewed enrichment claims.

    Matched on the locator rather than on the subject id, because a column claim's subject is the
    column node while the locator names the table on both.
    """
    from src.discovery.enrichment import CONCERNS_TOPIC, DESCRIBED_AS, HAS_SYNONYM

    predicates = {DESCRIBED_AS, HAS_SYNONYM, CONCERNS_TOPIC}
    items = services.review_queue.list_pending(ctx, limit=2000)
    return [i.assertion_id for i in items if i.predicate in predicates and i.table == full_name]


def subject_ids_for(source_id: str, full_name: str, column: str | None) -> str:
    """The node id a description attaches to. Built, never parsed."""
    if column:
        return column_node_id(source_id, full_name, column)
    return table_node_id(source_id, full_name)
