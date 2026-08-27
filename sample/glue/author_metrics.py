"""Author governed metrics against the Iceberg demo tables, and approve what runs.

Tier 1 reads `list_metrics(approved_only=True)`, so a draft answers nothing. The shipped pack in
`sample/metrics/legal.yaml` is examples for a fictional firm pointing at `legal_ops.*` tables that do not
exist, and it seeds as drafts deliberately -- so until now the tenant had no metric that could
answer anything, and the Athena wiring had nothing to prove itself with.

Each metric is compiled, then run, and only approved if it returned a figure. Approving first and
checking later would put a metric into service on the strength of it parsing, and a metric that
compiles but fails at answer time is worse than no metric: a question matches it, and the answer
is an error where a number should be.

Run inside the task, which is where the graph and Athena are both reachable:

    aws ecs execute-command ... --command "python /tmp/author_metrics.py"
"""

from __future__ import annotations

import sys

from src.config import load_config
from src.graph.client import GraphClient
from src.metrics.compiler import compile_metric
from src.metrics.graph_store import GraphMetricStore
from src.metrics.models import MetricDefinition, MetricRegistry, StaticCatalog

TENANT = "demo-firm"
AUTHOR = "author_metrics.py"

TIME_ENTRIES = "groundwork_legal.time_entries"
MATTERS = "groundwork_legal.matters"

#: Grain is what a question may slice by, and it is a restriction rather than a suggestion:
#: `parameters` is the closed set of columns a caller may filter on, so a metric cannot be
#: coerced into a cut its owner did not intend.
METRICS = [
    MetricDefinition(
        metric_id="lg_fees_billed",
        name="fees_billed",
        # Both a noun and a verb form of each idea. `MIN_DISTINCT_TERMS = 2` means a question
        # naming the metric once cannot match on that alone -- "what have we billed" reduces to
        # the single term "billed" -- so the vocabulary has to cover the words a question uses
        # *around* the metric too. This is the maintenance burden the vector router removes
        # rather than replaces: a paraphrase nobody listed still routes to tier 1.
        synonyms=[
            "fees",
            "fees billed",
            "billed",
            "billings",
            "invoiced",
            "invoice",
            "invoicing",
            "amount billed",
            "value billed",
            "charges",
        ],
        definition=(
            "Value of recorded time at each fee earner's charge-out rate, summed across the "
            "entries on a matter. The figure a partner means by 'what have we billed'."
        ),
        expression="SUM(amount_gbp)",
        source_table=TIME_ENTRIES,
        grain=["matter_id", "fee_earner", "grade"],
        time_grain_column="entry_date",
        time_grains=["day", "week", "month", "quarter", "year"],
        aggregation="additive",
        value_type="currency",
        unit="GBP",
        format="£#,##0",
        owner="finance-ops",
    ),
    MetricDefinition(
        metric_id="lg_hours_recorded",
        name="hours_recorded",
        synonyms=[
            "hours",
            "hours recorded",
            "hours worked",
            "time recorded",
            "time spent",
            "recorded time",
            "worked",
        ],
        definition="Hours of recorded time, by matter and fee earner.",
        expression="SUM(hours)",
        source_table=TIME_ENTRIES,
        grain=["matter_id", "fee_earner", "grade"],
        time_grain_column="entry_date",
        time_grains=["day", "week", "month", "quarter", "year"],
        aggregation="additive",
        value_type="number",
        unit="hours",
        format="#,##0.0",
        owner="finance-ops",
    ),
    MetricDefinition(
        metric_id="lg_matter_count",
        name="matter_count",
        synonyms=[
            "matters",
            "open matters",
            "matter count",
            "number of matters",
            "caseload",
            "matters open",
        ],
        definition=(
            "Distinct matters, by practice area and status. A count of matters rather than of "
            "rows, so a matter with many time entries is still one matter."
        ),
        expression="COUNT(DISTINCT matter_id)",
        source_table=MATTERS,
        grain=["practice_area", "status", "lead_partner"],
        time_grain_column="opened_date",
        time_grains=["month", "quarter", "year"],
        # A distinct count is immune to join fan-out but still not additive: summing distinct
        # counts across practice areas double-counts a matter appearing in two.
        aggregation="non_additive",
        value_type="count",
        unit="matters",
        owner="practice-management",
    ),
]


def _catalog(graph: GraphClient) -> StaticCatalog:
    """Schema read from the graph, which is where a catalog scan put it.

    The compiler needs column types to refuse a time grain on a non-temporal column, and primary
    keys to detect a join that would inflate a SUM. Neither is in the routing index -- that holds
    column *names* for matching -- so this is the read that has to come from the graph.
    """
    import boto3

    from src.discovery.glue_scanner import scan_catalog

    # `scan_catalog` already returns `TableSchema` -- the compiler's own view, with columns as a
    # name-to-type mapping and the primary keys resolved. Converting it would mean rebuilding the
    # shape it just handed over.
    result = scan_catalog(
        boto3.client("glue"),
        tenant_id=TENANT,
        source_id="glue-main",
        databases=["groundwork_legal"],
    )
    tables = {t.full_name: t for t in result.tables}
    print(f"catalog: {len(tables)} tables — {', '.join(sorted(tables))}")
    return StaticCatalog(tables=tables)


def main() -> int:
    cfg = load_config()
    graph = GraphClient(
        cfg.graph.uri,
        cfg.graph.user,
        cfg.graph.password,
        iam_auth=cfg.graph.iam_auth,
        region=cfg.graph.region,
    )
    store = GraphMetricStore(graph)
    catalog = _catalog(graph)
    registry = MetricRegistry.from_list(METRICS)

    # Built here rather than through `src.api.deps.build_athena_executor`, so this script runs
    # against a container image that predates that helper -- which is the situation whenever the
    # metrics need authoring before the next deploy. Same construction, same firewall.
    from src.executors.athena import AthenaConfig, AthenaExecutor
    from src.query.firewall import SQLFirewall

    bucket = cfg.structured.athena_results_bucket
    if not bucket:
        print("no Athena results bucket configured: nothing to verify against, nothing approved")
        return 1

    executor = AthenaExecutor(
        AthenaConfig(
            workgroup=cfg.structured.athena_workgroup or "primary",
            output_location=f"s3://{bucket}/athena-results/",
            region=cfg.graph.region or "",
        ),
        # The real allowlist, from what was actually scanned. A metric naming a table nobody
        # catalogued is refused here rather than at answer time.
        SQLFirewall(allowed_tables=set(catalog.tables)),
    )

    approved = 0
    for metric in METRICS:
        result = compile_metric(metric, catalog, registry=registry)
        if not result.is_valid:
            print(f"SKIP {metric.metric_id}: does not compile — {result.errors}")
            continue

        run = executor.execute(result.sql)
        if not run.success:
            print(f"SKIP {metric.metric_id}: compiled but did not run — {run.error}")
            continue

        head = run.rows[0] if run.rows else []
        print(f"  {metric.metric_id}: {run.row_count} rows, first = {head}")
        for w in result.warnings:
            print(f"    warning: {w}")

        store.save_metric(TENANT, metric, updated_by=AUTHOR)
        store.set_status(TENANT, metric.metric_id, "approved", updated_by=AUTHOR)
        approved += 1

    live = store.list_metrics(TENANT, approved_only=True)
    print(f"\napproved {approved} of {len(METRICS)}; tier 1 now sees {len(live)}")
    return 0 if approved else 1


if __name__ == "__main__":
    sys.exit(main())
