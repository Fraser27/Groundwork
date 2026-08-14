"""Tests for the metric compiler.

The two things worth testing here are determinism and governance. Determinism
because "no LLM writes this SQL" is the product claim; governance because the
compiler refusing a query is often the *correct* answer, and a compiler that
silently produces a wrong-but-valid number is worse than one that errors.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import sqlglot

from src.metrics.compiler import compile_metric, is_safe_scalar_expression
from src.metrics.loader import load_metrics
from src.metrics.models import (
    FilterClause,
    MetricDefinition,
    MetricRegistry,
    StaticCatalog,
)

SAMPLE = Path(__file__).resolve().parents[1] / "sample" / "metrics.yaml"

CATALOG = StaticCatalog.from_dicts(
    {
        "legal_ops.invoices": {
            "invoice_id": "string",
            "matter_id": "string",
            "invoice_amount": "decimal(18,2)",
            "invoice_date": "date",
            "invoice_status": "string",
            "practice_group": "string",
            "office": "string",
            "month": "string",
        },
        "legal_ops.time_entries": {
            "entry_id": "string",
            "matter_id": "string",
            "fee_earner_id": "string",
            "hours": "double",
            "standard_rate": "decimal(9,2)",
            "work_date": "date",
            "entry_status": "string",
            "is_billable": "boolean",
            "practice_group": "string",
        },
        "legal_ops.matter_wip_daily": {
            "matter_id": "string",
            "snapshot_date": "date",
            "unbilled_value": "decimal(18,2)",
            "matter_status": "string",
            "practice_group": "string",
            "office": "string",
        },
        "legal_ops.matters": {
            "matter_id": "string",
            "practice_group_id": "string",
            "opened_date": "date",
            "closed_date": "date",
            "client_id": "string",
        },
        "legal_ops.practice_groups": {
            "practice_group_id": "string",
            "practice_group": "string",
        },
        "legal_ops.time_narratives": {
            "entry_id": "string",
            "narrative": "string",
        },
    },
    primary_keys={
        "legal_ops.practice_groups": ["practice_group_id"],
        "legal_ops.matters": ["matter_id"],
    },
)


@pytest.fixture(scope="module")
def registry() -> MetricRegistry:
    result = load_metrics(SAMPLE)
    assert not result.errors, result.errors
    return result.registry


def metric(registry: MetricRegistry, metric_id: str) -> MetricDefinition:
    m = registry.get(metric_id)
    assert m is not None, metric_id
    return m


def parses(sql: str) -> bool:
    try:
        sqlglot.parse_one(sql, dialect="trino")
    except sqlglot.errors.ParseError:
        return False
    return True


class TestSamplePack:
    def test_sample_pack_loads_clean(self, registry):
        assert len(registry) == 6

    @pytest.mark.parametrize(
        "metric_id", ["lm_001", "lm_002", "lm_003", "lm_004", "lm_005", "lm_006"]
    )
    def test_every_sample_metric_compiles_to_valid_sql(self, registry, metric_id):
        result = compile_metric(metric(registry, metric_id), CATALOG, registry=registry)
        assert result.is_valid, result.errors
        assert parses(result.sql), result.sql


class TestDeterminism:
    """No LLM means byte-identical SQL for identical inputs. This is the product claim."""

    def test_same_inputs_produce_identical_sql(self, registry):
        m = metric(registry, "lm_001")
        kwargs = {
            "dimensions": ["practice_group", "invoice_date"],
            "filters": [FilterClause("office", "=", "London")],
            "time_grain": "month",
            "limit": 50,
        }
        first = compile_metric(m, CATALOG, **kwargs)
        second = compile_metric(m, CATALOG, **kwargs)
        assert first.sql == second.sql

    def test_dimension_order_is_preserved_not_normalised(self, registry):
        """Column order is the caller's, so a caller can rely on the output shape."""
        m = metric(registry, "lm_001")
        forward = compile_metric(m, CATALOG, dimensions=["practice_group", "office"])
        reverse = compile_metric(m, CATALOG, dimensions=["office", "practice_group"])
        assert forward.sql.index("practice_group") < forward.sql.index("office")
        assert reverse.sql.index("office") < reverse.sql.index("practice_group")


class TestTimeGrain:
    def test_requested_grain_becomes_date_trunc(self, registry):
        result = compile_metric(
            metric(registry, "lm_001"), CATALOG, dimensions=["invoice_date"], time_grain="month"
        )
        assert "DATE_TRUNC('month', invoice_date) AS invoice_date" in result.sql

    def test_group_by_uses_the_expression_not_the_alias(self, registry):
        """Trino rejects GROUP BY on an output alias, so the two clauses must differ."""
        result = compile_metric(
            metric(registry, "lm_001"), CATALOG, dimensions=["invoice_date"], time_grain="quarter"
        )
        group_by = result.sql.split("GROUP BY ")[1].splitlines()[0]
        assert group_by == "DATE_TRUNC('quarter', invoice_date)"
        assert " AS " not in group_by

    def test_undeclared_grain_is_refused(self, registry):
        result = compile_metric(
            metric(registry, "lm_003"), CATALOG, time_grain="day", registry=registry
        )
        assert not result.is_valid
        assert "not allowed" in result.errors[0]

    def test_unknown_grain_is_refused(self, registry):
        result = compile_metric(metric(registry, "lm_001"), CATALOG, time_grain="fortnight")
        assert not result.is_valid
        assert "Unsupported time_grain" in result.errors[0]

    def test_explicit_grain_adds_the_axis_when_no_dimensions_requested(self, registry):
        """Asking for a monthly series is asking to group by month.

        Without this the grain is silently dropped and the caller gets one ungrouped
        aggregate that looks like an answer.
        """
        result = compile_metric(
            metric(registry, "lm_001"), CATALOG, dimensions=[], time_grain="month"
        )
        assert result.is_valid, result.errors
        assert "DATE_TRUNC('month', invoice_date)" in result.sql
        assert "GROUP BY DATE_TRUNC('month', invoice_date)" in result.sql

    def test_declared_grains_supply_the_coarsest_default(self, registry):
        """A metric restricted to month+ must not leak its finer base grain."""
        result = compile_metric(
            metric(registry, "lm_005"), CATALOG, dimensions=["closed_date"]
        )
        assert result.time_grain == "year"
        assert "DATE_TRUNC('year', closed_date)" in result.sql

    def test_no_declared_grains_means_no_bucketing(self, registry):
        m = MetricDefinition(
            metric_id="x",
            name="raw_count",
            expression="COUNT(1)",
            source_table="legal_ops.invoices",
            grain=["invoice_date"],
        )
        result = compile_metric(m, CATALOG)
        assert "DATE_TRUNC" not in result.sql


class TestTimeAxisBypass:
    """A second time-like dimension would reintroduce a finer grain past the gate."""

    def test_other_temporal_column_is_refused(self, registry):
        result = compile_metric(
            metric(registry, "lm_005"),
            CATALOG,
            dimensions=["closed_date", "opened_date"],
        )
        assert not result.is_valid
        assert "governed time axis" in result.errors[0]

    def test_calendar_part_partition_column_is_refused(self, registry):
        """`month` is a string in Glue but still carries a time grain."""
        result = compile_metric(
            metric(registry, "lm_001"), CATALOG, dimensions=["invoice_date", "month"]
        )
        assert not result.is_valid
        assert "bypass" in result.errors[0]

    def test_non_temporal_dimensions_are_fine(self, registry):
        result = compile_metric(
            metric(registry, "lm_001"),
            CATALOG,
            dimensions=["invoice_date", "practice_group", "office"],
        )
        assert result.is_valid, result.errors

    def test_bypass_check_is_inert_without_declared_grains(self):
        """Nothing to bypass when the metric never restricted grains."""
        m = MetricDefinition(
            metric_id="x",
            name="raw_count",
            expression="COUNT(1)",
            source_table="legal_ops.matters",
        )
        result = compile_metric(m, CATALOG, dimensions=["opened_date", "closed_date"])
        assert result.is_valid, result.errors


class TestAdditivity:
    def test_semi_additive_sum_cannot_be_bucketed_across_time(self, registry):
        """Summing daily WIP up to a month double-counts the same unbilled time."""
        result = compile_metric(
            metric(registry, "lm_004"), CATALOG, dimensions=["snapshot_date"], time_grain="day"
        )
        assert not result.is_valid
        assert "semi_additive" in result.errors[0]

    def test_semi_additive_stays_at_base_grain_by_default(self, registry):
        """The coarsest-grain default must not fire where any rollup is invalid."""
        result = compile_metric(metric(registry, "lm_004"), CATALOG)
        assert result.is_valid, result.errors
        assert result.time_grain is None
        assert "DATE_TRUNC" not in result.sql

    def test_non_additive_metric_carries_a_warning(self, registry):
        result = compile_metric(
            metric(registry, "lm_006"), CATALOG, dimensions=["snapshot_date"], time_grain="month"
        )
        assert result.is_valid, result.errors
        assert any("non_additive" in w for w in result.warnings)

    def test_semi_additive_non_sum_may_be_bucketed(self):
        """The restriction is on SUM, not on the class — AVG of a balance is fine."""
        m = MetricDefinition(
            metric_id="x",
            name="avg_wip",
            expression="AVG(unbilled_value)",
            source_table="legal_ops.matter_wip_daily",
            grain=["snapshot_date"],
            time_grain_column="snapshot_date",
            time_grains=["day", "month"],
            aggregation="semi_additive",
        )
        result = compile_metric(m, CATALOG, dimensions=["snapshot_date"], time_grain="month")
        assert result.is_valid, result.errors


class TestFanoutDetection:
    def test_no_warning_when_join_key_is_the_targets_primary_key(self, registry):
        result = compile_metric(metric(registry, "lm_005"), CATALOG)
        assert result.is_valid, result.errors
        assert not any("inflate" in w for w in result.warnings)

    def test_warns_when_join_key_is_not_the_primary_key(self):
        m = MetricDefinition(
            metric_id="x",
            name="billed_with_narratives",
            expression="SUM(invoice_amount)",
            source_table="legal_ops.invoices",
            joins=[
                {
                    "table": "legal_ops.matters",
                    "source_column": "matter_id",
                    "target_column": "client_id",
                }
            ],
        )
        result = compile_metric(m, CATALOG)
        assert result.is_valid, result.errors
        assert any("legal_ops.matters" in w and "inflate" in w for w in result.warnings)

    def test_silent_when_target_has_no_primary_key_metadata(self):
        """Glue usually has no PK metadata; a warning on every join gets trained away."""
        m = MetricDefinition(
            metric_id="x",
            name="billed",
            expression="SUM(invoice_amount)",
            source_table="legal_ops.invoices",
            joins=[
                {
                    "table": "legal_ops.time_narratives",
                    "source_column": "invoice_id",
                    "target_column": "entry_id",
                }
            ],
        )
        result = compile_metric(m, CATALOG)
        assert not any("inflate" in w for w in result.warnings)

    def test_count_distinct_is_immune_to_fanout(self):
        m = MetricDefinition(
            metric_id="x",
            name="matter_count",
            expression="COUNT(DISTINCT matter_id)",
            source_table="legal_ops.invoices",
            joins=[
                {
                    "table": "legal_ops.matters",
                    "source_column": "matter_id",
                    "target_column": "client_id",
                }
            ],
        )
        result = compile_metric(m, CATALOG)
        assert not any("inflate" in w for w in result.warnings)


class TestFilterSurface:
    def test_undeclared_filter_column_is_refused(self, registry):
        result = compile_metric(
            metric(registry, "lm_001"), CATALOG, filters=[FilterClause("matter_id", "=", "M-1")]
        )
        assert not result.is_valid
        assert "not allowed" in result.errors[0]

    def test_required_parameter_must_be_supplied(self):
        m = MetricDefinition(
            metric_id="x",
            name="scoped_fees",
            expression="SUM(invoice_amount)",
            source_table="legal_ops.invoices",
            parameters=[{"column": "practice_group", "required": True}],
        )
        result = compile_metric(m, CATALOG)
        assert not result.is_valid
        assert "Required parameter" in result.errors[0]

    def test_string_literals_are_escaped(self, registry):
        result = compile_metric(
            metric(registry, "lm_001"),
            CATALOG,
            filters=[FilterClause("office", "=", "O'Hare")],
        )
        assert result.is_valid, result.errors
        assert "'O''Hare'" in result.sql
        assert parses(result.sql)

    def test_injection_attempt_stays_inside_the_literal(self, registry):
        result = compile_metric(
            metric(registry, "lm_001"),
            CATALOG,
            filters=[FilterClause("office", "=", "x' OR '1'='1")],
        )
        assert result.is_valid, result.errors
        assert "OR '1'='1'" not in result.sql
        assert parses(result.sql)

    def test_non_identifier_filter_column_is_refused(self, registry):
        result = compile_metric(
            metric(registry, "lm_001"),
            CATALOG,
            filters=[FilterClause("office = 'x' OR 1=1 --", "=", "y")],
        )
        assert not result.is_valid

    def test_in_filter_renders_a_list(self, registry):
        result = compile_metric(
            metric(registry, "lm_001"),
            CATALOG,
            filters=[FilterClause("office", "IN", ["London", "Leeds"])],
        )
        assert "office IN ('London', 'Leeds')" in result.sql

    def test_between_filter_needs_two_bounds(self, registry):
        result = compile_metric(
            metric(registry, "lm_001"),
            CATALOG,
            filters=[FilterClause("invoice_date", "BETWEEN", ["2026-01-01"])],
        )
        assert not result.is_valid
        assert "two bounds" in result.errors[0]

    def test_metric_filters_and_caller_filters_are_conjoined(self, registry):
        result = compile_metric(
            metric(registry, "lm_001"),
            CATALOG,
            filters=[FilterClause("office", "=", "London")],
        )
        where = result.sql.split("WHERE ")[1].splitlines()[0]
        assert "invoice_status = 'ISSUED'" in where
        assert "office = 'London'" in where
        assert " AND " in where


class TestExpressionSafety:
    @pytest.mark.parametrize(
        "expr",
        [
            "SUM(x); DROP TABLE t",
            "SUM(x) -- comment",
            "(SELECT max(x) FROM other)",
            "SUM(x) /* hidden */",
        ],
    )
    def test_unsafe_expressions_are_rejected(self, expr):
        assert not is_safe_scalar_expression(expr)

    @pytest.mark.parametrize(
        "expr",
        ["SUM(invoice_amount)", "SUM(hours * standard_rate)", "COUNT(DISTINCT matter_id)"],
    )
    def test_safe_expressions_pass(self, expr):
        assert is_safe_scalar_expression(expr)

    def test_metric_with_subquery_expression_will_not_compile(self):
        m = MetricDefinition(
            metric_id="x",
            name="sneaky",
            expression="(SELECT SUM(invoice_amount) FROM legal_ops.invoices)",
            source_table="legal_ops.invoices",
        )
        result = compile_metric(m, CATALOG)
        assert not result.is_valid
        assert "not a safe scalar" in result.errors[0]

    def test_unsafe_stored_filter_will_not_compile(self):
        m = MetricDefinition(
            metric_id="x",
            name="sneaky",
            expression="SUM(invoice_amount)",
            source_table="legal_ops.invoices",
            filters=["1=1 -- drop"],
        )
        result = compile_metric(m, CATALOG)
        assert not result.is_valid
        assert "not a safe predicate" in result.errors[0]


class TestDerived:
    def test_derived_metric_recomputes_each_side_as_a_cte(self, registry):
        result = compile_metric(
            metric(registry, "lm_003"),
            CATALOG,
            dimensions=["practice_group"],
            time_grain="quarter",
            registry=registry,
        )
        assert result.is_valid, result.errors
        assert "fees_billed AS (" in result.sql
        assert "standard_value AS (" in result.sql
        assert "(fees_billed / NULLIF(standard_value, 0)) AS realization_rate" in result.sql
        assert parses(result.sql)

    def test_each_base_buckets_its_own_time_axis_to_the_shared_period(self, registry):
        """The two sides measure time on different columns, so both must land on `period`.

        Without this the outer join has nothing to align on and realization silently
        compares one month's billings to another month's recorded time.
        """
        result = compile_metric(
            metric(registry, "lm_003"),
            CATALOG,
            dimensions=["practice_group"],
            time_grain="month",
            registry=registry,
        )
        assert result.is_valid, result.errors
        assert "DATE_TRUNC('month', invoice_date) AS period" in result.sql
        assert "DATE_TRUNC('month', work_date) AS period" in result.sql
        assert "standard_value.period" in result.sql

    def test_a_bases_own_date_column_is_not_a_valid_dimension(self, registry):
        """It has no meaning on the other side, so it would produce a broken join."""
        result = compile_metric(
            metric(registry, "lm_003"),
            CATALOG,
            dimensions=["practice_group", "invoice_date"],
            time_grain="month",
            registry=registry,
        )
        assert not result.is_valid
        assert "time axis" in result.errors[0]

    def test_grain_no_base_can_serve_is_refused(self, registry):
        """One side reporting a different period from the other is a wrong number."""
        m = MetricDefinition(
            metric_id="x",
            name="mismatched",
            type="derived",
            expression="fees_billed / open_matter_count",
            base_metrics=["lm_001", "lm_006"],
            time_grains=["year"],
        )
        result = compile_metric(m, CATALOG, registry=registry)
        assert not result.is_valid
        assert "open_matter_count" in result.errors[0]

    def test_derived_needs_a_registry(self, registry):
        result = compile_metric(metric(registry, "lm_003"), CATALOG)
        assert not result.is_valid
        assert "MetricRegistry" in result.errors[0]

    def test_unknown_base_metric_is_refused(self, registry):
        m = MetricDefinition(
            metric_id="x",
            name="bad_ratio",
            type="derived",
            expression="a / b",
            base_metrics=["lm_001", "nope"],
        )
        result = compile_metric(m, CATALOG, registry=registry)
        assert not result.is_valid
        assert "nope" in result.errors[0]

    def test_derived_of_derived_is_refused(self, registry):
        m = MetricDefinition(
            metric_id="x",
            name="nested",
            type="derived",
            expression="realization_rate * 2",
            base_metrics=["lm_003"],
        )
        result = compile_metric(m, CATALOG, registry=registry)
        assert not result.is_valid
        assert "derived" in result.errors[0]

    def test_derived_filter_must_name_a_projected_dimension(self, registry):
        result = compile_metric(
            metric(registry, "lm_003"),
            CATALOG,
            dimensions=["practice_group"],
            filters=[FilterClause("office", "=", "London")],
            registry=registry,
        )
        assert not result.is_valid
        assert "not available" in result.errors[0]

    def test_derived_dimensions_are_coalesced_across_ctes(self, registry):
        """A dimension present in one base but not the other must not blank the row."""
        result = compile_metric(
            metric(registry, "lm_003"),
            CATALOG,
            dimensions=["practice_group"],
            registry=registry,
        )
        assert "COALESCE(fees_billed.practice_group, standard_value.practice_group)" in result.sql
        assert "FULL OUTER JOIN" in result.sql

    def test_derived_with_no_keys_at_all_cross_joins(self):
        """Each CTE returns one row, so there is nothing to join on."""
        m = MetricDefinition(
            metric_id="x",
            name="lifetime_ratio",
            type="derived",
            expression="fees_billed / standard_value",
            base_metrics=["lm_001", "lm_002"],
        )
        registry = MetricRegistry.from_list(
            [
                MetricDefinition(
                    metric_id="lm_001",
                    name="fees_billed",
                    expression="SUM(invoice_amount)",
                    source_table="legal_ops.invoices",
                ),
                MetricDefinition(
                    metric_id="lm_002",
                    name="standard_value",
                    expression="SUM(hours * standard_rate)",
                    source_table="legal_ops.time_entries",
                ),
            ]
        )
        result = compile_metric(m, CATALOG, dimensions=[], registry=registry)
        assert result.is_valid, result.errors
        assert "CROSS JOIN standard_value" in result.sql
        assert "DATE_TRUNC" not in result.sql


class TestOrderAndLimit:
    def test_order_by_must_name_an_output_column(self, registry):
        result = compile_metric(
            metric(registry, "lm_001"), CATALOG, order_by=["(SELECT 1)"]
        )
        assert not result.is_valid
        assert "Invalid order_by" in result.errors[0]

    def test_order_by_accepts_direction(self, registry):
        result = compile_metric(
            metric(registry, "lm_001"),
            CATALOG,
            dimensions=["practice_group"],
            order_by=["fees_billed DESC"],
        )
        assert result.is_valid, result.errors
        assert result.sql.rstrip().endswith("ORDER BY fees_billed DESC")

    def test_order_by_rejects_a_bogus_direction(self, registry):
        result = compile_metric(
            metric(registry, "lm_001"), CATALOG, order_by=["fees_billed SIDEWAYS"]
        )
        assert not result.is_valid

    def test_limit_is_appended(self, registry):
        result = compile_metric(metric(registry, "lm_001"), CATALOG, limit=25)
        assert result.sql.rstrip().endswith("LIMIT 25")


class TestSchemaResolution:
    def test_unknown_dimensions_are_dropped_not_fatal(self, registry):
        """Exploring a metric should return the metric, not a stack trace."""
        result = compile_metric(
            metric(registry, "lm_001"), CATALOG, dimensions=["practice_group", "nonexistent"]
        )
        assert result.is_valid, result.errors
        assert "nonexistent" not in result.sql

    def test_empty_catalog_means_unknown_not_no_columns(self, registry):
        """Glue metadata is often absent; the compiler stays permissive."""
        result = compile_metric(
            metric(registry, "lm_001"),
            StaticCatalog({}),
            dimensions=["practice_group"],
        )
        assert result.is_valid, result.errors
        assert "practice_group" in result.sql

    def test_joined_table_columns_are_resolvable(self, registry):
        result = compile_metric(
            metric(registry, "lm_005"), CATALOG, dimensions=["practice_group"]
        )
        assert result.is_valid, result.errors
        assert "practice_group" in result.sql

    def test_reported_tables_cover_source_and_joins(self, registry):
        result = compile_metric(metric(registry, "lm_005"), CATALOG)
        assert result.tables == ["legal_ops.matters", "legal_ops.practice_groups"]

    def test_derived_reports_every_base_table(self, registry):
        result = compile_metric(metric(registry, "lm_003"), CATALOG, registry=registry)
        assert result.tables == ["legal_ops.invoices", "legal_ops.time_entries"]
