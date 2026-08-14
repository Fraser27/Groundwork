"""Governed metric -> Athena SQL. Deterministic, no LLM, ever.

This is the whole point of a governed metric: two people asking "what was
realization last quarter" get byte-identical SQL, and when a regulator asks why the
number is what it is, the answer is a metric definition under version control
rather than a sampled token stream.

Everything the compiler emits comes from one of three places: a validated
`MetricDefinition`, a `SchemaCatalog` lookup, or a literal escaped by
`_sql_literal`. Nothing else reaches the output string.

The governance the compiler enforces, beyond producing valid SQL:

* **Time grain.** A metric that declares `time_grains` is served at one of them or
  not at all. Silently answering a monthly metric at day grain is a wrong number
  that looks right.
* **Additivity.** A `semi_additive` SUM cannot be bucketed across time —
  summing daily WIP into a month double-counts. Refused, not warned.
* **Fan-out.** An aggregate over a join whose key is not the target's primary key
  will inflate. Warned, because the join may legitimately be one-to-one and we only
  ever have partial PK metadata.
* **Filter surface.** If a metric declares `parameters`, those are the only
  columns a caller may filter on.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp

from src.metrics.models import (
    IDENTIFIER_RE,
    OPERATORS,
    TIME_GRAIN_ORDER,
    TIME_GRAIN_UNITS,
    FilterClause,
    MetricDefinition,
    MetricJoin,
    MetricRegistry,
    SchemaCatalog,
)

logger = logging.getLogger(__name__)

DIALECT = "trino"

#: Column data-type prefixes treated as temporal for bucketing.
_TEMPORAL_TYPE_PREFIXES = ("date", "timestamp", "time")

#: Columns whose *name* denotes a calendar part even when stored as string/int —
#: Glue partition columns like `month='03'`. Grouping by one carries a time grain
#: implicitly, which is how a caller would otherwise sidestep declared time_grains.
_CALENDAR_PART_NAMES = frozenset(TIME_GRAIN_UNITS)

#: Nodes that make a fragment unsafe to inline: a fragment must be a scalar, not a
#: statement, and must not smuggle in a subquery or set operation.
_UNSAFE_EXPR_NODES = (exp.Select, exp.Subquery, exp.Union, exp.Except, exp.Intersect)

#: Output name for the shared time bucket in a derived metric. Composed metrics
#: usually measure time on differently-named columns, so the bucket needs one name
#: both sides agree on before the outer join can align them.
PERIOD_ALIAS = "period"


class MetricCompilationError(ValueError):
    """Raised for a metric definition the compiler refuses to emit SQL for."""


@dataclass
class CompilationResult:
    sql: str
    source_table: str
    metric_name: str | None = None
    is_valid: bool = True
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    time_grain: str | None = None
    """The grain actually applied — may differ from the request when a metric's
    declared grains supply a default."""

    tables: list[str] = field(default_factory=list)
    """Physical tables referenced, for the firewall to check."""


def _fail(message: str, *, table: str = "", name: str | None = None) -> CompilationResult:
    return CompilationResult(
        sql="", source_table=table, metric_name=name, is_valid=False, errors=[message]
    )


# ── SQL fragment safety ───────────────────────────────────────────────────────


def _has_comment_or_separator(raw: str) -> bool:
    return ";" in raw or "--" in raw or "/*" in raw or "*/" in raw


def is_safe_scalar_expression(expr: str) -> bool:
    """True if `expr` is one safe scalar SQL expression, e.g. `SUM(billed_value)`.

    Metric expressions are authored by humans in YAML, not generated, but they are
    interpolated verbatim, so they get parsed and inspected rather than trusted.
    """
    if not expr or _has_comment_or_separator(expr):
        return False
    try:
        parsed = sqlglot.parse(expr, dialect=DIALECT)
    except sqlglot.errors.ParseError:
        return False
    if len(parsed) != 1 or parsed[0] is None:
        return False
    node = parsed[0]
    if isinstance(node, _UNSAFE_EXPR_NODES):
        return False
    return not any(node.find(t) for t in _UNSAFE_EXPR_NODES)


def _sql_literal(value: str | float | bool) -> str:
    # bool before the numeric branch: bool is a subclass of int, and emitting `1`
    # for TRUE would silently change the meaning of a boolean filter.
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return str(value)
    return "'" + str(value).replace("'", "''") + "'"


def _check_filter_shape(filters: list[FilterClause]) -> str | None:
    """Column names and operators come from a caller, so neither is trusted.

    Values are escaped by `_sql_literal`, but a column name is interpolated bare —
    it has to be an identifier before it gets anywhere near the output.
    """
    for f in filters:
        if not IDENTIFIER_RE.match(f.column):
            return f"Filter column is not a valid identifier: '{f.column}'"
        if f.operator.upper() not in OPERATORS:
            return f"Unsupported filter operator '{f.operator}'; allowed: {sorted(OPERATORS)}"
    return None


def _build_filter_clauses(filters: list[FilterClause], prefix: str = "") -> list[str]:
    clauses: list[str] = []
    for f in filters:
        column = f"{prefix}{f.column}"
        op = f.operator.upper()
        if op in ("IN", "NOT IN"):
            values = f.value if isinstance(f.value, (list, tuple)) else [f.value]
            if not values:
                raise MetricCompilationError(f"{op} filter on '{f.column}' has no values")
            rendered = ", ".join(_sql_literal(v) for v in values)
            clauses.append(f"{column} {op} ({rendered})")
        elif op == "BETWEEN":
            if not isinstance(f.value, (list, tuple)) or len(f.value) != 2:
                raise MetricCompilationError(
                    f"BETWEEN filter on '{f.column}' needs exactly two bounds"
                )
            lo, hi = f.value
            clauses.append(f"{column} BETWEEN {_sql_literal(lo)} AND {_sql_literal(hi)}")
        else:
            clauses.append(f"{column} {op} {_sql_literal(f.value)}")
    return clauses


def _validate_order_by(order_by: list[str], known: set[str]) -> str | None:
    """Order_by entries must name an output column, optionally with ASC/DESC."""
    for entry in order_by:
        parts = entry.split()
        if len(parts) > 2 or (len(parts) == 2 and parts[1].upper() not in {"ASC", "DESC"}):
            return entry
        if parts[0] not in known:
            return entry
    return None


# ── Schema resolution ─────────────────────────────────────────────────────────


def _column_types(metric: MetricDefinition, catalog: SchemaCatalog) -> dict[str, str]:
    """Column -> type across the source table and every join target.

    Flattened deliberately: a metric's dimensions are unqualified column names, so
    the compiler resolves them the same way the query engine will.
    """
    types: dict[str, str] = {}
    for table in metric.tables:
        schema = catalog.table(table)
        if schema is not None:
            types.update(schema.columns)
    return types


def _is_temporal_type(data_type: str) -> bool:
    return data_type.startswith(_TEMPORAL_TYPE_PREFIXES)


def _is_time_like(column: str, col_types: dict[str, str]) -> bool:
    return _is_temporal_type(col_types.get(column, "")) or column.lower() in _CALENDAR_PART_NAMES


def _resolve_time_axis(metric: MetricDefinition, col_types: dict[str, str]) -> str | None:
    """The one column `time_grains` governs.

    Explicit `time_grain_column` wins; otherwise the first temporal column in the
    metric's grain, so metrics authored without the explicit field still work.
    """
    if metric.time_grain_column:
        return metric.time_grain_column
    for dim in metric.grain:
        if _is_temporal_type(col_types.get(dim, "")):
            return dim
    return None


def _default_time_grain(declared: list[str]) -> str | None:
    """Coarsest declared grain.

    The safe default: a metric restricted to ['quarter'] must never emit daily rows
    because the caller said nothing.
    """
    valid = [g for g in declared if g in TIME_GRAIN_UNITS]
    return max(valid, key=lambda g: TIME_GRAIN_ORDER[g]) if valid else None


# ── Governance checks ─────────────────────────────────────────────────────────


def _check_time_axis_bypass(
    dimensions: list[str],
    time_axis: str | None,
    declared: list[str],
    col_types: dict[str, str],
) -> str | None:
    """Refuse time-slicing dimensions other than the governed axis.

    `time_grains` only controls DATE_TRUNC on the axis. Grouping by *any other*
    time-like column — a raw timestamp, or a `month` partition string — reintroduces
    a finer grain through the back door, so it fails loudly instead of answering at
    the wrong grain.
    """
    if not declared:
        return None
    for d in dimensions:
        if d == time_axis or not _is_time_like(d, col_types):
            continue
        hint = (
            f"group by '{time_axis}' with a time_grain from {sorted(declared)} instead"
            if time_axis
            else f"this metric declares time_grains {sorted(declared)} but has no time axis column"
        )
        return (
            f"Dimension '{d}' groups by time but is not this metric's governed time axis "
            f"— that would bypass the declared time_grains {sorted(declared)}; {hint}."
        )
    return None


def _apply_time_grain(
    dimensions: list[str],
    grain: str | None,
    declared: list[str],
    time_axis: str | None,
) -> tuple[list[str], list[str], str | None]:
    """Rewrite the time axis to `DATE_TRUNC(unit, col)`.

    Returns (select_dims, group_dims, error). The two differ because Trino rejects
    GROUP BY on an output alias: SELECT carries `DATE_TRUNC(...) AS col`, GROUP BY
    carries the bare expression.
    """
    if not grain:
        return dimensions, dimensions, None
    unit = TIME_GRAIN_UNITS.get(grain.lower())
    if unit is None:
        return dimensions, dimensions, f"Unsupported time_grain '{grain}'"
    if declared and grain.lower() not in declared:
        return dimensions, dimensions, (
            f"time_grain '{grain}' not allowed; this metric declares {sorted(declared)}"
        )
    if not time_axis:
        return dimensions, dimensions, (
            "No time axis available to apply time_grain — set the metric's "
            "time_grain_column, or include a date/timestamp column in its grain"
        )
    if time_axis not in dimensions:
        return dimensions, dimensions, (
            f"time_grain '{grain}' applies to time axis '{time_axis}', "
            f"which is not among the queried dimensions {dimensions}"
        )
    trunc = f"DATE_TRUNC('{unit}', {time_axis})"
    select_dims = [f"{trunc} AS {time_axis}" if d == time_axis else d for d in dimensions]
    group_dims = [trunc if d == time_axis else d for d in dimensions]
    return select_dims, group_dims, None


def _is_sum(expression: str) -> bool:
    """True if the top-level aggregate is a SUM — the semi-additive trap."""
    try:
        node = sqlglot.parse_one(expression, dialect=DIALECT)
    except sqlglot.errors.ParseError:
        return False
    return isinstance(node, exp.Sum)


_FANOUT_SENSITIVE = (exp.Sum, exp.Count, exp.Avg)


def _is_fanout_sensitive(expression: str) -> bool:
    """True if row duplication from a one-to-many join would distort the aggregate.

    SUM/COUNT/AVG all inflate when a join multiplies source rows.
    COUNT(DISTINCT ...) is immune and excluded.
    """
    try:
        node = sqlglot.parse_one(expression, dialect=DIALECT)
    except sqlglot.errors.ParseError:
        return False
    for agg in node.find_all(*_FANOUT_SENSITIVE):
        if isinstance(agg, exp.Count) and agg.args.get("this") and agg.find(exp.Distinct):
            continue
        return True
    return False


def _detect_fanout_joins(
    expression: str, joins: list[MetricJoin], catalog: SchemaCatalog
) -> list[str]:
    """Join targets that can fan out an additive measure.

    Only flags on *positive* evidence: the target has PK metadata AND the join key
    is not among those PKs. Glue rarely carries PK metadata, and a warning on every
    join would be trained away within a week.
    """
    if not _is_fanout_sensitive(expression):
        return []
    risky = []
    for j in joins:
        schema = catalog.table(j.table)
        if schema and schema.primary_keys and j.target_column not in schema.primary_keys:
            risky.append(j.table)
    return risky


# ── Compilation ───────────────────────────────────────────────────────────────


def _make_alias(table: str, used: set[str]) -> str:
    segment = table.split(".")[-1] if table else ""
    short = segment[0] if segment else "t"
    alias, i = short, 2
    while alias in used:
        alias, i = f"{short}{i}", i + 1
    used.add(alias)
    return alias


def _validate_expressions(metric: MetricDefinition) -> str | None:
    if not is_safe_scalar_expression(metric.expression):
        return (
            f"Metric '{metric.name}' expression is not a safe scalar SQL expression: "
            f"'{metric.expression}'"
        )
    for f in metric.filters:
        if not is_safe_scalar_expression(f):
            return f"Metric '{metric.name}' filter is not a safe predicate: '{f}'"
    return None


def _validate_filters(
    metric: MetricDefinition, filters: list[FilterClause], col_types: dict[str, str]
) -> str | None:
    """A caller may filter only on declared parameters, and must supply required ones.

    When a metric declares no parameters the filter surface is its real columns —
    still a closed set, just a wider one.
    """
    if metric.parameters:
        allowed = {p.column for p in metric.parameters}
        for f in filters:
            if f.column not in allowed:
                return (
                    f"Filter on '{f.column}' not allowed — metric '{metric.name}' declares "
                    f"parameters {sorted(allowed)}"
                )
        provided = {f.column for f in filters}
        missing = [p.column for p in metric.parameters if p.required and p.column not in provided]
        if missing:
            return f"Required parameter(s) missing for '{metric.name}': {sorted(missing)}"
    elif col_types:
        for f in filters:
            if f.column not in col_types:
                return f"Filter column '{f.column}' is not a known column of '{metric.source_table}'"
    return None


def compile_metric(
    metric: MetricDefinition,
    catalog: SchemaCatalog,
    *,
    dimensions: list[str] | None = None,
    filters: list[FilterClause] | None = None,
    order_by: list[str] | None = None,
    limit: int | None = None,
    time_grain: str | None = None,
    registry: MetricRegistry | None = None,
) -> CompilationResult:
    """Compile one metric to Athena SQL. Same inputs always give the same output."""
    filters = list(filters or [])

    if metric.type == "derived":
        if registry is None:
            return _fail(
                f"Derived metric '{metric.metric_id}' needs a MetricRegistry to resolve "
                f"its base metrics {metric.base_metrics}",
                name=metric.name,
            )
        return _compile_derived(
            metric,
            catalog,
            registry,
            dimensions=dimensions,
            filters=filters,
            order_by=order_by,
            limit=limit,
            time_grain=time_grain,
        )

    table = metric.source_table
    name = metric.name

    if err := _validate_expressions(metric):
        return _fail(err, table=table, name=name)
    if err := _check_filter_shape(filters):
        return _fail(err, table=table, name=name)

    col_types = _column_types(metric, catalog)

    if err := _validate_filters(metric, filters, col_types):
        return _fail(err, table=table, name=name)

    dims = list(dimensions) if dimensions is not None else list(metric.grain)

    # Unknown dimensions are dropped rather than fatal: a caller exploring a metric
    # should get the metric, not a stack trace. An unknown *filter* is fatal, because
    # dropping it would silently widen the result set.
    if col_types:
        unknown = [d for d in dims if d not in col_types]
        if unknown:
            logger.warning("Metric '%s': dropping unknown dimensions %s", metric.metric_id, unknown)
            dims = [d for d in dims if d in col_types]

    declared = list(metric.time_grains)
    time_axis = _resolve_time_axis(metric, col_types)
    semi_additive_sum = metric.aggregation == "semi_additive" and _is_sum(metric.expression)

    # An explicit time_grain implies grouping by the axis it applies to — asking for
    # a monthly series is asking to group by month. Adding the axis beats refusing
    # over an argument the caller should not have to repeat.
    if time_grain and time_axis and time_axis not in dims:
        dims = [*dims, time_axis]

    if err := _check_time_axis_bypass(dims, time_axis, declared, col_types):
        return _fail(err, table=table, name=name)

    # A metric that restricts grains is always served at one of them: with no grain
    # named, fall back to the coarsest declared rather than leaking the finer base
    # grain. Skipped for semi-additive sums, where any cross-time rollup is invalid —
    # those stay at base grain instead of auto-applying a grain we'd then reject.
    effective_grain = time_grain
    if not effective_grain and declared and time_axis in dims and not semi_additive_sum:
        effective_grain = _default_time_grain(declared)

    select_dims, group_dims, err = _apply_time_grain(dims, effective_grain, declared, time_axis)
    if err:
        return _fail(err, table=table, name=name)

    if effective_grain and semi_additive_sum:
        return _fail(
            f"Metric '{name}' is semi_additive (a point-in-time snapshot) and cannot be "
            f"summed across a time grain — bucketing daily values up to "
            f"'{effective_grain}' double-counts. Use a last-value or average over the "
            f"period, or query at base grain with no time_grain.",
            table=table,
            name=name,
        )

    if order_by and (bad := _validate_order_by(order_by, set(dims) | {name})):
        return _fail(f"Invalid order_by entry: '{bad}'", table=table, name=name)

    used: set[str] = set()
    source_alias = _make_alias(table, used)
    aliases = {table: source_alias}
    for j in metric.joins:
        aliases.setdefault(j.table, _make_alias(j.table, used))

    lines = [
        f"SELECT {', '.join([*select_dims, f'{metric.expression} AS {name}'])}",
        f"FROM {table} {source_alias}",
    ]
    for j in metric.joins:
        lines.append(
            f"{j.join_type} JOIN {j.table} {aliases[j.table]} "
            f"ON {source_alias}.{j.source_column} = {aliases[j.table]}.{j.target_column}"
        )

    try:
        where = [*metric.filters, *_build_filter_clauses(filters)]
    except MetricCompilationError as e:
        return _fail(str(e), table=table, name=name)
    if where:
        lines.append(f"WHERE {' AND '.join(where)}")
    if group_dims:
        lines.append(f"GROUP BY {', '.join(group_dims)}")
    if order_by:
        lines.append(f"ORDER BY {', '.join(order_by)}")
    elif select_dims:
        lines.append(f"ORDER BY {name} DESC")
    if limit:
        lines.append(f"LIMIT {limit}")

    warnings: list[str] = []
    if risky := _detect_fanout_joins(metric.expression, metric.joins, catalog):
        warnings.append(
            f"Metric '{name}' aggregates across join(s) to {risky} whose join key is not "
            f"that table's primary key: a one-to-many match duplicates source rows and "
            f"inflates the result. Verify the join is one-to-one, or pre-aggregate."
        )
    if metric.aggregation == "non_additive":
        warnings.append(
            f"Metric '{name}' is non_additive: correct at the queried grain, but the "
            f"returned rows must not be summed or averaged further."
        )

    return CompilationResult(
        sql="\n".join(lines),
        source_table=table,
        metric_name=name,
        warnings=warnings,
        time_grain=effective_grain,
        tables=sorted(set(metric.tables)),
    )


def _compile_cte_body(
    metric: MetricDefinition,
    catalog: SchemaCatalog,
    dimensions: list[str],
    time_grain: str | None,
) -> str:
    """One base metric as a CTE body, bucketed to the shared grain.

    The time bucket is aliased to `PERIOD_ALIAS` rather than to the base's own axis
    column. Composed metrics rarely share a column name for time — realization is
    invoices.invoice_date over time_entries.work_date — so the bucket has to become a
    shared name for the outer join to line the two sides up at all.
    """
    if err := _validate_expressions(metric):
        raise MetricCompilationError(err)

    col_types = _column_types(metric, catalog)
    declared = list(metric.time_grains)
    time_axis = _resolve_time_axis(metric, col_types)

    if err := _check_time_axis_bypass(dimensions, time_axis, declared, col_types):
        raise MetricCompilationError(f"Metric '{metric.name}': {err}")

    select_dims = list(dimensions)
    group_dims = list(dimensions)
    if time_grain:
        unit = TIME_GRAIN_UNITS.get(time_grain.lower())
        if unit is None:
            raise MetricCompilationError(f"Unsupported time_grain '{time_grain}'")
        if declared and time_grain.lower() not in declared:
            raise MetricCompilationError(
                f"Metric '{metric.name}' does not declare time_grain '{time_grain}'; "
                f"it allows {sorted(declared)}"
            )
        if not time_axis:
            raise MetricCompilationError(
                f"Metric '{metric.name}' has no time axis, so it cannot be composed at "
                f"'{time_grain}' grain — set its time_grain_column"
            )
        if metric.aggregation == "semi_additive" and _is_sum(metric.expression):
            raise MetricCompilationError(
                f"Metric '{metric.name}' is semi_additive and cannot be summed across a "
                f"'{time_grain}' grain"
            )
        trunc = f"DATE_TRUNC('{unit}', {time_axis})"
        select_dims.append(f"{trunc} AS {PERIOD_ALIAS}")
        group_dims.append(trunc)

    used: set[str] = set()
    source_alias = _make_alias(metric.source_table, used)
    aliases = {metric.source_table: source_alias}
    for j in metric.joins:
        aliases.setdefault(j.table, _make_alias(j.table, used))

    select = ", ".join([*select_dims, f"{metric.expression} AS {metric.name}"])
    lines = [f"SELECT {select}", f"  FROM {metric.source_table} {source_alias}"]
    for j in metric.joins:
        lines.append(
            f"  {j.join_type} JOIN {j.table} {aliases[j.table]} "
            f"ON {source_alias}.{j.source_column} = {aliases[j.table]}.{j.target_column}"
        )
    if metric.filters:
        lines.append(f"  WHERE {' AND '.join(metric.filters)}")
    if group_dims:
        lines.append(f"  GROUP BY {', '.join(group_dims)}")
    return "\n".join(lines)


def _compile_derived(
    metric: MetricDefinition,
    catalog: SchemaCatalog,
    registry: MetricRegistry,
    *,
    dimensions: list[str] | None,
    filters: list[FilterClause],
    order_by: list[str] | None,
    limit: int | None,
    time_grain: str | None,
) -> CompilationResult:
    """A derived metric as CTEs joined on shared dimensions, then its expression.

    Each base is recomputed from its own base rows at the shared grain rather than
    rolled up from a coarser result. That is what makes a ratio of two additive
    measures correct at any grain — and it is why the bases may sit on entirely
    different tables with differently-named date columns.

    Time appears in the output as `period`, not as either base's date column, since
    neither name is meaningful for the other side.
    """
    bases: list[MetricDefinition] = []
    missing: list[str] = []
    for ref in metric.base_metrics:
        base = registry.resolve(ref)
        (bases.append(base) if base else missing.append(ref))
    if missing:
        return _fail(
            f"Derived metric '{metric.metric_id}' references unknown base metric(s) {missing}",
            name=metric.name,
        )
    if any(b.type == "derived" for b in bases):
        return _fail(
            f"Derived metric '{metric.metric_id}' composes another derived metric; "
            f"only one level of composition is supported so the emitted SQL stays reviewable",
            name=metric.name,
        )

    requested = list(dimensions) if dimensions is not None else list(metric.grain)
    source_table = bases[0].source_table

    if err := _validate_expressions(metric):
        return _fail(err, table=source_table, name=metric.name)
    if err := _check_filter_shape(filters):
        return _fail(err, table=source_table, name=metric.name)

    declared = list(metric.time_grains)
    # A derived metric's own time_grains are the outer gate; the bases then have to
    # be able to serve whatever survives it.
    if time_grain and declared and time_grain.lower() not in declared:
        return _fail(
            f"time_grain '{time_grain}' not allowed; metric '{metric.name}' declares "
            f"{sorted(declared)}",
            table=source_table,
            name=metric.name,
        )

    # Only non-time dimensions are grouped by name. A caller naming a base's date
    # column would be grouping by a column the other base does not have, so time is
    # expressed as the grain plus the `period` output.
    dims: list[str] = []
    for d in requested:
        if d == PERIOD_ALIAS:
            continue
        base_axes = {
            _resolve_time_axis(b, _column_types(b, catalog)) for b in bases
        }
        if d in base_axes:
            return _fail(
                f"Dimension '{d}' is one base metric's time axis and has no meaning on the "
                f"others. Derived metrics slice time via time_grain, which is returned as "
                f"'{PERIOD_ALIAS}'.",
                table=source_table,
                name=metric.name,
            )
        dims.append(d)

    # Every base must carry every dimension. Unlike a simple metric, an unknown
    # dimension cannot just be dropped here: the outer join lines the CTEs up on
    # these columns, so one base missing a dimension is a broken query, not a
    # narrower one.
    for base in bases:
        base_cols = _column_types(base, catalog)
        if not base_cols:
            continue
        if absent := [d for d in dims if d not in base_cols]:
            return _fail(
                f"Derived metric '{metric.name}' cannot be grouped by {sorted(absent)}: "
                f"base metric '{base.name}' has no such column",
                table=source_table,
                name=metric.name,
            )

    effective = time_grain
    if not effective and declared:
        effective = _default_time_grain(declared)
    if effective:
        # The grain has to be servable by every base, or one side silently reports a
        # different period from the other.
        unservable = [b.name for b in bases if b.time_grains and effective not in b.time_grains]
        if unservable:
            return _fail(
                f"Derived metric '{metric.name}' cannot be served at '{effective}' grain: "
                f"base metric(s) {sorted(unservable)} do not declare it",
                table=source_table,
                name=metric.name,
            )

    # `period` is projected only when a grain applies, so it joins the key list here
    # rather than in `dims`.
    keys = [*dims, PERIOD_ALIAS] if effective else list(dims)

    for f in filters:
        if f.column not in keys:
            return _fail(
                f"Filter on '{f.column}' is not available on derived metric "
                f"'{metric.name}' — filterable dimensions are {sorted(keys)}",
                table=source_table,
                name=metric.name,
            )

    cte_parts: list[str] = []
    names: list[str] = []
    tables: list[str] = []
    for base in bases:
        try:
            body = _compile_cte_body(base, catalog, dims, effective)
        except MetricCompilationError as e:
            return _fail(str(e), table=base.source_table, name=metric.name)
        cte_parts.append(f"{base.name} AS (\n  {body}\n)")
        names.append(base.name)
        tables.extend(base.tables)

    if order_by and (bad := _validate_order_by(order_by, set(keys) | {metric.name})):
        return _fail(f"Invalid order_by entry: '{bad}'", table=source_table, name=metric.name)

    inner_select: list[str] = []
    for k in keys:
        # COALESCE across every CTE: a key value present in one base but not another
        # must not blank the output row.
        inner_select.append(f"COALESCE({', '.join(f'{c}.{k}' for c in names)}) AS {k}")
    inner_select.extend(f"{c}.{c}" for c in names)

    inner_lines = [f"SELECT {', '.join(inner_select)}", f"FROM {names[0]}"]
    for idx, cte in enumerate(names[1:], start=1):
        if keys:
            prior = names[:idx]
            on = " AND ".join(
                f"{prior[0]}.{k} = {cte}.{k}"
                if len(prior) == 1
                else f"COALESCE({', '.join(f'{c}.{k}' for c in prior)}) = {cte}.{k}"
                for k in keys
            )
            inner_lines.append(f"FULL OUTER JOIN {cte} ON {on}")
        else:
            # No shared keys: each CTE returns exactly one row.
            inner_lines.append(f"CROSS JOIN {cte}")

    inner = "\n".join(inner_lines)
    outer = [f"sub.{k}" for k in keys] + [f"({metric.expression}) AS {metric.name}"]
    ctes = ",\n".join(cte_parts)
    lines = [f"SELECT {', '.join(outer)}", f"FROM (\nWITH {ctes}\n{inner}\n) sub"]

    try:
        where = _build_filter_clauses(filters, prefix="sub.")
    except MetricCompilationError as e:
        return _fail(str(e), table=source_table, name=metric.name)
    if where:
        lines.append(f"WHERE {' AND '.join(where)}")
    if order_by:
        lines.append(f"ORDER BY {', '.join(order_by)}")
    elif keys:
        lines.append(f"ORDER BY {metric.name} DESC")
    if limit:
        lines.append(f"LIMIT {limit}")

    warnings = []
    units = {b.unit for b in bases if b.unit}
    if len(units) > 1:
        warnings.append(
            f"Derived metric '{metric.name}' combines base metrics with different units "
            f"{sorted(units)}; check the result is meaningful."
        )
    if metric.aggregation == "non_additive":
        warnings.append(
            f"Metric '{metric.name}' is non_additive: correct at the queried grain, but the "
            f"returned rows must not be summed or averaged further."
        )

    return CompilationResult(
        sql="\n".join(lines),
        source_table=source_table,
        metric_name=metric.name,
        warnings=warnings,
        time_grain=effective,
        tables=sorted(set(tables)),
    )
