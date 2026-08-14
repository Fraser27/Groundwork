"""Metric definitions and the schema seam the compiler resolves names against.

Identifiers are validated *here*, at the model boundary, rather than in the
compiler. The compiler interpolates table, column and alias names straight into
SQL text, so its safety argument is "every name reaching the output was validated
on the way in" — which only holds if a MetricDefinition cannot exist with a name
that isn't a plain identifier.

The compiler reads column types and primary keys through `SchemaCatalog` rather
than a graph client. Two reasons: no module outside `src/graph/` may build Cypher,
and a compiler that needs a live database to be tested stops being tested.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel, Field, field_validator, model_validator

IDENTIFIER_RE = re.compile(r"^[A-Za-z_][A-Za-z_0-9]*$")
#: catalog.database.table, database.table, or table.
QUALIFIED_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z_0-9]*(\.[A-Za-z_][A-Za-z_0-9]*){0,2}$")

#: Grains the compiler can bucket to, mapped to the unit Trino's DATE_TRUNC wants.
TIME_GRAIN_UNITS: Mapping[str, str] = {
    "hour": "hour",
    "day": "day",
    "week": "week",
    "month": "month",
    "quarter": "quarter",
    "year": "year",
}
#: Coarseness ranking — picks the default when a metric declares several grains.
TIME_GRAIN_ORDER: Mapping[str, int] = {
    "hour": 0,
    "day": 1,
    "week": 2,
    "month": 3,
    "quarter": 4,
    "year": 5,
}

AGGREGATIONS = frozenset({"additive", "semi_additive", "non_additive"})
METRIC_TYPES = frozenset({"simple", "derived"})
JOIN_TYPES = frozenset({"INNER", "LEFT", "RIGHT", "FULL", "CROSS"})
OPERATORS = frozenset({"=", "!=", ">", "<", ">=", "<=", "IN", "NOT IN", "LIKE", "BETWEEN"})


class MetricJoin(BaseModel):
    table: str
    source_column: str
    target_column: str
    join_type: str = "INNER"

    @field_validator("table")
    @classmethod
    def _qualified(cls, v: str) -> str:
        if not QUALIFIED_NAME_RE.match(v):
            raise ValueError(f"not a valid table name: {v!r}")
        return v

    @field_validator("source_column", "target_column")
    @classmethod
    def _identifier(cls, v: str) -> str:
        if not IDENTIFIER_RE.match(v):
            raise ValueError(f"not a valid column name: {v!r}")
        return v

    @field_validator("join_type")
    @classmethod
    def _known_join(cls, v: str) -> str:
        up = v.upper()
        if up not in JOIN_TYPES:
            raise ValueError(f"join_type must be one of {sorted(JOIN_TYPES)}, got {v!r}")
        return up


class MetricParameter(BaseModel):
    """A column a caller is permitted to filter on. Anything else is refused."""

    column: str
    operator: str = "="
    required: bool = False
    description: str = ""

    @field_validator("column")
    @classmethod
    def _identifier(cls, v: str) -> str:
        if not IDENTIFIER_RE.match(v):
            raise ValueError(f"not a valid column name: {v!r}")
        return v

    @field_validator("operator")
    @classmethod
    def _known_operator(cls, v: str) -> str:
        up = v.upper()
        if up not in OPERATORS:
            raise ValueError(f"operator must be one of {sorted(OPERATORS)}, got {v!r}")
        return up


class MetricDefinition(BaseModel):
    metric_id: str
    name: str
    """Becomes the output column alias, so it must be a plain identifier."""

    synonyms: list[str] = Field(default_factory=list)
    definition: str = ""
    type: str = "simple"
    expression: str
    """A scalar SQL aggregate (`SUM(billed_value)`), or for a derived metric an
    expression over its base metrics' output names (`billed_value / hours`)."""

    source_table: str = ""
    joins: list[MetricJoin] = Field(default_factory=list)
    base_metrics: list[str] = Field(default_factory=list)
    filters: list[str] = Field(default_factory=list)
    grain: list[str] = Field(default_factory=list)
    parameters: list[MetricParameter] = Field(default_factory=list)

    time_grains: list[str] = Field(default_factory=list)
    """The grains a caller may ask for. Empty means the metric is not time-governed;
    a non-empty list is a hard restriction, and the compiler will serve one of these
    or refuse."""

    time_grain_column: str = ""
    """The single column `time_grains` applies to. Empty falls back to the first
    temporal column in `grain`."""

    aggregation: str = "additive"
    """additive — safe to sum across every dimension (a flow, e.g. fees billed).
    semi_additive — additive except across time (a point-in-time snapshot like WIP;
      summing daily balances into a month double-counts).
    non_additive — never summable (ratios, averages, distinct counts); only correct
      when recomputed from base rows at the target grain."""

    value_type: str = "number"
    unit: str = ""
    format: str = ""
    owner: str = ""

    @field_validator("name")
    @classmethod
    def _name_is_identifier(cls, v: str) -> str:
        if not IDENTIFIER_RE.match(v):
            raise ValueError(f"metric name must be a plain SQL identifier, got {v!r}")
        return v

    @field_validator("source_table")
    @classmethod
    def _table_qualified(cls, v: str) -> str:
        if v and not QUALIFIED_NAME_RE.match(v):
            raise ValueError(f"not a valid table name: {v!r}")
        return v

    @field_validator("grain")
    @classmethod
    def _grain_identifiers(cls, v: list[str]) -> list[str]:
        for col in v:
            if not IDENTIFIER_RE.match(col):
                raise ValueError(f"grain column is not a valid identifier: {col!r}")
        return v

    @field_validator("time_grain_column")
    @classmethod
    def _axis_identifier(cls, v: str) -> str:
        if v and not IDENTIFIER_RE.match(v):
            raise ValueError(f"time_grain_column is not a valid identifier: {v!r}")
        return v

    @field_validator("time_grains")
    @classmethod
    def _known_grains(cls, v: list[str]) -> list[str]:
        out = []
        for g in v:
            low = g.lower()
            if low not in TIME_GRAIN_UNITS:
                raise ValueError(f"unsupported time grain {g!r}; known: {sorted(TIME_GRAIN_UNITS)}")
            out.append(low)
        return out

    @field_validator("aggregation")
    @classmethod
    def _known_aggregation(cls, v: str) -> str:
        low = v.lower()
        if low not in AGGREGATIONS:
            raise ValueError(f"aggregation must be one of {sorted(AGGREGATIONS)}, got {v!r}")
        return low

    @field_validator("type")
    @classmethod
    def _known_type(cls, v: str) -> str:
        low = v.lower()
        if low not in METRIC_TYPES:
            raise ValueError(f"type must be one of {sorted(METRIC_TYPES)}, got {v!r}")
        return low

    @model_validator(mode="after")
    def _shape_matches_type(self) -> MetricDefinition:
        if self.type == "derived":
            if not self.base_metrics:
                raise ValueError(f"derived metric {self.metric_id!r} has no base_metrics")
        elif not self.source_table:
            raise ValueError(f"simple metric {self.metric_id!r} has no source_table")
        return self

    @property
    def tables(self) -> list[str]:
        """Source table plus join targets, in the order they appear in the query."""
        return [self.source_table, *(j.table for j in self.joins)] if self.source_table else []


@dataclass(frozen=True)
class MetricRegistry:
    """Metric lookup for derived composition. Resolves by id, then by name."""

    metrics: Mapping[str, MetricDefinition]

    @classmethod
    def from_list(cls, metrics: Iterable[MetricDefinition]) -> MetricRegistry:
        return cls({m.metric_id: m for m in metrics})

    def get(self, metric_id: str) -> MetricDefinition | None:
        return self.metrics.get(metric_id)

    def resolve(self, ref: str) -> MetricDefinition | None:
        hit = self.metrics.get(ref)
        if hit is not None:
            return hit
        return next((m for m in self.metrics.values() if m.name == ref), None)

    def __len__(self) -> int:
        return len(self.metrics)

    def __iter__(self):
        return iter(self.metrics.values())


@dataclass(frozen=True)
class TableSchema:
    """What the compiler needs to know about a physical table.

    An empty `columns` means *unknown*, not *no columns* — Glue metadata is often
    absent, and the compiler stays permissive rather than refusing every metric on
    a table it cannot see.
    """

    full_name: str
    columns: Mapping[str, str]
    """column name -> data type, lowercased."""

    primary_keys: frozenset[str] = frozenset()
    """Empty means "no PK metadata", which is the common Glue case. Fan-out
    detection therefore only fires on positive evidence."""


class SchemaCatalog(Protocol):
    def table(self, full_name: str) -> TableSchema | None: ...


@dataclass(frozen=True)
class StaticCatalog:
    """In-memory `SchemaCatalog` — used by tests and by a graph-backed adapter."""

    tables: Mapping[str, TableSchema]

    @classmethod
    def from_dicts(
        cls,
        columns_by_table: Mapping[str, Mapping[str, str]],
        primary_keys: Mapping[str, Iterable[str]] | None = None,
    ) -> StaticCatalog:
        pks = primary_keys or {}
        return cls(
            {
                name: TableSchema(
                    full_name=name,
                    columns={c: t.lower() for c, t in cols.items()},
                    primary_keys=frozenset(pks.get(name, ())),
                )
                for name, cols in columns_by_table.items()
            }
        )

    def table(self, full_name: str) -> TableSchema | None:
        return self.tables.get(full_name)


@dataclass(frozen=True)
class FilterClause:
    """A caller-supplied predicate. The value is always escaped, never trusted."""

    column: str
    operator: str
    value: str | int | float | bool | list | tuple
