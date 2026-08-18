"""AST-level SQL allowlist. The last thing between a generated query and Athena.

Two decisions worth stating, because both are the opposite of the obvious one:

**AST, not regex.** Table references hide in CTE bodies, correlated subqueries,
UNION arms and lateral joins. `find_all(exp.Table)` recurses; a regex over the query
text does not, and the gap between those two is exactly where an exfiltration query
lives.

**Fail closed, always.** An unparseable query is denied, not passed through — if we
cannot see what a query touches we cannot claim it is safe. An *empty* allowlist
denies everything rather than allowing everything, so a failed catalog load
degrades to "no queries" instead of "all queries". `allow_all` is the only way to
disable enforcement, and it has to be typed out.

Two further rules, `require_aggregate` and `require_limit`, are off by default and on for
model-written SQL. They exist because the table allowlist is not a sufficient control there:
`SELECT * FROM matters` names a legitimate table, so the allowlist permits it, and Athena rows
carry no assertion or matter id, so `blocks.py` cannot screen them row-wise either. An aggregate
is the only shape that cannot hand back an individual walled matter's row. They are off for
compiled metric SQL because that was approved by a human who could see what it exposed, and
because a metric may legitimately return an unbounded list.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp

logger = logging.getLogger(__name__)

DIALECT = "trino"

#: Statement types allowed at all. A read layer has no business emitting anything
#: else, and an allowlisted table does not make DELETE acceptable.
_READ_ONLY_STATEMENTS = (exp.Select, exp.Union, exp.Except, exp.Intersect)

#: Rows a bounded statement may return when none was asked for. Large enough that an aggregate
#: over a real firm's matters is not silently clipped, small enough to cap an Athena scan.
DEFAULT_ROW_LIMIT = 1000


def _aggregates(select: exp.Select) -> bool:
    """Whether this SELECT reduces the rows it reads.

    `parent_select` is the load-bearing part: without it a subquery's SUM would vouch for the
    SELECT above it, and `SELECT name, (SELECT SUM(h) FROM te) FROM matters` still returns one
    row per matter.
    """
    if select.args.get("group"):
        return True
    return any(agg.parent_select is select for agg in select.find_all(exp.AggFunc))


def _row_wise_select(statement: exp.Expression, cte_names: set[str]) -> str:
    """The first table read row-wise with nothing aggregating it, or "" when none is.

    Checked per SELECT rather than on the statement, because the exposure is a SELECT that reads a
    real table and neither reduces those rows itself nor sits under something that does. A derived
    table is fine -- `SELECT COUNT(*) FROM (SELECT * FROM matters)` never emits the inner rows --
    so an enclosing aggregate satisfies the rule for everything beneath it.
    """
    for select in statement.find_all(exp.Select):
        sources = [
            t
            for t in select.find_all(exp.Table)
            if t.parent_select is select
            and not (not t.catalog and not t.db and t.name.lower() in cte_names)
        ]
        if not sources or _aggregates(select):
            continue
        enclosing = select.parent_select
        while enclosing is not None:
            if _aggregates(enclosing):
                break
            enclosing = enclosing.parent_select
        else:
            return sources[0].sql()
    return ""


@dataclass
class ValidationResult:
    allowed: bool
    denied_tables: list[str] = field(default_factory=list)
    reason: str = ""
    tables: list[str] = field(default_factory=list)
    """Every table the query touches, useful for audit even when allowed."""


class SQLFirewall:
    """Validates SQL against an allowlist of fully-qualified table names.

    The allowlist may be static (`allowed_tables`) or live (`allowlist_provider`), so
    tables discovered by a catalog scan *after* startup become queryable without a
    restart. Provider results are cached for `cache_ttl` seconds, and if the provider
    raises, the last good snapshot is reused — never widened, never emptied.
    """

    def __init__(
        self,
        allowed_tables: set[str] | None = None,
        *,
        allowlist_provider: Callable[[], set[str]] | None = None,
        allow_all: bool = False,
        cache_ttl: float = 30.0,
        require_aggregate: bool = False,
        require_limit: bool = False,
    ) -> None:
        self._static = {t.lower() for t in (allowed_tables or set())}
        self._provider = allowlist_provider
        self.allow_all = allow_all
        self._cache_ttl = cache_ttl
        self._cache: set[str] | None = None
        self._cache_ts = 0.0
        # Both off by default, so the compiled-metric path is unchanged. `SqlGenerator`'s firewall
        # turns them on: see the module docstring for why the allowlist alone is not enough there.
        self.require_aggregate = require_aggregate
        self.require_limit = require_limit

    @property
    def allowed_tables(self) -> set[str]:
        allowed = set(self._static)
        if self._provider is not None:
            now = time.monotonic()
            if self._cache is None or (now - self._cache_ts) >= self._cache_ttl:
                try:
                    self._cache = {t.lower() for t in self._provider() if t}
                    self._cache_ts = now
                except Exception as e:
                    logger.warning("firewall: allowlist provider failed, %s", e)
                    if self._cache is None:
                        self._cache = set()
            allowed |= self._cache
        return allowed

    def validate(self, sql: str) -> ValidationResult:
        if self.allow_all:
            return ValidationResult(allowed=True)

        allowed = self.allowed_tables

        try:
            statements = sqlglot.parse(sql, dialect=DIALECT)
        except sqlglot.errors.ParseError as e:
            return ValidationResult(allowed=False, reason=f"could not parse SQL: {e}")

        statements = [s for s in statements if s is not None]
        if not statements:
            return ValidationResult(allowed=False, reason="no statement found")
        if len(statements) > 1:
            return ValidationResult(
                allowed=False,
                reason=f"expected a single statement, got {len(statements)}",
            )

        statement = statements[0]
        if not isinstance(statement, _READ_ONLY_STATEMENTS):
            return ValidationResult(
                allowed=False,
                reason=f"only read statements are permitted, got {type(statement).__name__.upper()}",
            )

        # CTE names surface as exp.Table references. An unqualified reference matching
        # a CTE defined in this statement is an internal alias, not an external table.
        # Only the NAME is exempt — real tables inside the CTE body are still found by
        # find_all and validated normally.
        cte_names = {
            cte.alias_or_name.lower() for cte in statement.find_all(exp.CTE) if cte.alias_or_name
        }

        seen: list[str] = []
        denied: list[str] = []
        for table in statement.find_all(exp.Table):
            parts = [p for p in (table.catalog, table.db, table.name) if p]
            if not parts:
                continue
            qualified = ".".join(parts)
            if not table.catalog and not table.db and table.name.lower() in cte_names:
                continue
            seen.append(qualified)
            if not self._is_allowed(qualified.lower(), allowed):
                denied.append(qualified)

        if denied:
            unique = sorted(set(denied))
            return ValidationResult(
                allowed=False,
                denied_tables=unique,
                reason=f"unauthorized tables: {', '.join(unique)}",
                tables=sorted(set(seen)),
            )

        tables = sorted(set(seen))
        if self.require_aggregate and (row_wise := _row_wise_select(statement, cte_names)):
            return ValidationResult(
                allowed=False,
                reason=(
                    "only aggregate queries are permitted here: this one reads rows directly "
                    f"from {row_wise}. An aggregate cannot hand back an individual matter's row, "
                    "which is the only control available over warehouse rows -- they carry no "
                    "matter id, so the ethical wall cannot screen them."
                ),
                tables=tables,
            )
        if self.require_limit and statement.args.get("limit") is None:
            return ValidationResult(
                allowed=False,
                reason=(
                    f"a LIMIT is required here; add LIMIT {DEFAULT_ROW_LIMIT} or fewer. Athena "
                    "is billed by bytes scanned and an unbounded query has no cost ceiling."
                ),
                tables=tables,
            )
        return ValidationResult(allowed=True, tables=tables)

    @staticmethod
    def _is_allowed(qualified: str, allowed: set[str]) -> bool:
        """Match exactly, or by suffix when one side omits the database.

        Athena resolves an unqualified name against the session database, so
        `matters` and `legal_ops.matters` can be the same table. Suffix matching keeps
        the firewall from rejecting legitimate queries; it only ever matches on a full
        dotted segment, so `legal_ops.matters` never satisfies `secret_matters`.
        """
        if qualified in allowed:
            return True
        return any(a.endswith(f".{qualified}") or qualified.endswith(f".{a}") for a in allowed)
