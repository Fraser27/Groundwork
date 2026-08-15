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
    ) -> None:
        self._static = {t.lower() for t in (allowed_tables or set())}
        self._provider = allowlist_provider
        self.allow_all = allow_all
        self._cache_ttl = cache_ttl
        self._cache: set[str] | None = None
        self._cache_ts = 0.0

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
        return ValidationResult(allowed=True, tables=sorted(set(seen)))

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
