"""Tier 1: match a question to a governed metric.

Matching is lexical and deterministic — name, synonyms, definition text. No model
decides which metric answers a question, because a model picking the wrong metric
produces a confidently wrong number that looks governed.

A match must be *unambiguous*. If two metrics score equally the tier declines and
lets the question fall through, on the grounds that an arbitrary tie-break is worse
than an honest miss: the user can rephrase, but cannot detect a silent wrong pick.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from src.metrics.compiler import compile_metric
from src.metrics.models import MetricDefinition, MetricRegistry, SchemaCatalog
from src.query.graph_reader import terms_of

logger = logging.getLogger(__name__)

#: Minimum weighted score. Name hits count double, so this is one name hit.
MIN_SCORE = 2

#: Minimum number of *distinct* question terms that must match, regardless of weight.
#: Without this, "which matters involve Acme Corporation" selected `open_matter_count`
#: on the single word "matters" — one shared noun is coincidence, not intent, and a
#: wrong metric answers with a confident number.
MIN_DISTINCT_TERMS = 2


@dataclass
class MetricMatch:
    """A matched metric, ready to compile."""

    metric: MetricDefinition
    score: int
    matched_on: list[str]
    catalog: SchemaCatalog
    registry: MetricRegistry
    time_grain: str | None = None

    warnings: list[str] = field(default_factory=list)
    """Populated by `compile()`. The compiler's governance warnings — fan-out risk,
    non-additive aggregation — belong in the answer, not just a log line."""

    def compile(self) -> str:
        result = compile_metric(
            self.metric,
            self.catalog,
            time_grain=self.time_grain,
            registry=self.registry,
        )
        if not result.is_valid:
            raise ValueError(f"{self.metric.metric_id} failed to compile: {result.errors}")
        self.warnings = list(result.warnings)
        return result.sql

    def run(self, sql: str) -> Any:
        """Execution is the executor's job; tier 1 stops at the SQL.

        Returning None rather than raising keeps `execute=True` honest: the SQL is
        real and reviewable, there is simply no Athena to run it against locally.
        """
        logger.info("metric %s compiled; no executor wired", self.metric.metric_id)
        return None


#: Time words in a question, mapped to the grain a metric may be sliced by. Checked
#: against the metric's own `time_grains`, so a metric that forbids daily reporting
#: cannot be coerced into it by phrasing.
_GRAIN_WORDS = {
    "daily": "day",
    "day": "day",
    "weekly": "week",
    "week": "week",
    "monthly": "month",
    "month": "month",
    "quarterly": "quarter",
    "quarter": "quarter",
    "yearly": "year",
    "annual": "year",
    "annually": "year",
    "year": "year",
}


class MetricMatcher:
    """Lexical matcher over a metric pack."""

    def __init__(
        self,
        metrics: list[MetricDefinition],
        catalog: SchemaCatalog,
    ) -> None:
        self._metrics = metrics
        self._catalog = catalog
        self._registry = MetricRegistry.from_list(metrics)

    @property
    def catalog(self) -> SchemaCatalog:
        """The schema the compiler validates against. Exposed so the compile endpoint uses
        the same catalog tier 1 does, rather than assembling a second one that could
        disagree about a column's type."""
        return self._catalog

    @property
    def metrics(self) -> list[MetricDefinition]:
        """The pack this matcher matches against. Read-only: the copy stops a caller
        listing metrics from also editing what tier 1 will serve."""
        return list(self._metrics)

    def _vocabulary(self, metric: MetricDefinition) -> tuple[set[str], set[str]]:
        """Split into what the metric *is called* and what its prose mentions.

        The split exists because a derived metric's definition describes its inputs:
        `realization_rate` is documented as "fees billed over standard value", so a
        flat bag of words made it tie with `fees_billed` on the question "fees billed
        by month" — and the tie made the matcher decline both. Naming beats mention.
        """
        names: set[str] = set()
        names.update(terms_of(metric.name))
        names.update(terms_of(metric.metric_id.replace("_", " ")))
        for syn in getattr(metric, "synonyms", ()) or ():
            names.update(terms_of(syn))

        prose: set[str] = set()
        if getattr(metric, "definition", None):
            prose.update(terms_of(metric.definition))
        return names, prose - names

    def match(self, question: str) -> MetricMatch | None:
        terms = terms_of(question)
        if not terms:
            return None

        # Weighted: a name or synonym hit counts double a definition-prose hit, so
        # the metric a question names outranks one that merely mentions it.
        scored: list[tuple[int, list[str], MetricDefinition]] = []
        for metric in self._metrics:
            names, prose = self._vocabulary(metric)
            name_hits = [t for t in terms if t in names]
            prose_hits = [t for t in terms if t in prose]
            matched = name_hits + prose_hits
            weight = 2 * len(name_hits) + len(prose_hits)
            if weight >= MIN_SCORE and len(set(matched)) >= MIN_DISTINCT_TERMS:
                scored.append((weight, matched, metric))

        if not scored:
            return None

        scored.sort(key=lambda row: -row[0])
        if len(scored) > 1 and scored[0][0] == scored[1][0]:
            # Ambiguous. Declining is the safe failure: a wrong metric returns a
            # number that looks authoritative.
            logger.info(
                "declining ambiguous metric match: %s tied at %d",
                [m.metric_id for _, _, m in scored[:3]],
                scored[0][0],
            )
            return None

        score, matched, metric = scored[0]
        return MetricMatch(
            metric=metric,
            score=score,
            matched_on=matched,
            catalog=self._catalog,
            registry=self._registry,
            time_grain=self._grain_for(metric, terms),
        )

    def _grain_for(self, metric: MetricDefinition, terms: list[str]) -> str | None:
        allowed = set(getattr(metric, "time_grains", ()) or ())
        for term in terms:
            grain = _GRAIN_WORDS.get(term)
            if grain and grain in allowed:
                return grain
        return None
