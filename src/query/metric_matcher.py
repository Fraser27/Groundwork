"""Tier 1: match a question to a governed metric.

The primary path is lexical and deterministic — name, synonyms, definition text — and it keeps
priority, because an exact word is stronger evidence of intent than cosine proximity.

A match must be *unambiguous*. If two metrics score equally the tier declines and
lets the question fall through, on the grounds that an arbitrary tie-break is worse
than an honest miss: the user can rephrase, but cannot detect a silent wrong pick.

**When keywords find nothing, a routing candidate may be used instead, and it is labelled
differently.** "Whats the money charged for a matter" shares no word with `fees_billed`'s
vocabulary and so reached no metric at all, which silently downgraded a governed question to an
ungoverned answer. The tier router already scores a metric layer for exactly this question, so its
hits become candidates here.

That path is honest about what it is. Embedding the question is a model call, so the *choice* of
metric is not reproducible even though the SQL is still compiled from an approved definition with
no model in the path. `MetricMatch.selected_by` carries the difference and every caller reports it,
because a number labelled fully deterministic when a model chose which number to compute is the
overstatement this whole module exists to avoid.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from src.graph.scope import AuthContext
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

SELECTED_BY_KEYWORD = "keyword"
SELECTED_BY_ROUTER = "router"

#: How similar a routing description must be before it may *select* a metric.
#:
#: Deliberately well above `router_min_similarity`, which decides something much cheaper: whether
#: tier 1 is worth trying at all. Being wrong there costs a lane that finds nothing; being wrong
#: here returns a number that looks authoritative. In production the metric layer scored 0.55 on
#: the question this path exists for, so this admits that and little below it.
MIN_CANDIDATE_SIMILARITY = 0.45

#: How far ahead of the runner-up the winner must be. Same reasoning as the keyword tie-break --
#: an arbitrary pick between two plausible metrics is worse than an honest miss -- and an absolute
#: gap rather than a ratio because cosines compress near the top of the range.
MIN_CANDIDATE_MARGIN = 0.05

#: Candidates to pull. Two would be enough for the margin; a few more so a stale record naming a
#: deprecated metric cannot crowd out the live ones.
CANDIDATE_TOP_K = 5


@dataclass(frozen=True)
class MetricCandidate:
    """A metric the routing index thinks the question is about.

    A *candidate*, never a selection: the thresholds in this module decide, and a candidate whose
    id this matcher's pack does not hold is dropped rather than trusted, because a routing record
    outlives the metric it describes.
    """

    metric_id: str
    similarity: float
    label: str = ""


@dataclass
class MetricMatch:
    """A matched metric, ready to compile."""

    metric: MetricDefinition
    score: int
    matched_on: list[str]
    catalog: SchemaCatalog
    registry: MetricRegistry
    time_grain: str | None = None
    selected_by: str = SELECTED_BY_KEYWORD
    """How this metric was chosen. `router` means an embedding model narrowed it, so the SQL is
    still compiled deterministically but the *choice* is not reproducible from the question alone.
    Every caller reports this rather than collapsing it into "governed"."""

    similarity: float | None = None
    """Cosine that selected it, on the router path. None on the keyword path, where `score` and
    `matched_on` are the explanation."""

    _executor: Any | None = None
    """Injected by the matcher. Absent means the SQL is returned unrun."""

    warnings: list[str] = field(default_factory=list)
    """Populated by `compile()`. The compiler's governance warnings — fan-out risk,
    non-additive aggregation — belong in the answer, not just a log line."""

    @property
    def selected_deterministically(self) -> bool:
        """Whether the *choice* of metric is reproducible from the question alone.

        Distinct from whether the SQL is deterministic, which it always is. Collapsing the two
        would let an answer a model steered inherit the guarantee of one it did not."""
        return self.selected_by == SELECTED_BY_KEYWORD

    @property
    def is_runnable(self) -> bool:
        """Whether there is a warehouse behind this match at all.

        Read rather than inferred from a `None` answer. A *failed* query returns None too, so the
        caller that inferred reported a 403 on the lake as "no query engine is configured" and threw
        away the error naming the permission.
        """
        return self._executor is not None

    @property
    def selection_note(self) -> str:
        """One sentence saying how this metric was reached, for the answer rather than the log."""
        if self.selected_deterministically:
            return (
                f"Matched the approved metric '{self.metric.name}' on the words "
                f"{', '.join(sorted(set(self.matched_on)))}."
            )
        similarity = f"{self.similarity:.2f}" if self.similarity is not None else "unknown"
        return (
            f"No approved metric shares a word with this question, so '{self.metric.name}' was "
            f"chosen by similarity to its description ({similarity}). The SQL below is still "
            "compiled from the approved definition with no AI involved, but the choice of metric "
            "was not, so a differently worded question could reach a different metric."
        )

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
        """Run the compiled SQL, or return None when there is no warehouse to run it against.

        None rather than raising keeps `execute=True` honest locally: the SQL is real and
        reviewable, there is simply nothing to execute it. For most of this project's life that
        was the *only* path -- no executor was ever constructed -- so a governed metric compiled
        correctly and never returned a figure.

        A failed query returns its result object rather than None, because "the warehouse refused
        this" and "there is no warehouse" are different answers and the first one names a
        column or a permission the reader can go and fix.
        """
        if self._executor is None:
            logger.info("metric %s compiled; no executor wired", self.metric.metric_id)
            return None

        result = self._executor.execute(sql)
        if not result.success:
            logger.warning(
                "metric %s failed to run (%s): %s",
                self.metric.metric_id,
                result.error_code,
                result.error,
            )
            self.warnings.append(
                f"The metric compiled but the query did not run ({result.error_code}): "
                f"{result.error}"
            )
            return None
        if result.truncated:
            # A silently truncated aggregate is a wrong number, not a partial one.
            self.warnings.append(
                f"Only the first {result.row_count} rows were returned, so any total below is "
                "computed over a prefix rather than the whole result."
            )
        return {"columns": result.columns, "rows": result.rows}


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
    """Lexical matcher over a metric pack, with a similarity fallback when no word matches."""

    def __init__(
        self,
        metrics: list[MetricDefinition],
        catalog: SchemaCatalog,
        executor: Any | None = None,
        *,
        candidate_source: Any | None = None,
    ) -> None:
        self._metrics = metrics
        self._catalog = catalog
        self._registry = MetricRegistry.from_list(metrics)
        # Optional. Without it a metric still compiles and returns its SQL, which is the
        # reviewable half; the number is what needs a warehouse to reach.
        self._executor = executor
        # Anything with `metric_candidates(ctx, question, top_k=...)`; in practice the tier router.
        # Absent is today's behaviour: keyword or nothing.
        self._candidate_source = candidate_source

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

    def match(
        self,
        question: str,
        ctx: AuthContext | None = None,
        settings: Any | None = None,
    ) -> MetricMatch | None:
        """The keyword match, or a routing candidate when no word matched.

        `ctx` and `settings` are optional so every existing caller keeps working: without them
        there is no similarity fallback, which is exactly the keyword-only behaviour of a
        deployment with no vector store.
        """
        terms = terms_of(question)
        if not terms:
            return None

        scored = self._scored(terms)
        if len(scored) > 1 and scored[0][0] == scored[1][0]:
            # Ambiguous. Declining is the safe failure: a wrong metric returns a
            # number that looks authoritative.
            #
            # And the router is deliberately *not* consulted here. A tie means two metrics are
            # each lexically plausible, so letting a cosine break it is the arbitrary tie-break
            # this refuses to make -- the fallback exists for questions no metric words match.
            logger.info(
                "declining ambiguous metric match: %s tied at %d",
                [m.metric_id for _, _, m in scored[:3]],
                scored[0][0],
            )
            return None

        if not scored:
            return self._by_similarity(question, terms, ctx, settings)

        score, matched, metric = scored[0]
        return MetricMatch(
            metric=metric,
            score=score,
            matched_on=matched,
            catalog=self._catalog,
            registry=self._registry,
            time_grain=self._grain_for(metric, terms),
            _executor=self._executor,
        )

    def _scored(self, terms: list[str]) -> list[tuple[int, list[str], MetricDefinition]]:
        """Metrics clearing both floors, best first.

        Weighted: a name or synonym hit counts double a definition-prose hit, so the metric a
        question names outranks one that merely mentions it.
        """
        scored: list[tuple[int, list[str], MetricDefinition]] = []
        for metric in self._metrics:
            names, prose = self._vocabulary(metric)
            name_hits = [t for t in terms if t in names]
            prose_hits = [t for t in terms if t in prose]
            matched = name_hits + prose_hits
            weight = 2 * len(name_hits) + len(prose_hits)
            if weight >= MIN_SCORE and len(set(matched)) >= MIN_DISTINCT_TERMS:
                scored.append((weight, matched, metric))
        scored.sort(key=lambda row: -row[0])
        return scored

    def _by_similarity(
        self,
        question: str,
        terms: list[str],
        ctx: AuthContext | None,
        settings: Any | None,
    ) -> MetricMatch | None:
        """A metric no word matched, reached by how close the question is to its description.

        The recall this recovers is real -- "whats the money charged for a matter" shares nothing
        with `fees_billed`'s vocabulary -- and so is the cost: the choice is a model's, so the
        match says so and the thresholds here are stricter than the router's own.
        """
        if self._candidate_source is None or ctx is None:
            return None
        if settings is not None and not getattr(settings, "router_enabled", True):
            # An administrator who turned routing off has asked for keyword matching, and a
            # switch that still let similarity pick the metric would be decorative.
            return None

        try:
            candidates = self._candidate_source.metric_candidates(
                ctx, question, top_k=CANDIDATE_TOP_K
            )
        except Exception as e:  # noqa: BLE001
            # A miss, never an error: tier 1 finding nothing is a fall-through, and a failed
            # similarity search must not be worse than not having one.
            logger.warning("metric candidates unavailable: %s", e)
            return None

        known = {m.metric_id: m for m in self._metrics}
        # Filtered against the pack before ranking. A routing record outlives the metric it
        # describes, and a stale one winning would select a metric this matcher cannot compile.
        ranked = sorted(
            (c for c in candidates if c.metric_id in known),
            key=lambda c: -c.similarity,
        )
        if not ranked:
            return None

        top = ranked[0]
        if top.similarity < MIN_CANDIDATE_SIMILARITY:
            logger.info(
                "declining metric candidate %s: %.3f is below the %.2f selection floor",
                top.metric_id,
                top.similarity,
                MIN_CANDIDATE_SIMILARITY,
            )
            return None
        if len(ranked) > 1 and top.similarity - ranked[1].similarity < MIN_CANDIDATE_MARGIN:
            logger.info(
                "declining ambiguous metric candidates: %s at %.3f vs %s at %.3f",
                top.metric_id,
                top.similarity,
                ranked[1].metric_id,
                ranked[1].similarity,
            )
            return None

        metric = known[top.metric_id]
        logger.info(
            "metric %s selected by similarity %.3f, not by keyword", metric.metric_id, top.similarity
        )
        return MetricMatch(
            metric=metric,
            score=0,
            matched_on=[],
            catalog=self._catalog,
            registry=self._registry,
            # Still lexical, and still a hard restriction the metric's own `time_grains` gates.
            time_grain=self._grain_for(metric, terms),
            selected_by=SELECTED_BY_ROUTER,
            similarity=top.similarity,
            _executor=self._executor,
        )

    def _grain_for(self, metric: MetricDefinition, terms: list[str]) -> str | None:
        allowed = set(getattr(metric, "time_grains", ()) or ())
        for term in terms:
            grain = _GRAIN_WORDS.get(term)
            if grain and grain in allowed:
                return grain
        return None


def match_metric(
    matcher: Any,
    question: str,
    ctx: AuthContext | None = None,
    settings: Any | None = None,
) -> MetricMatch | None:
    """Ask a matcher for a metric, passing scope only if it can take it.

    Both query endpoints call this rather than `matcher.match` directly. They have disagreed about
    whether a question is governed before, and one of them forgetting to thread `ctx` would mean
    `/query` reached tier 1 by similarity while `/query/compose` did not -- a difference in a
    governance claim caused by an argument list.

    The signature check exists because a matcher is injected and need not be `MetricMatcher`:
    tests and the MCP server pass their own. Inspected rather than caught as a `TypeError`, which
    would swallow a real `TypeError` raised inside `match` and report it as a miss.
    """
    call = getattr(matcher, "match", None)
    if call is None:
        return None
    if ctx is not None and _accepts_scope(call):
        return call(question, ctx, settings)
    return call(question)


def _accepts_scope(call: Any) -> bool:
    import inspect

    try:
        inspect.signature(call).bind("q", None, None)
    except (TypeError, ValueError):
        return False
    return True


def chosen_deterministically(match: Any) -> bool:
    """Whether the choice of metric is reproducible from the question alone.

    Read through here rather than off the attribute, because a matcher is injected and need not be
    `MetricMatcher`. Absence means keyword: a matcher that knows nothing of a router did not use
    one, so treating the attribute's absence as non-deterministic would understate every
    keyword match made by an injected matcher.
    """
    return bool(getattr(match, "selected_deterministically", True))


def selection_of(match: MetricMatch | Any) -> dict[str, Any]:
    """How a metric was chosen, as the API reports it.

    One shape from one place, because two endpoints describing the same selection differently is
    how a caller comes to believe one of them.
    """
    deterministic = chosen_deterministically(match)
    return {
        "metric_id": getattr(getattr(match, "metric", None), "metric_id", ""),
        "selected_by": getattr(
            match, "selected_by", SELECTED_BY_KEYWORD if deterministic else SELECTED_BY_ROUTER
        ),
        "deterministic": deterministic,
        "similarity": getattr(match, "similarity", None),
        "matched_on": list(getattr(match, "matched_on", ()) or ()),
        "note": getattr(match, "selection_note", ""),
    }
