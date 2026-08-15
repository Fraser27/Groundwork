"""Load and validate metric YAML.

Individually invalid metrics are skipped with a warning rather than aborting the
file: a metrics pack is edited by analysts, and one bad entry taking down every
governed metric would push people toward ungoverned SQL, which is the outcome this
whole layer exists to avoid.

Cross-metric problems are different. A derived metric pointing at a base that does
not exist, or a duplicate id, is a defect in the *pack* and is reported as such.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from pydantic import ValidationError

from src.metrics.models import MetricDefinition, MetricRegistry

logger = logging.getLogger(__name__)


@dataclass
class LoadResult:
    metrics: list[MetricDefinition] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    """Entries rejected. Non-empty does not mean nothing loaded."""

    @property
    def registry(self) -> MetricRegistry:
        return MetricRegistry.from_list(self.metrics)


def load_metrics(path: str | Path) -> LoadResult:
    p = Path(path)
    if not p.exists():
        return LoadResult(errors=[f"metrics file not found: {p}"])

    try:
        raw = yaml.safe_load(p.read_text()) or {}
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        where = f" at line {mark.line + 1}, column {mark.column + 1}" if mark else ""
        return LoadResult(errors=[f"malformed YAML in {p}{where}: {getattr(exc, 'problem', exc)}"])

    if not isinstance(raw, dict):
        return LoadResult(errors=[f"{p}: expected a mapping at the top level, got {type(raw).__name__}"])

    result = LoadResult()
    seen: set[str] = set()

    for idx, entry in enumerate(raw.get("metrics") or []):
        if not isinstance(entry, dict):
            result.errors.append(f"metric #{idx}: expected a mapping, got {type(entry).__name__}")
            continue
        entry_id = entry.get("metric_id")
        label = f"metric_id={entry_id!r}" if entry_id else f"metric #{idx}"

        if entry_id and entry_id in seen:
            result.errors.append(f"{label}: duplicate metric_id, keeping the first occurrence")
            continue

        try:
            metric = MetricDefinition(**entry)
        except (ValidationError, TypeError) as exc:
            result.errors.append(f"{label}: {_terse(exc)}")
            logger.warning("Skipping invalid metric (%s) in %s: %s", label, p, exc)
            continue

        seen.add(metric.metric_id)
        result.metrics.append(metric)

    result.errors.extend(_check_references(result.metrics))
    logger.info(
        "Loaded %d metrics from %s (%d rejected)", len(result.metrics), p, len(result.errors)
    )
    return result


def _check_references(metrics: list[MetricDefinition]) -> list[str]:
    """Derived metrics must resolve, and must not compose other derived metrics.

    The depth limit is deliberate: a two-level composition produces SQL nobody will
    read closely, and "the reviewer can follow the SQL" is a governance requirement,
    not a nicety.
    """
    by_ref: dict[str, MetricDefinition] = {}
    for m in metrics:
        by_ref[m.metric_id] = m
        by_ref.setdefault(m.name, m)

    errors: list[str] = []
    for m in metrics:
        if m.type != "derived":
            continue
        for ref in m.base_metrics:
            base = by_ref.get(ref)
            if base is None:
                errors.append(f"metric_id={m.metric_id!r}: unknown base metric {ref!r}")
            elif base.type == "derived":
                errors.append(
                    f"metric_id={m.metric_id!r}: base metric {ref!r} is itself derived; "
                    f"only one level of composition is supported"
                )
    return errors


def _terse(exc: Exception) -> str:
    """Pydantic's default rendering is multi-line; loader errors want one line each."""
    if isinstance(exc, ValidationError):
        return "; ".join(
            f"{'.'.join(str(x) for x in e['loc']) or '<root>'}: {e['msg']}" for e in exc.errors()
        )
    return str(exc)
