"""How a set of similarity scores becomes a decision about which tiers to run.

Deliberately pure: numbers in, numbers out. No OpenSearch, no `AuthContext`, no settings
object. This is the part of the router most likely to be wrong and most likely to be retuned
once real questions have been through it, so it is replaceable without touching the router and
testable without a cluster.

Two things here are easy to get wrong and expensive to notice.

**A raw OpenSearch score is not a cosine.** With `cosinesimil` the score is `1 / (2 - cos)`,
which ranges about 0.33 to 1.0. An administrator who sets a threshold of 0.8 believing it is a
cosine actually gets 0.75. Every number is converted before it is scored, compared or shown.

**Similarity is not calibrated.** There is no value of cosine that means "relevant" across
questions of different lengths and phrasings, so the selection rule is *relative*: a layer is
searched when it scores close enough to the best layer. The absolute floor exists only to
answer a different question -- did anything look relevant at all -- because when the answer is
no, the honest response is to try everything rather than to pick the least bad option.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

#: How many of a layer's best hits are averaged into its score.
#:
#: One (i.e. max) is brittle: a single lucky item promotes a whole layer. Averaging everything
#: is worse in the other direction, because a layer with one excellent match and a long
#: mediocre tail scores below a layer of uniformly middling hits. Two asks for corroboration
#: without demanding breadth, which is the right trade when a layer may legitimately hold only
#: one relevant thing -- a tenant with a single governed metric, say.
DEFAULT_TOP_N = 2


def cosine_of(raw_score: float) -> float:
    """Recover the cosine from an OpenSearch `cosinesimil` score.

    `_score = 1 / (2 - cos)`, so `cos = 2 - 1/_score`. Guarded against a zero or negative
    score, which the engine does not produce but a fake client in a test might.
    """
    if raw_score <= 0:
        return -1.0
    return 2.0 - 1.0 / raw_score


@dataclass(frozen=True)
class LayerScore:
    """One layer's score and, in words, why it was selected or dropped.

    `reason` is not decoration. This ends up in a diagram an auditor expands, and a number with
    no explanation beside it is not an explanation -- "0.31" says nothing, "0.31 is below the
    0.46 needed to stay within the margin of the best layer" says what to change.
    """

    kind: str
    score: float
    raw_score: float
    """Before the boost, so the trace can show what the boost did rather than just its effect."""

    boost: float
    hit_count: int
    """Reported, and deliberately not part of the score: forty mediocre hits must not outrank
    two good ones."""

    relative: float
    selected: bool
    reason: str


def score_layers(
    hits: Mapping[str, Sequence[float]],
    *,
    min_similarity: float,
    margin: float,
    boosts: Mapping[str, float] | None = None,
    top_n: int = DEFAULT_TOP_N,
) -> list[LayerScore]:
    """Score each layer and decide which to search.

    `hits` maps a layer kind to its cosines. Order does not matter -- they are sorted here, so a
    caller cannot change the outcome by forgetting to sort.

    `boosts` is a mapping rather than a single argument for governed metrics, so a future domain
    pack can favour a different layer without a signature change. A boost is additive so the
    trace can read "0.66 +0.05 governed-metric boost"; a multiplier would be unreadable at a
    glance and would compress differences near the top of the range.

    Returns every layer that was asked about, including the dropped ones. A caller needs the
    losers to explain the decision, and dropping them here would make the trace a record of the
    outcome rather than of the reasoning.
    """
    boosts = boosts or {}
    scored: list[tuple[str, float, float, float, int]] = []

    for kind, values in hits.items():
        above = sorted((v for v in values if v >= min_similarity), reverse=True)
        if not above:
            scored.append((kind, 0.0, 0.0, boosts.get(kind, 0.0), 0))
            continue
        head = above[: max(1, top_n)]
        raw = sum(head) / len(head)
        boost = boosts.get(kind, 0.0)
        # Clamped: a boost that pushed a layer past 1.0 would make `relative` exceed 1 for
        # something other than the best layer, and the margin arithmetic stops meaning anything.
        scored.append((kind, min(1.0, raw + boost), raw, boost, len(above)))

    best = max((s for _, s, _, _, _ in scored), default=0.0)
    # `1 - margin` of nothing is still nothing. With no layer above the floor every layer is
    # reported as dropped, and the router reads that as "nothing looked relevant" and degrades
    # to trying everything -- which is why this returns cleanly rather than raising.
    cutoff = best * (1.0 - margin) if best > 0 else 0.0

    out: list[LayerScore] = []
    for kind, score, raw, boost, count in scored:
        relative = score / best if best > 0 else 0.0
        selected = score > 0 and score >= cutoff
        out.append(
            LayerScore(
                kind=kind,
                score=round(score, 4),
                raw_score=round(raw, 4),
                boost=boost,
                hit_count=count,
                relative=round(relative, 4),
                selected=selected,
                reason=_reason(
                    kind=kind,
                    score=score,
                    raw=raw,
                    boost=boost,
                    count=count,
                    best=best,
                    cutoff=cutoff,
                    selected=selected,
                    min_similarity=min_similarity,
                ),
            )
        )
    out.sort(key=lambda layer: -layer.score)
    return out


def _reason(
    *,
    kind: str,
    score: float,
    raw: float,
    boost: float,
    count: int,
    best: float,
    cutoff: float,
    selected: bool,
    min_similarity: float,
) -> str:
    """One sentence an auditor can act on, naming the number that decided it."""
    if count == 0:
        return f"nothing scored above the {min_similarity:.2f} similarity floor"

    boosted = f" ({raw:.2f} +{boost:.2f} boost)" if boost else ""
    hits = f"{count} hit" if count == 1 else f"{count} hits"

    if not selected:
        return f"{score:.2f}{boosted} from {hits} is below the {cutoff:.2f} margin cutoff"
    if score >= best:
        return f"best match at {score:.2f}{boosted} from {hits}"
    return f"{score:.2f}{boosted} from {hits} is within the {cutoff:.2f} margin cutoff"
