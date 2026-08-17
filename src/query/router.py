"""Which tiers a question is worth asking, decided by similarity rather than by keyword.

The resolver walks tiers in order and takes the first that answers. That is cheap and it is also
how a governed metric gets missed: `MetricMatcher` needs a keyword overlap, so "what did we
invoice last quarter" misses `fees_billed` unless someone has already added "invoiced" to a
synonym list by hand. The question then falls through to tier 4, where a model writes the SQL. A
maintenance gap silently downgrades governance, which is the wrong direction for a failure.

So the question is embedded once and matched against a routing index of the things it could be
about, and only the layers that scored well are searched. **Recall is what this widens; nothing
here decides an answer.** A metric still has to be chosen by the deterministic matcher and
compiled with no model in the path, so the SQL remains exactly as reproducible as before.

Three things this module is careful about, in descending order of how much they would cost:

**A tier the tenant has forbidden is never queried.** Resolved before the search, not filtered
out of the results. Filtering afterwards would put a forbidden layer's items in the trace -- which
is a disclosure of data the tenant told the system not to use -- and would let that layer's score
raise `best_score`, narrowing the margin and dropping a *permitted* layer that would have
answered.

**A routing hit for an entity is a subject name.** The trace is expandable, so the matter wall has
to apply to the routing search itself, before anything reaches the trace. Screening the answer but
not the router would leak a screened party's name in the step above the wall.

**Nothing routes when routing fails.** An empty index, no hit above the floor, an unreachable
collection -- every one of those degrades to the resolver's own ordering rather than to an empty
answer. A router is an optimisation, and an optimisation that can refuse to answer is a liability.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from src.governance import KNOWN_TIERS, GovernanceSettings
from src.graph.scope import AuthContext
from src.query.router_index import KIND_ENTITY, KIND_METRIC, KIND_TABLE
from src.query.router_scoring import LayerScore, cosine_of, score_layers

logger = logging.getLogger(__name__)

#: The passage layer, scored from the chunk index rather than the routing index.
#:
#: Nothing in the routing index can score tier 3: passages live in the document index, and a
#: summary of them would be a second copy of text that is already embedded. So the same query
#: vector probes both, and this is the kind the chunk hits are filed under.
KIND_PASSAGES = "passages"

#: Which tiers a layer justifies running.
#:
#: `table` maps to both 2 and 3 because a catalogued table is reachable two ways: the graph holds
#: its schema as DECLARED facts, and tier 3 reads that schema alongside passages. It used to map
#: to 4, which was the tier where a model wrote SQL -- retired, because it was never built.
_LAYER_TIERS: dict[str, tuple[int, ...]] = {
    KIND_METRIC: (1,),
    KIND_ENTITY: (2,),
    KIND_TABLE: (2, 3),
    KIND_PASSAGES: (3,),
}

#: How many items a layer contributes to the trace. Enough to see why a layer won; not so many
#: that the expandable step becomes a wall of near-identical rows.
TRACE_ITEMS = 5

#: How many hits to pull per probe. Above the scorer's top-2 so the floor has something to reject,
#: and bounded because a routing search is on the latency path of every question.
SEARCH_TOP_K = 10


@dataclass
class RouterItem:
    """One matched item, as the trace shows it."""

    kind: str
    item_id: str
    label: str
    similarity: float
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "item_id": self.item_id,
            "label": self.label,
            "similarity": round(self.similarity, 4),
            "detail": self.detail,
        }


@dataclass
class RouterDecision:
    """Which tiers to run, and the reasoning, in a form a reader can expand.

    `degraded` is the honest signal that the router did not decide. A degraded decision carries
    every permitted tier, so the answer is the resolver's own and the trace says why.
    """

    tiers: list[int]
    layers: list[LayerScore] = field(default_factory=list)
    items: dict[str, list[RouterItem]] = field(default_factory=dict)
    dropped: dict[str, str] = field(default_factory=dict)
    forbidden: set[str] = field(default_factory=set)
    """Tiers the tenant does not permit, as a set rather than left to be inferred from the wording
    of `dropped`. "Your administrator turned this off" and "this did not look relevant" are
    different facts about the system, and a reader has to be able to tell them apart -- so the
    distinction is data, not a phrase a UI has to pattern-match."""

    degraded: bool = False
    reason: str | None = None
    enabled: bool = True
    best_score: float = 0.0
    margin: float = 0.0
    min_similarity: float = 0.0
    metric_boost: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "degraded": self.degraded,
            "reason": self.reason,
            "margin": self.margin,
            "min_similarity": self.min_similarity,
            "metric_boost": self.metric_boost,
            "best_score": round(self.best_score, 4),
            "layers": [
                {
                    "kind": layer.kind,
                    # A layer maps to one or two tiers; the first is the one a reader thinks of it
                    # as, and `tiers` carries the rest so the mapping is not hidden.
                    "tier": _LAYER_TIERS.get(layer.kind, ())[0]
                    if _LAYER_TIERS.get(layer.kind)
                    else None,
                    "tiers": list(_LAYER_TIERS.get(layer.kind, ())),
                    "score": layer.score,
                    "raw_score": layer.raw_score,
                    "boost": layer.boost,
                    "hit_count": layer.hit_count,
                    "relative": layer.relative,
                    "selected": layer.selected,
                    "reason": layer.reason,
                    "items": [i.to_dict() for i in self.items.get(layer.kind, [])],
                }
                for layer in self.layers
            ],
            "tiers_selected": list(self.tiers),
            "tiers_dropped": dict(self.dropped),
            "tiers_forbidden": sorted(self.forbidden),
        }


class TierRouter:
    """Scores the layers and returns the tiers worth running.

    Every dependency is injected and each is optional, because a deployment without a vector store
    must still answer questions -- it simply answers them the way it did before this existed.
    """

    def __init__(
        self,
        *,
        routing_index: Any | None = None,
        chunk_search: Any | None = None,
        embedder: Any | None = None,
    ) -> None:
        self._routing = routing_index
        self._chunks = chunk_search
        self._embedder = embedder

    def route(
        self, ctx: AuthContext, question: str, settings: GovernanceSettings
    ) -> RouterDecision:
        """Decide which tiers to run. Never raises."""
        permitted = sorted(int(t) for t in settings.allowed_tiers)
        # Over the tiers that exist, not a literal range. A hardcoded 4 here reported retired
        # tier 4 as "not permitted for this tenant" on every single question, which reads as an
        # administrator's decision rather than as a capability that was never built.
        forbidden = {
            str(t): f"tier {t} is not permitted for this tenant"
            for t in sorted(KNOWN_TIERS)
            if t not in permitted
        }
        base = RouterDecision(
            tiers=permitted,
            dropped=dict(forbidden),
            forbidden=set(forbidden),
            enabled=bool(getattr(settings, "router_enabled", True)),
            margin=float(getattr(settings, "router_margin", 0.0)),
            min_similarity=float(getattr(settings, "router_min_similarity", 0.0)),
            metric_boost=float(getattr(settings, "router_metric_boost", 0.0)),
        )

        if not base.enabled:
            base.degraded = True
            base.reason = "routing is turned off, so every permitted tier was tried in order"
            return base
        if self._routing is None or self._embedder is None:
            base.degraded = True
            base.reason = "no vector store is configured, so every permitted tier was tried"
            return base

        # A layer whose only tiers are forbidden is not searched at all. This is the ordering the
        # whole disclosure argument rests on: not queried, rather than queried and discarded.
        kinds = {
            kind
            for kind, tiers in _LAYER_TIERS.items()
            if any(t in permitted for t in tiers) and kind != KIND_PASSAGES
        }
        probe_passages = any(t in permitted for t in _LAYER_TIERS[KIND_PASSAGES])
        if not kinds and not probe_passages:
            base.degraded = True
            base.reason = "no tier is permitted for this tenant"
            return base

        try:
            vector = self._embedder.embed_query(question)
        except Exception as e:
            logger.warning("router could not embed the question: %s", e)
            base.degraded = True
            base.reason = f"the question could not be embedded ({e}), so every tier was tried"
            return base

        cosines: dict[str, list[float]] = {}
        items: dict[str, list[RouterItem]] = {}

        if kinds:
            self._probe_routing(ctx, vector, kinds, cosines, items)
        if probe_passages:
            self._probe_passages(ctx, vector, cosines, items)

        if not any(cosines.values()):
            base.degraded = True
            base.reason = "nothing in the index resembled the question, so every tier was tried"
            base.items = items
            return base

        layers = score_layers(
            cosines,
            min_similarity=base.min_similarity,
            margin=base.margin,
            boosts={KIND_METRIC: base.metric_boost},
        )

        chosen: set[int] = set()
        dropped = dict(forbidden)
        for layer in layers:
            tiers = [t for t in _LAYER_TIERS.get(layer.kind, ()) if t in permitted]
            if layer.selected:
                chosen.update(tiers)
            else:
                for t in tiers:
                    # Only if no selected layer also justifies it. A table layer losing must not
                    # drop tier 2 when the entity layer won it.
                    dropped.setdefault(str(t), f"{layer.kind}: {layer.reason}")

        # A tier some layer selected is never reported as dropped, whatever another layer said.
        for t in chosen:
            dropped.pop(str(t), None)

        if not chosen:
            base.degraded = True
            base.reason = "no layer cleared the margin, so every permitted tier was tried"
            base.layers = layers
            base.items = items
            return base

        # Re-applied rather than trusted. Step one excluded these already, so this is the assertion
        # that the mapping above did not reintroduce one.
        tiers = sorted(t for t in chosen if t in permitted)
        return RouterDecision(
            tiers=tiers,
            layers=layers,
            items=items,
            dropped=dropped,
            forbidden=set(forbidden),
            degraded=False,
            reason=None,
            enabled=True,
            best_score=max((layer.score for layer in layers), default=0.0),
            margin=base.margin,
            min_similarity=base.min_similarity,
            metric_boost=base.metric_boost,
        )

    def _probe_routing(
        self,
        ctx: AuthContext,
        vector: Sequence[float],
        kinds: set[str],
        cosines: dict[str, list[float]],
        items: dict[str, list[RouterItem]],
    ) -> None:
        """Metrics, entities and tables, in one search over one index."""
        from src.query.router_index import routing_index_name

        try:
            hits = self._routing.search(
                routing_index_name(ctx),
                vector,
                top_k=SEARCH_TOP_K,
                kinds=frozenset(kinds),
                matter_allowlist=ctx.matter_allowlist,
                matter_denylist=ctx.matter_denylist,
            )
        except Exception as e:
            # One probe failing is not the whole decision failing: the other layer may still
            # decide, and `cosines` simply has no entry for these kinds.
            logger.warning("router could not search the routing index: %s", e)
            return

        for hit in hits:
            kind = hit.record.kind
            cos = cosine_of(hit.raw_score)
            cosines.setdefault(kind, []).append(cos)
            bucket = items.setdefault(kind, [])
            if len(bucket) < TRACE_ITEMS:
                bucket.append(
                    RouterItem(
                        kind=kind,
                        item_id=hit.record.item_id,
                        label=hit.record.label,
                        similarity=cos,
                        detail=dict(hit.record.detail or {}),
                    )
                )

    def _probe_passages(
        self,
        ctx: AuthContext,
        vector: Sequence[float],
        cosines: dict[str, list[float]],
        items: dict[str, list[RouterItem]],
    ) -> None:
        """Tier 3 is scored from real chunk hits, because that is where passages are."""
        if self._chunks is None:
            return
        from src.documents.embed import index_name

        try:
            hits = self._chunks.search(
                index_name(ctx),
                vector,
                top_k=SEARCH_TOP_K,
                matter_allowlist=ctx.matter_allowlist,
                matter_denylist=ctx.matter_denylist,
            )
        except Exception as e:
            logger.warning("router could not search the chunk index: %s", e)
            return

        for hit in hits:
            cos = cosine_of(hit.score)
            cosines.setdefault(KIND_PASSAGES, []).append(cos)
            bucket = items.setdefault(KIND_PASSAGES, [])
            if len(bucket) < TRACE_ITEMS:
                record = hit.record
                bucket.append(
                    RouterItem(
                        kind=KIND_PASSAGES,
                        item_id=record.vector_id,
                        # The filename is what a reader recognises; the chunk id is not.
                        label=f"{record.document_id} p{record.page}",
                        similarity=cos,
                        detail={"page": record.page, "document_id": record.document_id},
                    )
                )
