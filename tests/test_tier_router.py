"""Deciding which tiers a question is worth asking.

The property this file exists for: **a tier the tenant has forbidden is never queried**, asserted
on the request that went to OpenSearch rather than on what came back. Filtering forbidden items out
of the results would look identical in a passing test and be wrong twice over -- the items would
reach the trace, which an auditor expands, and the forbidden layer's score would raise
`best_score`, narrowing the margin and dropping a *permitted* layer that would have answered.

The second property: **routing never costs an answer.** An empty index, an unreachable collection,
a question that embeds to nothing relevant -- each degrades to the resolver's own tier order. A
router is an optimisation, and an optimisation able to refuse an answer is a liability.

No AWS. Every dependency is injected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.governance import GovernanceSettings
from src.graph.scope import AuthContext
from src.query.router import KIND_PASSAGES, TierRouter
from src.query.router_index import KIND_ENTITY, KIND_METRIC, KIND_TABLE

TENANT = "demo-firm"


def ctx(**over) -> AuthContext:
    return AuthContext(user_id="alice@firm.example", tenant_id=TENANT, **over)


def settings(**over) -> GovernanceSettings:
    base = {"router_margin": 0.35, "router_min_similarity": 0.25, "router_metric_boost": 0.0}
    base.update(over)
    return GovernanceSettings(**base)


@dataclass
class FakeRecord:
    kind: str
    item_id: str
    label: str
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class FakeHit:
    record: FakeRecord
    raw_score: float


class FakeRoutingIndex:
    """Records the search it was asked to run, which is what most of these assert on."""

    def __init__(self, by_kind: dict[str, float] | None = None) -> None:
        self.by_kind = by_kind or {}
        self.calls: list[dict[str, Any]] = []

    def search(self, index, vector, *, top_k=10, kinds=None, **kw):
        self.calls.append({"index": index, "kinds": kinds, "top_k": top_k, **kw})
        allowed = set(kinds) if kinds is not None else set(self.by_kind)
        return [
            FakeHit(FakeRecord(kind=k, item_id=f"{k}-1", label=f"{k} one"), raw_score=raw)
            for k, raw in self.by_kind.items()
            if k in allowed
        ]


@dataclass
class FakeChunkRecord:
    vector_id: str = "c1"
    document_id: str = "doc-1"
    page: int = 1


@dataclass
class FakeChunkHit:
    record: FakeChunkRecord
    score: float


class FakeChunks:
    def __init__(self, raw: float | None = None) -> None:
        self.raw = raw
        self.calls: list[dict[str, Any]] = []

    def search(self, index, vector, *, top_k=10, **kw):
        self.calls.append({"index": index, **kw})
        return [] if self.raw is None else [FakeChunkHit(FakeChunkRecord(), self.raw)]


class FakeEmbedder:
    def __init__(self, fail: bool = False) -> None:
        self.fail = fail
        self.embedded: list[str] = []

    def embed_query(self, text: str):
        if self.fail:
            raise RuntimeError("bedrock unreachable")
        self.embedded.append(text)
        return [0.1, 0.2, 0.3]


def router(routing=None, chunks=None, embedder=None) -> TierRouter:
    return TierRouter(
        routing_index=routing if routing is not None else FakeRoutingIndex(),
        chunk_search=chunks if chunks is not None else FakeChunks(),
        embedder=embedder if embedder is not None else FakeEmbedder(),
    )


# A raw score of ~0.83 is cos 0.8; ~0.63 is cos 0.4. Written as raw scores because that is what
# the engine returns and what the router has to convert.
STRONG = 0.833
WEAK = 0.625


class TestAForbiddenTierIsNeverQueried:
    """Resolved before the search, not filtered out of the results."""

    def test_a_forbidden_kind_is_excluded_from_the_search_itself(self):
        routing = FakeRoutingIndex({KIND_METRIC: STRONG, KIND_ENTITY: STRONG})
        router(routing).route(ctx(), "anything", settings(allowed_tiers=frozenset({2, 3})))

        assert routing.calls, "the router did not search at all"
        assert KIND_METRIC not in routing.calls[0]["kinds"]

    def test_its_items_never_reach_the_trace(self):
        """The trace is expandable, so an item in it is a disclosure of data the tenant told the
        system not to use."""
        routing = FakeRoutingIndex({KIND_METRIC: STRONG, KIND_ENTITY: WEAK})
        decision = router(routing).route(
            ctx(), "anything", settings(allowed_tiers=frozenset({2, 3}))
        )

        assert KIND_METRIC not in decision.items
        assert all(layer.kind != KIND_METRIC for layer in decision.layers)

    def test_a_forbidden_layer_cannot_drop_a_permitted_one(self):
        """The subtle half. A forbidden metric layer scoring 0.8 would set `best_score` to 0.8 and
        push the 0.4 entity layer outside a 0.35 margin -- so forbidding tier 1 would silently cost
        tier 2 as well."""
        routing = FakeRoutingIndex({KIND_METRIC: STRONG, KIND_ENTITY: WEAK})
        decision = router(routing).route(
            ctx(), "anything", settings(allowed_tiers=frozenset({2, 3}))
        )

        assert 2 in decision.tiers

    def test_the_reason_says_forbidden_rather_than_low_scoring(self):
        """ "Your administrator turned this off" and "this did not look relevant" are different
        facts about the system, and a reader has to be able to tell them apart."""
        routing = FakeRoutingIndex({KIND_ENTITY: STRONG})
        decision = router(routing).route(
            ctx(), "anything", settings(allowed_tiers=frozenset({2, 3}))
        )

        assert "not permitted" in decision.dropped["1"]

    def test_the_chunk_index_is_not_probed_when_tier_3_is_forbidden(self):
        chunks = FakeChunks(STRONG)
        routing = FakeRoutingIndex({KIND_ENTITY: STRONG})
        router(routing, chunks).route(ctx(), "anything", settings(allowed_tiers=frozenset({1, 2})))

        assert chunks.calls == []

    def test_the_router_never_widens_past_allowed_tiers(self):
        routing = FakeRoutingIndex({k: STRONG for k in (KIND_METRIC, KIND_ENTITY, KIND_TABLE)})
        decision = router(routing, FakeChunks(STRONG)).route(
            ctx(), "anything", settings(allowed_tiers=frozenset({2}))
        )

        assert decision.tiers == [2]

    def test_a_degraded_decision_does_not_resurrect_a_forbidden_tier(self):
        """The failure path is the easiest place to lose a restriction."""
        decision = TierRouter().route(ctx(), "anything", settings(allowed_tiers=frozenset({2, 3})))

        assert decision.degraded is True
        assert decision.tiers == [2, 3]


class TestRoutingNeverCostsAnAnswer:
    def test_no_vector_store_degrades_to_every_permitted_tier(self):
        decision = TierRouter().route(ctx(), "anything", settings())

        assert decision.degraded is True
        assert decision.tiers == [1, 2, 3, 4]
        assert decision.reason and "no vector store" in decision.reason

    def test_an_empty_index_degrades(self):
        decision = router(FakeRoutingIndex({})).route(ctx(), "anything", settings())

        assert decision.degraded is True
        assert decision.tiers == [1, 2, 3, 4]

    def test_nothing_above_the_floor_degrades(self):
        """Similarity is not calibrated, so "everything scored badly" means the question is not
        about anything indexed -- and guessing a layer would be worse than trying them all."""
        routing = FakeRoutingIndex({KIND_METRIC: 0.51})  # cos ~0.04
        decision = router(routing).route(ctx(), "anything", settings(router_min_similarity=0.5))

        assert decision.degraded is True
        assert decision.tiers == [1, 2, 3, 4]

    def test_an_unreachable_index_degrades_rather_than_raising(self):
        class Broken:
            def search(self, *a, **kw):
                raise RuntimeError("collection unreachable")

        decision = router(Broken()).route(ctx(), "anything", settings())

        assert decision.degraded is True
        assert decision.tiers == [1, 2, 3, 4]

    def test_a_failed_embedding_degrades(self):
        decision = router(embedder=FakeEmbedder(fail=True)).route(ctx(), "anything", settings())

        assert decision.degraded is True
        assert decision.reason and "embedded" in decision.reason

    def test_routing_can_be_turned_off(self):
        routing = FakeRoutingIndex({KIND_METRIC: STRONG})
        decision = router(routing).route(ctx(), "anything", settings(router_enabled=False))

        assert decision.degraded is True
        assert routing.calls == [], "a disabled router still searched"

    def test_one_probe_failing_leaves_the_other_deciding(self):
        """A chunk index that is down must not veto a metric that matched."""

        class BrokenChunks:
            def search(self, *a, **kw):
                raise RuntimeError("down")

        routing = FakeRoutingIndex({KIND_METRIC: STRONG})
        decision = router(routing, BrokenChunks()).route(ctx(), "anything", settings())

        assert decision.degraded is False
        assert decision.tiers == [1]


class TestTheDecision:
    def test_a_strong_metric_match_selects_tier_1(self):
        routing = FakeRoutingIndex({KIND_METRIC: STRONG, KIND_ENTITY: WEAK})
        decision = router(routing).route(ctx(), "fees billed last quarter", settings())

        assert 1 in decision.tiers

    def test_a_table_layer_justifies_both_tier_2_and_tier_4(self):
        """A catalogued table is reachable two ways: its schema is in the graph, and tier 4 writes
        SQL against it."""
        routing = FakeRoutingIndex({KIND_TABLE: STRONG})
        decision = router(routing).route(ctx(), "anything", settings())

        assert decision.tiers == [2, 4]

    def test_passages_are_scored_from_the_chunk_index(self):
        """Nothing in the routing index can score tier 3, so a summary of passages there would be a
        second copy of text that is already embedded."""
        routing = FakeRoutingIndex({KIND_METRIC: WEAK})
        decision = router(routing, FakeChunks(STRONG)).route(ctx(), "anything", settings())

        assert 3 in decision.tiers
        assert KIND_PASSAGES in decision.items

    def test_a_tier_two_layers_justify_is_not_reported_as_dropped(self):
        """A losing table layer must not drop tier 2 when the entity layer won it."""
        routing = FakeRoutingIndex({KIND_ENTITY: STRONG, KIND_TABLE: 0.55})
        decision = router(routing).route(ctx(), "anything", settings(router_margin=0.05))

        assert 2 in decision.tiers
        assert "2" not in decision.dropped

    def test_the_question_is_embedded_once(self):
        """Both probes share the vector. Embedding twice would double the latency the router adds
        to every question."""
        embedder = FakeEmbedder()
        routing = FakeRoutingIndex({KIND_METRIC: STRONG})
        router(routing, FakeChunks(STRONG), embedder).route(ctx(), "a question", settings())

        assert embedder.embedded == ["a question"]

    def test_tiers_come_back_in_order(self):
        """The resolver walks them in order, so tier 4 must stay last."""
        routing = FakeRoutingIndex({k: STRONG for k in (KIND_METRIC, KIND_ENTITY, KIND_TABLE)})
        decision = router(routing, FakeChunks(STRONG)).route(ctx(), "anything", settings())

        assert decision.tiers == sorted(decision.tiers)
        assert decision.tiers[-1] == 4


class TestTheMatterWallAppliesToRoutingToo:
    """A routing hit for an entity is a subject name, and the trace shows it above the wall."""

    def test_the_allowlist_is_passed_to_the_routing_search(self):
        routing = FakeRoutingIndex({KIND_ENTITY: STRONG})
        router(routing).route(ctx(matter_allowlist=frozenset({"M-1"})), "anything", settings())

        assert routing.calls[0]["matter_allowlist"] == frozenset({"M-1"})

    def test_the_denylist_is_passed_to_the_routing_search(self):
        routing = FakeRoutingIndex({KIND_ENTITY: STRONG})
        router(routing).route(ctx(matter_denylist=frozenset({"M-2"})), "anything", settings())

        assert routing.calls[0]["matter_denylist"] == frozenset({"M-2"})

    def test_the_wall_reaches_the_chunk_probe_as_well(self):
        chunks = FakeChunks(STRONG)
        router(FakeRoutingIndex({}), chunks).route(
            ctx(matter_denylist=frozenset({"M-2"})), "anything", settings()
        )

        assert chunks.calls[0]["matter_denylist"] == frozenset({"M-2"})


class TestTheTrace:
    def test_it_carries_the_scores_and_the_reasons(self):
        routing = FakeRoutingIndex({KIND_METRIC: STRONG, KIND_ENTITY: WEAK})
        body = router(routing).route(ctx(), "anything", settings()).to_dict()

        assert body["tiers_selected"]
        assert all(layer["reason"] for layer in body["layers"])

    def test_a_layer_names_every_tier_it_justifies(self):
        routing = FakeRoutingIndex({KIND_TABLE: STRONG})
        body = router(routing).route(ctx(), "anything", settings()).to_dict()

        table = next(layer for layer in body["layers"] if layer["kind"] == KIND_TABLE)
        assert table["tiers"] == [2, 4]

    def test_items_are_capped(self):
        """Enough to see why a layer won; not a wall of near-identical rows."""
        from src.query.router import TRACE_ITEMS

        routing = FakeRoutingIndex({KIND_ENTITY: STRONG})
        decision = router(routing).route(ctx(), "anything", settings())

        assert len(decision.items.get(KIND_ENTITY, [])) <= TRACE_ITEMS

    def test_similarity_is_reported_as_a_cosine_not_a_raw_score(self):
        """An admin reading 0.83 where the cosine is 0.80 would tune against the wrong number."""
        routing = FakeRoutingIndex({KIND_ENTITY: STRONG})
        decision = router(routing).route(ctx(), "anything", settings())

        assert decision.items[KIND_ENTITY][0].similarity < 0.83
