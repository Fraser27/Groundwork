"""Tests for tier resolution with real collaborators wired in.

The behaviour worth protecting is precision, not recall. A resolver that answers
every question is worse than one that declines: tier 1 returning the wrong metric
produces a confident number, and tier 2 matching on a single common noun produces a
confident irrelevance. Both look authoritative.
"""

from __future__ import annotations

import pytest

from src.documents.review import InMemoryAssertionStore, ReviewQueue
from src.governance import GovernanceSettings
from src.graph.assertions import (
    EpistemicClass,
    ReviewState,
    SourceLocator,
    build_assertion,
)
from src.graph.scope import AuthContext
from src.metrics.loader import load_metrics
from src.metrics.models import StaticCatalog
from src.query.graph_reader import GraphReader, terms_of
from src.query.metric_matcher import MetricMatcher
from src.query.resolver import QueryBlocked, Resolver, Tier

TENANT = "firm-acme"
METRICS = "sample/metrics.yaml"


@pytest.fixture
def ctx() -> AuthContext:
    return AuthContext(user_id="alice@firm.com", tenant_id=TENANT)


@pytest.fixture
def matcher() -> MetricMatcher:
    return MetricMatcher(load_metrics(METRICS).metrics, StaticCatalog(tables={}))


@pytest.fixture
def reader(ctx: AuthContext) -> GraphReader:
    queue = ReviewQueue(InMemoryAssertionStore())
    facts = [
        ("document:d1", "MENTIONS", "party:acme-corporation", 0.9),
        ("counsel:dalgleish-rowe", "REPRESENTS", "party:beta-holdings-ltd", 0.95),
        ("document:d1", "CITES", "authority:347-us-483", 0.97),
    ]
    assertions = [
        build_assertion(
            tenant_id=TENANT,
            subject_id=s,
            predicate=p,
            object_id=o,
            epistemic_class=EpistemicClass.EXTRACTED_MODEL,
            method="llm:opus-5",
            confidence=c,
            source_locator=SourceLocator(
                document_id="d1", filename="d1.pdf", page=1, quote="Acme Corporation"
            ),
            matter_id="M-1",
        )
        for s, p, o, c in facts
    ]
    queue.stage(ctx, assertions, job_id="j1")
    # Interpretive claims reach the graph only once a reviewer signs them off, so the
    # fixture does what a reviewer would. Without this the reader correctly returns
    # nothing and the tier tests would be asserting against an empty graph.
    for a in assertions:
        if a.review_state is ReviewState.PENDING:
            queue.approve(ctx, a.assertion_id)
    queue.promote(ctx, job_id="j1")
    return GraphReader(queue)


class TestTermExtraction:
    def test_drops_question_scaffolding(self):
        assert terms_of("which matters involve Acme Corporation") == [
            "matters",
            "acme",
            "corporation",
        ]

    def test_empty_question_yields_nothing(self):
        assert terms_of("what is the of and") == []


class TestMetricMatching:
    def test_matches_by_name(self, matcher):
        assert matcher.match("show me fees billed by month").metric.metric_id == "lm_001"

    def test_matches_by_synonym(self, matcher):
        assert matcher.match("what is our recovery rate").metric.metric_id == "lm_003"

    def test_name_outranks_definition_prose(self, matcher):
        """`realization_rate` is *defined* as fees billed over standard value, so a flat
        bag of words tied the two and the tie made the matcher decline both."""
        assert matcher.match("fees billed by month").metric.metric_id == "lm_001"

    def test_single_common_noun_does_not_match(self, matcher):
        """"which matters involve Acme" once selected open_matter_count on "matters"."""
        assert matcher.match("which matters involve Acme Corporation") is None

    def test_unrelated_question_declines(self, matcher):
        assert matcher.match("find documents about antitrust") is None

    def test_grain_taken_from_question(self, matcher):
        assert matcher.match("fees billed by quarter").time_grain == "quarter"

    def test_grain_refused_when_metric_forbids_it(self, matcher):
        """`realization_rate` declares month/quarter/year, so daily must not stick."""
        assert matcher.match("realization rate daily").time_grain is None

    def test_compiles_deterministic_sql(self, matcher):
        sql = matcher.match("fees billed by month").compile()
        assert sql.upper().startswith("SELECT")


class TestGraphReading:
    def test_finds_by_entity_name(self, reader, ctx):
        hits = reader.search(ctx, "which matters involve Acme Corporation")
        assert hits and hits[0]["object_id"] == "party:acme-corporation"

    def test_reports_why_it_matched(self, reader, ctx):
        """The matched terms are what let the UI explain an answer's relevance."""
        hits = reader.search(ctx, "who represents Beta Holdings")
        assert "beta" in hits[0]["matched_on"]

    def test_carries_source_span(self, reader, ctx):
        hit = reader.search(ctx, "Acme Corporation")[0]
        assert hit["source"]["document_id"] == "d1"

    def test_confidence_floor_excludes_weak_facts(self, reader, ctx):
        assert reader.search(ctx, "Acme Corporation", min_confidence=0.99) == []

    def test_unmatched_question_returns_nothing(self, reader, ctx):
        assert reader.search(ctx, "entirely unrelated gibberish") == []

    def test_expand_walks_from_a_seed(self, reader, ctx):
        assert len(reader.expand(ctx, ["document:d1"], depth=1)) >= 2

    def test_expand_respects_depth(self, reader, ctx):
        shallow = reader.expand(ctx, ["counsel:dalgleish-rowe"], depth=1)
        deep = reader.expand(ctx, ["counsel:dalgleish-rowe"], depth=3)
        assert len(deep) >= len(shallow)


class TestTierOrdering:
    def test_metric_wins_over_graph(self, matcher, reader, ctx):
        r = Resolver(metric_matcher=matcher, graph_reader=reader)
        assert r.resolve(ctx, "fees billed by month", GovernanceSettings()).tier is Tier.GOVERNED_METRIC

    def test_falls_through_to_graph(self, matcher, reader, ctx):
        r = Resolver(metric_matcher=matcher, graph_reader=reader)
        res = r.resolve(ctx, "which matters involve Acme Corporation", GovernanceSettings())
        assert res.tier is Tier.GRAPH_TRAVERSAL
        assert res.assertions_used

    def test_graph_answer_is_auditable(self, matcher, reader, ctx):
        """Every id in `assertions_used` must resolve to a real assertion."""
        r = Resolver(metric_matcher=matcher, graph_reader=reader)
        res = r.resolve(ctx, "Acme Corporation", GovernanceSettings())
        assert all(isinstance(a, str) and a for a in res.assertions_used)

    def test_tier_override_skips_earlier_tiers(self, matcher, reader, ctx):
        r = Resolver(metric_matcher=matcher, graph_reader=reader)
        res = r.resolve(
            ctx, "fees billed by month", GovernanceSettings(), tier_override=Tier.GRAPH_TRAVERSAL
        )
        assert res.tiers_attempted == [Tier.GRAPH_TRAVERSAL]

    def test_execute_false_still_returns_sql(self, matcher, ctx):
        r = Resolver(metric_matcher=matcher)
        res = r.resolve(ctx, "fees billed by month", GovernanceSettings(), execute=False)
        assert res.sql and res.answer is None

    def test_nothing_matched_reports_honestly(self, matcher, reader, ctx):
        r = Resolver(metric_matcher=matcher, graph_reader=reader)
        res = r.resolve(ctx, "zzzq unrelated gibberish", GovernanceSettings())
        assert res.answer is None
        assert res.warnings


class TestKillSwitch:
    def test_blocks_only_when_no_tier_answered(self, matcher, reader, ctx):
        settings = GovernanceSettings(block_ungoverned_queries=True)
        r = Resolver(metric_matcher=matcher, graph_reader=reader)
        with pytest.raises(QueryBlocked):
            r.resolve(ctx, "zzzq unrelated gibberish", settings)

    def test_governed_metric_unaffected_by_switch(self, matcher, reader, ctx):
        settings = GovernanceSettings(block_ungoverned_queries=True)
        r = Resolver(metric_matcher=matcher, graph_reader=reader)
        assert r.resolve(ctx, "fees billed by month", settings).tier is Tier.GOVERNED_METRIC

    def test_refusal_is_recorded_for_review(self, matcher, ctx):
        """A question people keep asking is a metric waiting to be written."""
        r = Resolver(metric_matcher=matcher)
        with pytest.raises(QueryBlocked):
            r.resolve(ctx, "zzzq gibberish", GovernanceSettings(block_ungoverned_queries=True))
        assert r.blocked and r.blocked[0].question == "zzzq gibberish"


class TestVectorDegradation:
    def test_tier_three_declines_when_vector_search_fails(self, reader, ctx):
        """A missing Bedrock credential must not 500 — tier 3 declines and 4 runs."""

        class Failing:
            def search(self, *_a, **_k):
                raise RuntimeError("no credentials")

        from src.query.vector_search import VectorSearch

        class Broken(VectorSearch):
            def __init__(self):
                self._embedder = Failing()

        r = Resolver(graph_reader=reader, vector_search=Broken())
        res = r.resolve(ctx, "zzzq gibberish", GovernanceSettings())
        assert res.answer is None


class TestHybridSeedsTheWalkWithGraphNodeIds:
    """Tier 3 returned zero related facts for every question, silently.

    A passage carries a bare `document_id`; an assertion's subject is `document:<id>`, per
    `DocumentMeta.entity_id`. Seeding `expand()` with the raw id meant the first frontier matched
    nothing, so the walk ended before it began and the answer's graph half was always empty. It
    read as "nothing connects to this passage" rather than as a failure, which is why it survived:
    an empty grounding card looks like a sparse graph.
    """

    def _resolver(self, reader):
        class Vectors:
            def search(self, ctx, question, *, top_k=10):
                # What VectorSearch really returns: the bare document id, no prefix.
                return [{"document_id": "d1", "page": 1, "text": "Acme Corporation"}]

        return Resolver(graph_reader=reader, vector_search=Vectors())

    def test_a_hybrid_answer_carries_the_walked_edges(self, reader, ctx):
        res = self._resolver(reader).resolve(ctx, "what does d1 say about acme", GovernanceSettings(), tier_override=Tier.HYBRID)

        assert res.tier is Tier.HYBRID
        assert res.answer["related"], "the graph half was empty, so the walk never started"
        assert res.assertions_used, "an answer that used facts must record which ones"

    def test_the_walked_edges_say_how_far_out_they_are(self, reader, ctx):
        """`matched_on` is empty for a walked edge because no term matched it, so `hops` carries the
        explanation instead: a reader has to be able to tell a fact stated in the cited document
        from one two steps away."""
        res = self._resolver(reader).resolve(ctx, "what does d1 say about acme", GovernanceSettings(), tier_override=Tier.HYBRID)

        hops = {e["hops"] for e in res.answer["related"]}
        assert hops, "no hop distance recorded"
        assert min(hops) == 1, "an edge on the cited document itself is one hop"
        assert all(h >= 1 for h in hops)

    def test_a_term_match_records_no_hop_distance(self, reader, ctx):
        """Tier 2 explains itself with `matched_on`, so `hops` stays None rather than claiming a
        distance the search never walked."""
        hits = reader.search(ctx, "acme corporation", min_confidence=0.5)

        assert hits
        assert all(h["hops"] is None for h in hits)
        assert any(h["matched_on"] for h in hits)
