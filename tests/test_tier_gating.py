"""Which tiers may answer a question, and who decides.

Three controls that are deliberately not interchangeable:

`allowed_tiers` is the tenant's hard cap, set by an admin. A tier outside it never runs.
`tiers` is the caller's subset, honoured only within the cap.
`tier_override` pins one tier.

The property worth protecting is that **asking for a forbidden tier is refused, not
silently answered a different way**. A user who pins "governed metric only" and receives
an LLM-written answer labelled tier 4 has been given something they explicitly declined,
and if that looks the same as a successful call the label is the only warning.
"""

from __future__ import annotations

import pytest

from src.governance import GovernanceSettings
from src.graph.scope import AuthContext
from src.query.resolver import QueryBlocked, Resolver, Tier

TENANT = "demo-firm"


@pytest.fixture
def ctx() -> AuthContext:
    return AuthContext(tenant_id=TENANT, user_id="alice")


@pytest.fixture
def resolver() -> Resolver:
    """Nothing wired, so every tier misses and `tiers_attempted` reports what was tried.

    That is the point: this tests tier *selection*, not what a tier returns.
    """
    return Resolver(
        metric_matcher=None,
        graph_reader=None,
        vector_search=None,
        sql_generator=None,
        firewall=None,
    )


def attempted(resolution) -> list[int]:
    return [int(t) for t in resolution.tiers_attempted]


class TestDefaults:
    def test_all_four_tiers_are_tried_by_default(self, resolver, ctx):
        assert attempted(resolver.resolve(ctx, "q", GovernanceSettings())) == [1, 2, 3, 4]

    def test_tiers_are_tried_most_precise_first(self, resolver, ctx):
        """Order is the governance property: a governed metric must get the chance to
        answer before an LLM writes SQL for the same question."""
        assert attempted(resolver.resolve(ctx, "q", GovernanceSettings())) == sorted(
            attempted(resolver.resolve(ctx, "q", GovernanceSettings()))
        )


class TestTenantCap:
    def test_a_capped_tenant_never_runs_the_excluded_tier(self, resolver, ctx):
        capped = GovernanceSettings(allowed_tiers=frozenset({1, 2}))
        assert attempted(resolver.resolve(ctx, "q", capped)) == [1, 2]

    def test_a_cap_can_exclude_a_middle_tier(self, resolver, ctx):
        """A firm may want metrics and graph traversal but not hybrid retrieval, so the
        cap is a set rather than a maximum."""
        capped = GovernanceSettings(allowed_tiers=frozenset({1, 2, 4}))
        assert attempted(resolver.resolve(ctx, "q", capped)) == [1, 2, 4]

    def test_an_empty_cap_refuses_every_question(self, resolver, ctx):
        """Answering anyway would make the setting decorative."""
        with pytest.raises(QueryBlocked, match="no resolution tier is permitted"):
            resolver.resolve(ctx, "q", GovernanceSettings(allowed_tiers=frozenset()))


class TestCallerSubset:
    def test_a_subset_runs_only_those_tiers(self, resolver, ctx):
        res = resolver.resolve(
            ctx, "q", GovernanceSettings(), tiers_requested=[Tier.GRAPH_TRAVERSAL, Tier.HYBRID]
        )
        assert attempted(res) == [2, 3]

    def test_a_subset_is_still_ordered_most_precise_first(self, resolver, ctx):
        res = resolver.resolve(
            ctx, "q", GovernanceSettings(), tiers_requested=[Tier.LLM_SQL, Tier.GOVERNED_METRIC]
        )
        assert attempted(res) == [1, 4]

    def test_a_duplicate_tier_is_tried_once(self, resolver, ctx):
        res = resolver.resolve(
            ctx, "q", GovernanceSettings(), tiers_requested=[Tier.HYBRID, Tier.HYBRID]
        )
        assert attempted(res) == [3]


class TestRefusalIsVisible:
    def test_asking_for_a_forbidden_tier_is_refused(self, resolver, ctx):
        """Not silently answered at another tier. A caller who pinned tier 4 and got a
        tier 2 answer has been given something they did not ask for."""
        capped = GovernanceSettings(allowed_tiers=frozenset({1, 2}))
        with pytest.raises(QueryBlocked, match="tier 4 is not permitted"):
            resolver.resolve(ctx, "q", capped, tier_override=Tier.LLM_SQL)

    def test_the_refusal_names_what_is_permitted(self, resolver, ctx):
        """An administrator controls this, so the user needs to know what to ask for
        instead rather than just that they were refused."""
        capped = GovernanceSettings(allowed_tiers=frozenset({1, 2}))
        with pytest.raises(QueryBlocked, match="Permitted tiers: 1, 2"):
            resolver.resolve(ctx, "q", capped, tier_override=Tier.LLM_SQL)

    def test_a_subset_containing_a_forbidden_tier_is_refused(self, resolver, ctx):
        capped = GovernanceSettings(allowed_tiers=frozenset({1, 2}))
        with pytest.raises(QueryBlocked, match="not permitted"):
            resolver.resolve(ctx, "q", capped, tiers_requested=[Tier.GOVERNED_METRIC, Tier.HYBRID])

    def test_an_unrequested_cap_narrows_silently(self, resolver, ctx):
        """The distinction: a caller who asked for nothing specific gets whatever the cap
        allows without an error, because they expressed no preference to contradict."""
        capped = GovernanceSettings(allowed_tiers=frozenset({1}))
        assert attempted(resolver.resolve(ctx, "q", capped)) == [1]


class TestOverrideStillWorks:
    def test_a_permitted_override_pins_one_tier(self, resolver, ctx):
        res = resolver.resolve(ctx, "q", GovernanceSettings(), tier_override=Tier.HYBRID)
        assert attempted(res) == [3]

    def test_override_takes_precedence_over_a_subset(self, resolver, ctx):
        """Both are caller intent, so the more specific one wins rather than being
        intersected into something neither asked for."""
        res = resolver.resolve(
            ctx,
            "q",
            GovernanceSettings(),
            tier_override=Tier.GOVERNED_METRIC,
            tiers_requested=[Tier.HYBRID, Tier.LLM_SQL],
        )
        assert attempted(res) == [1]


class TestEnvParsing:
    def test_a_comma_separated_cap_parses(self, monkeypatch):
        monkeypatch.setenv("LEXGRAPH_ALLOWED_TIERS", "1,2")
        assert GovernanceSettings.from_env().allowed_tiers == frozenset({1, 2})

    def test_an_unparseable_cap_falls_back_to_all_tiers(self, monkeypatch):
        """A typo in a cap must not stop the API starting, and must not silently forbid
        everything either, which would look like a total outage."""
        monkeypatch.setenv("LEXGRAPH_ALLOWED_TIERS", "nonsense")
        assert GovernanceSettings.from_env().allowed_tiers == frozenset({1, 2, 3, 4})

    def test_an_unset_cap_permits_all_tiers(self, monkeypatch):
        monkeypatch.delenv("LEXGRAPH_ALLOWED_TIERS", raising=False)
        assert GovernanceSettings.from_env().allowed_tiers == frozenset({1, 2, 3, 4})
