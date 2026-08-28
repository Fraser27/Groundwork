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

from src.governance import GovernanceError, GovernanceSettings
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
        catalog=None,
        sql_lane=None,
    )


def attempted(resolution) -> list[int]:
    return [int(t) for t in resolution.tiers_attempted]


class TestDefaults:
    def test_the_default_is_metrics_then_vector_first(self, resolver, ctx):
        """Not every tier: tiers 2 and 3 are one search in opposite directions and the default
        picks vector first. A default naming both would make the order they were tried in the
        thing that decided which direction a tenant got."""
        assert attempted(resolver.resolve(ctx, "q", GovernanceSettings())) == [1, 3]

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
        """A firm may want metrics and vector-first retrieval but not graph-first traversal, so
        the cap is a set rather than a maximum."""
        capped = GovernanceSettings(allowed_tiers=frozenset({1, 3}))
        assert attempted(resolver.resolve(ctx, "q", capped)) == [1, 3]

    def test_an_empty_cap_refuses_every_question(self, resolver, ctx):
        """Answering anyway would make the setting decorative."""
        with pytest.raises(QueryBlocked, match="no resolution tier is permitted"):
            resolver.resolve(ctx, "q", GovernanceSettings(allowed_tiers=frozenset()))


class TestCallerSubset:
    def test_a_subset_runs_only_those_tiers(self, resolver, ctx):
        graph_first = GovernanceSettings(allowed_tiers=frozenset({1, 2}))
        res = resolver.resolve(ctx, "q", graph_first, tiers_requested=[Tier.GRAPH_TRAVERSAL])
        assert attempted(res) == [2]

    def test_a_subset_is_still_ordered_most_precise_first(self, resolver, ctx):
        res = resolver.resolve(
            ctx, "q", GovernanceSettings(), tiers_requested=[Tier.HYBRID, Tier.GOVERNED_METRIC]
        )
        assert attempted(res) == [1, 3]

    def test_a_duplicate_tier_is_tried_once(self, resolver, ctx):
        res = resolver.resolve(
            ctx, "q", GovernanceSettings(), tiers_requested=[Tier.HYBRID, Tier.HYBRID]
        )
        assert attempted(res) == [3]


class TestRefusalIsVisible:
    def test_asking_for_a_forbidden_tier_is_refused(self, resolver, ctx):
        """Not silently answered at another tier. A caller who pinned tier 3 and got a
        tier 2 answer has been given something they did not ask for."""
        capped = GovernanceSettings(allowed_tiers=frozenset({1, 2}))
        with pytest.raises(QueryBlocked, match="tier 3 is not permitted"):
            resolver.resolve(ctx, "q", capped, tier_override=Tier.HYBRID)

    def test_the_refusal_names_what_is_permitted(self, resolver, ctx):
        """An administrator controls this, so the user needs to know what to ask for
        instead rather than just that they were refused."""
        capped = GovernanceSettings(allowed_tiers=frozenset({1, 2}))
        with pytest.raises(QueryBlocked, match="Permitted tiers: 1, 2"):
            resolver.resolve(ctx, "q", capped, tier_override=Tier.HYBRID)

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
            tiers_requested=[Tier.GRAPH_TRAVERSAL, Tier.HYBRID],
        )
        assert attempted(res) == [1]


class TestEnvParsing:
    def test_a_comma_separated_cap_parses(self, monkeypatch):
        monkeypatch.setenv("GROUNDWORK_ALLOWED_TIERS", "1,2")
        assert GovernanceSettings.from_env().allowed_tiers == frozenset({1, 2})

    def test_an_unparseable_cap_falls_back_to_the_default(self, monkeypatch):
        """A typo in a cap must not stop the API starting, and must not silently forbid
        everything either, which would look like a total outage."""
        monkeypatch.setenv("GROUNDWORK_ALLOWED_TIERS", "nonsense")
        assert GovernanceSettings.from_env().allowed_tiers == frozenset({1, 3})

    def test_an_unset_cap_gets_the_default(self, monkeypatch):
        monkeypatch.delenv("GROUNDWORK_ALLOWED_TIERS", raising=False)
        assert GovernanceSettings.from_env().allowed_tiers == frozenset({1, 3})

    def test_an_env_var_naming_both_directions_is_coerced_not_refused(self, monkeypatch):
        """A deployment carrying the old `1,2,3` must still start. Refusing it would take the
        whole API down over a value that was legal for most of this project's life."""
        monkeypatch.setenv("GROUNDWORK_ALLOWED_TIERS", "1,2,3")
        assert GovernanceSettings.from_env().allowed_tiers == frozenset({1, 3})

    def test_a_graph_first_env_var_survives(self, monkeypatch):
        """The coercion drops a direction only when both are named. A tenant who asked for
        graph first keeps it."""
        monkeypatch.setenv("GROUNDWORK_ALLOWED_TIERS", "1,2")
        assert GovernanceSettings.from_env().allowed_tiers == frozenset({1, 2})


class TestTheFourthTierStaysRetired:
    """It was never implemented, and the ways it could come back are all quiet ones.

    A number that used to be a tier is more dangerous than one that never was: env vars,
    persisted settings rows and stale callers all still carry 4, and `Tier(4)` now raises. Each
    of these is a path by which a retired tier reappears as a 500 rather than a refusal.
    """

    def test_the_enum_has_no_fourth_member(self):
        assert [int(t) for t in Tier] == [1, 2, 3]

    def test_every_tier_has_an_explanation(self):
        """A missing entry is a KeyError on `Resolution.explanation`, which is on every answer."""
        from src.query.resolver import TIER_EXPLANATION

        assert set(TIER_EXPLANATION) == set(Tier)

    def test_a_stale_env_var_does_not_resurrect_it(self, monkeypatch):
        monkeypatch.setenv("GROUNDWORK_ALLOWED_TIERS", "1,2,3,4")
        assert GovernanceSettings.from_env().allowed_tiers == frozenset({1, 3})

    def test_a_persisted_row_does_not_resurrect_it(self):
        """A tenant whose row was written while the fourth tier existed still has [1,2,3,4] in
        DynamoDB. Passing it through would let storage reintroduce a tier the code lacks."""
        from src.governance_store import _decode

        decoded = _decode("allowed_tiers", [1, 2, 3, 4], frozenset({1, 3}))
        assert decoded == frozenset({1, 3})

    def test_asking_for_it_is_refused_not_crashed(self, resolver, ctx):
        """Through the enum rather than the route validator: `Tier(4)` raises ValueError, and a
        stale caller deserves a refusal rather than a 500."""
        with pytest.raises(ValueError):
            Tier(4)

    def test_the_known_tier_set_matches_the_enum(self):
        """Two lists that can drift is how a retired tier survived in `from_env`'s default for a
        day. `governance` cannot import `Tier` -- the resolver imports governance -- so the
        duplication is deliberate and this is what keeps it honest."""
        from src.governance import KNOWN_TIERS

        assert KNOWN_TIERS == {int(t) for t in Tier}

    def test_a_retired_tier_in_the_cap_is_refused_rather_than_stored(self):
        """It reached the Admin page as `[1,2,3,4]` because nothing validated the members --
        `_decode` narrowed a persisted row and `from_env`'s default still carried a 4, so the one
        path nobody guarded was the one that leaked."""
        with pytest.raises(GovernanceError, match="not a resolution tier"):
            GovernanceSettings(allowed_tiers=frozenset({1, 2, 3, 4}))


class TestOneDirectionOnly:
    """Tiers 2 and 3 are the same three stores searched in opposite orders, so a tenant picks one.

    The property: **the cap is where the direction is chosen**, and it is chosen explicitly. A cap
    naming both would leave the direction to whichever tier the resolver's loop reached first, and
    provenance that depends on iteration order is not provenance -- a passage would be attributed
    to the graph vouching for it on one deploy and to similarity on the next.
    """

    def test_naming_both_directions_is_refused(self):
        with pytest.raises(GovernanceError, match="not both"):
            GovernanceSettings(allowed_tiers=frozenset({1, 2, 3}))

    def test_the_refusal_says_which_cap_means_which_direction(self):
        """An administrator has to know what to set instead, not just that they were refused."""
        with pytest.raises(GovernanceError, match=r"\{1, 2\} is graph first"):
            GovernanceSettings(allowed_tiers=frozenset({2, 3}))

    def test_graph_first_is_permitted(self):
        assert GovernanceSettings(allowed_tiers=frozenset({1, 2})).retrieval_direction == (
            "graph_first"
        )

    def test_vector_first_is_permitted(self):
        assert GovernanceSettings(allowed_tiers=frozenset({1, 3})).retrieval_direction == (
            "vector_first"
        )

    def test_metrics_only_has_no_direction(self):
        """Not defaulted to one. A tenant with no traversal permitted searches nothing, and naming
        a direction it cannot use would read as a capability."""
        assert GovernanceSettings(allowed_tiers=frozenset({1})).retrieval_direction == (
            "metrics_only"
        )

    def test_a_stored_row_naming_both_is_coerced_rather_than_lost(self):
        """A row written before the rule existed is legal storage. Refusing it would take that
        tenant's whole settings document down and revert their floor and kill switch too."""
        from src.governance_store import _decode

        assert _decode("allowed_tiers", [1, 2, 3], frozenset({1, 3})) == frozenset({1, 3})

    def test_a_stored_graph_first_row_keeps_graph_traversal(self):
        """The regression this guards: narrowing a stored row against the *default* would compute
        `{1,2} & {1,3}` and silently take graph traversal away from every tenant that chose it."""
        from src.governance_store import _decode

        assert _decode("allowed_tiers", [1, 2], frozenset({1, 3})) == frozenset({1, 2})

    def test_the_direction_is_reported_to_the_admin_screen(self):
        assert GovernanceSettings().to_dict()["retrieval_direction"] == "vector_first"

    def test_applying_a_patch_survives_the_derived_key(self):
        """`to_dict` carries a key that is not a constructor argument, and `apply` rebuilds the
        object -- so reading the wrong one of the two makes every Admin save a 500."""
        applied = GovernanceSettings().apply({"allowed_tiers": frozenset({1, 2})}, updated_by="a")
        assert applied.retrieval_direction == "graph_first"
