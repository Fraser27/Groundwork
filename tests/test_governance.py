"""Tests for runtime governance settings.

The cap/floor relationship is the one that matters. It is defence in depth rather
than policy: the gap between them is what keeps an unreviewed model assertion out of
answers even if the review gate itself were bypassed.
"""

from __future__ import annotations

from dataclasses import fields

import pytest

from src.governance import (
    FIELD_HELP,
    GRAPH_EXPAND_LIMIT_CEILING,
    MAX_COMPOSE_CALLS_CEILING,
    GovernanceError,
    GovernanceSettings,
)
from src.graph.assertions import answerable_confidence


class TestCapFloorInvariant:
    def test_defaults_leave_a_gap(self):
        s = GovernanceSettings()
        assert s.model_confidence_cap < s.min_confidence_floor

    def test_cap_equal_to_floor_rejected(self):
        with pytest.raises(GovernanceError, match="must stay below"):
            GovernanceSettings(min_confidence_floor=0.8, model_confidence_cap=0.8)

    def test_cap_above_floor_rejected(self):
        with pytest.raises(GovernanceError, match="must stay below"):
            GovernanceSettings(min_confidence_floor=0.7, model_confidence_cap=0.9)

    def test_raising_floor_alone_is_allowed(self):
        """Widening the gap is always safe."""
        s = GovernanceSettings().apply({"min_confidence_floor": 0.95}, updated_by="admin")
        assert s.min_confidence_floor == 0.95
        assert s.model_confidence_cap == 0.79

    def test_lowering_floor_into_the_cap_is_refused(self):
        """The trap: lowering the floor without lowering the cap closes the gap."""
        with pytest.raises(GovernanceError, match="must stay below"):
            GovernanceSettings().apply({"min_confidence_floor": 0.5}, updated_by="admin")

    def test_both_can_move_together(self):
        s = GovernanceSettings().apply(
            {"min_confidence_floor": 0.5, "model_confidence_cap": 0.4}, updated_by="admin"
        )
        assert (s.min_confidence_floor, s.model_confidence_cap) == (0.5, 0.4)

    def test_the_gap_no_longer_makes_approved_facts_unreachable(self):
        """The gap is kept, but it stopped being the thing that hid approved facts.

        It used to double as a permanent ceiling: a fact capped at 0.79 stayed at 0.79 after
        approval, under a 0.80 floor, and an admin could not lower the floor to reach it
        because doing so closes the gap and is refused. So the cap's only live effect was
        blocking *approved* facts -- which is not what it is for. Approval now rescales instead,
        which needs no settings change at all.
        """
        s = GovernanceSettings()
        capped = s.effective_model_confidence(0.99)
        assert capped < s.min_confidence_floor
        assert answerable_confidence(capped) >= s.min_confidence_floor
        # And the admin still cannot reach it by lowering the floor, which is why the rescale
        # had to be the fix rather than a configuration change.
        with pytest.raises(GovernanceError):
            s.apply({"min_confidence_floor": 0.5}, updated_by="admin")

    def test_an_unreviewed_claim_still_cannot_clear_the_floor_on_its_own(self):
        """What the gap is actually for, and it has to keep holding."""
        s = GovernanceSettings()
        for claimed in (0.8, 0.95, 1.0):
            assert s.effective_model_confidence(claimed) < s.min_confidence_floor


class TestClamping:
    def test_overconfident_model_is_clamped(self):
        assert GovernanceSettings().effective_model_confidence(0.99) == 0.79

    def test_modest_model_confidence_is_preserved(self):
        assert GovernanceSettings().effective_model_confidence(0.42) == 0.42

    def test_clamp_follows_configured_cap(self):
        s = GovernanceSettings(min_confidence_floor=0.6, model_confidence_cap=0.3)
        assert s.effective_model_confidence(0.9) == 0.3


class TestApplySemantics:
    def test_rejected_patch_leaves_original_untouched(self):
        s = GovernanceSettings()
        with pytest.raises(GovernanceError):
            s.apply({"min_confidence_floor": 0.1}, updated_by="admin")
        assert s.min_confidence_floor == 0.8

    def test_unknown_key_rejected(self):
        with pytest.raises(GovernanceError, match="unknown settings"):
            GovernanceSettings().apply({"nonexistent": 1}, updated_by="admin")

    def test_updated_by_recorded(self):
        s = GovernanceSettings().apply({"vector_top_k": 5}, updated_by="alice@firm.com")
        assert s.updated_by == "alice@firm.com"

    def test_apply_returns_new_instance(self):
        original = GovernanceSettings()
        assert GovernanceSettings().apply({"vector_top_k": 9}, updated_by="a") is not original


class TestBounds:
    @pytest.mark.parametrize("depth", [0, 6, 99])
    def test_absurd_traversal_depth_rejected(self, depth):
        with pytest.raises(GovernanceError, match="graph_expand_depth"):
            GovernanceSettings(graph_expand_depth=depth)

    def test_zero_top_k_rejected(self):
        with pytest.raises(GovernanceError, match="vector_top_k"):
            GovernanceSettings(vector_top_k=0)

    @pytest.mark.parametrize("floor", [-0.1, 1.5])
    def test_floor_outside_unit_range_rejected(self, floor):
        with pytest.raises(GovernanceError):
            GovernanceSettings(min_confidence_floor=floor)

    @pytest.mark.parametrize("calls", [0, MAX_COMPOSE_CALLS_CEILING + 1, 300])
    def test_search_budget_outside_range_rejected(self, calls):
        """Bounded above as well as below, unlike the other reach settings. This one is a spend cap,
        so a typo'd 300 is not a stricter setting but a far laxer one, and what it invites is a bill
        rather than a refusal."""
        with pytest.raises(GovernanceError, match="max_compose_calls"):
            GovernanceSettings(max_compose_calls=calls)

    def test_the_ceiling_itself_is_allowed(self):
        assert GovernanceSettings(max_compose_calls=MAX_COMPOSE_CALLS_CEILING).max_compose_calls

    @pytest.mark.parametrize("limit", [0, GRAPH_EXPAND_LIMIT_CEILING + 1])
    def test_walk_limit_outside_range_rejected(self, limit):
        """Bounded above because past the ceiling the walk stops being evidence and becomes a dump,
        and bounded below because the cap applies to the whole walk: set it under the number of
        edges at one hop and `graph_expand_depth` silently stops meaning anything."""
        with pytest.raises(GovernanceError, match="graph_expand_limit"):
            GovernanceSettings(graph_expand_limit=limit)

    def test_the_walk_limit_ceiling_itself_is_allowed(self):
        settings = GovernanceSettings(graph_expand_limit=GRAPH_EXPAND_LIMIT_CEILING)
        assert settings.graph_expand_limit == GRAPH_EXPAND_LIMIT_CEILING


class TestKillSwitches:
    def test_default_permissive_for_queries(self):
        """Ungoverned queries are allowed until an admin decides otherwise."""
        assert GovernanceSettings().block_ungoverned_queries is False

    def test_switches_are_settable(self):
        s = GovernanceSettings().apply(
            {"block_ungoverned_queries": True, "block_model_extraction": True},
            updated_by="admin",
        )
        assert s.block_ungoverned_queries and s.block_model_extraction


class TestUiHelp:
    def test_every_configurable_field_has_help(self):
        """A control a lawyer cannot understand is a control they will set wrongly."""
        # Fields, not `to_dict()`: that also carries `retrieval_direction`, which is derived from
        # `allowed_tiers` and has nothing for an administrator to set.
        configurable = {f.name for f in fields(GovernanceSettings)} - {"updated_by", "updated_at"}
        assert configurable == set(FIELD_HELP)

    def test_a_derived_key_is_not_offered_as_a_control(self):
        """`to_dict` carries more than the fields. An Admin form built from it would render a
        read-only value as an editable one, and a write of it would be silently ignored."""
        derived = set(GovernanceSettings().to_dict()) - {f.name for f in fields(GovernanceSettings)}
        assert derived == {"retrieval_direction"}
        assert derived.isdisjoint(FIELD_HELP)

    def test_embedding_model_help_warns_about_migration(self):
        assert "RE-PROCESSING" in FIELD_HELP["embedding_model"]

    def test_help_avoids_jargon(self):
        """Written for a lawyer-administrator, not a data engineer."""
        for text in FIELD_HELP.values():
            lowered = text.lower()
            assert "epistemic" not in lowered
            assert "cypher" not in lowered
