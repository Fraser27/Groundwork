"""Tests for runtime governance settings.

The cap/floor relationship is the one that matters. It is defence in depth rather
than policy: the gap between them is what keeps an unreviewed model assertion out of
answers even if the review gate itself were bypassed.
"""

from __future__ import annotations

import pytest

from src.governance import FIELD_HELP, GovernanceError, GovernanceSettings


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
        configurable = set(GovernanceSettings().to_dict()) - {"updated_by", "updated_at"}
        assert configurable == set(FIELD_HELP)

    def test_embedding_model_help_warns_about_migration(self):
        assert "RE-PROCESSING" in FIELD_HELP["embedding_model"]

    def test_help_avoids_jargon(self):
        """Written for a lawyer-administrator, not a data engineer."""
        for text in FIELD_HELP.values():
            lowered = text.lower()
            assert "epistemic" not in lowered
            assert "cypher" not in lowered
