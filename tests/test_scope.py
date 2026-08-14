"""Tests for tenant/matter scoping.

The property under test is the one that matters most in a whitelabel deployment:
*no query can be built without scoping*. A regression here is a cross-firm
privilege breach, not a bug report.
"""

from __future__ import annotations

import pytest

from src.access import AccessDecision, AccessManager
from src.graph.assertions import EpistemicClass
from src.graph.scope import AuthContext, ScopeViolation, edge_scope, node_scope

CTX = AuthContext(user_id="alice@firm.com", tenant_id="firm-acme")


class TestTenantIsolation:
    def test_tenant_filter_always_present(self):
        assert "tenant_id = $scope_tenant" in edge_scope(CTX).where

    def test_tenant_param_bound(self):
        assert edge_scope(CTX).params["scope_tenant"] == "firm-acme"

    def test_cluster_key_is_tenant_derived(self):
        """A seam for per-tenant storage split; today used for resource naming."""
        assert CTX.cluster_key() == "tenant-firm-acme"

    @pytest.mark.parametrize(
        "bad",
        [
            "Firm-ACME",              # uppercase
            "firm acme",              # space
            "a",                      # too short
            "firm'; DROP GRAPH--",    # injection attempt
            "../other-tenant",        # traversal attempt
            "",
        ],
    )
    def test_malformed_tenant_ids_rejected(self, bad):
        """Tenant ids reach a graph name, so they are validated, not escaped."""
        with pytest.raises(ScopeViolation):
            AuthContext(user_id="u", tenant_id=bad)

    def test_tenant_rejection_carries_no_screen_decision(self):
        """Cross-tenant stays silent. A tenant-level refusal must not arrive at the HTTP
        layer looking like a disclosable screen, or it would be answered 403 and named."""
        with pytest.raises(ScopeViolation) as e:
            AuthContext(user_id="u", tenant_id="Firm-ACME")
        assert e.value.decision is None
        assert e.value.is_screen is False

    def test_narrowing_cannot_widen(self):
        scoped = edge_scope(CTX).and_("r.confidence > 0.9")
        assert "tenant_id = $scope_tenant" in scoped.where


#: A screened context with the reason and contact a real screen carries.
SCREENED_CTX = AuthContext(
    user_id="bob@firm.com",
    tenant_id="firm-acme",
    matter_denylist=frozenset({"M-2291"}),
    screen_reasons={"M-2291": "acted for the opposing party in 2024"},
    screen_contacts={"M-2291": "risk@firm.com"},
)


class TestMatterWalls:
    def test_no_allowlist_means_whole_tenant(self):
        assert "scope_matters" not in edge_scope(CTX).params

    def test_allowlist_restricts(self):
        ctx = AuthContext(user_id="u", tenant_id="firm-acme", matter_allowlist=frozenset({"M-1"}))
        assert ctx.can_read_matter("M-1")
        assert not ctx.can_read_matter("M-2")

    def test_denylist_beats_allowlist(self):
        """Cedar `forbid` semantics: a wall must not be defeatable by a broad role."""
        ctx = AuthContext(
            user_id="u",
            tenant_id="firm-acme",
            matter_allowlist=frozenset({"M-1", "M-2"}),
            matter_denylist=frozenset({"M-2"}),
        )
        assert ctx.can_read_matter("M-1")
        assert not ctx.can_read_matter("M-2")

    def test_denylist_beats_full_tenant_access(self):
        ctx = AuthContext(
            user_id="u", tenant_id="firm-acme", matter_denylist=frozenset({"M-9"})
        )
        assert ctx.can_read_matter("M-1")
        assert not ctx.can_read_matter("M-9")

    def test_decision_survives_a_truthiness_check(self):
        """`can_read_matter` returns a decision now. Every existing `if not ...` call site
        depends on that staying falsey for a refusal."""
        assert bool(SCREENED_CTX.can_read_matter("M-1")) is True
        assert bool(SCREENED_CTX.can_read_matter("M-2291")) is False


class TestScreensAreNamed:
    """The behaviour change: within one firm a screen is disclosed, not silently applied.

    "No conflicts found" when three screened matters matched is the harm the screen exists
    to prevent, so the refusal names the matter and routes the person somewhere.
    """

    def test_screened_matter_is_named_with_reason_and_contact(self):
        with pytest.raises(ScopeViolation) as e:
            SCREENED_CTX.assert_can_read_matter("M-2291")
        msg = str(e.value)
        assert "M-2291" in msg
        assert "acted for the opposing party in 2024" in msg
        assert "risk@firm.com" in msg

    def test_screen_fields_are_structured_not_only_prose(self):
        """The HTTP layer builds a JSON body from these. Parsing the sentence back apart is
        how the 403 and the message eventually disagree."""
        with pytest.raises(ScopeViolation) as e:
            SCREENED_CTX.assert_can_read_matter("M-2291")
        assert e.value.decision is AccessDecision.SCREENED
        assert e.value.matter_id == "M-2291"
        assert e.value.contact == "risk@firm.com"
        assert e.value.reason == "acted for the opposing party in 2024"
        assert e.value.is_screen is True

    def test_a_screen_without_a_contact_still_routes_somewhere(self):
        """A wall with nowhere to appeal is not a documented screen."""
        ctx = AuthContext(
            user_id="u",
            tenant_id="firm-acme",
            matter_denylist=frozenset({"M-7"}),
            screen_reasons={"M-7": "conflict"},
        )
        with pytest.raises(ScopeViolation) as e:
            ctx.assert_can_read_matter("M-7")
        assert "risk team" in str(e.value)

    def test_not_assigned_reads_differently_from_screened(self):
        """An assignment gap is somebody forgetting to staff you; a screen is a decision.
        Telling a lawyer to "contact risk" about the first would be nonsense."""
        gap = AuthContext(
            user_id="u", tenant_id="firm-acme", matter_allowlist=frozenset({"M-1"})
        )
        with pytest.raises(ScopeViolation) as e:
            gap.assert_can_read_matter("M-2")
        msg = str(e.value)
        assert e.value.decision is AccessDecision.NOT_ASSIGNED
        assert e.value.is_screen is False
        assert "screened" not in msg.lower()
        assert "matter owner" in msg

    def test_withheld_matters_are_named_for_reporting(self):
        assert SCREENED_CTX.withheld_matters() == [
            {
                "matter_id": "M-2291",
                "reason": "acted for the opposing party in 2024",
                "contact": "risk@firm.com",
            }
        ]

    def test_an_unscreened_user_withholds_nothing(self):
        assert CTX.withheld_matters() == []

    def test_context_from_resolved_access_carries_the_reasons(self):
        """`AuthContext` cannot explain a screen it was not told about, so the seam between
        `AccessManager` and scoping has to carry the reason across."""
        manager = AccessManager()
        manager.assign("firm-acme", "bob", "M-1", actor="owner")
        manager.screen(
            "firm-acme",
            "bob",
            "M-2291",
            actor="risk@firm.com",
            reason="acted for the opposing party",
            contact="risk@firm.com",
        )
        ctx = AuthContext.from_access(manager.resolve("firm-acme", "bob"))

        assert ctx.can_read_matter("M-1") is AccessDecision.ALLOWED
        with pytest.raises(ScopeViolation) as e:
            ctx.assert_can_read_matter("M-2291")
        assert "acted for the opposing party" in str(e.value)
        assert e.value.contact == "risk@firm.com"

    def test_a_context_built_from_bare_grants_still_refuses(self):
        """Degrades rather than fails: a denylist with no reasons attached must still be a
        wall, just a less well explained one."""
        ctx = AuthContext(user_id="u", tenant_id="firm-acme", matter_denylist=frozenset({"M-9"}))
        with pytest.raises(ScopeViolation) as e:
            ctx.assert_can_read_matter("M-9")
        assert e.value.is_screen is True
        assert "M-9" in str(e.value)


class TestEpistemicFiltering:
    def test_predicted_excluded_by_default(self):
        assert EpistemicClass.PREDICTED.value not in edge_scope(CTX).params["scope_classes"]

    def test_predicted_included_only_when_opted_in(self):
        ctx = AuthContext(user_id="u", tenant_id="firm-acme", include_suggestions=True)
        assert EpistemicClass.PREDICTED.value in edge_scope(ctx).params["scope_classes"]

    def test_caller_cannot_smuggle_predicted_in(self):
        """Even an explicit trusted_classes request cannot bypass the flag."""
        scoped = edge_scope(CTX, trusted_classes=frozenset({EpistemicClass.PREDICTED}))
        assert EpistemicClass.PREDICTED.value not in scoped.params["scope_classes"]

    def test_confidence_floor_applied(self):
        assert edge_scope(CTX).params["scope_min_conf"] == 0.8

    def test_pending_review_excluded_by_default(self):
        assert "review_state IN $scope_states" in edge_scope(CTX).where

    def test_review_queue_can_see_pending(self):
        assert "review_state" not in edge_scope(CTX, include_pending=True).where


class TestBitemporalReads:
    def test_current_only_by_default(self):
        assert "superseded_at IS NULL" in edge_scope(CTX).where

    def test_as_of_reconstructs_past_belief(self):
        """'What did the file show when we advised?' — including facts since retracted."""
        scoped = edge_scope(CTX, as_of="2026-01-01T00:00:00Z")
        assert "recorded_at <= $scope_as_of" in scoped.where
        assert "superseded_at > $scope_as_of" in scoped.where
        assert scoped.params["scope_as_of"] == "2026-01-01T00:00:00Z"


class TestNodeScope:
    def test_node_scope_is_tenant_bound(self):
        assert "n.tenant_id = $scope_tenant" in node_scope(CTX).where
