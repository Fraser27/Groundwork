"""A question may be stricter than its tenant, never laxer.

The Ask page has sent `min_confidence` since the trust-floor control was built. Nothing declared
it on either request model, and Pydantic ignores unknown fields, so it was dropped in silence --
then the page rendered "no fact cleared the trust floor of 0.85" from its own state, naming a
number no read had ever used. A control that reports itself applied while doing nothing is worse
than an absent one, because the reader trusts it.

The clamp is one-way for the same reason the tier router narrows and never widens: the floor is a
governance control, so a caller able to lower it could opt out of one from the Ask page.
"""

from __future__ import annotations

from src.api.routes_query import ComposeRequest, QueryRequest
from src.governance import GovernanceSettings


def settings(**over) -> GovernanceSettings:
    base = {"min_confidence_floor": 0.8, "model_confidence_cap": 0.79}
    base.update(over)
    return GovernanceSettings(**base)


class TestTheFieldSurvivesTheRequest:
    """Declared on both models, because the drop was silent and cost the whole feature."""

    def test_query_accepts_it(self):
        assert QueryRequest(query="q", min_confidence=0.85).min_confidence == 0.85

    def test_compose_accepts_it(self):
        assert ComposeRequest(query="q", min_confidence=0.85).min_confidence == 0.85

    def test_both_models_declare_it(self):
        """Named explicitly: a field on one endpoint and not the other is how `/query` and
        `/query/compose` come to disagree about how strict the same question was."""
        assert "min_confidence" in QueryRequest.model_fields
        assert "min_confidence" in ComposeRequest.model_fields

    def test_absent_means_the_tenant_floor(self):
        assert QueryRequest(query="q").min_confidence is None


class TestTheClampIsOneWay:
    def test_a_higher_floor_is_honoured(self):
        assert settings().with_raised_floor(0.95).min_confidence_floor == 0.95

    def test_a_lower_floor_is_ignored(self):
        """The load-bearing case. Honouring it would let anyone drop the tenant's floor from the
        Ask page, which is opting out of a governance control rather than configuring one."""
        assert settings().with_raised_floor(0.1).min_confidence_floor == 0.8

    def test_an_absent_floor_changes_nothing(self):
        assert settings().with_raised_floor(None).min_confidence_floor == 0.8

    def test_the_tenant_settings_are_not_mutated(self):
        """Returns a copy, like `apply`: a per-question override that edited the tenant's own
        settings would persist one caller's strictness onto everybody else."""
        base = settings()
        base.with_raised_floor(0.95)
        assert base.min_confidence_floor == 0.8

    def test_nothing_else_is_disturbed(self):
        raised = settings(vector_top_k=7).with_raised_floor(0.95)
        assert raised.vector_top_k == 7
        assert raised.model_confidence_cap == 0.79

    def test_a_raised_floor_does_not_break_the_cap_invariant(self):
        """`model_confidence_cap` must stay below the floor. Raising the floor widens that gap,
        so it can never close it -- which is why only this direction is safe to allow at all."""
        raised = settings().with_raised_floor(0.99)
        assert raised.model_confidence_cap < raised.min_confidence_floor
