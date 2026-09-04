"""One place builds a planner, so two callers cannot disagree about what a question got.

`/query/compose` and the `compose` MCP tool both need a `Planner` wired to seven
collaborators. When the route built its own, adding the MCP tool meant copying that wiring,
and the copy would drift in the direction that shows least: one caller getting a `sql_lane`
and the other not means the same question is governed differently depending on whether a
person or an agent asked it. Nothing would fail; the answers would just differ.
"""

from __future__ import annotations

from src.api.deps import build_services
from src.config import AuthConfig, GraphConfig, GroundworkConfig

TENANT = "demo-firm"


def _services():
    cfg = GroundworkConfig(
        environment="local",
        auth=AuthConfig(dev_bypass_tenant=TENANT),
        graph=GraphConfig(uri="bolt://127.0.0.1:1", user="none", password="none"),
    )
    cfg.validate()
    return build_services(cfg)


class TestTheResolverAndPlannerAgree:
    def test_both_get_the_same_kind_of_sql_lane(self):
        """The lane that decides whether a question can reach model-written SQL. If one caller
        has it and the other does not, `governed` means different things per endpoint."""
        services = _services()
        resolver_lane = services.build_resolver(TENANT)._sql
        planner_lane = services.build_planner(TENANT)._sql

        assert type(resolver_lane) is type(planner_lane)
        assert planner_lane is not None

    def test_both_get_the_same_kind_of_router(self):
        services = _services()
        assert type(services.build_resolver(TENANT)._router) is type(
            services.build_planner(TENANT)._router
        )

    def test_an_injected_matcher_reaches_both(self):
        """Tests inject a matcher to exercise tier 1 without a graph. Honouring it in one
        builder and not the other would make a tier-1 test pass through one caller and fail
        through the other for reasons that have nothing to do with tier 1."""
        services = _services()
        sentinel = object()
        services.metric_matcher = sentinel

        assert services.build_resolver(TENANT)._metrics is sentinel
        assert services.build_planner(TENANT)._metrics is sentinel


class TestSynthesisIsTheOneDeliberateDifference:
    def test_the_web_route_gets_a_writer_by_default(self):
        """A person reading the page wants prose over the parts."""
        services = _services()
        services.config.models.synthesis_model = "some.model"
        assert services.build_planner(TENANT)._synthesiser is not None

    def test_an_agent_can_refuse_one(self):
        """The agent is the writer. A second model's paragraph underneath its own would be a
        second ungoverned layer, and nobody could say which of them added a claim."""
        services = _services()
        services.config.models.synthesis_model = "some.model"
        assert services.build_planner(TENANT, synthesise=False)._synthesiser is None

    def test_no_model_configured_is_not_an_error(self):
        """Without Bedrock the parts and their citations are still the answer. Refusing the
        question would trade a complete result for no result."""
        services = _services()
        services.config.models.synthesis_model = ""
        assert services.build_planner(TENANT)._synthesiser is None


class TestTheTenantsSynthesisModelIsHonoured:
    """Admin has offered this dropdown all along while `build_synthesiser` read only the
    deployment config, so the save failed and the value would have been ignored anyway."""

    def test_the_tenants_choice_reaches_the_synthesiser(self):
        services = _services()
        services.config.models.synthesis_model = "deployment.model"
        settings = services.settings_for(TENANT)
        services.save_settings(
            TENANT, settings.apply({"synthesis_model": "tenant.model"}, updated_by="t")
        )

        assert services.build_planner(TENANT)._synthesiser.model_id == "tenant.model"

    def test_an_unset_deployment_model_cannot_be_overruled_by_a_tenant(self):
        """No synthesis model deployed means no Bedrock access for it, and a firm cannot grant
        itself access by picking from a dropdown. The deployment vetoes, the tenant chooses."""
        services = _services()
        services.config.models.synthesis_model = ""
        settings = services.settings_for(TENANT)
        services.save_settings(
            TENANT, settings.apply({"synthesis_model": "tenant.model"}, updated_by="t")
        )

        assert services.build_planner(TENANT)._synthesiser is None

    def test_saving_it_is_not_an_unknown_setting(self):
        """The reported bug: `unknown settings: ['synthesis_model']` from a control the page
        renders next to three that do save."""
        settings = _services().settings_for(TENANT)
        assert settings.apply({"synthesis_model": "x"}, updated_by="t").synthesis_model == "x"
