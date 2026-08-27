"""The ungoverned-queries kill switch, on the lane it actually controls.

For most of this project's life it controlled nothing: it gated a fourth tier that had never been
built, was projected to the Admin UI as `kill_switch_active`, and was read nowhere in `src/query/`.
A governance toggle that reports itself active while controlling nothing is the worst direction for
a failure in a legal product, so what these tests assert is the wiring, not the setting.

Two properties carry the file:

- **on removes the SQL lane and nothing else.** Passages and graph facts still come back, and the
  response is 200. Refusing the whole tier would turn a switch that removes an ungoverned
  capability into one that removes governed answers.
- **both endpoints agree.** `/query` and `/query/compose` call the same `SqlLane`, so a question
  cannot be governed on one and ungoverned on the other.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.discovery.catalog_store import CatalogColumn, CatalogTable
from src.governance import GovernanceSettings
from src.graph.scope import AuthContext
from src.query.planner import Lane, Planner, Provenance
from src.query.resolver import UNGOVERNED_BLOCKED, Resolver, Tier
from src.query.sql_generation import SqlGenerator, SqlLane

TENANT = "demo-firm"

MATTERS = CatalogTable(
    full_name="groundwork_legal.matters",
    name="matters",
    database="groundwork_legal",
    source_id="glue",
    description="One row per matter",
    columns=(CatalogColumn("matter_id", "string"), CatalogColumn("client_id", "string")),
)


@pytest.fixture
def ctx() -> AuthContext:
    return AuthContext(tenant_id=TENANT, user_id="alice")


class FakeBedrock:
    def __init__(self) -> None:
        self.calls = 0

    def converse(self, **kwargs):
        self.calls += 1
        sql = "SELECT COUNT(*) FROM groundwork_legal.matters LIMIT 10"
        return {"output": {"message": {"content": [{"text": sql}]}}}


class FakeCatalog:
    def tables(self, tenant_id: str) -> list[CatalogTable]:
        return [MATTERS]


class FakeGraph:
    def search(self, ctx: AuthContext, question: str, **kw: Any) -> list[dict]:
        return []

    def expand(self, ctx: AuthContext, seeds: list[str], **kw: Any) -> list[dict]:
        return [{"assertion_id": "a1", "subject_id": "matter:m1", "epistemic_class": "DECLARED"}]

    def blocking_facts(self, ctx: AuthContext, seeds: list[str], **kw: Any) -> list[dict]:
        return []


class FakeVectors:
    def search(self, ctx: AuthContext, question: str, **kw: Any) -> list[dict]:
        return [{"document_id": "d1", "page": 2, "char_start": 0, "char_end": 40}]


def _lane(bedrock: FakeBedrock) -> SqlLane:
    """No executor: the SQL is generated and not run, which is enough to test the gate."""
    return SqlLane(generator=SqlGenerator(model_id="m", bedrock=bedrock))


def _resolver(bedrock: FakeBedrock) -> Resolver:
    return Resolver(
        graph_reader=FakeGraph(),
        vector_search=FakeVectors(),
        catalog=FakeCatalog(),
        sql_lane=_lane(bedrock),
    )


def _planner(bedrock: FakeBedrock) -> Planner:
    return Planner(
        graph_reader=FakeGraph(),
        vector_search=FakeVectors(),
        catalog=FakeCatalog(),
        sql_lane=_lane(bedrock),
    )


class TestTheSwitchIsActuallyRead:
    def test_off_generates_sql(self, ctx):
        bedrock = FakeBedrock()
        res = _resolver(bedrock).resolve(ctx, "how many matters", GovernanceSettings())
        assert res.generated_sql is not None
        assert bedrock.calls == 1

    def test_on_never_calls_the_model(self, ctx):
        """Not "generates and discards". A blocked lane must not spend a Bedrock call either."""
        bedrock = FakeBedrock()
        settings = GovernanceSettings(block_ungoverned_queries=True)
        res = _resolver(bedrock).resolve(ctx, "how many matters", settings)
        assert res.generated_sql is None
        assert bedrock.calls == 0

    def test_on_leaves_the_answer_governed(self, ctx):
        """A refused lane produced no model-written SQL, so what is left is governed and says so."""
        settings = GovernanceSettings(block_ungoverned_queries=True)
        res = _resolver(FakeBedrock()).resolve(ctx, "how many matters", settings)
        assert res.is_governed

    def test_off_makes_the_answer_ungoverned(self, ctx):
        res = _resolver(FakeBedrock()).resolve(ctx, "how many matters", GovernanceSettings())
        assert not res.is_governed


class TestTheOtherLanesSurvive:
    """The reason this is a lane gate and not a tier refusal."""

    def test_passages_and_facts_still_come_back_on_query(self, ctx):
        settings = GovernanceSettings(block_ungoverned_queries=True)
        res = _resolver(FakeBedrock()).resolve(ctx, "how many matters", settings)
        assert res.tier is Tier.HYBRID
        assert res.answer["passages"]
        assert res.answer["related"]
        assert "generated" not in res.answer

    def test_the_tier_is_not_refused(self, ctx):
        """A raise here would take the passages and the graph facts down with the generator."""
        settings = GovernanceSettings(block_ungoverned_queries=True)
        res = _resolver(FakeBedrock()).resolve(ctx, "how many matters", settings)
        assert res.answer is not None

    def test_compose_keeps_every_other_lane(self, ctx):
        settings = GovernanceSettings(block_ungoverned_queries=True)
        answer = _planner(FakeBedrock()).plan(ctx, "how many matters", settings)
        assert Lane.PASSAGES in answer.lanes_run
        assert Lane.GRAPH in answer.lanes_run
        assert Lane.CATALOG in answer.lanes_run
        assert Lane.SQL not in answer.lanes_run


class TestTheSkipIsNamedAsTheSwitch:
    def test_compose_names_the_reason(self, ctx):
        """"Your administrator turned this off" and "this did not look relevant" are different
        facts, and the first must never be reported in the words of the second."""
        settings = GovernanceSettings(block_ungoverned_queries=True)
        answer = _planner(FakeBedrock()).plan(ctx, "how many matters", settings)
        assert answer.lanes_skipped[Lane.SQL.value] == UNGOVERNED_BLOCKED

    def test_a_tier_cap_is_named_as_the_cap_instead(self, ctx):
        """Removing tier 3 removes this lane too, but for a different reason, and it says so."""
        settings = GovernanceSettings(allowed_tiers=frozenset({1, 2}))
        answer = _planner(FakeBedrock()).plan(ctx, "how many matters", settings)
        assert "not permitted for this tenant" in answer.lanes_skipped[Lane.SQL.value]


class TestTheRefusalIsRecordedAsABacklog:
    """A question people keep asking is a governed metric waiting to be written."""

    def test_the_resolver_records_it(self, ctx):
        settings = GovernanceSettings(block_ungoverned_queries=True)
        resolver = _resolver(FakeBedrock())
        resolver.resolve(ctx, "how many matters", settings)
        assert [b.question for b in resolver.blocked] == ["how many matters"]
        assert resolver.blocked[0].reason == UNGOVERNED_BLOCKED

    def test_the_planner_records_it(self, ctx):
        settings = GovernanceSettings(block_ungoverned_queries=True)
        planner = _planner(FakeBedrock())
        planner.plan(ctx, "how many matters", settings)
        assert [b.question for b in planner.blocked] == ["how many matters"]

    def test_nothing_is_recorded_when_the_switch_is_off(self, ctx):
        resolver = _resolver(FakeBedrock())
        resolver.resolve(ctx, "how many matters", GovernanceSettings())
        assert resolver.blocked == []


class TestBothEndpointsAgree:
    """A question governed on one endpoint and ungoverned on the other would make `governed` mean
    different things depending on which was asked. This repo has hit that bug class repeatedly."""

    @pytest.mark.parametrize("blocked", [False, True])
    def test_the_two_endpoints_agree_on_governed(self, ctx, blocked):
        settings = GovernanceSettings(block_ungoverned_queries=blocked)
        res = _resolver(FakeBedrock()).resolve(ctx, "how many matters", settings)
        answer = _planner(FakeBedrock()).plan(
            ctx, "how many matters", settings, allow_synthesis=False
        )
        composed_ungoverned = any(
            p.provenance is Provenance.MODEL_WRITTEN for p in answer.parts
        )
        assert res.is_governed is not composed_ungoverned

    def test_both_produce_the_same_sql_for_the_same_question(self, ctx):
        res = _resolver(FakeBedrock()).resolve(ctx, "how many matters", GovernanceSettings())
        answer = _planner(FakeBedrock()).plan(
            ctx, "how many matters", GovernanceSettings(), allow_synthesis=False
        )
        sql_part = next(p for p in answer.parts if p.lane is Lane.SQL)
        assert sql_part.sql == res.generated_sql.sql


class TestTheComposedLabelStopsSayingGoverned:
    def test_a_model_written_part_is_never_fully_deterministic(self, ctx):
        answer = _planner(FakeBedrock()).plan(
            ctx, "how many matters", GovernanceSettings(), allow_synthesis=False
        )
        assert not answer.is_fully_deterministic

    def test_the_label_names_it_in_a_lawyers_words(self, ctx):
        """`model_written` is a wire identifier, not a phrase to put in front of a client."""
        answer = _planner(FakeBedrock()).plan(
            ctx, "how many matters", GovernanceSettings(), allow_synthesis=False
        )
        assert "written by AI" in answer.governance_label
        assert "model_written" not in answer.governance_label


class TestOverHttp:
    """200, not 403. The one thing an administrator turning this on must not do is stop the firm
    getting answers."""

    def _client(self):
        from fastapi.testclient import TestClient

        from src.api.app import create_app
        from src.api.deps import get_services
        from src.config import AuthConfig, GraphConfig, GroundworkConfig

        cfg = GroundworkConfig(
            environment="local",
            auth=AuthConfig(dev_bypass_tenant=TENANT),
            graph=GraphConfig(uri="bolt://127.0.0.1:1", user="none", password="none"),
        )
        cfg.validate()
        client = TestClient(create_app(cfg))
        services = get_services()
        services.catalog = FakeCatalog()
        services.graph_reader = FakeGraph()
        return client, services

    def _switch_on(self, client):
        response = client.patch(
            f"/api/tenants/{TENANT}/governance", json={"block_ungoverned_queries": True}
        )
        assert response.status_code == 200, response.text

    def test_query_answers_200_with_the_switch_on(self):
        client, _ = self._client()
        self._switch_on(client)
        response = client.post(f"/api/tenants/{TENANT}/query", json={"query": "how many matters"})
        assert response.status_code == 200, response.text

    def test_compose_answers_200_with_the_switch_on(self):
        client, _ = self._client()
        self._switch_on(client)
        response = client.post(
            f"/api/tenants/{TENANT}/query/compose",
            json={"query": "how many matters", "synthesise": False},
        )
        assert response.status_code == 200, response.text

    def test_compose_names_the_sql_lane_as_skipped(self):
        client, _ = self._client()
        self._switch_on(client)
        body = client.post(
            f"/api/tenants/{TENANT}/query/compose",
            json={"query": "how many matters", "synthesise": False},
        ).json()
        assert body["lanes_skipped"].get("sql") == UNGOVERNED_BLOCKED

    def test_the_refusal_reaches_the_governance_backlog(self):
        """`record_blocked` is drained on the success path now. It was drained only on the
        exception path, so a refusal that returned 200 was lost and the Governance screen showed an
        empty backlog for exactly the tenants most in need of a new metric."""
        client, services = self._client()
        self._switch_on(client)
        client.post(f"/api/tenants/{TENANT}/query", json={"query": "how many matters"})
        # Among them rather than the only one: this deployment has no vector store either, so
        # "no tier could answer" is also recorded and is also true. Both are backlog signals.
        assert UNGOVERNED_BLOCKED in [
            e["reason"] for e in services.blocked_queries.get(TENANT, [])
        ]

    def test_compose_drains_its_backlog_too(self):
        client, services = self._client()
        self._switch_on(client)
        client.post(
            f"/api/tenants/{TENANT}/query/compose",
            json={"query": "how many matters", "synthesise": False},
        )
        assert services.blocked_queries.get(TENANT)
