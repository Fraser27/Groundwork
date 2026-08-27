"""What a reset is allowed to remove.

Reset exists to prove that S3 and Glue are authoritative: throw the derived tiers away, replay,
get the same graph. Metrics break that symmetry. A metric definition is authored in this app
and has no upstream source, so including it in a "rebuild from source" operation is
destruction wearing a rebuild's clothes.

That was found by deploying: a reset wiped six real metric definitions and the response
cheerfully said replay would reconstruct what was removed. It would not.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.admin_ops import ResetScope, reset_derived
from src.api.app import create_app
from src.config import AuthConfig, GraphConfig, GroundworkConfig
from src.graph.scope import AuthContext

TENANT = "demo-firm"
RESET = f"/api/tenants/{TENANT}/admin/reset"


def _services(**over: Any) -> Any:
    class Assertions:
        def all_for_tenant(self, t: str) -> list[int]:
            return [1, 2, 3]

        def drop_tenant(self, t: str) -> int:
            return 3

    class Catalog:
        def __init__(self) -> None:
            self.cleared = False

        def tables(self, t: str) -> list[int]:
            return [1, 2]

        def clear(self, t: str) -> None:
            self.cleared = True

    class Jobs:
        def drop_tenant(self, t: str) -> int:
            return 4

    class Vectors:
        def drop_tenant(self, t: str) -> int:
            return 9

    base = {
        "review_queue": SimpleNamespace(store=Assertions()),
        "graph": None,
        "embedder": Vectors(),
        "job_store": Jobs(),
        "catalog": Catalog(),
    }
    base.update(over)
    return SimpleNamespace(**base)


@pytest.fixture
def ctx() -> AuthContext:
    return AuthContext(tenant_id=TENANT, user_id="alice")


class TestDefaultsPreserveAuthoredWork:
    def test_metrics_are_not_dropped_by_default(self):
        assert ResetScope().metrics is False

    def test_everything_else_is_dropped_by_default(self):
        scope = ResetScope()
        assert (scope.graph, scope.vectors, scope.jobs, scope.catalog) == (True, True, True, True)

    def test_only_metrics_count_as_unrecoverable(self):
        assert ResetScope().destroys_unrecoverable_work is False
        assert ResetScope(metrics=True).destroys_unrecoverable_work is True


class TestScopeIsHonoured:
    def test_each_box_can_be_unticked(self, ctx):
        services = _services()
        report = reset_derived(services, ctx, ResetScope(catalog=False, jobs=False, vectors=False))
        assert report.tables_forgotten == 0
        assert report.jobs_dropped == 0
        assert report.vectors_dropped == 0
        assert services.catalog.cleared is False

    def test_a_full_scope_drops_everything(self, ctx):
        report = reset_derived(_services(), ctx, ResetScope())
        assert report.assertions_dropped == 3
        assert report.vectors_dropped == 9
        assert report.jobs_dropped == 4
        assert report.tables_forgotten == 2


class TestTheNoteTellsTheTruth:
    def test_a_default_reset_says_replay_reconstructs_it(self, ctx):
        note = reset_derived(_services(), ctx).to_dict()["note"]
        assert "still in S3" in note
        assert "CANNOT" not in note

    def test_dropping_metrics_says_they_cannot_be_reconstructed(self, ctx):
        """The old note claimed replay would rebuild what was removed, which was false for a
        metric somebody wrote by hand."""
        report = reset_derived(_services(), ctx, ResetScope(metrics=True))
        report.metrics_dropped = 6
        assert "CANNOT be reconstructed" in report.to_dict()["note"]


class TestTheApiRequiresASecondConfirmation:
    @pytest.fixture
    def client(self) -> TestClient:
        cfg = GroundworkConfig(
            environment="local",
            auth=AuthConfig(dev_bypass_tenant=TENANT),
            graph=GraphConfig(uri="bolt://127.0.0.1:1", user="none", password="none"),
        )
        cfg.validate()
        return TestClient(create_app(cfg))

    def test_a_default_reset_needs_no_confirmation(self, client):
        assert client.post(RESET, json={}).status_code == 200

    def test_dropping_metrics_without_confirming_is_refused(self, client):
        """ "Reset derived data" does not read like "delete work nobody can recover", so the
        checkbox alone is not informed consent."""
        r = client.post(RESET, json={"metrics": True})
        assert r.status_code == 400
        assert "cannot be undone" in r.json()["detail"]

    def test_the_refusal_explains_why_replay_will_not_help(self, client):
        detail = client.post(RESET, json={"metrics": True}).json()["detail"]
        assert "no replay reconstructs them" in detail

    def test_confirming_allows_it(self, client):
        r = client.post(RESET, json={"metrics": True, "confirm_metric_loss": True})
        assert r.status_code == 200
