"""API integration tests.

These exercise the boundary rather than the units: the point is that authentication,
scoping, the review gate and the governance validator survive being wired together
through HTTP. Several of these would pass at the unit level and fail here.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.deps import get_services
from src.config import AuthConfig, GraphConfig, LexGraphConfig
from src.graph.assertions import EpistemicClass, SourceLocator, build_assertion
from src.graph.scope import AuthContext

TENANT = "dev-tenant"


def _config(**over) -> LexGraphConfig:
    cfg = LexGraphConfig(
        environment="local",
        auth=AuthConfig(dev_bypass_tenant=TENANT),
        graph=GraphConfig(uri="bolt://127.0.0.1:1", user="none", password="none"),
    )
    for k, v in over.items():
        setattr(cfg, k, v)
    cfg.validate()
    return cfg


def _with_example_metrics() -> None:
    """Give the app under test a metric matcher.

    Metrics now live in the graph and the matcher is built per request from a tenant's
    approved metrics, so nothing is loaded at startup. These tests are about tier 1
    behaviour rather than about where a definition is stored, so the example pack is
    injected directly. `Services.metric_matcher` exists for exactly this.
    """
    from src.api.deps import load_example_pack
    from src.metrics.models import StaticCatalog
    from src.query.metric_matcher import MetricMatcher

    metrics = load_example_pack()
    if metrics:
        get_services().metric_matcher = MetricMatcher(metrics, StaticCatalog(tables={}))


@pytest.fixture
def client() -> TestClient:
    app = create_app(_config())
    # No lifespan: the graph is intentionally unreachable here, and these tests are
    # about the HTTP boundary rather than graph connectivity.
    _with_example_metrics()
    return TestClient(app)


@pytest.fixture
def ctx() -> AuthContext:
    return AuthContext(user_id="dev@localhost", tenant_id=TENANT)


def _stage_model_assertion(confidence: float = 0.7, matter_id: str = "M-1") -> str:
    services = get_services()
    ctx = AuthContext(user_id="dev@localhost", tenant_id=TENANT)
    a = build_assertion(
        tenant_id=TENANT,
        subject_id="Doc-1",
        predicate="CONCERNS_TOPIC",
        object_id="Topic-Antitrust",
        epistemic_class=EpistemicClass.EXTRACTED_MODEL,
        method="llm:claude-sonnet-5",
        confidence=confidence,
        source_locator=SourceLocator(
            document_id="doc-1",
            filename="memorandum.pdf",
            page=2,
            chunk_id="doc-1:c1",
            quote="the Adverse Party",
        ),
        matter_id=matter_id,
    )
    services.review_queue.stage(ctx, [a])
    return a.assertion_id


class TestHealth:
    def test_health_reports_degraded_without_graph(self, client):
        body = client.get("/health").json()
        assert body["status"] == "degraded"
        assert body["graph"] == "disconnected"

    def test_health_flags_dev_bypass(self, client):
        """An operator must be able to see that auth is bypassed."""
        assert client.get("/health").json()["dev_auth_bypass"] is True


class TestOntologyEndpoint:
    def test_splits_governing_from_descriptive(self):
        """Split rather than flat-with-a-boolean: "which predicates are closed?" is the
        question an administrator asks, and the Glossary renders the two lists apart."""
        body = TestClient(create_app(_config())).get("/api/ontology/legal").json()
        assert body["domain"] == "legal"
        assert any(p["id"] == "ADVERSE_TO" for p in body["governing_predicates"])
        assert any(p["id"] == "CONCERNS_TOPIC" for p in body["descriptive_predicates"])
        assert all(p["governing"] for p in body["governing_predicates"])
        assert not any(p["governing"] for p in body["descriptive_predicates"])

    def test_predicates_carry_domain_range_and_help(self, client):
        body = client.get("/api/ontology/legal").json()
        adverse = next(p for p in body["governing_predicates"] if p["id"] == "ADVERSE_TO")
        assert adverse["symmetric"] is True
        assert adverse["range"] == ["Party"]
        assert adverse["help"]

    def test_rules_expose_premise_floor(self, client):
        """The Glossary tells a reader which class a rule will fire on."""
        rules = client.get("/api/ontology/legal").json()["rules"]
        assert all(r["min_premise_class"] for r in rules)

    def test_unknown_domain_404s(self, client):
        assert client.get("/api/ontology/aerospace").status_code == 404

    def test_healthcare_pack_also_serves(self, client):
        assert client.get("/api/ontology/healthcare").json()["domain"] == "healthcare"


class TestTenantIsolation:
    def test_path_tenant_must_match_token(self, client):
        """404, not 403 — confirming another tenant exists is itself a leak."""
        assert client.get("/api/tenants/other-firm/dashboard").status_code == 404

    def test_own_tenant_is_reachable(self, client):
        assert client.get(f"/api/tenants/{TENANT}/dashboard").status_code == 200


class TestReviewQueue:
    def test_pending_assertion_appears(self, client):
        aid = _stage_model_assertion()
        body = client.get(f"/api/tenants/{TENANT}/assertions").json()
        assert aid in [a["assertion_id"] for a in body["assertions"]]

    def test_below_floor_is_flagged(self, client):
        """A reviewer needs to know the claim is not currently shaping answers."""
        _stage_model_assertion(confidence=0.7)
        body = client.get(f"/api/tenants/{TENANT}/assertions").json()
        assert body["assertions"][0]["below_floor"] is True
        assert body["confidence_floor"] == 0.8

    def test_queue_is_least_confident_first(self, client):
        _stage_model_assertion(confidence=0.75, matter_id="M-A")
        _stage_model_assertion(confidence=0.5, matter_id="M-B")
        body = client.get(f"/api/tenants/{TENANT}/assertions").json()
        confidences = [a["confidence"] for a in body["assertions"]]
        assert confidences == sorted(confidences)

    def test_approve_transitions_state(self, client):
        aid = _stage_model_assertion()
        r = client.post(f"/api/tenants/{TENANT}/assertions/{aid}/approve")
        assert r.status_code == 200
        assert r.json()["review_state"] == "APPROVED"

    def test_reject_requires_a_reason(self, client):
        aid = _stage_model_assertion()
        assert (
            client.post(f"/api/tenants/{TENANT}/assertions/{aid}/reject", json={}).status_code
            == 422
        )

    def test_reject_records_reason(self, client):
        aid = _stage_model_assertion()
        r = client.post(
            f"/api/tenants/{TENANT}/assertions/{aid}/reject",
            json={"reason": "misread the caption"},
        )
        assert r.json()["review_state"] == "REJECTED"

    def test_rejected_cannot_be_approved(self, client):
        """409: the claim existed and was withdrawn, which is a conflict not a 404."""
        aid = _stage_model_assertion()
        client.post(f"/api/tenants/{TENANT}/assertions/{aid}/reject", json={"reason": "wrong"})
        assert client.post(f"/api/tenants/{TENANT}/assertions/{aid}/approve").status_code == 409

    def test_unknown_assertion_404s(self, client):
        assert client.post(f"/api/tenants/{TENANT}/assertions/nope/approve").status_code == 404


class TestProvenance:
    def test_returns_the_citation(self, client):
        """File, page and quote — what a reviewer opens the PDF and searches for."""
        aid = _stage_model_assertion()
        body = client.get(f"/api/tenants/{TENANT}/assertions/{aid}/provenance").json()
        src = body["assertion"]["source_locator"]
        assert src["document_id"] == "doc-1"
        assert src["filename"] == "memorandum.pdf"
        assert src["page"] == 2
        assert src["quote"] == "the Adverse Party"

    def test_explanation_is_plain_language(self, client):
        """The whole product is answering "why does it believe this?" to a lawyer."""
        aid = _stage_model_assertion()
        body = client.get(f"/api/tenants/{TENANT}/assertions/{aid}/provenance").json()
        explanation = body["explanation"].lower()
        assert "ai model" in explanation
        assert "epistemic" not in explanation


class TestGovernance:
    def test_settings_ship_with_help(self, client):
        body = client.get(f"/api/tenants/{TENANT}/governance").json()
        assert set(body["help"]) >= {"min_confidence_floor", "model_confidence_cap"}

    def test_valid_patch_applies(self, client):
        r = client.patch(f"/api/tenants/{TENANT}/governance", json={"min_confidence_floor": 0.9})
        assert r.json()["settings"]["min_confidence_floor"] == 0.9

    def test_closing_the_cap_floor_gap_is_refused(self, client):
        """The invariant that keeps unreviewed model output out of answers."""
        r = client.patch(f"/api/tenants/{TENANT}/governance", json={"min_confidence_floor": 0.5})
        assert r.status_code == 422
        assert "must stay below" in r.json()["detail"]

    def test_embedding_model_change_warns_about_migration(self, client):
        r = client.patch(
            f"/api/tenants/{TENANT}/governance", json={"embedding_model": "other-model"}
        )
        assert any("re-process" in w.lower() for w in r.json()["warnings"])

    def test_disabling_closed_vocabulary_warns(self, client):
        r = client.patch(
            f"/api/tenants/{TENANT}/governance", json={"enforce_closed_vocabulary": False}
        )
        assert any("conflict check" in w.lower() for w in r.json()["warnings"])

    def test_unknown_setting_rejected(self, client):
        r = client.patch(f"/api/tenants/{TENANT}/governance", json={"nope": 1})
        assert r.status_code == 422


class TestQuery:
    def test_kill_switch_refuses_only_ungoverned(self, client):
        """403 for a question no tier can answer — well-formed, deliberately refused.

        The phrasing matters: a question a *governed metric* can answer is unaffected
        by the switch, which is the whole point of the switch. So this uses a question
        with no metric and no matching assertions.
        """
        client.patch(f"/api/tenants/{TENANT}/governance", json={"block_ungoverned_queries": True})
        r = client.post(
            f"/api/tenants/{TENANT}/query",
            json={"query": "zzzq nonexistent gibberish topic"},
        )
        assert r.status_code == 403
        assert "administrator" in r.json()["detail"].lower()

    def test_kill_switch_does_not_block_governed_metrics(self, client):
        """A governed answer is still allowed while ungoverned queries are off."""
        client.patch(f"/api/tenants/{TENANT}/governance", json={"block_ungoverned_queries": True})
        r = client.post(
            f"/api/tenants/{TENANT}/query", json={"query": "what is our realization rate"}
        )
        assert r.status_code == 200
        body = r.json()
        assert body["tier"] == 1
        assert body["governed"] is True
        assert "SELECT" in body["sql"]

    def test_governed_metric_compiles_deterministic_sql(self, client):
        """Tier 1 returns SQL a reviewer can read, with no model in the path."""
        r = client.post(
            f"/api/tenants/{TENANT}/query",
            json={"query": "show me fees billed by month", "execute": False},
        )
        body = r.json()
        assert body["tier_name"] == "GOVERNED_METRIC"
        assert "SELECT" in body["sql"]
        assert "no AI involved" in body["explanation"]

    def test_query_reports_tier_and_governance(self, client):
        r = client.post(f"/api/tenants/{TENANT}/query", json={"query": "anything"})
        body = r.json()
        assert "tier" in body and "governed" in body and "explanation" in body

    def test_empty_query_rejected(self, client):
        assert client.post(f"/api/tenants/{TENANT}/query", json={"query": ""}).status_code == 422


class TestGraphNeighbourhood:
    def test_returns_edges_around_a_node(self, client):
        _stage_model_assertion()
        client.get(f"/api/tenants/{TENANT}/assertions")
        r = client.get(f"/api/tenants/{TENANT}/graph/neighbourhood", params={"node_id": "Doc-1"})
        assert r.status_code == 200
        assert r.json()["confidence_floor"] == 0.8

    def test_depth_is_bounded(self, client):
        r = client.get(
            f"/api/tenants/{TENANT}/graph/neighbourhood",
            params={"node_id": "Doc-1", "depth": 9},
        )
        assert r.status_code == 422

    def test_no_node_returns_an_overview_rather_than_422(self, client):
        """The explorer opens before anything is selected, so its first request names no node.
        Requiring one meant that request was a 422 and the page never rendered -- which reads as
        "the graph is empty" rather than "you have not picked a starting point"."""
        _stage_model_assertion()
        r = client.get(f"/api/tenants/{TENANT}/graph/neighbourhood")

        assert r.status_code == 200
        body = r.json()
        assert len(body["edges"]) >= 1
        assert len(body["nodes"]) >= 1

    def test_the_overview_is_capped(self, client):
        """A firm's whole graph is not a diagram: drawing every edge is an unreadable hairball
        that also freezes the browser."""
        _stage_model_assertion()
        r = client.get(f"/api/tenants/{TENANT}/graph/neighbourhood", params={"limit": 1})

        body = r.json()
        assert len(body["edges"]) == 1
        assert body["total_edges"] >= 1

    def test_the_overview_says_when_it_truncated(self, client):
        """A silently truncated graph is worse than a small one, because the reader believes they
        are looking at everything.

        Distinct subjects, not repeated confidences: assertion ids are content-addressed, so the
        same claim staged three times collapses to one edge and there is nothing to truncate.
        """
        services = get_services()
        ctx = AuthContext(user_id="dev@localhost", tenant_id=TENANT)
        for i in range(3):
            services.review_queue.stage(
                ctx,
                [
                    build_assertion(
                        tenant_id=TENANT,
                        subject_id=f"Doc-{i}",
                        predicate="CONCERNS_TOPIC",
                        object_id=f"Topic-{i}",
                        epistemic_class=EpistemicClass.EXTRACTED_MODEL,
                        method="llm:test@v1",
                        confidence=0.7,
                        source_locator=SourceLocator(
                            document_id=f"doc-{i}", filename="f.pdf", page=1, quote="a quote"
                        ),
                    )
                ],
            )
        r = client.get(f"/api/tenants/{TENANT}/graph/neighbourhood", params={"limit": 1})

        body = r.json()
        assert body["truncated"] is True
        assert body["total_edges"] >= 3


SCREEN_REASON = "acted for the opposing party in 2024"
SCREEN_CONTACT = "risk@firm.com"
SCREENED_MATTER = "M-2291"


def _client_as(ctx: AuthContext) -> TestClient:
    """A client authenticating as `ctx`.

    Patches `authenticate` rather than the access store because that is the one seam that
    leaves claim extraction and `AuthContext` construction exactly as they ship. `create_app`
    installs a fresh `Services`, so stage assertions *after* calling this or they go into the
    review queue this app just replaced.
    """
    app = create_app(_config())
    services = get_services()
    original = services.authenticator.authenticate

    def as_ctx(*args, **kw):
        _, grants = original(*args, **kw)
        return ctx, grants

    services.authenticator.authenticate = as_ctx
    return TestClient(app)


def _screened_client() -> TestClient:
    """A client whose caller is screened from `SCREENED_MATTER`, with a reason and contact."""
    return _client_as(
        AuthContext(
            user_id="bob@firm.com",
            tenant_id=TENANT,
            matter_denylist=frozenset({SCREENED_MATTER}),
            screen_reasons={SCREENED_MATTER: SCREEN_REASON},
            screen_contacts={SCREENED_MATTER: SCREEN_CONTACT},
        )
    )


class TestMatters:
    def test_unscreened_caller_withholds_nothing(self, client):
        body = client.get(f"/api/tenants/{TENANT}/matters").json()
        assert body["withheld"] == []
        assert body["withheld_count"] == 0

    def test_screened_matters_are_named_with_reason_and_contact(self):
        """The behaviour change. A count alone does not tell a lawyer which client to ask
        about, and "no conflicts found" over a filtered list is the harm a screen prevents."""
        body = _screened_client().get(f"/api/tenants/{TENANT}/matters").json()
        assert body["withheld"] == [
            {
                "matter_id": SCREENED_MATTER,
                "reason": SCREEN_REASON,
                "contact": SCREEN_CONTACT,
            }
        ]
        assert body["withheld_count"] == 1

    def test_screened_matters_are_never_mixed_into_the_visible_list(self):
        """Two lists rather than one with a flag: a caller that iterates `matters` must not
        be able to reach a screened matter by forgetting to check a boolean."""
        screened = _screened_client()
        _stage_model_assertion(matter_id=SCREENED_MATTER)
        _stage_model_assertion(matter_id="M-OPEN")
        body = screened.get(f"/api/tenants/{TENANT}/matters").json()
        assert SCREENED_MATTER not in [m["matter_id"] for m in body["matters"]]
        assert "M-OPEN" in [m["matter_id"] for m in body["matters"]]
        assert all("walled" not in m for m in body["matters"])


class TestScreenedRefusalIsNamed:
    """403 with a reason and a contact, not the old blanket 404.

    A 404 for a screen tells a lawyer the matter does not exist, which is misleading about
    a decision their own firm documented and acknowledged.
    """

    def test_screened_assertion_is_403_not_404(self):
        screened = _screened_client()
        aid = _stage_model_assertion(matter_id=SCREENED_MATTER)
        r = screened.get(f"/api/tenants/{TENANT}/assertions/{aid}/provenance")
        assert r.status_code == 403

    def test_the_403_names_the_matter_the_reason_and_the_contact(self):
        screened = _screened_client()
        aid = _stage_model_assertion(matter_id=SCREENED_MATTER)
        body = screened.get(f"/api/tenants/{TENANT}/assertions/{aid}/provenance").json()
        assert body["decision"] == "SCREENED"
        assert body["matter_id"] == SCREENED_MATTER
        assert body["reason"] == SCREEN_REASON
        assert body["contact"] == SCREEN_CONTACT
        assert SCREENED_MATTER in body["detail"]

    def test_approving_a_screened_assertion_is_403(self):
        screened = _screened_client()
        aid = _stage_model_assertion(matter_id=SCREENED_MATTER)
        r = screened.post(f"/api/tenants/{TENANT}/assertions/{aid}/approve")
        assert r.status_code == 403
        assert SCREEN_CONTACT in r.json()["detail"]

    def test_unknown_assertion_still_404s_and_reveals_nothing(self):
        """The disclosure is about screens, not about ids. An id that does not exist must
        not become a way to enumerate what does."""
        r = _screened_client().get(f"/api/tenants/{TENANT}/assertions/no-such-id/provenance")
        assert r.status_code == 404
        assert SCREENED_MATTER not in r.text

    def test_an_assignment_gap_is_404_not_403(self):
        """Only a screen is disclosed. An allowlist that simply omits a matter says nothing
        about whether the matter is real, because nobody decided anything about this user."""
        narrow = _client_as(
            AuthContext(
                user_id="bob@firm.com",
                tenant_id=TENANT,
                matter_allowlist=frozenset({"M-MINE"}),
            )
        )
        aid = _stage_model_assertion(matter_id="M-OTHER")
        r = narrow.get(f"/api/tenants/{TENANT}/assertions/{aid}/provenance")
        assert r.status_code == 404

    def test_cross_tenant_is_still_silent_and_still_404(self):
        """Must not regress. A boundary between two firms is confidentiality, not a screen
        agreed with anyone, so it names nothing."""
        r = _screened_client().get("/api/tenants/other-firm/matters")
        assert r.status_code == 404
        assert "other-firm" not in r.json()["detail"]
        assert SCREENED_MATTER not in r.text
