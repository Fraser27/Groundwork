"""What was asked, which tier answered, and which facts the answer rested on.

The gap this closes is asymmetric. `graph_audit` records how beliefs *changed*, so a wipe is
traceable; nothing recorded that anyone ever *read* the graph, so the two questions a partner
actually asks had no answer: "what did we tell the client, and on what basis?", and its inverse,
"this fact was wrong -- which advice rested on it?".

Two properties are worth more than the rest.

**A refusal is not recorded here.** The kill switch already logs it, it produced no answer, and
double-recording it would make the question log's counts wrong while looking like diligence.

**The assertion ids are stored, not just counted.** The inverse lookup filters on that list, so a
count alone would make "no question used this fact" and "we did not keep the ids" indistinguishable
-- and the first is exculpatory while the second is a hole in the record.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.deps import get_services
from src.config import AuthConfig, GraphConfig, GroundworkConfig
from src.graph_audit import MAX_STORED_IDS
from src.agent.loop import CAPPED_REASONS, RunResult
from src.query.planner import ComposedAnswer, Lane, Part, Provenance
from src.query.resolver import Resolution, Tier
from src.query_audit import (
    ASK_PREFIX,
    MAX_SCAN,
    InMemoryQueryAudit,
    QueryAudit,
    QueryEvent,
    _to_event,
    asked_pk,
    event_for,
    event_for_composed,
    event_for_run,
)

TENANT = "dev-tenant"
OTHER = "other-firm"


def asked(**kw: Any) -> QueryEvent:
    base: dict[str, Any] = {
        "tenant_id": TENANT,
        "actor": "partner@firm.example",
        "question": "who acts for Calder?",
        "tier": 2,
        "tier_name": "GRAPH_TRAVERSAL",
        "governed": True,
        "answered": True,
    }
    return QueryEvent(**{**base, **kw})


class TestWhatIsRecorded:
    def test_a_graph_answer_records_the_facts_it_rested_on(self):
        """The ids, not a count. The inverse lookup has nothing to filter on otherwise."""
        r = Resolution(tier=Tier.GRAPH_TRAVERSAL, answer=[{"x": 1}], assertions_used=["a-1", "a-2"])
        e = event_for(TENANT, "alice", "who acts for Calder?", r)

        assert e.assertion_ids == ("a-1", "a-2")
        assert e.facts_used == 2
        assert e.uses("a-1")
        assert not e.uses("a-9")

    def test_the_tier_is_recorded_not_inferred_later(self):
        """An approved metric and a model-written query are different claims about
        trustworthiness, and reconstructing which one from the SQL afterwards is guessing."""
        governed = event_for(
            TENANT,
            "alice",
            "fees by month",
            Resolution(tier=Tier.GOVERNED_METRIC, answer=[], sql="SELECT 1"),
        )
        assert (governed.tier, governed.tier_name, governed.governed) == (
            1,
            "GOVERNED_METRIC",
            True,
        )

        # Ungoverned is now a property of the answer rather than of the tier: `generated_sql`
        # set means a model wrote the query. It was `tier is not LLM_SQL`, which held only while
        # exactly one tier could involve a model.
        model = event_for(
            TENANT,
            "alice",
            "fees by month",
            Resolution(
                tier=Tier.HYBRID, answer=[], sql="SELECT 1", generated_sql={"sql": "SELECT 1"}
            ),
        )
        assert (model.tier, model.governed) == (3, False)

    def test_an_empty_answer_is_still_recorded_and_marked(self):
        """Having no answer is part of the record. Dropping it would make the log read as though
        the question was never asked."""
        e = event_for(TENANT, "alice", "q", Resolution(tier=Tier.GRAPH_TRAVERSAL, answer=None))
        assert e.answered is False
        assert e.facts_used == 0

    def test_cited_documents_are_recorded_once_each(self):
        """A hybrid answer quotes several passages from one document; the document is one source."""
        r = Resolution(
            tier=Tier.HYBRID,
            answer={},
            citations=[
                {"document_id": "doc-1", "page": 2},
                {"document_id": "doc-1", "page": 7},
                {"document_id": "doc-2", "page": 1},
                {"page": 3},
            ],
        )
        assert event_for(TENANT, "alice", "q", r).document_ids == ("doc-1", "doc-2")

    def test_the_sql_is_kept_so_the_answer_is_reproducible(self):
        r = Resolution(tier=Tier.GOVERNED_METRIC, answer=[], sql="SELECT sum(fees) FROM t")
        assert event_for(TENANT, "alice", "q", r).sql == "SELECT sum(fees) FROM t"


class TestTheInverseLookup:
    def test_it_names_the_questions_that_used_a_fact(self):
        audit = InMemoryQueryAudit()
        audit.append(asked(at="2026-01-01T00:00:00Z", question="a", assertion_ids=("a-1",)))
        audit.append(asked(at="2026-01-02T00:00:00Z", question="b", assertion_ids=("a-2", "a-1")))
        audit.append(asked(at="2026-01-03T00:00:00Z", question="c", assertion_ids=("a-9",)))

        hit = [e.question for e in audit.questions(TENANT) if e.uses("a-1")]
        assert hit == ["b", "a"]

    def test_a_fact_nothing_used_returns_nothing_rather_than_everything(self):
        audit = InMemoryQueryAudit()
        audit.append(asked(assertion_ids=("a-1",)))
        assert [e for e in audit.questions(TENANT) if e.uses("a-2")] == []


class TestTenantIsolation:
    def test_another_firms_questions_are_not_returned(self):
        """The questions a firm asks are as confidential as the answers."""
        audit = InMemoryQueryAudit()
        audit.append(asked(tenant_id=OTHER, question="their strategy"))
        assert audit.questions(TENANT) == []

    def test_the_key_is_scoped_to_the_tenant(self):
        items: list[dict] = []

        class FakeTable:
            def put_item(self, **kw):
                items.append(kw["Item"])
                return {}

            def query(self, **kw):
                return {"Items": []}

        QueryAudit(table=FakeTable()).append(asked())
        assert items[0]["PK"] == asked_pk(TENANT)
        assert items[0]["SK"].startswith(ASK_PREFIX)


class TestItIsAppendOnly:
    def test_a_duplicate_key_is_refused_rather_than_overwritten(self):
        """What makes it append-only rather than merely append-shaped: no later write can rewrite
        the record of what was asked."""
        conditions: list[str] = []

        class FakeTable:
            def put_item(self, **kw):
                conditions.append(kw.get("ConditionExpression", ""))
                return {}

            def query(self, **kw):
                return {"Items": []}

        QueryAudit(table=FakeTable()).append(asked())
        assert "attribute_not_exists" in conditions[0]

    def test_newest_first(self):
        audit = InMemoryQueryAudit()
        for i in range(3):
            audit.append(asked(at=f"2026-01-0{i + 1}T00:00:00Z", question=f"q{i}"))
        assert [e.question for e in audit.questions(TENANT)] == ["q2", "q1", "q0"]

    def test_a_huge_id_list_is_capped_but_the_count_is_exact(self):
        """Same 400KB item limit as the graph log. The count has to stay right even when the list
        is clipped, and the clipping has to be visible or the inverse lookup lies quietly."""
        items: list[dict] = []

        class FakeTable:
            def put_item(self, **kw):
                items.append(kw["Item"])
                return {}

            def query(self, **kw):
                return {"Items": []}

        many = tuple(f"a-{i}" for i in range(MAX_STORED_IDS + 50))
        QueryAudit(table=FakeTable()).append(asked(assertion_ids=many))

        assert len(items[0]["assertion_ids"]) == MAX_STORED_IDS
        assert items[0]["facts_used"] == MAX_STORED_IDS + 50
        assert items[0]["ids_truncated"] is True

    def test_a_stored_row_reads_back_as_the_event_it_was(self):
        """DynamoDB returns numbers as Decimal and absent attributes as None, so the round trip is
        where an audit read quietly starts returning the wrong tier."""
        from decimal import Decimal

        stored: list[dict] = []

        class FakeTable:
            def put_item(self, **kw):
                item = dict(kw["Item"])
                item["tier"] = Decimal(str(item["tier"]))
                stored.append(item)
                return {}

            def query(self, **kw):
                return {"Items": stored}

        audit = QueryAudit(table=FakeTable())
        audit.append(asked(tier=4, tier_name="LLM_SQL", governed=False, assertion_ids=("a-1",)))

        got = audit.questions(TENANT)[0]
        assert got.tier == 4
        assert got.governed is False
        assert got.assertion_ids == ("a-1",)
        assert got.question == "who acts for Calder?"


class TestOverHttp:
    """The wiring, which is where these go wrong."""

    @pytest.fixture
    def client(self) -> TestClient:
        cfg = GroundworkConfig(
            environment="local",
            auth=AuthConfig(dev_bypass_tenant=TENANT),
            graph=GraphConfig(uri="bolt://127.0.0.1:1", user="none", password="none"),
        )
        cfg.validate()
        app = create_app(cfg)
        from src.api.deps import load_example_pack
        from src.metrics.models import StaticCatalog
        from src.query.metric_matcher import MetricMatcher

        metrics = load_example_pack()
        if metrics:
            get_services().metric_matcher = MetricMatcher(metrics, StaticCatalog(tables={}))
        get_services().query_audit = InMemoryQueryAudit()
        return TestClient(app)

    def test_asking_a_question_records_it(self, client):
        client.post(f"/api/tenants/{TENANT}/query", json={"query": "what is our realization rate"})

        log = client.get(f"/api/tenants/{TENANT}/audit/questions").json()
        assert log["count"] == 1
        row = log["questions"][0]
        assert row["question"] == "what is our realization rate"
        assert row["actor"] == "dev@localhost"
        assert row["tier"] == 1
        assert row["governed"] is True
        assert row["at"]

    def test_an_unanswerable_question_reaches_both_logs(self, client):
        """Two logs, two meanings. The question log records what was answered and on what basis;
        the backlog records what nobody could answer, because a question people keep asking is a
        governed metric waiting to be written.

        This used to depend on the kill switch raising. The switch has nothing to refuse until
        tier 3 generates SQL, so the backlog is now written on the ordinary no-answer path -- and
        the route has to drain it there too, or it would be lost for every tenant with the switch
        off, which is the majority and the ones most likely to need a new metric.
        """
        r = client.post(
            f"/api/tenants/{TENANT}/query", json={"query": "zzzq nonexistent gibberish topic"}
        )
        assert r.status_code == 200
        assert r.json()["answer"] is None

        # In both logs, for different reasons. The question log records that somebody asked and
        # got nothing -- `answered: false` exists for exactly this -- because a read that leaves no
        # trace cannot answer "what did we tell the client". The backlog records it as a metric
        # worth writing. Neither is a duplicate of the other.
        asked_log = client.get(f"/api/tenants/{TENANT}/audit/questions").json()
        assert asked_log["count"] == 1
        assert asked_log["questions"][0]["answered"] is False

        blocked = client.get(f"/api/tenants/{TENANT}/governance/blocked").json()
        assert blocked["count"] == 1

    def test_the_inverse_lookup_over_http(self, client):
        services = get_services()
        services.query_audit.append(asked(question="advice A", assertion_ids=("a-1", "a-2")))
        services.query_audit.append(asked(question="advice B", assertion_ids=("a-9",)))

        r = client.get(f"/api/tenants/{TENANT}/audit/questions?assertion_id=a-1").json()
        assert r["count"] == 1
        assert r["questions"][0]["question"] == "advice A"
        assert r["assertion_id"] == "a-1"

    def test_the_inverse_lookup_reports_how_far_it_read(self, client):
        """Bounded, so the response must say so. A filtered list presented as complete would let
        somebody conclude no advice rested on a fact when the window simply ended."""
        get_services().query_audit.append(asked(assertion_ids=("a-1",)))
        r = client.get(f"/api/tenants/{TENANT}/audit/questions?assertion_id=a-1").json()
        assert r["scanned"] >= r["count"]
        assert "window" in r["note"]

    def test_newest_first_over_http(self, client):
        for i in range(3):
            get_services().query_audit.append(
                asked(at=f"2026-01-0{i + 1}T00:00:00Z", question=f"q{i}")
            )
        got = client.get(f"/api/tenants/{TENANT}/audit/questions").json()["questions"]
        assert [q["question"] for q in got] == ["q2", "q1", "q0"]

    def test_the_limit_is_bounded(self, client):
        assert client.get(f"/api/tenants/{TENANT}/audit/questions?limit=5000").status_code == 422

    def test_an_unconfigured_log_says_so_rather_than_returning_nothing(self, client):
        """An empty list and "there is no log" must not look the same: the first says no question
        was asked, the second that the record does not exist."""
        get_services().query_audit = None
        assert client.get(f"/api/tenants/{TENANT}/audit/questions").status_code == 503

    def test_an_unrecorded_question_is_still_answered_and_says_so(self, client):
        """An audit write that fails must not turn a good answer into a 500 -- but the caller has
        to be told the answer is not in the record, or an unauditable answer looks identical to an
        auditable one."""

        class Broken:
            def append(self, event):
                raise RuntimeError("dynamodb is down")

            def questions(self, tenant_id, *, limit=200):
                return []

        get_services().query_audit = Broken()
        r = client.post(
            f"/api/tenants/{TENANT}/query", json={"query": "what is our realization rate"}
        )
        assert r.status_code == 200
        assert any("not recorded" in w for w in r.json()["warnings"])

    def test_the_scan_window_is_bounded(self):
        assert MAX_SCAN <= 500


class TestEverySurfaceLeavesARow:
    """The gap this closes: for weeks only `/query` recorded anything.

    `/query/compose`, the Retrieval agent and every MCP tool call answered questions and left no
    trace, which is the failure this whole module exists to prevent -- and it was invisible,
    because the Audit page looked healthy while showing one surface out of four.
    """

    def composed(self, **over: Any) -> ComposedAnswer:
        answer = ComposedAnswer()
        answer.parts.append(
            Part(
                lane=Lane.GRAPH,
                provenance=Provenance.INFERRED,
                tier=Tier.GRAPH_TRAVERSAL,
                content=[{"assertion_id": "a-1"}],
                assertion_ids=["a-1"],
            )
        )
        answer.parts.append(
            Part(
                lane=Lane.PASSAGES,
                provenance=Provenance.VERBATIM,
                tier=Tier.HYBRID,
                content=[{"text": "..."}],
                citations=[{"document_id": "doc-7"}],
            )
        )
        for k, v in over.items():
            setattr(answer, k, v)
        return answer

    def test_a_composed_answer_records_the_highest_tier_that_ran(self):
        """The furthest the answer reached, not the first lane. A row saying tier 2 for an answer
        that also read passages would understate what was permitted."""
        e = event_for_composed(TENANT, "me", "q", self.composed())
        assert e.tier == 3
        assert e.tier_name == "HYBRID"

    def test_a_composed_answer_records_its_basis_in_words_not_a_boolean(self):
        """`governed` cannot express a mixed answer. A run over a verbatim passage and a model's
        reading is neither governed nor ungoverned, and picking one would record a claim the
        answer itself refuses to make."""
        e = event_for_composed(TENANT, "me", "q", self.composed())
        assert e.governance == "inferred + verbatim"
        assert e.governed is False
        assert e.basis == "inferred + verbatim"

    def test_a_composed_answer_collects_facts_and_documents_from_every_part(self):
        e = event_for_composed(TENANT, "me", "q", self.composed())
        assert e.assertion_ids == ("a-1",)
        assert e.document_ids == ("doc-7",)

    def test_a_single_tier_row_still_reads_as_governed(self):
        """The boolean is the whole truth for a `Resolution`, so nothing about the old rows or the
        old lookup changes."""
        r = Resolution(tier=Tier.GOVERNED_METRIC, answer=[{"x": 1}])
        e = event_for(TENANT, "me", "q", r)
        assert e.governance == ""
        assert e.basis == "governed"
        assert e.surface == "query"

    def test_an_agent_run_is_one_row_carrying_its_tool_calls(self):
        """Not a row per call. A run is one question with one answer, and eight rows would read as
        eight pieces of advice."""
        result = RunResult(
            run_id="run:abc",
            events=[
                {"kind": "tool_call", "tool": "describe_ontology"},
                {"kind": "tool_result", "tool": "describe_ontology", "result": {"units": []}},
                {"kind": "tool_call", "tool": "compose"},
                {
                    "kind": "tool_result",
                    "tool": "compose",
                    "result": {
                        "governance": "verbatim + inferred",
                        "parts": [
                            {
                                "tier": 3,
                                "assertion_ids": ["a-1", "a-2"],
                                "citations": [{"document_id": "doc-1"}],
                            }
                        ],
                    },
                },
            ],
            answer="Grounded.",
            stop_reason="end_turn",
        )
        e = event_for_run(TENANT, "me", "does acting for Calder conflict?", result)

        assert e.surface == "retrieval_agent"
        assert e.run_id == "run:abc"
        assert e.tools_called == ("describe_ontology", "compose")
        assert e.tier == 3
        assert e.assertion_ids == ("a-1", "a-2")
        assert e.document_ids == ("doc-1",)
        assert e.answered is True

    def test_an_agent_run_is_never_governed_however_governed_its_sources(self):
        """The prose is the agent's own. A run that read nothing but compiled metrics still
        produced an answer no human approved the wording of."""
        result = RunResult(
            run_id="run:abc",
            events=[
                {"kind": "tool_call", "tool": "compose"},
                {
                    "kind": "tool_result",
                    "tool": "compose",
                    "result": {"governance": "governed", "parts": [{"tier": 1}]},
                },
            ],
        )
        e = event_for_run(TENANT, "me", "q", result)
        assert e.governed is False
        assert e.governance == "governed + written by agent"

    def test_a_capped_run_says_so_in_its_basis(self):
        """A run cut off at a cap answered from less than it meant to read, and that belongs in
        the record rather than only in the transcript."""
        # Taken from the loop's own set rather than written out, so a renamed reason fails here
        # instead of silently dropping the caveat from every capped row.
        for reason in CAPPED_REASONS:
            result = RunResult(
                run_id="run:abc",
                events=[{"kind": "tool_call", "tool": "compose"}],
                stop_reason=reason,
            )
            assert "stopped at a cap" in event_for_run(TENANT, "me", "q", result).governance

    def test_a_run_that_finished_does_not_claim_it_was_capped(self):
        result = RunResult(
            run_id="run:abc",
            events=[{"kind": "tool_call", "tool": "compose"}],
            stop_reason="end_turn",
        )
        assert "cap" not in event_for_run(TENANT, "me", "q", result).governance

    def test_an_errored_tool_result_contributes_no_evidence(self):
        """A failed call must not lend its ids to the row. The audit's claim is that the agent was
        shown this evidence, and a refusal showed it nothing."""
        result = RunResult(
            run_id="run:abc",
            events=[
                {"kind": "tool_call", "tool": "compose"},
                {
                    "kind": "tool_result",
                    "tool": "compose",
                    "is_error": True,
                    "result": {"parts": [{"tier": 3, "assertion_ids": ["a-9"]}]},
                },
            ],
        )
        e = event_for_run(TENANT, "me", "q", result)
        assert e.assertion_ids == ()
        assert e.answered is False

    def test_a_failed_run_is_still_recorded(self):
        """ "The agent could not answer this" is part of the record. A log holding only successes
        overstates how well the surface works."""
        result = RunResult(
            run_id="run:abc",
            events=[{"kind": "run_failed", "error": "bedrock timed out"}],
            stop_reason="error",
        )
        e = event_for_run(TENANT, "me", "q", result)
        assert e.answered is False
        assert e.governance == "no governed source + written by agent"

    def test_the_new_fields_survive_a_dynamodb_round_trip(self):
        """Written and read back, because a field the store drops is a field the page cannot show
        and nothing else would notice."""
        stored: list[dict] = []

        class FakeTable:
            def put_item(self, **kw):
                stored.append(dict(kw["Item"]))
                return {}

            def query(self, **kw):
                return {"Items": stored}

        audit = QueryAudit(table=FakeTable())
        audit.append(
            asked(
                surface="retrieval_agent",
                governance="verbatim + written by agent",
                run_id="run:xyz",
                tools_called=("compose", "get_provenance"),
            )
        )
        back = audit.questions(TENANT)[0]
        assert back.surface == "retrieval_agent"
        assert back.governance == "verbatim + written by agent"
        assert back.run_id == "run:xyz"
        assert back.tools_called == ("compose", "get_provenance")

    def test_a_row_written_before_these_fields_reads_as_an_ask_row(self):
        """Which is what it was. An absent surface must not render as blank or unknown."""
        e = _to_event({"tenant_id": TENANT, "question": "old", "tier": 1, "governed": True})
        assert e.surface == "query"
        assert e.basis == "governed"
        assert e.run_id is None

    @pytest.fixture
    def client_for_compose(self) -> TestClient:
        cfg = GroundworkConfig(
            environment="local",
            auth=AuthConfig(dev_bypass_tenant=TENANT),
            graph=GraphConfig(uri="bolt://127.0.0.1:1", user="none", password="none"),
        )
        cfg.validate()
        app = create_app(cfg)
        get_services().query_audit = InMemoryQueryAudit()
        return TestClient(app)

    def test_the_compose_route_records_the_question(self, client_for_compose):
        """The wiring, which is the half that was missing. `event_for_composed` being correct is
        worth nothing if the route never calls it."""
        client = client_for_compose
        r = client.post(f"/api/tenants/{TENANT}/query/compose", json={"query": "who acts for x"})
        assert r.status_code == 200

        log = client.get(f"/api/tenants/{TENANT}/audit/questions").json()
        assert log["count"] == 1
        assert log["questions"][0]["question"] == "who acts for x"
        assert log["questions"][0]["surface"] == "compose"

    def test_an_unrecorded_composed_answer_says_so(self, client_for_compose):
        """Same contract as `/query`: the answer still returns, and the caller is told it is not in
        the record. An unauditable answer must not look identical to an auditable one."""

        class Broken:
            def append(self, event):
                raise RuntimeError("dynamodb is down")

            def questions(self, tenant_id, *, limit=200):
                return []

        get_services().query_audit = Broken()
        r = client_for_compose.post(
            f"/api/tenants/{TENANT}/query/compose", json={"query": "who acts for x"}
        )
        assert r.status_code == 200
        assert any("not recorded" in w for w in r.json()["warnings"])
