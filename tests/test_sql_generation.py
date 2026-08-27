"""SQL a model writes, and the boundary that makes it safe to run.

The properties worth more than the rest of this file:

- **the allowlist is built from the tables the prompt was offered**, so a table in the tenant's
  catalog but not in the prompt is unexecutable rather than merely discouraged
- **the aggregate and limit rules are enforced on the query, not asked for in the prompt**, so a
  model that ignores the instruction is still refused
- **a refused query never reaches Athena**, asserted on the injected client
- **an error is carried, never flattened to an empty result**, because an empty result reads as
  "no data" for a query that never ran
- **an approved synonym reaches table selection**, so a question that shares no word with a table
  is not answered with silence, which is the failure class this repo exists to prevent

No Bedrock and no AWS. Both clients are injected.
"""

from __future__ import annotations

import json
import logging

import pytest

from src.discovery.catalog_store import CatalogColumn, CatalogTable
from src.governance import GovernanceSettings
from src.graph.scope import AuthContext
from src.query.firewall import SQLFirewall
from src.query.planner import Lane, Planner
from src.query.resolver import Resolver
from src.query.sql_generation import (
    MAX_TABLES,
    SqlGenerator,
    SqlLane,
    build_prompt,
    relevant_tables,
)

TENANT = "demo-firm"


def _table(name: str, *columns: CatalogColumn, description: str = "") -> CatalogTable:
    return CatalogTable(
        full_name=f"groundwork_legal.{name}",
        name=name,
        database="groundwork_legal",
        source_id="glue",
        description=description,
        columns=columns,
    )


MATTERS = _table(
    "matters",
    CatalogColumn("matter_id", "string", "The firm's matter reference", is_primary_key=True),
    CatalogColumn("client_id", "string", "Which client"),
    CatalogColumn("opened_date", "date", is_partition=True),
    description="One row per matter the firm has opened",
)

TIME_ENTRIES = _table(
    "time_entries",
    CatalogColumn("matter_id", "string"),
    CatalogColumn("hours", "double", "Hours recorded"),
    description="Recorded time against matters",
)

PAYROLL = _table("payroll", CatalogColumn("salary", "double"), description="Staff compensation")

RETURNS = CatalogTable(
    full_name="anycorp.returns",
    name="returns",
    database="anycorp",
    source_id="glue",
    description="Return requests and refund decisions",
    columns=(CatalogColumn("order_id", "string"), CatalogColumn("refunded", "double")),
)
"""The retail case that motivated synonyms: "how many items did shoppers send back" shares not one
word with this table's name or description, so word overlap alone generated no SQL at all."""

SHIPMENTS = CatalogTable(
    full_name="anycorp.shipments",
    name="shipments",
    database="anycorp",
    source_id="glue",
    description="Outbound parcels",
    columns=(CatalogColumn("order_id", "string"),),
)

SENT_BACK = {"anycorp.returns": ["send back", "items returned"]}


class FakeBedrock:
    """Returns the SQL it was told to, and records what it was asked."""

    def __init__(self, text: str = "SELECT COUNT(*) FROM groundwork_legal.matters LIMIT 10") -> None:
        self.text = text
        self.request: dict | None = None

    def converse(self, **kwargs):
        self.request = kwargs
        return {"output": {"message": {"content": [{"text": self.text}]}}}


class Unreachable:
    def converse(self, **kwargs):
        raise RuntimeError("no route to bedrock")


class FakeAthena:
    """Records every query that reached it, which is the point of several of these tests."""

    def __init__(self, *, state: str = "SUCCEEDED", reason: str = "COLUMN_NOT_FOUND") -> None:
        self.state = state
        self.reason = reason
        self.started: list[str] = []

    def start_query_execution(self, **kwargs):
        self.started.append(kwargs["QueryString"])
        return {"QueryExecutionId": "q-1"}

    def get_query_execution(self, QueryExecutionId):
        return {
            "QueryExecution": {
                "Status": {"State": self.state, "StateChangeReason": self.reason},
                "Statistics": {"DataScannedInBytes": 1024},
            }
        }

    def get_paginator(self, name):
        page = {
            "ResultSet": {
                "ResultSetMetadata": {"ColumnInfo": [{"Name": "n"}]},
                "Rows": [
                    {"Data": [{"VarCharValue": "n"}]},
                    {"Data": [{"VarCharValue": "42"}]},
                ],
            }
        }
        return type("P", (), {"paginate": lambda self, **kw: [page]})()


def _lane(bedrock, client, *, allowed: set[str] | None = None) -> SqlLane:
    """A lane wired to the injected clients. `allowed` overrides what the firewall permits, which
    is how the "offered to the prompt" property is tested from the other side."""
    from src.executors.athena import AthenaConfig, AthenaExecutor

    def factory(offered: set[str]):
        return AthenaExecutor(
            AthenaConfig(output_location="s3://b/athena-results/"),
            SQLFirewall(
                allowed if allowed is not None else offered,
                require_aggregate=True,
                require_limit=True,
            ),
            client=client,
        )

    return SqlLane(
        generator=SqlGenerator(model_id="m", bedrock=bedrock), executor_factory=factory
    )


def _sent(bedrock: FakeBedrock) -> str:
    return json.dumps(bedrock.request)


class TestTheGeneratorCannotReachACatalog:
    def test_tables_is_a_required_keyword(self):
        """The signature is the control: no call path can generate SQL without a catalog."""
        with pytest.raises(TypeError):
            SqlGenerator(model_id="m", bedrock=FakeBedrock()).generate("how many matters")

    def test_no_tables_means_no_query_and_no_model_call(self):
        bedrock = FakeBedrock()
        assert SqlGenerator(model_id="m", bedrock=bedrock).generate("q", tables=[]) is None
        assert bedrock.request is None


class TestThePromptCarriesTheSchema:
    def test_columns_types_and_descriptions_all_reach_the_model(self):
        prompt = build_prompt("how many hours", tables=[MATTERS])
        assert "groundwork_legal.matters" in prompt
        assert "matter_id string" in prompt
        assert "The firm's matter reference" in prompt
        assert "One row per matter the firm has opened" in prompt

    def test_key_and_partition_markers_are_present(self):
        """A partition column is how a scan stays cheap and a PK is how a join is written."""
        prompt = build_prompt("how many hours", tables=[MATTERS])
        assert "PK" in prompt
        assert "PARTITION" in prompt

    def test_a_table_not_passed_in_is_absent_from_the_prompt(self):
        prompt = build_prompt("how many hours", tables=[MATTERS])
        assert "payroll" not in prompt

    def test_the_table_count_is_bounded(self):
        """A Glue scan takes every visible database, so an unrelated table must not be able to push
        the firm's own tables out of the window."""
        many = [_table(f"t{i}") for i in range(MAX_TABLES + 5)]
        prompt = build_prompt("q", tables=many)
        assert f"groundwork_legal.t{MAX_TABLES + 4}" not in prompt


class TestTemperatureIsNeverSent:
    def test_no_temperature_by_default(self):
        """Sonnet 5 answers `ValidationException: temperature is deprecated for this model`, which
        silently failed every summary this deployment attempted before synthesis.py stopped
        sending it."""
        bedrock = FakeBedrock()
        SqlGenerator(model_id="m", bedrock=bedrock).generate("q", tables=[MATTERS])
        assert "temperature" not in (bedrock.request or {})["inferenceConfig"]


class TestTheModelsAnswerIsNormalised:
    def test_a_markdown_fence_is_stripped(self):
        bedrock = FakeBedrock("```sql\nSELECT COUNT(*) FROM groundwork_legal.matters LIMIT 5\n```")
        out = SqlGenerator(model_id="m", bedrock=bedrock).generate("q", tables=[MATTERS])
        assert out is not None
        assert out.sql.startswith("SELECT")
        assert "```" not in out.sql

    def test_declining_is_honoured_rather_than_treated_as_a_failure(self):
        """A query over the wrong column is worse than no query, so NO_QUERY is a real answer."""
        bedrock = FakeBedrock("NO_QUERY")
        assert SqlGenerator(model_id="m", bedrock=bedrock).generate("q", tables=[MATTERS]) is None

    def test_an_unreachable_model_returns_none_rather_than_raising(self):
        """The SQL lane is one lane of an answer that also has passages and graph facts. A Bedrock
        outage must not take those down."""
        gen = SqlGenerator(model_id="m", bedrock=Unreachable())
        assert gen.generate("q", tables=[MATTERS]) is None

    def test_the_tables_offered_are_recorded_on_the_result(self):
        out = SqlGenerator(model_id="m", bedrock=FakeBedrock()).generate(
            "q", tables=[MATTERS, TIME_ENTRIES]
        )
        assert out is not None
        assert out.tables_offered == (
            "groundwork_legal.matters",
            "groundwork_legal.time_entries",
        )


class TestTheAllowlistIsWhatWasOffered:
    """Not the tenant's whole catalog. This is the difference between "the model named a table it
    was not offered" being unexecutable and being merely discouraged."""

    def test_a_table_in_the_catalog_but_not_offered_is_denied(self):
        """`payroll` is a real scanned table for this tenant. The question does not mention it, so
        it never reaches the prompt, so it is not on the allowlist, so a query naming it cannot
        run -- and Athena is never called."""
        bedrock = FakeBedrock("SELECT SUM(salary) FROM groundwork_legal.payroll LIMIT 10")
        client = FakeAthena()
        result = _lane(bedrock, client).run(
            "how many matters do we have", tables=[MATTERS, TIME_ENTRIES, PAYROLL]
        )
        assert result is not None
        assert result.error_code == "blocked"
        assert "payroll" in (result.error or "")
        assert client.started == []

    def test_the_unoffered_table_is_absent_from_the_prompt_too(self):
        bedrock = FakeBedrock()
        _lane(bedrock, FakeAthena()).run(
            "how many matters do we have", tables=[MATTERS, PAYROLL]
        )
        assert "payroll" not in _sent(bedrock)

    def test_an_offered_table_runs(self):
        bedrock = FakeBedrock()
        client = FakeAthena()
        result = _lane(bedrock, client).run("how many matters", tables=[MATTERS, PAYROLL])
        assert result is not None
        assert result.error is None
        assert result.rows == {"columns": ["n"], "rows": [["42"]]}
        assert len(client.started) == 1


class TestTheFirewallRulesAreEnforcedNotRequested:
    """A prompt instruction is a request. These assert on what happens when the model ignores it."""

    def test_a_row_wise_query_is_refused_and_never_reaches_athena(self):
        bedrock = FakeBedrock("SELECT * FROM groundwork_legal.matters LIMIT 10")
        client = FakeAthena()
        result = _lane(bedrock, client).run("how many matters", tables=[MATTERS])
        assert result is not None
        assert result.error_code == "blocked"
        assert "aggregate" in (result.error or "")
        assert client.started == []

    def test_an_unbounded_query_is_refused_and_never_reaches_athena(self):
        bedrock = FakeBedrock("SELECT COUNT(*) FROM groundwork_legal.matters")
        client = FakeAthena()
        result = _lane(bedrock, client).run("how many matters", tables=[MATTERS])
        assert result is not None
        assert result.error_code == "blocked"
        assert "LIMIT" in (result.error or "")
        assert client.started == []

    def test_the_sql_is_still_reported_when_it_was_refused(self):
        """The reader needs to see what was refused. A blocked query with no SQL beside it is an
        unexplained refusal."""
        bedrock = FakeBedrock("SELECT * FROM groundwork_legal.matters LIMIT 10")
        result = _lane(bedrock, FakeAthena()).run("how many matters", tables=[MATTERS])
        assert result is not None
        assert result.generated.sql.startswith("SELECT *")


class TestAFailureIsNeverAnEmptyResult:
    def test_a_hallucinated_column_surfaces_as_an_error_not_as_no_rows(self):
        """The firewall validates tables, not columns, so this reaches Athena and errors. Reporting
        it as an empty result would read as "no data" -- the silent failure scope.py exists to
        prevent."""
        bedrock = FakeBedrock("SELECT COUNT(nonexistent) FROM groundwork_legal.matters LIMIT 10")
        client = FakeAthena(state="FAILED")
        result = _lane(bedrock, client).run("how many matters", tables=[MATTERS])
        assert result is not None
        assert result.rows is None
        assert result.error_code == "query_error"
        assert "COLUMN_NOT_FOUND" in (result.error or "")

    def test_no_executor_returns_the_sql_unrun_rather_than_nothing(self):
        """The same posture as a compiled metric with no executor: the SQL is the reviewable
        artefact and there is simply nowhere to run it."""
        lane = SqlLane(generator=SqlGenerator(model_id="m", bedrock=FakeBedrock()))
        result = lane.run("how many matters", tables=[MATTERS])
        assert result is not None
        assert result.rows is None
        assert result.error is None
        assert result.generated.sql


class TestWhichTablesAQuestionReaches:
    def test_a_question_sharing_no_words_reaches_nothing(self):
        """The 21 unrelated tables a real Glue scan picks up are excluded by this, not by scope."""
        assert relevant_tables("what is the weather", [MATTERS, TIME_ENTRIES]) == []

    def test_the_lane_declines_rather_than_prompting_with_everything(self):
        bedrock = FakeBedrock()
        assert _lane(bedrock, FakeAthena()).run("what is the weather", tables=[MATTERS]) is None
        assert bedrock.request is None

    def test_a_description_word_is_enough(self):
        """`time_entries` is named for its table, but a lawyer asks about hours."""
        assert relevant_tables("how many hours recorded", [MATTERS, TIME_ENTRIES]) == [
            TIME_ENTRIES
        ]


QUESTION = "how many items did shoppers send back"


class TestAnApprovedSynonymReachesTableSelection:
    """`approved_synonyms` existed with no callers, so approving one changed nothing. The cost of
    that was a silent empty: no candidate tables, no prompt, no SQL, and no reason given."""

    def test_a_question_sharing_no_word_with_a_table_finds_it_by_synonym(self):
        assert relevant_tables(QUESTION, [RETURNS], synonyms=SENT_BACK) == [RETURNS]

    def test_the_same_question_finds_nothing_without_the_synonym(self):
        """The bug, asserted so the fix cannot regress into looking like a coincidence."""
        assert relevant_tables(QUESTION, [RETURNS]) == []

    def test_the_lane_generates_sql_it_previously_declined_to(self):
        bedrock = FakeBedrock("SELECT COUNT(*) FROM anycorp.returns LIMIT 10")
        lane = SqlLane(generator=SqlGenerator(model_id="m", bedrock=bedrock))
        assert lane.run(QUESTION, tables=[RETURNS], synonyms=SENT_BACK) is not None
        assert lane.run(QUESTION, tables=[RETURNS]) is None

    def test_a_synonym_is_keyed_by_full_name_not_by_graph_id(self):
        """Catalog ids are built and never parsed, so this module must not learn their format. A
        mapping keyed the graph's way silently matches nothing."""
        by_graph_id = {"table:glue:anycorp.returns": ["send back"]}
        assert relevant_tables(QUESTION, [RETURNS], synonyms=by_graph_id) == []

    def test_a_synonym_does_not_drag_in_an_unrelated_table(self):
        """Widening selection is the point; widening it to everything would put the tenant's whole
        catalog in the prompt."""
        synonyms = {"groundwork_legal.payroll": ["wages", "salaries"]}
        assert relevant_tables(
            "how many hours recorded", [MATTERS, TIME_ENTRIES, PAYROLL], synonyms=synonyms
        ) == [TIME_ENTRIES]


class TestAnExactNameStillWins:
    """Someone naming a table explicitly must never lose the window to a synonym match."""

    def test_a_name_match_is_ordered_ahead_of_a_synonym_match(self):
        synonyms = {"anycorp.shipments": ["returns", "sent back"]}
        assert relevant_tables(
            "how many returns did we take", [SHIPMENTS, RETURNS], synonyms=synonyms
        ) == [RETURNS, SHIPMENTS]

    def test_the_strongest_match_survives_the_cap_whatever_the_catalog_order(self):
        """Unranked selection made "which 20" a matter of catalog order, so an unrelated database
        scanned first could push the table the question named out of the prompt."""
        filler = [_table(f"t{i}", description="opened by staff") for i in range(MAX_TABLES + 4)]
        lane = SqlLane(generator=SqlGenerator(model_id="m", bedrock=FakeBedrock()))
        result = lane.run("how many matters opened", tables=[*filler, MATTERS])
        assert result is not None
        assert result.generated.tables_offered[0] == "groundwork_legal.matters"

    def test_synonyms_none_leaves_the_selected_set_unchanged(self):
        tables = [MATTERS, TIME_ENTRIES, PAYROLL, RETURNS]
        question = "how many hours recorded against matters"
        assert relevant_tables(question, tables, synonyms=None) == relevant_tables(
            question, tables
        )
        assert set(relevant_tables(question, tables)) == {MATTERS, TIME_ENTRIES}


class TestTheCapIsNeverSilent:
    def test_dropping_tables_is_logged_with_the_count(self, caplog):
        """A silent cap reads as "covered everything", so a question the dropped tables would have
        answered looks like a question the schema could not answer."""
        many = [_table(f"t{i}", description="opened by staff") for i in range(MAX_TABLES + 4)]
        lane = SqlLane(generator=SqlGenerator(model_id="m", bedrock=FakeBedrock()))
        with caplog.at_level(logging.WARNING, logger="src.query.sql_generation"):
            result = lane.run("how many matters opened", tables=many)
        assert result is not None
        assert len(result.generated.tables_offered) == MAX_TABLES
        assert "4 dropped" in caplog.text
        assert "groundwork_legal.t23" in caplog.text

    def test_nothing_is_logged_when_everything_fits(self, caplog):
        with caplog.at_level(logging.WARNING, logger="src.query.sql_generation"):
            SqlGenerator(model_id="m", bedrock=FakeBedrock()).generate("q", tables=[MATTERS])
        assert caplog.text == ""


class FakeGraph:
    def search(self, ctx, question, **kw):
        return []

    def expand(self, ctx, seeds, **kw):
        return [{"assertion_id": "a1", "subject_id": "matter:m1", "epistemic_class": "DECLARED"}]

    def blocking_facts(self, ctx, seeds, **kw):
        return []


class FakeVectors:
    def search(self, ctx, question, **kw):
        return [{"document_id": "d1", "page": 2, "char_start": 0, "char_end": 40}]


class Catalog:
    def __init__(self, *tables) -> None:
        self._tables = list(tables)

    def tables(self, tenant_id: str):
        return self._tables


def _ctx() -> AuthContext:
    return AuthContext(tenant_id=TENANT, user_id="alice")


def _wired(kind: str, *, catalog, synonyms_for=None):
    build = Resolver if kind == "resolver" else Planner
    return build(
        graph_reader=FakeGraph(),
        vector_search=FakeVectors(),
        catalog=catalog,
        sql_lane=SqlLane(generator=SqlGenerator(model_id="m", bedrock=FakeBedrock())),
        synonyms_for=synonyms_for,
    )


class TestBothEndpointsGetTheSynonyms:
    """Injected, not looked up: neither the resolver nor the planner may learn where approved
    synonyms live. And both must get them, or a question is answerable on one endpoint only."""

    def test_the_resolver_passes_them_to_the_lane(self):
        resolver = _wired("resolver", catalog=Catalog(RETURNS), synonyms_for=lambda ctx: SENT_BACK)
        res = resolver.resolve(_ctx(), QUESTION, GovernanceSettings())
        assert res.generated_sql is not None

    def test_the_resolver_without_them_declines_as_it_did_before(self):
        resolver = _wired("resolver", catalog=Catalog(RETURNS))
        res = resolver.resolve(_ctx(), QUESTION, GovernanceSettings())
        assert res.generated_sql is None

    def test_the_planner_runs_both_schema_lanes_on_the_same_selection(self):
        """The catalog lane shows the reader what the SQL lane was given. One lane widened by a
        synonym and the other not would show schema a query was not written over."""
        planner = _wired("planner", catalog=Catalog(RETURNS), synonyms_for=lambda ctx: SENT_BACK)
        answer = planner.plan(_ctx(), QUESTION, GovernanceSettings(), allow_synthesis=False)
        assert Lane.CATALOG in answer.lanes_run
        assert Lane.SQL in answer.lanes_run


class TestASynonymOutageNeverCostsAnAnswer:
    """Synonyms widen table selection. A graph that cannot be reached must cost the widening and
    nothing else -- turning an answer word overlap already found into an error would be worse than
    the gap synonyms were added to close."""

    def _broken(self, ctx):
        raise RuntimeError("no route to the graph")

    def test_the_resolver_still_generates_sql(self, caplog):
        resolver = _wired("resolver", catalog=Catalog(MATTERS), synonyms_for=self._broken)
        with caplog.at_level(logging.WARNING, logger="src.query.resolver"):
            res = resolver.resolve(_ctx(), "how many matters", GovernanceSettings())
        assert res.generated_sql is not None
        assert "no approved synonyms" in caplog.text

    def test_the_planner_still_runs_both_schema_lanes(self, caplog):
        planner = _wired("planner", catalog=Catalog(MATTERS), synonyms_for=self._broken)
        with caplog.at_level(logging.WARNING, logger="src.query.planner"):
            answer = planner.plan(
                _ctx(), "how many matters", GovernanceSettings(), allow_synthesis=False
            )
        assert Lane.CATALOG in answer.lanes_run
        assert Lane.SQL in answer.lanes_run
        assert "no approved synonyms" in caplog.text
