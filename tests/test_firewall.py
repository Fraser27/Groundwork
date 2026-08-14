"""Tests for the SQL firewall.

The interesting cases are all the ones a regex allowlist gets wrong: a table hidden
in a CTE body, a UNION arm, a correlated subquery. Those are also exactly where an
exfiltration query would sit, so they are tested individually rather than as one
"complex query" case.
"""

from __future__ import annotations

import pytest

from src.executors.athena import AthenaConfig, AthenaExecutor
from src.query.firewall import SQLFirewall

ALLOWED = {"legal_ops.invoices", "legal_ops.matters", "legal_ops.time_entries"}


@pytest.fixture
def firewall() -> SQLFirewall:
    return SQLFirewall(ALLOWED)


class TestBasicAllowDeny:
    def test_allowed_table_passes(self, firewall):
        assert firewall.validate("SELECT * FROM legal_ops.invoices").allowed

    def test_unknown_table_is_denied(self, firewall):
        result = firewall.validate("SELECT * FROM hr.salaries")
        assert not result.allowed
        assert result.denied_tables == ["hr.salaries"]

    def test_reason_names_the_denied_table(self, firewall):
        assert "hr.salaries" in firewall.validate("SELECT * FROM hr.salaries").reason

    def test_allowed_tables_are_reported_for_audit(self, firewall):
        result = firewall.validate(
            "SELECT * FROM legal_ops.invoices i JOIN legal_ops.matters m ON i.matter_id = m.matter_id"
        )
        assert result.tables == ["legal_ops.invoices", "legal_ops.matters"]


class TestRecursiveExtraction:
    """find_all recurses; a regex over the query text does not. This is the gap."""

    def test_denied_table_inside_a_join(self, firewall):
        result = firewall.validate(
            "SELECT * FROM legal_ops.invoices i JOIN hr.salaries s ON i.id = s.id"
        )
        assert not result.allowed
        assert result.denied_tables == ["hr.salaries"]

    def test_denied_table_inside_a_cte_body(self, firewall):
        result = firewall.validate(
            "WITH leak AS (SELECT * FROM hr.salaries) SELECT * FROM leak"
        )
        assert not result.allowed
        assert result.denied_tables == ["hr.salaries"]

    def test_denied_table_inside_a_nested_cte(self, firewall):
        result = firewall.validate(
            "WITH a AS (SELECT 1 AS x), b AS (SELECT * FROM a JOIN hr.salaries s ON s.x = a.x) "
            "SELECT * FROM b"
        )
        assert not result.allowed
        assert result.denied_tables == ["hr.salaries"]

    def test_denied_table_inside_a_scalar_subquery(self, firewall):
        result = firewall.validate(
            "SELECT (SELECT max(pay) FROM hr.salaries) FROM legal_ops.invoices"
        )
        assert not result.allowed
        assert result.denied_tables == ["hr.salaries"]

    def test_denied_table_inside_a_where_in_subquery(self, firewall):
        result = firewall.validate(
            "SELECT * FROM legal_ops.invoices WHERE matter_id IN (SELECT id FROM hr.salaries)"
        )
        assert not result.allowed

    def test_denied_table_in_a_union_arm(self, firewall):
        result = firewall.validate(
            "SELECT id FROM legal_ops.invoices UNION ALL SELECT id FROM hr.salaries"
        )
        assert not result.allowed
        assert result.denied_tables == ["hr.salaries"]

    def test_denied_table_in_a_derived_table(self, firewall):
        result = firewall.validate(
            "SELECT * FROM (SELECT * FROM hr.salaries) x"
        )
        assert not result.allowed

    def test_every_denied_table_is_reported_not_just_the_first(self, firewall):
        result = firewall.validate(
            "SELECT * FROM hr.salaries s JOIN finance.ledger l ON s.id = l.id"
        )
        assert result.denied_tables == ["finance.ledger", "hr.salaries"]


class TestCTEAliases:
    """The compiler names CTEs after metrics; sqlglot surfaces those as tables."""

    def test_cte_name_is_not_treated_as_a_table(self, firewall):
        result = firewall.validate(
            "WITH fees_billed AS (SELECT SUM(invoice_amount) AS v FROM legal_ops.invoices) "
            "SELECT * FROM fees_billed"
        )
        assert result.allowed, result.reason

    def test_cte_exemption_only_covers_unqualified_references(self, firewall):
        """A real table that happens to share a CTE's name is still validated."""
        result = firewall.validate(
            "WITH salaries AS (SELECT 1 AS x) SELECT * FROM hr.salaries"
        )
        assert not result.allowed
        assert result.denied_tables == ["hr.salaries"]

    def test_cte_exemption_does_not_leak_across_statements(self, firewall):
        """A CTE defined in one query must not whitelist a bare name in another."""
        assert firewall.validate("WITH salaries AS (SELECT 1 AS x) SELECT * FROM salaries").allowed
        assert not firewall.validate("SELECT * FROM salaries").allowed


class TestFailClosed:
    def test_unparseable_sql_is_denied(self, firewall):
        result = firewall.validate("SELEKT * FRUM")
        assert not result.allowed

    def test_empty_allowlist_denies_everything(self):
        """An empty list must never be read as 'no restrictions'."""
        result = SQLFirewall(set()).validate("SELECT * FROM legal_ops.invoices")
        assert not result.allowed

    def test_allow_all_must_be_explicit(self):
        assert SQLFirewall(set(), allow_all=True).validate("SELECT * FROM anything.at_all").allowed

    def test_empty_query_is_denied(self, firewall):
        assert not firewall.validate("").allowed

    @pytest.mark.parametrize(
        "sql",
        [
            "DELETE FROM legal_ops.invoices",
            "DROP TABLE legal_ops.invoices",
            "INSERT INTO legal_ops.invoices VALUES (1)",
            "UPDATE legal_ops.invoices SET invoice_amount = 0",
            "CREATE TABLE legal_ops.x AS SELECT 1",
        ],
    )
    def test_writes_are_denied_even_on_allowed_tables(self, firewall, sql):
        """An allowlisted table does not make DELETE acceptable."""
        result = firewall.validate(sql)
        assert not result.allowed
        assert "read" in result.reason

    def test_stacked_statements_are_denied(self, firewall):
        result = firewall.validate(
            "SELECT * FROM legal_ops.invoices; SELECT * FROM legal_ops.matters"
        )
        assert not result.allowed
        assert "single statement" in result.reason


class TestQualification:
    def test_unqualified_name_matches_a_qualified_allowlist_entry(self, firewall):
        """Athena resolves bare names against the session database."""
        assert firewall.validate("SELECT * FROM invoices").allowed

    def test_suffix_matching_respects_dotted_boundaries(self):
        """`legal_ops.matters` must never satisfy an allowlist of `secret_matters`."""
        result = SQLFirewall({"vault.secret_matters"}).validate("SELECT * FROM legal_ops.matters")
        assert not result.allowed

    def test_matching_is_case_insensitive(self, firewall):
        assert firewall.validate("SELECT * FROM LEGAL_OPS.INVOICES").allowed


class TestLiveAllowlist:
    def test_provider_results_are_used(self):
        firewall = SQLFirewall(allowlist_provider=lambda: {"legal_ops.invoices"})
        assert firewall.validate("SELECT * FROM legal_ops.invoices").allowed

    def test_provider_is_cached_within_the_ttl(self):
        calls = []

        def provider():
            calls.append(1)
            return {"legal_ops.invoices"}

        firewall = SQLFirewall(allowlist_provider=provider, cache_ttl=60)
        firewall.validate("SELECT 1 FROM legal_ops.invoices")
        firewall.validate("SELECT 1 FROM legal_ops.invoices")
        assert len(calls) == 1

    def test_provider_failure_reuses_the_last_good_snapshot(self):
        """A momentarily unavailable catalog must not widen or empty the allowlist."""
        state = {"fail": False}

        def provider():
            if state["fail"]:
                raise RuntimeError("neptune unavailable")
            return {"legal_ops.invoices"}

        firewall = SQLFirewall(allowlist_provider=provider, cache_ttl=0)
        assert firewall.validate("SELECT 1 FROM legal_ops.invoices").allowed
        state["fail"] = True
        assert firewall.validate("SELECT 1 FROM legal_ops.invoices").allowed
        assert not firewall.validate("SELECT 1 FROM hr.salaries").allowed

    def test_provider_failing_on_the_first_call_denies_everything(self):
        firewall = SQLFirewall(allowlist_provider=lambda: (_ for _ in ()).throw(RuntimeError("x")))
        assert not firewall.validate("SELECT 1 FROM legal_ops.invoices").allowed

    def test_static_and_provider_allowlists_are_unioned(self):
        firewall = SQLFirewall(
            {"legal_ops.invoices"}, allowlist_provider=lambda: {"legal_ops.matters"}
        )
        assert firewall.validate("SELECT 1 FROM legal_ops.invoices").allowed
        assert firewall.validate("SELECT 1 FROM legal_ops.matters").allowed


class TestCompilerOutput:
    """The firewall's real job is passing legitimate compiled metric SQL."""

    def test_compiled_derived_metric_sql_passes(self, firewall):
        sql = (
            "SELECT sub.practice_group, sub.period, (fees_billed / standard_value) AS rate\n"
            "FROM (\nWITH fees_billed AS (\n"
            "  SELECT practice_group, DATE_TRUNC('month', invoice_date) AS period, "
            "SUM(invoice_amount) AS fees_billed\n"
            "  FROM legal_ops.invoices i GROUP BY practice_group, "
            "DATE_TRUNC('month', invoice_date)\n),\n"
            "standard_value AS (\n"
            "  SELECT practice_group, DATE_TRUNC('month', work_date) AS period, "
            "SUM(hours) AS standard_value\n"
            "  FROM legal_ops.time_entries t GROUP BY practice_group, "
            "DATE_TRUNC('month', work_date)\n)\n"
            "SELECT fees_billed.period, fees_billed.fees_billed, standard_value.standard_value\n"
            "FROM fees_billed FULL OUTER JOIN standard_value "
            "ON fees_billed.period = standard_value.period\n) sub"
        )
        result = firewall.validate(sql)
        assert result.allowed, result.reason
        assert result.tables == ["legal_ops.invoices", "legal_ops.time_entries"]


class FakeAthena:
    """Enough of the Athena client to drive the executor's happy and sad paths."""

    def __init__(self, *, states: list[str] | None = None, rows: list[list[str]] | None = None):
        self._states = states or ["SUCCEEDED"]
        self._rows = rows if rows is not None else [["London", "100"]]
        self.started: list[str] = []
        self.stopped: list[str] = []

    def start_query_execution(self, **kwargs):
        self.started.append(kwargs["QueryString"])
        return {"QueryExecutionId": "q-1"}

    def get_query_execution(self, QueryExecutionId):
        state = self._states.pop(0) if len(self._states) > 1 else self._states[0]
        return {
            "QueryExecution": {
                "Status": {"State": state, "StateChangeReason": "COLUMN_NOT_FOUND"},
                "Statistics": {"DataScannedInBytes": 2048},
            }
        }

    def stop_query_execution(self, QueryExecutionId):
        self.stopped.append(QueryExecutionId)

    def get_paginator(self, name):
        assert name == "get_query_results"
        header = {"Data": [{"VarCharValue": "office"}, {"VarCharValue": "total"}]}
        data = [{"Data": [{"VarCharValue": v} for v in row]} for row in self._rows]
        return _ResultsPaginator(
            {
                "ResultSet": {
                    "ResultSetMetadata": {
                        "ColumnInfo": [{"Name": "office"}, {"Name": "total"}]
                    },
                    "Rows": [header, *data],
                }
            }
        )


class _ResultsPaginator:
    def __init__(self, page):
        self._page = page

    def paginate(self, **kwargs):
        return [self._page]


def executor(client, firewall: SQLFirewall, **config) -> AthenaExecutor:
    return AthenaExecutor(AthenaConfig(**config), firewall, client=client)


class TestExecutorEnforcement:
    """The firewall runs inside execute — a caller cannot be trusted to have run it."""

    def test_a_denied_query_never_reaches_athena(self, firewall):
        client = FakeAthena()
        result = executor(client, firewall).execute("SELECT * FROM hr.salaries")
        assert not result.success
        assert result.error_code == "blocked"
        assert client.started == []

    def test_an_allowed_query_runs(self, firewall):
        client = FakeAthena()
        result = executor(client, firewall).execute("SELECT * FROM legal_ops.invoices")
        assert result.success, result.error
        assert result.columns == ["office", "total"]
        assert result.rows == [["London", "100"]]
        assert result.bytes_scanned == 2048

    def test_the_header_row_is_not_returned_as_data(self, firewall):
        client = FakeAthena(rows=[["London", "100"], ["Leeds", "50"]])
        result = executor(client, firewall).execute("SELECT * FROM legal_ops.invoices")
        assert result.row_count == 2
        assert ["office", "total"] not in result.rows

    def test_truncation_is_reported_not_hidden(self, firewall):
        """A silently truncated result is a wrong number presented as an answer."""
        client = FakeAthena(rows=[[str(i), str(i)] for i in range(10)])
        result = executor(client, firewall).execute(
            "SELECT * FROM legal_ops.invoices", max_rows=3
        )
        assert result.truncated
        assert result.row_count == 3

    def test_an_exact_fit_is_not_flagged_as_truncated(self, firewall):
        client = FakeAthena(rows=[[str(i), str(i)] for i in range(3)])
        result = executor(client, firewall).execute(
            "SELECT * FROM legal_ops.invoices", max_rows=3
        )
        assert not result.truncated

    def test_a_failed_query_surfaces_athenas_reason(self, firewall):
        client = FakeAthena(states=["FAILED"])
        result = executor(client, firewall).execute("SELECT * FROM legal_ops.invoices")
        assert not result.success
        assert result.error_code == "query_error"
        assert "COLUMN_NOT_FOUND" in result.error

    def test_a_timeout_cancels_the_query(self, firewall):
        """Leaving it running after we stop caring costs money and holds a slot."""
        client = FakeAthena(states=["RUNNING"])
        result = executor(client, firewall, timeout_seconds=0.01).execute(
            "SELECT * FROM legal_ops.invoices"
        )
        assert result.error_code == "timeout"
        assert client.stopped == ["q-1"]
