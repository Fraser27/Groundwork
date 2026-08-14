"""Tests for the Glue catalog scanner.

The scanner's job is not "read Glue" — boto3 does that. Its job is to turn a catalog
into assertions that carry the right epistemic class, and to keep partial failure
partial. Those are what is tested here.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.discovery.enrichment import (
    DESCRIBED_AS,
    HAS_SYNONYM,
    TableEnrichment,
    build_enrichment_assertions,
    enrich_tables,
    method_for,
)
from src.discovery.glue_scanner import (
    HAS_COLUMN,
    HAS_TABLE,
    METHOD,
    PARTITIONED_BY,
    scan_catalog,
    table_node_id,
)
from src.graph.assertions import EpistemicClass, ReviewState
from src.metrics.compiler import compile_metric
from src.metrics.models import MetricDefinition

TENANT = "firm-acme"
SOURCE = "glue-prod"

INVOICES = {
    "Name": "invoices",
    "Description": "Issued client invoices",
    "UpdateTime": datetime(2026, 3, 1, tzinfo=UTC),
    "Parameters": {"primary_key": "invoice_id"},
    "StorageDescriptor": {
        "Location": "s3://firm-lake/invoices/",
        "Columns": [
            {"Name": "invoice_id", "Type": "string", "Comment": "Invoice reference"},
            {"Name": "invoice_amount", "Type": "decimal(18,2)"},
            {"Name": "invoice_date", "Type": "date"},
        ],
    },
    "PartitionKeys": [{"Name": "month", "Type": "string"}],
}

MATTERS = {
    "Name": "matters",
    "StorageDescriptor": {
        "Columns": [
            {"Name": "matter_id", "Type": "string"},
            {"Name": "opened_date", "Type": "timestamp"},
        ]
    },
}

ICEBERG = {
    "Name": "time_entries",
    "Parameters": {"table_type": "ICEBERG"},
    "StorageDescriptor": {"Columns": [{"Name": "entry_id", "Type": "string"}]},
}


class FakeGlue:
    """Minimal stand-in for the boto3 Glue client, paginators included."""

    def __init__(self, tables_by_db: dict[str, list[dict]], *, fail: set[str] | None = None):
        self._tables = tables_by_db
        self._fail = fail or set()

    def get_paginator(self, operation: str):
        if operation == "get_databases":
            return _Paginator(
                lambda: [{"DatabaseList": [{"Name": db} for db in self._tables]}]
            )
        if operation == "get_tables":
            return _Paginator(self._tables_page)
        raise AssertionError(f"unexpected paginator {operation}")

    def _tables_page(self, DatabaseName: str):
        if DatabaseName in self._fail:
            raise RuntimeError(f"AccessDeniedException: {DatabaseName}")
        return [{"TableList": self._tables.get(DatabaseName, [])}]


class _Paginator:
    def __init__(self, pages):
        self._pages = pages

    def paginate(self, **kwargs):
        return self._pages(**kwargs)


@pytest.fixture
def glue() -> FakeGlue:
    return FakeGlue({"legal_ops": [INVOICES, MATTERS], "warehouse": [ICEBERG]})


@pytest.fixture
def scan(glue):
    return scan_catalog(glue, tenant_id=TENANT, source_id=SOURCE)


def assertions_for(scan, predicate: str):
    return [a for a in scan.assertions if a.predicate == predicate]


class TestDiscovery:
    def test_all_databases_are_discovered_when_none_are_named(self, scan):
        assert {t.full_name for t in scan.tables} == {
            "legal_ops.invoices",
            "legal_ops.matters",
            "warehouse.time_entries",
        }

    def test_explicit_database_list_narrows_the_scan(self, glue):
        scan = scan_catalog(glue, tenant_id=TENANT, source_id=SOURCE, databases=["warehouse"])
        assert {t.full_name for t in scan.tables} == {"warehouse.time_entries"}

    def test_partition_keys_are_scanned_as_columns(self, scan):
        invoices = next(t for t in scan.tables if t.full_name == "legal_ops.invoices")
        assert "month" in invoices.columns

    def test_partition_columns_get_their_own_predicate(self, scan):
        partitioned = assertions_for(scan, PARTITIONED_BY)
        assert len(partitioned) == 1
        assert partitioned[0].source_locator.column == "month"

    def test_iceberg_tables_are_identified(self, scan):
        node = next(n for n in scan.nodes if n.props.get("full_name") == "warehouse.time_entries")
        assert node.props["catalog_type"] == "iceberg"

    def test_primary_key_parameter_is_honoured(self, scan):
        invoices = next(t for t in scan.tables if t.full_name == "legal_ops.invoices")
        assert invoices.primary_keys == frozenset({"invoice_id"})

    def test_missing_primary_key_metadata_stays_empty_not_guessed(self, scan):
        matters = next(t for t in scan.tables if t.full_name == "legal_ops.matters")
        assert matters.primary_keys == frozenset()


class TestEpistemicClass:
    """A catalog IS a system of record, so everything here is DECLARED."""

    def test_every_assertion_is_declared(self, scan):
        assert scan.assertions
        assert all(a.epistemic_class is EpistemicClass.DECLARED for a in scan.assertions)

    def test_declared_assertions_auto_assert(self, scan):
        """Nothing here needs a human — there is no judgement involved."""
        assert all(a.review_state is ReviewState.AUTO_ASSERTED for a in scan.assertions)

    def test_confidence_is_certain(self, scan):
        assert all(a.confidence == 1.0 for a in scan.assertions)

    def test_method_is_the_versioned_scan_identity(self, scan):
        assert all(a.method == METHOD for a in scan.assertions)
        assert METHOD == "glue:catalog_scan"

    def test_no_premises_because_nothing_is_inferred(self, scan):
        assert all(a.premises == () for a in scan.assertions)


class TestProvenance:
    def test_every_assertion_is_tenanted(self, scan):
        assert all(a.tenant_id == TENANT for a in scan.assertions)

    def test_table_assertions_locate_the_table(self, scan):
        table_edges = assertions_for(scan, HAS_TABLE)
        locators = {a.source_locator.table for a in table_edges}
        assert "legal_ops.invoices" in locators
        assert all(a.source_locator.source_id == SOURCE for a in table_edges)

    def test_column_assertions_locate_the_column(self, scan):
        column_edges = assertions_for(scan, HAS_COLUMN)
        edge = next(a for a in column_edges if a.source_locator.column == "invoice_amount")
        assert edge.source_locator.table == "legal_ops.invoices"
        assert edge.source_locator.source_id == SOURCE

    def test_glue_update_time_becomes_world_time(self, scan):
        """When the schema became true, which is not when we happened to scan it."""
        edge = next(
            a
            for a in assertions_for(scan, HAS_TABLE)
            if a.object_id == table_node_id(SOURCE, "legal_ops.invoices")
        )
        assert edge.valid_from.startswith("2026-03-01")
        assert edge.recorded_at != edge.valid_from

    def test_columns_hang_off_their_table_not_the_source(self, scan):
        edge = next(a for a in assertions_for(scan, HAS_COLUMN))
        assert edge.subject_id.startswith("table:")


class TestIdempotence:
    def test_rescanning_produces_identical_assertion_ids(self, glue):
        """Content-addressed ids make a re-scan a no-op rather than a duplicate."""
        first = scan_catalog(glue, tenant_id=TENANT, source_id=SOURCE)
        second = scan_catalog(glue, tenant_id=TENANT, source_id=SOURCE)
        assert [a.assertion_id for a in first.assertions] == [
            a.assertion_id for a in second.assertions
        ]

    def test_a_different_tenant_gets_different_assertions(self, glue):
        mine = scan_catalog(glue, tenant_id="firm-a", source_id=SOURCE)
        theirs = scan_catalog(glue, tenant_id="firm-b", source_id=SOURCE)
        assert not {a.assertion_id for a in mine.assertions} & {
            a.assertion_id for a in theirs.assertions
        }


class TestPartialFailure:
    def test_an_inaccessible_database_does_not_abort_the_scan(self):
        glue = FakeGlue(
            {"legal_ops": [INVOICES], "hr": [MATTERS]},
            fail={"hr"},
        )
        scan = scan_catalog(glue, tenant_id=TENANT, source_id=SOURCE)
        assert {t.full_name for t in scan.tables} == {"legal_ops.invoices"}
        assert any("hr" in e for e in scan.errors)

    def test_a_malformed_table_does_not_abort_its_database(self):
        glue = FakeGlue({"legal_ops": [{"NoName": True}, INVOICES]})
        scan = scan_catalog(glue, tenant_id=TENANT, source_id=SOURCE)
        assert {t.full_name for t in scan.tables} == {"legal_ops.invoices"}
        assert len(scan.errors) == 1

    def test_a_table_with_no_columns_still_produces_a_table_assertion(self):
        glue = FakeGlue({"legal_ops": [{"Name": "empty"}]})
        scan = scan_catalog(glue, tenant_id=TENANT, source_id=SOURCE)
        assert len(assertions_for(scan, HAS_TABLE)) == 1
        assert not assertions_for(scan, HAS_COLUMN)


class TestDownstreamHandoff:
    def test_scan_feeds_the_firewall_allowlist(self, scan):
        assert scan.allowed_tables() == {
            "legal_ops.invoices",
            "legal_ops.matters",
            "warehouse.time_entries",
        }

    def test_scan_feeds_the_metric_compiler(self, scan):
        """The scan is the compiler's only source of truth about columns and types."""
        m = MetricDefinition(
            metric_id="x",
            name="fees_billed",
            expression="SUM(invoice_amount)",
            source_table="legal_ops.invoices",
            grain=["invoice_date"],
            time_grain_column="invoice_date",
            time_grains=["month"],
        )
        result = compile_metric(m, scan.schema_catalog(), dimensions=["invoice_date"])
        assert result.is_valid, result.errors
        assert "DATE_TRUNC('month', invoice_date)" in result.sql

    def test_scanned_partition_column_blocks_a_time_grain_bypass(self, scan):
        """The `month` partition is only known to be time-like because it was scanned."""
        m = MetricDefinition(
            metric_id="x",
            name="fees_billed",
            expression="SUM(invoice_amount)",
            source_table="legal_ops.invoices",
            time_grain_column="invoice_date",
            time_grains=["month"],
        )
        result = compile_metric(
            m, scan.schema_catalog(), dimensions=["invoice_date", "month"]
        )
        assert not result.is_valid
        assert "bypass" in result.errors[0]

    def test_scanned_primary_key_silences_the_fanout_warning(self, scan):
        m = MetricDefinition(
            metric_id="x",
            name="fees_billed",
            expression="SUM(invoice_amount)",
            source_table="legal_ops.matters",
            joins=[
                {
                    "table": "legal_ops.invoices",
                    "source_column": "matter_id",
                    "target_column": "invoice_id",
                }
            ],
        )
        result = compile_metric(m, scan.schema_catalog())
        assert not any("inflate" in w for w in result.warnings)


MODEL = "anthropic.claude-sonnet-5"


class FakeBedrock:
    """Stand-in for bedrock-runtime's Converse API."""

    def __init__(self, replies: list[str] | None = None, error: Exception | None = None):
        self._replies = replies or []
        self._error = error
        self.prompts: list[str] = []

    def converse(self, *, modelId, messages, inferenceConfig):
        self.prompts.append(messages[0]["content"][0]["text"])
        if self._error is not None:
            raise self._error
        reply = self._replies.pop(0) if self._replies else "{}"
        return {"output": {"message": {"content": [{"text": reply}]}}}


class TestEnrichmentEpistemics:
    """The same metadata, from a model instead of the catalog, must land in review.

    This is the contrast that justifies enrichment being a separate pipeline from the
    scan rather than one pass: identical subject matter, different epistemic class.
    """

    @pytest.fixture
    def enriched(self, scan):
        table = next(t for t in scan.tables if t.full_name == "legal_ops.invoices")
        return build_enrichment_assertions(
            TableEnrichment(
                table_description="Invoices issued to clients.",
                column_descriptions={"invoice_amount": "Amount invoiced."},
                synonyms=["bills", "client bills"],
                topics=["billing"],
            ),
            table,
            tenant_id=TENANT,
            source_id=SOURCE,
            model_id=MODEL,
        )

    def test_model_output_is_extracted_model(self, enriched):
        assert enriched.assertions
        assert all(
            a.epistemic_class is EpistemicClass.EXTRACTED_MODEL for a in enriched.assertions
        )

    def test_model_output_lands_in_review_not_live(self, enriched):
        assert all(a.review_state is ReviewState.PENDING for a in enriched.assertions)

    def test_method_records_which_model_said_it(self, enriched):
        assert all(a.method == method_for(MODEL) for a in enriched.assertions)
        assert MODEL in method_for(MODEL)

    def test_enrichment_and_scan_never_collide_on_the_same_edge(self, scan, enriched):
        assert not {a.assertion_id for a in scan.assertions} & {
            a.assertion_id for a in enriched.assertions
        }

    def test_descriptions_and_synonyms_are_separate_predicates(self, enriched):
        predicates = {a.predicate for a in enriched.assertions}
        assert DESCRIBED_AS in predicates
        assert HAS_SYNONYM in predicates

    def test_column_description_is_located_to_the_column(self, enriched):
        edge = next(
            a for a in enriched.assertions if a.source_locator.column == "invoice_amount"
        )
        assert edge.source_locator.table == "legal_ops.invoices"

    def test_hallucinated_columns_are_dropped(self, scan):
        """The catalog, not the model, is the authority on what columns exist."""
        table = next(t for t in scan.tables if t.full_name == "legal_ops.invoices")
        result = build_enrichment_assertions(
            TableEnrichment(column_descriptions={"nonexistent_col": "Something."}),
            table,
            tenant_id=TENANT,
            source_id=SOURCE,
            model_id=MODEL,
        )
        assert not result.assertions

    def test_identical_descriptions_share_one_node(self, scan):
        """Approving the text once approves it everywhere it was proposed."""
        tables = [t for t in scan.tables if t.full_name != "warehouse.time_entries"]
        ids = set()
        for table in tables:
            result = build_enrichment_assertions(
                TableEnrichment(table_description="A legal operations table."),
                table,
                tenant_id=TENANT,
                source_id=SOURCE,
                model_id=MODEL,
            )
            ids |= {n.node_id for n in result.nodes}
        assert len(ids) == 1


class TestEnrichmentRun:
    def test_existing_descriptions_are_not_regenerated(self, scan):
        """A model must never overwrite what a person already wrote."""
        bedrock = FakeBedrock()
        table = next(t for t in scan.tables if t.full_name == "legal_ops.matters")
        have = {"legal_ops.matters": "Curated by the practice management team."}
        have.update({f"legal_ops.matters.{c}": "documented" for c in table.columns})
        enrich_tables(
            bedrock,
            [table],
            tenant_id=TENANT,
            source_id=SOURCE,
            model_id=MODEL,
            domain="legal",
        existing_descriptions=have,
        )
        assert bedrock.prompts == []

    def test_only_undescribed_columns_are_asked_about(self, scan):
        bedrock = FakeBedrock(['{"table_description": "Invoices."}'])
        table = next(t for t in scan.tables if t.full_name == "legal_ops.invoices")
        enrich_tables(
            bedrock,
            [table],
            tenant_id=TENANT,
            source_id=SOURCE,
            model_id=MODEL,
            existing_descriptions={"legal_ops.invoices.invoice_amount": "Amount."},
        )
        ask = bedrock.prompts[0].split("columns: ")[-1].splitlines()[0]
        assert "invoice_id" in ask
        assert "invoice_amount" not in ask

    def test_json_wrapped_in_prose_is_still_parsed(self, scan):
        bedrock = FakeBedrock(
            ['Here you go:\n```json\n{"table_description": "Client invoices."}\n```']
        )
        table = next(t for t in scan.tables if t.full_name == "legal_ops.invoices")
        result = enrich_tables(
            bedrock, [table], tenant_id=TENANT, source_id=SOURCE, model_id=MODEL
        )
        assert any(n.props.get("text") == "Client invoices." for n in result.nodes)

    def test_unparseable_response_yields_nothing_rather_than_garbage(self, scan):
        bedrock = FakeBedrock(["I'm afraid I can't help with that."])
        table = next(t for t in scan.tables if t.full_name == "legal_ops.invoices")
        result = enrich_tables(
            bedrock, [table], tenant_id=TENANT, source_id=SOURCE, model_id=MODEL
        )
        assert not result.assertions

    def test_a_model_failure_is_recorded_and_does_not_abort_the_run(self, scan):
        bedrock = FakeBedrock(error=RuntimeError("ValidationException"))
        result = enrich_tables(
            bedrock, list(scan.tables), tenant_id=TENANT, source_id=SOURCE, model_id=MODEL
        )
        assert len(result.errors) == len(scan.tables)
        assert not result.assertions

    def test_enrichment_failure_leaves_the_declared_catalog_intact(self, scan):
        """The scan is the floor; enrichment is strictly additive."""
        bedrock = FakeBedrock(error=RuntimeError("ValidationException"))
        enrich_tables(
            bedrock, list(scan.tables), tenant_id=TENANT, source_id=SOURCE, model_id=MODEL
        )
        assert scan.allowed_tables()
        assert all(a.review_state is ReviewState.AUTO_ASSERTED for a in scan.assertions)

    def test_a_non_retryable_error_is_not_retried(self, scan):
        """Retrying a ValidationException just burns the same failure three times."""
        bedrock = FakeBedrock(error=RuntimeError("ValidationException: bad model id"))
        table = next(iter(scan.tables))
        enrich_tables(
            bedrock, [table], tenant_id=TENANT, source_id=SOURCE, model_id=MODEL
        )
        assert len(bedrock.prompts) == 1

    def test_throttling_is_retried(self, scan, monkeypatch):
        monkeypatch.setattr("src.discovery.enrichment.time.sleep", lambda _: None)
        bedrock = FakeBedrock(error=RuntimeError("ThrottlingException"))
        table = next(iter(scan.tables))
        result = enrich_tables(
            bedrock, [table], tenant_id=TENANT, source_id=SOURCE, model_id=MODEL
        )
        assert len(bedrock.prompts) == 3
        assert result.errors
