"""Catalog enrichment: the nodes get written, and only approved descriptions reach the planner.

Two gaps this pins, both of which were silent.

**The nodes were built and discarded.** `scan_catalog` and `enrich_tables` each return a `nodes`
list and nothing wrote it, so a live tenant held 43 `:Entity` and zero `:Table`. That broke metric
lineage (`LINK_METRIC_TO_TABLE` matches `(t:Table {full_name})`) and it meant a `DESCRIBED_AS` edge
pointed at a node whose `text` had never been stored, making an approved description unrecoverable
without paying for the model again.

**The read loop was open.** `sql_generation.build_prompt` already emitted `-- description` for
tables and columns, but read it off `CatalogTable`, which is populated only from Glue comments. So
approving an enrichment changed nothing about the generated SQL. The end-to-end test here is the
one that would have caught that: PENDING is absent from the prompt, APPROVED is present.
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from src.discovery.catalog_overlay import (
    SOURCE_GLUE,
    SOURCE_HUMAN,
    SOURCE_MODEL,
    SOURCE_NONE,
    EnrichedCatalog,
    overlay_tables,
    source_of,
)
from src.discovery.catalog_store import CatalogColumn, CatalogTable
from src.discovery.enrichment import (
    CONCERNS_TOPIC,
    DESCRIBED_AS,
    TableEnrichment,
    build_enrichment_assertions,
    is_catalog_claim,
)
from src.discovery.glue_scanner import CatalogNode, column_node_id, table_node_id
from src.discovery.graph_store import CatalogGraphStore, DescriptionText
from src.graph import catalog_queries as q
from src.graph.assertions import (
    DESCRIPTIVE_CONFIDENCE,
    EpistemicClass,
    SourceLocator,
    build_assertion,
)
from src.graph.scope import DEFAULT_MIN_CONFIDENCE, AuthContext
from src.metrics.models import TableSchema
from src.ontology.loader import available_domains, load_ontology
from src.query.sql_generation import build_prompt

TENANT = "demo-firm"
SOURCE = "glue-main"
FULL_NAME = "fin.facilities"


def ctx() -> AuthContext:
    return AuthContext(user_id="probe@firm.example", tenant_id=TENANT)


class FakeGraph:
    """Records writes and replays canned rows. No database, per the boto3 rule."""

    def __init__(self, rows: list[dict[str, Any]] | None = None, fail: bool = False) -> None:
        self.batches: list[tuple[str, list[dict[str, Any]]]] = []
        self.rows = rows or []
        self.fail = fail

    def write_batch(self, cypher: str, batch: list[dict[str, Any]]) -> None:
        if self.fail:
            raise RuntimeError("neptune is unreachable")
        self.batches.append((cypher, batch))

    def read_scoped(self, template: str, scope: Any, params: Any = None) -> list[dict[str, Any]]:
        assert "{scope}" in template, "a scoped read must carry the token"
        return self.rows

    def written_props(self) -> list[dict[str, Any]]:
        return [item["props"] for _, batch in self.batches for item in batch]


def table(description: str = "", column_description: str = "") -> CatalogTable:
    return CatalogTable(
        full_name=FULL_NAME,
        name="facilities",
        database="fin",
        source_id=SOURCE,
        description=description,
        columns=(
            CatalogColumn(name="bal", data_type="double", description=column_description),
            CatalogColumn(name="fac_id", data_type="string", is_primary_key=True),
        ),
    )


def model_text(text: str) -> DescriptionText:
    return DescriptionText(
        text=text,
        epistemic_class="EXTRACTED_MODEL",
        confidence=DESCRIPTIVE_CONFIDENCE,
        method="llm:m1",
    )


def human_text(text: str) -> DescriptionText:
    return DescriptionText(
        text=text, epistemic_class="DECLARED", confidence=DESCRIPTIVE_CONFIDENCE, method="admin:me"
    )


class TestTheNodesActuallyGetWritten:
    """The regression guard. `grep write_batch src/` used to return only its own definition."""

    def test_every_node_reaches_a_batch(self):
        graph = FakeGraph()
        nodes = [
            CatalogNode("table:x:t", ("Table",), {"tenant_id": TENANT, "full_name": "t"}),
            CatalogNode("column:x:t.a", ("Column",), {"tenant_id": TENANT, "name": "a"}),
            CatalogNode("description:d1", ("Description",), {"tenant_id": TENANT, "text": "Hi."}),
        ]
        assert CatalogGraphStore(graph).persist(nodes) == 3
        assert len(graph.written_props()) == 3

    def test_the_description_text_is_stored(self):
        """The failure that made this urgent: the edge pointed at a node whose text was never
        written, so an approved description could not be read back at any price."""
        graph = FakeGraph()
        CatalogGraphStore(graph).persist(
            [CatalogNode("description:d1", ("Description",), {"tenant_id": TENANT, "text": "Hi."})]
        )
        assert graph.written_props()[0]["text"] == "Hi."

    def test_it_merges_on_the_key_an_assertion_edge_uses(self):
        """`upsert_edge` merges `(:Entity {tenant_id, entity_id})`. Merging on anything else builds
        a parallel node set the edges do not point at, which is this bug wearing a fix."""
        graph = FakeGraph()
        CatalogGraphStore(graph).persist(
            [CatalogNode("table:x:t", ("Table",), {"tenant_id": TENANT})]
        )
        cypher = graph.batches[0][0]
        assert "MERGE (n:Entity {tenant_id: item.tenant_id, entity_id: item.node_id})" in cypher
        assert "SET n:Table" in cypher

    def test_one_batch_per_label(self):
        """`write_batch` prepends a single UNWIND, so one statement serves one label."""
        graph = FakeGraph()
        CatalogGraphStore(graph).persist(
            [
                CatalogNode("table:x:t", ("Table",), {"tenant_id": TENANT}),
                CatalogNode("column:x:t.a", ("Column",), {"tenant_id": TENANT}),
                CatalogNode("column:x:t.b", ("Column",), {"tenant_id": TENANT}),
            ]
        )
        labels = [re.search(r"SET n:(\w+)", c).group(1) for c, _ in graph.batches]
        assert sorted(labels) == ["Column", "Table"]

    def test_a_list_valued_prop_is_refused_before_the_write(self):
        """Neptune rejects list-valued properties where Neo4j accepts them, so this would pass
        locally and fail only once deployed."""
        graph = FakeGraph()
        with pytest.raises(ValueError, match="scalars"):
            CatalogGraphStore(graph).persist(
                [CatalogNode("table:x:t", ("Table",), {"tenant_id": TENANT, "tags": ["a"]})]
            )
        assert graph.batches == []

    def test_a_node_without_a_tenant_is_refused(self):
        graph = FakeGraph()
        with pytest.raises(ValueError, match="tenant_id"):
            CatalogGraphStore(graph).persist([CatalogNode("table:x:t", ("Table",), {})])

    def test_an_unlabelled_node_is_skipped_not_written_as_a_mystery(self):
        graph = FakeGraph()
        assert (
            CatalogGraphStore(graph).persist([CatalogNode("x:1", (), {"tenant_id": TENANT})]) == 0
        )


class TestTheOverlayPrecedence:
    """Three sources can describe a column and they are not equally authoritative."""

    def test_a_person_beats_the_glue_comment(self):
        """Somebody edits a description precisely because the comment was wrong. Demoting their
        edit below it would make the edit look broken."""
        out = overlay_tables(
            [table(column_description="Glue said this.")],
            {column_node_id(SOURCE, FULL_NAME, "bal"): human_text("A person said this.")},
        )
        assert out[0].columns[0].description == "A person said this."

    def test_a_model_does_not_beat_the_glue_comment(self):
        """A comment somebody wrote upstream is still a human statement."""
        out = overlay_tables(
            [table(column_description="Glue said this.")],
            {column_node_id(SOURCE, FULL_NAME, "bal"): model_text("A model guessed this.")},
        )
        assert out[0].columns[0].description == "Glue said this."

    def test_a_model_fills_a_gap(self):
        out = overlay_tables(
            [table()], {column_node_id(SOURCE, FULL_NAME, "bal"): model_text("Balance.")}
        )
        assert out[0].columns[0].description == "Balance."

    def test_the_table_description_is_overlaid_too(self):
        out = overlay_tables([table()], {table_node_id(SOURCE, FULL_NAME): model_text("Loans.")})
        assert out[0].description == "Loans."

    def test_nothing_to_overlay_leaves_the_tables_alone(self):
        original = table(description="As scanned.")
        assert overlay_tables([original], {})[0] is original

    def test_the_reported_source_matches_what_was_used(self):
        assert source_of("Glue.", model_text("m")) == SOURCE_GLUE
        assert source_of("Glue.", human_text("h")) == SOURCE_HUMAN
        assert source_of("", model_text("m")) == SOURCE_MODEL
        assert source_of("", None) == SOURCE_NONE

    def test_two_model_descriptions_resolve_deterministically(self):
        """Two `DESCRIBED_AS` edges from one subject are legitimate, so the winner must not depend
        on row order."""
        rows = [
            {
                "subject_id": "table:t",
                "text": "second",
                "epistemic_class": "EXTRACTED_MODEL",
                "confidence": 0.8,
                "method": "llm:b",
            },
            {
                "subject_id": "table:t",
                "text": "first",
                "epistemic_class": "EXTRACTED_MODEL",
                "confidence": 0.9,
                "method": "llm:a",
            },
        ]
        forward = CatalogGraphStore(FakeGraph(rows)).approved_descriptions(ctx())
        backward = CatalogGraphStore(FakeGraph(list(reversed(rows)))).approved_descriptions(ctx())
        assert forward["table:t"].text == backward["table:t"].text == "first"

    def test_a_person_wins_over_a_more_confident_model(self):
        rows = [
            {
                "subject_id": "table:t",
                "text": "model",
                "epistemic_class": "EXTRACTED_MODEL",
                "confidence": 1.0,
                "method": "llm:a",
            },
            {
                "subject_id": "table:t",
                "text": "person",
                "epistemic_class": "DECLARED",
                "confidence": 0.8,
                "method": "admin:me",
            },
        ]
        best = CatalogGraphStore(FakeGraph(rows)).approved_descriptions(ctx())
        assert best["table:t"].text == "person"

    def test_a_graph_failure_degrades_to_the_bare_catalog(self):
        """An outage must not empty the Tables page. It does silently stop descriptions reaching
        the prompt, which is why `_descriptions` logs at warning."""

        class Broken:
            def approved_descriptions(self, _ctx):
                raise RuntimeError("graph down")

        class Store:
            def tables(self, _tenant):
                return [table(description="As scanned.")]

        enriched = EnrichedCatalog(Store(), Broken(), lambda t: ctx())
        assert enriched.tables(TENANT)[0].description == "As scanned."


class TestOnlyApprovedDescriptionsReachThePlanner:
    """The whole point of the feature, and the half that was missing."""

    def schema(self, tables: list[CatalogTable]) -> str:
        return build_prompt("what is the balance", tables=tables)

    def test_a_pending_description_is_absent_from_the_prompt(self):
        """`approved_descriptions` reads through `edge_scope` with `include_pending=False`, so a
        proposal nobody signed off never reaches the model."""
        prompt = self.schema(overlay_tables([table()], {}))
        assert "Outstanding balance" not in prompt

    def test_an_approved_description_is_present_in_the_prompt(self):
        overlaid = overlay_tables(
            [table()],
            {column_node_id(SOURCE, FULL_NAME, "bal"): model_text("Outstanding balance.")},
        )
        assert "-- Outstanding balance." in self.schema(overlaid)

    def test_the_table_description_reaches_the_prompt_header(self):
        overlaid = overlay_tables(
            [table()], {table_node_id(SOURCE, FULL_NAME): model_text("Credit facilities.")}
        )
        assert "-- Credit facilities." in self.schema(overlaid)


class TestTheFloorArithmetic:
    def test_an_approved_descriptive_claim_sits_exactly_on_the_floor(self):
        """If either constant moves, an approved description silently disappears from every prompt
        instead of failing here."""
        assert DESCRIPTIVE_CONFIDENCE == DEFAULT_MIN_CONFIDENCE

    def test_the_scoped_read_uses_a_floor_comparison_that_admits_it(self):
        from src.graph.scope import edge_scope

        assert ">= $scope_min_conf" in edge_scope(ctx()).where


class TestTheClosedVocabularyAcceptsThem:
    """An enrichment write rejected at write time looks exactly like a model that found nothing."""

    @pytest.mark.parametrize("domain", available_domains())
    def test_every_enrichment_predicate_is_declared(self, domain):
        onto = load_ontology(domain)
        known = onto.governing_predicates | onto.descriptive_predicates
        for predicate in (DESCRIBED_AS, "HAS_SYNONYM", "CONCERNS_TOPIC"):
            assert predicate in known, f"{domain} does not declare {predicate}"

    @pytest.mark.parametrize("domain", available_domains())
    def test_they_are_descriptive_not_governing(self, domain):
        """Governing would rescale an approved description into the answerable band, where it would
        outrank a conflict a partner personally approved."""
        onto = load_ontology(domain)
        for predicate in (DESCRIBED_AS, "HAS_SYNONYM", "CONCERNS_TOPIC"):
            assert predicate not in onto.governing_predicates

    @pytest.mark.parametrize("domain", available_domains())
    def test_a_real_enrichment_write_is_accepted(self, domain):
        """Catches a later pack edit adding `domain:`/`range:`, which would turn on endpoint
        validation and reject every one of these, since no pack declares a Description kind."""
        onto = load_ontology(domain)
        schema = TableSchema(
            full_name=FULL_NAME, columns={"bal": "double"}, primary_keys=frozenset()
        )
        result = build_enrichment_assertions(
            TableEnrichment(
                table_description="Credit facilities.",
                column_descriptions={"bal": "Outstanding balance."},
                synonyms=["loans"],
                topics=["credit"],
            ),
            schema,
            tenant_id=TENANT,
            source_id=SOURCE,
            model_id="m1",
        )
        assert result.assertions
        for staged in result.assertions:
            build_assertion(
                tenant_id=TENANT,
                subject_id=staged.subject_id,
                predicate=staged.predicate,
                object_id=staged.object_id,
                epistemic_class=EpistemicClass.EXTRACTED_MODEL,
                method="llm:m1",
                confidence=DESCRIPTIVE_CONFIDENCE,
                source_locator=SourceLocator(source_id=SOURCE, table=FULL_NAME),
                allowed_predicates=onto.allowed_for(staged.predicate),
                endpoint_kinds=onto.endpoint_kinds(staged.predicate),
            )


class TestTheCypherShape:
    """Single tokens a careless edit removes silently, as in `test_metric_versions_cypher`."""

    @pytest.mark.parametrize("name", sorted(q.ALL_CATALOG_QUERIES))
    def test_every_read_is_scoped(self, name):
        assert "{scope}" in q.ALL_CATALOG_QUERIES[name]

    def test_the_catalog_reads_reach_a_column_through_its_edge(self):
        """`c.table` is a convenience copy of the parent name. Reading it instead of walking
        `HAS_COLUMN` would return columns no assertion ever attached to a table."""
        assert "-[r:HAS_COLUMN]->" in q.COLUMNS_FOR_TENANT
        assert "c.table" not in q.COLUMNS_FOR_TENANT
        assert "-[r:HAS_TABLE]->" in q.TABLES_FOR_TENANT

    def test_the_columns_read_names_its_parent_table(self):
        """Without `full_name` the rows cannot be grouped and every table hydrates with no
        columns, which looks like a scan that found empty tables."""
        assert "t.full_name AS full_name" in q.COLUMNS_FOR_TENANT

    def test_the_node_write_is_tenant_keyed(self):
        cypher = q.upsert_node("Table")
        assert "item.tenant_id" in cypher
        # `write_batch` prepends `UNWIND $batch AS item`, so a `$name` parameter binds nothing.
        assert "$" not in cypher

    def test_no_statement_writes_a_second_id_property(self):
        """`entity_id` is already the id. A `node_id` property would be a second identity."""
        assert "node_id:" not in q.upsert_node("Table")

    def test_an_unsafe_label_is_refused_rather_than_escaped(self):
        from src.graph.assertion_queries import UnsafeRelationshipType

        for bad in ("Table {x}", "Table`", "Table Column", ""):
            with pytest.raises(UnsafeRelationshipType):
                q.upsert_node(bad)


class TestTheRoutes:
    """The wiring, which is the half that was missing: `enrich_tables` had no caller at all."""

    @pytest.fixture
    def client(self):
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
        app = create_app(cfg)
        services = get_services()
        services.catalog._tables[TENANT] = {FULL_NAME: table()}
        return TestClient(app)

    def test_the_settings_projection_carries_the_enrichment_model(self, client):
        """A field missing here does not merely fail to display. `updateSettings` patches
        governance then re-reads this projection, so the Admin picker would silently revert the
        control the user just moved."""
        body = client.get(f"/api/tenants/{TENANT}/settings").json()
        assert "enrichment_model" in body
        assert body["enrichment_model"] in {m["id"] for m in body["available_models"]}

    def test_enriching_an_unknown_table_is_refused(self, client):
        r = client.post(f"/api/tenants/{TENANT}/sources/enrich", json={"tables": ["nope.missing"]})
        assert r.status_code == 404
        assert "Scan the source first" in r.json()["detail"]

    def test_the_table_detail_reports_where_each_description_came_from(self, client):
        """Without this the review gate is invisible: a proposal exists, does nothing, and there is
        nowhere to approve it from."""
        body = client.get(f"/api/tenants/{TENANT}/tables/{FULL_NAME}").json()
        assert "description_source" in body
        assert "pending_enrichment" in body
        assert all("description_source" in c for c in body["columns"])

    def test_an_empty_description_is_refused_rather_than_stored(self, client):
        r = client.patch(
            f"/api/tenants/{TENANT}/tables/{FULL_NAME}/description", json={"text": "   "}
        )
        assert r.status_code == 400

    def test_describing_an_unknown_column_is_refused(self, client):
        r = client.patch(
            f"/api/tenants/{TENANT}/tables/{FULL_NAME}/description",
            json={"column": "nope", "text": "Something."},
        )
        assert r.status_code == 404

    def test_a_human_description_is_declared_and_sits_on_the_floor(self):
        """Not REVIEWER_CONFIDENCE. At 0.98 a description would outrank an ADVERSE_TO a partner
        approved at 0.9, which is the inversion DESCRIPTIVE_CONFIDENCE exists to undo."""
        from src.documents.review import REVIEWER_CONFIDENCE

        assert DESCRIPTIVE_CONFIDENCE < REVIEWER_CONFIDENCE
        onto = load_ontology("fintech")
        assertion = build_assertion(
            tenant_id=TENANT,
            subject_id=table_node_id(SOURCE, FULL_NAME),
            predicate=DESCRIBED_AS,
            object_id="description:abc",
            epistemic_class=EpistemicClass.DECLARED,
            method="admin:me@firm.example",
            confidence=DESCRIPTIVE_CONFIDENCE,
            source_locator=SourceLocator(source_id=SOURCE, table=FULL_NAME),
            allowed_predicates=onto.allowed_for(DESCRIBED_AS),
        )
        # DECLARED plus a non-governing predicate derives AUTO_ASSERTED, so it needs no review.
        assert assertion.review_state.value == "AUTO_ASSERTED"
        assert assertion.confidence == DESCRIPTIVE_CONFIDENCE

    def test_the_queue_item_says_which_table_a_claim_is_about(self):
        """`QueueItem` carried document_id and page but not table/column, so "every pending
        description for this table" was not a question the queue could answer."""
        from src.documents.review import AssertionRecord, QueueItem

        assertion = build_assertion(
            tenant_id=TENANT,
            subject_id=table_node_id(SOURCE, FULL_NAME),
            predicate=DESCRIBED_AS,
            object_id="description:abc",
            epistemic_class=EpistemicClass.EXTRACTED_MODEL,
            method="llm:m1",
            confidence=DESCRIPTIVE_CONFIDENCE,
            source_locator=SourceLocator(source_id=SOURCE, table=FULL_NAME, column="bal"),
        )
        item = QueueItem.of(AssertionRecord(assertion=assertion))
        assert item.table == FULL_NAME
        assert item.column == "bal"


class TestTheRunIsBounded:
    def test_the_table_cap_keeps_a_run_under_the_store_limit(self):
        """`GraphAssertionStore.DEFAULT_LIMIT` is 5000 and `all_for_tenant` truncates there. One
        table is roughly 1 + N columns + up to 14 terms, so an uncapped catalog-wide run would
        silently truncate the review queue it exists to fill."""
        from src.discovery.enrichment_run import MAX_TABLES_PER_RUN
        from src.graph.assertion_store import DEFAULT_LIMIT

        worst_case_per_table = 1 + 40 + 8 + 6
        assert MAX_TABLES_PER_RUN * worst_case_per_table < DEFAULT_LIMIT

    def test_existing_descriptions_are_not_re_guessed(self):
        """Built from the overlaid tables, so an approved model description or a person's edit is
        left alone. That is what makes "a model never overwrites a human" hold across runs."""
        from src.discovery.enrichment_run import existing_descriptions

        have = existing_descriptions([table(description="T.", column_description="C.")], SOURCE)
        assert have[FULL_NAME] == "T."
        assert have[f"{FULL_NAME}.bal"] == "C."
        assert f"{FULL_NAME}.fac_id" not in have


class TestCatalogClaimsStayOutOfTheReviewQueue:
    """One enrichment run took demo-firm's queue to 21 pending rows, all catalog metadata, with zero
    real claims left in it. That is verbatim the failure `AUTO_ASSERT_CLASSES` warns about: "a queue
    nobody can clear gets rubber-stamped, which destroys the guarantee the queue exists to give"."""

    def described(self, predicate=DESCRIBED_AS, table=FULL_NAME, column=None):
        return build_assertion(
            tenant_id=TENANT,
            subject_id=table_node_id(SOURCE, FULL_NAME),
            predicate=predicate,
            object_id="description:abc",
            epistemic_class=EpistemicClass.EXTRACTED_MODEL,
            method="llm:m1",
            confidence=DESCRIPTIVE_CONFIDENCE,
            source_locator=SourceLocator(source_id=SOURCE, table=table, column=column),
        )

    def from_a_document(self, predicate):
        return build_assertion(
            tenant_id=TENANT,
            subject_id="document:d1",
            predicate=predicate,
            object_id="topic:shipping",
            epistemic_class=EpistemicClass.EXTRACTED_MODEL,
            method="llm:m1",
            confidence=DESCRIPTIVE_CONFIDENCE,
            source_locator=SourceLocator(document_id="doc-1", page=3, quote="the parties agree"),
        )

    def test_a_description_is_a_catalog_claim(self):
        assert is_catalog_claim(self.described())
        assert is_catalog_claim(self.described(predicate="HAS_SYNONYM"))

    def test_a_topic_tag_on_a_table_is_a_catalog_claim(self):
        assert is_catalog_claim(self.described(predicate=CONCERNS_TOPIC, column="bal"))

    def test_a_topic_tag_on_a_document_is_not(self):
        """`CONCERNS_TOPIC` is the pack's general subject-matter tag, so document extraction writes
        it too. Filtering on the predicate alone hid real claims -- three existing tests went red,
        which is how this was caught."""
        assert not is_catalog_claim(self.from_a_document(CONCERNS_TOPIC))

    def test_a_document_claim_is_never_filtered(self):
        for predicate in ("MENTIONS", "REPRESENTS", "ADVERSE_TO"):
            assert not is_catalog_claim(self.from_a_document(predicate))

    def test_the_predicate_set_excludes_the_shared_one(self):
        """Pinned, because adding CONCERNS_TOPIC back here would silently hide document claims."""
        from src.discovery.enrichment import CATALOG_PREDICATES

        assert CONCERNS_TOPIC not in CATALOG_PREDICATES
        assert CATALOG_PREDICATES == {DESCRIBED_AS, "HAS_SYNONYM"}


class RouteGraph:
    """A graph for the table detail route: synonyms through the scope, metrics through `query`.

    The synonym rows carry a review state and this honours the scope's filter, so "only approved
    reaches the page" is tested rather than assumed. A fake that returned every row would pass
    whatever the route did.
    """

    def __init__(
        self,
        *,
        synonyms: list[dict[str, Any]] | None = None,
        approved_metrics: list[dict[str, Any]] | None = None,
        measuring: list[dict[str, Any]] | None = None,
        fail: bool = False,
    ) -> None:
        self.synonyms = synonyms or []
        self.approved_metrics = approved_metrics or []
        self.measuring = measuring or []
        self.fail = fail
        self.batches: list[tuple[str, list[dict[str, Any]]]] = []
        self.reads: list[str] = []

    def write_batch(self, cypher: str, batch: list[dict[str, Any]]) -> None:
        self.batches.append((cypher, batch))

    def read_scoped(self, template: str, scope: Any, params: Any = None) -> list[dict[str, Any]]:
        assert "{scope}" in template, "a scoped read must carry the token"
        if template != q.APPROVED_SYNONYMS:
            return []
        self.reads.append("synonyms")
        states = scope.params.get("scope_states")
        return [
            {"subject_id": r["subject_id"], "name": r["name"]}
            for r in self.synonyms
            if states is None or r["review_state"] in states
        ]

    def query(self, cypher: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        from src.graph import metric_queries as mq

        if self.fail:
            raise RuntimeError("neptune is unreachable")
        if cypher == mq.LIST_APPROVED_METRICS:
            return list(self.approved_metrics)
        if cypher == mq.METRICS_MEASURING_TABLE:
            assert params["full_name"] == FULL_NAME
            return list(self.measuring)
        return []


def metric_row(metric_id: str = "utilisation") -> dict[str, Any]:
    return {
        "metric_id": metric_id,
        "name": metric_id,
        "definition": "Share of available hours billed.",
        "expression": "SUM(billed) / SUM(available)",
        "source_table": FULL_NAME,
    }


def topic(
    object_id: str = "topic:credit-risk",
    *,
    from_a_document: bool = False,
    column: str | None = None,
) -> Any:
    locator = (
        SourceLocator(document_id="doc-1", page=2, quote="the parties agree")
        if from_a_document
        else SourceLocator(source_id=SOURCE, table=FULL_NAME, column=column)
    )
    return build_assertion(
        tenant_id=TENANT,
        subject_id="document:doc-1" if from_a_document else table_node_id(SOURCE, FULL_NAME),
        predicate=CONCERNS_TOPIC,
        object_id=object_id,
        epistemic_class=EpistemicClass.EXTRACTED_MODEL,
        method="llm:m1",
        confidence=DESCRIPTIVE_CONFIDENCE,
        source_locator=locator,
    )


class TestTheTableDetailContract:
    """What the table page renders about a table beyond its columns.

    The `metrics` key is the one to be careful with. Present and empty means "the lineage was read
    and nothing measures this table", which the page states as a fact about coverage. Absent means
    nobody could look. An empty list standing in for an unanswered question would tell a reader a
    table is unvetted on the strength of a graph timeout.
    """

    def api(self, graph: Any) -> tuple[Any, Any]:
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
        app = create_app(cfg)
        services = get_services()
        services.catalog._tables[TENANT] = {FULL_NAME: table()}
        services.graph = graph
        return TestClient(app), services

    def detail(self, graph: Any, staged: list[Any] | None = None) -> dict[str, Any]:
        client, services = self.api(graph)
        for assertion in staged or []:
            services.review_queue.stage(ctx(), [assertion], job_id="enrich")
        return client.get(f"/api/tenants/{TENANT}/tables/{FULL_NAME}").json()

    def approved(self, graph: Any, staged: list[Any]) -> dict[str, Any]:
        client, services = self.api(graph)
        services.review_queue.stage(ctx(), staged, job_id="enrich")
        for assertion in staged:
            services.review_queue.approve(ctx(), assertion.assertion_id, note="looks right")
        services.review_queue.promote(ctx(), job_id="enrich")
        return client.get(f"/api/tenants/{TENANT}/tables/{FULL_NAME}").json()

    def test_an_unmeasured_table_says_so_rather_than_saying_nothing(self):
        body = self.detail(RouteGraph())
        assert body["metrics"] == []

    def test_a_metric_that_reads_the_table_is_named(self):
        body = self.detail(RouteGraph(approved_metrics=[metric_row()], measuring=[metric_row()]))
        assert body["metrics"] == [
            {
                "metric_id": "utilisation",
                "name": "utilisation",
                "definition": "Share of available hours billed.",
            }
        ]

    def test_a_draft_metric_is_not_coverage(self):
        """The page says no *approved* metric reads this table, and `metrics_measuring` filters on
        no status at all, so a draft counted here would make that claim false."""
        body = self.detail(RouteGraph(approved_metrics=[], measuring=[metric_row()]))
        assert body["metrics"] == []

    def test_unreadable_lineage_omits_the_key_rather_than_sending_none(self):
        body = self.detail(RouteGraph(fail=True))
        assert "metrics" not in body

    def test_no_graph_omits_the_key_too(self):
        body = self.detail(None)
        assert "metrics" not in body

    def test_an_approved_synonym_is_shown(self):
        graph = RouteGraph(
            synonyms=[
                {
                    "subject_id": table_node_id(SOURCE, FULL_NAME),
                    "name": "credit lines",
                    "review_state": "APPROVED",
                }
            ]
        )
        assert self.detail(graph)["synonyms"] == ["credit lines"]

    def test_a_pending_synonym_is_not(self):
        """A model's guess must not widen what a table is called until somebody agrees with it."""
        graph = RouteGraph(
            synonyms=[
                {
                    "subject_id": table_node_id(SOURCE, FULL_NAME),
                    "name": "revolvers",
                    "review_state": "PENDING",
                }
            ]
        )
        assert self.detail(graph)["synonyms"] == []

    def test_another_tables_synonym_does_not_leak(self):
        graph = RouteGraph(
            synonyms=[
                {
                    "subject_id": table_node_id(SOURCE, "fin.other"),
                    "name": "elsewhere",
                    "review_state": "APPROVED",
                }
            ]
        )
        assert self.detail(graph)["synonyms"] == []

    def test_an_approved_topic_is_shown_as_a_label(self):
        body = self.approved(RouteGraph(), [topic()])
        assert body["topics"] == ["credit risk"]

    def test_a_pending_topic_is_not(self):
        body = self.detail(RouteGraph(), staged=[topic()])
        assert body["topics"] == []

    def test_a_topic_a_document_mentions_never_appears_on_a_table(self):
        """`CONCERNS_TOPIC` is shared with document extraction, so the predicate alone does not say
        what the claim is about. Without the locator check every filing's subject matter would be
        listed as a property of a table."""
        body = self.approved(RouteGraph(), [topic(from_a_document=True)])
        assert body["topics"] == []

    def test_the_three_fields_are_always_sent_when_the_graph_answers(self):
        """The UI reads `metrics` undefaulted and the other two with `?? []`. A field that is
        sometimes absent renders as an assertion about the table rather than about the API."""
        body = self.detail(RouteGraph())
        for field in ("metrics", "synonyms", "topics"):
            assert field in body


class Recorder:
    """An indexer that records what it was asked to rebuild. `catalog` is left unset so the
    container does not swap it for the enriched one, which would say nothing here."""

    def __init__(self, fail: bool = False) -> None:
        self.calls: list[dict[str, bool]] = []
        self.fail = fail
        self.catalog = None
        self.graph = None
        self.metric_store = None

    def rebuild(self, ctx: Any, *, metrics: bool, tables: bool, entities: bool) -> Any:
        if self.fail:
            raise RuntimeError("the embedding model timed out")
        self.calls.append({"metrics": metrics, "tables": tables, "entities": entities})
        return type("R", (), {"errors": [], "to_dict": lambda self: {}})()


class TestApprovalRefreshesTheRoutingIndex:
    """Approving a description changed nothing a question could reach until somebody pressed
    Rebuild: the routing index holds the words that were embedded, not the words now approved.

    After the response, never inside it. An approval that fails because an embedding call is slow is
    worse than an index that is a moment stale, and the approval is already durable by then.
    """

    def api(self, recorder: Recorder | None) -> tuple[Any, Any]:
        client, services = TestTheTableDetailContract().api(RouteGraph())
        services.router_indexer = recorder
        return client, services

    def pending(self, services: Any) -> None:
        assertion = build_assertion(
            tenant_id=TENANT,
            subject_id=table_node_id(SOURCE, FULL_NAME),
            predicate=DESCRIBED_AS,
            object_id="description:abc",
            epistemic_class=EpistemicClass.EXTRACTED_MODEL,
            method="llm:m1",
            confidence=DESCRIPTIVE_CONFIDENCE,
            source_locator=SourceLocator(source_id=SOURCE, table=FULL_NAME),
        )
        services.review_queue.stage(ctx(), [assertion], job_id="enrich")

    def test_approving_refreshes_the_table_layer_only(self):
        """A catalog approval cannot change a metric definition or a fact, and the entity layer is
        the expensive one."""
        recorder = Recorder()
        client, services = self.api(recorder)
        self.pending(services)

        r = client.post(f"/api/tenants/{TENANT}/tables/{FULL_NAME}/enrichment/approve")

        assert r.json()["approved"] == 1
        assert recorder.calls == [{"metrics": False, "tables": True, "entities": False}]

    def test_approving_nothing_reindexes_nothing(self):
        recorder = Recorder()
        client, _ = self.api(recorder)

        r = client.post(f"/api/tenants/{TENANT}/tables/{FULL_NAME}/enrichment/approve")

        assert r.json()["approved"] == 0
        assert recorder.calls == []

    def test_a_human_description_refreshes_it_too(self):
        """It is live the moment it is written, so the index is the only thing left stale."""
        recorder = Recorder()
        client, _ = self.api(recorder)

        r = client.patch(
            f"/api/tenants/{TENANT}/tables/{FULL_NAME}/description",
            json={"text": "Revolving credit facilities, one row each."},
        )

        assert r.json()["live"] is True
        assert recorder.calls == [{"metrics": False, "tables": True, "entities": False}]

    def test_an_index_that_cannot_be_refreshed_does_not_fail_the_approval(self):
        client, services = self.api(Recorder(fail=True))
        self.pending(services)

        r = client.post(f"/api/tenants/{TENANT}/tables/{FULL_NAME}/enrichment/approve")

        assert r.status_code == 200
        assert r.json()["approved"] == 1

    def test_no_routing_index_is_not_an_error_either(self):
        """Vector search off is a supported deployment, so there is nothing to refresh."""
        client, services = self.api(None)
        self.pending(services)

        r = client.post(f"/api/tenants/{TENANT}/tables/{FULL_NAME}/enrichment/approve")

        assert r.status_code == 200
        assert r.json()["approved"] == 1


class TestTheReviewQueueRefreshesItToo:
    """The same description can be approved from the Review Queue, which is a different route.

    Hooking only the per-table action left the generic path silently stale: a reviewer clearing the
    queue would approve wording that never reached tier selection, and nothing said so.
    """

    def api(self, recorder: Recorder | None) -> tuple[Any, Any, str]:
        client, services = TestApprovalRefreshesTheRoutingIndex().api(recorder)
        assertion = build_assertion(
            tenant_id=TENANT,
            subject_id=table_node_id(SOURCE, FULL_NAME),
            predicate=DESCRIBED_AS,
            object_id="description:abc",
            epistemic_class=EpistemicClass.EXTRACTED_MODEL,
            method="llm:m1",
            confidence=DESCRIPTIVE_CONFIDENCE,
            source_locator=SourceLocator(source_id=SOURCE, table=FULL_NAME),
        )
        services.review_queue.stage(ctx(), [assertion], job_id="enrich")
        return client, services, assertion.assertion_id

    def test_approving_one_from_the_queue_refreshes_the_table_layer(self):
        recorder = Recorder()
        client, _, assertion_id = self.api(recorder)

        r = client.post(f"/api/tenants/{TENANT}/assertions/{assertion_id}/approve")

        assert r.status_code == 200
        assert recorder.calls == [{"metrics": False, "tables": True, "entities": False}]

    def test_approving_in_bulk_refreshes_it_once(self):
        recorder = Recorder()
        client, _, assertion_id = self.api(recorder)

        r = client.post(
            f"/api/tenants/{TENANT}/assertions/approve",
            json={"assertion_ids": [assertion_id]},
        )

        assert r.status_code == 200
        assert recorder.calls == [{"metrics": False, "tables": True, "entities": False}]

    def test_approving_a_document_fact_does_not_reindex_the_catalog(self):
        """Most of what a reviewer approves is a fact read off a page, and re-embedding the schema
        for one of those would pay for the table layer on every approval in the product."""
        recorder = Recorder()
        client, services = TestApprovalRefreshesTheRoutingIndex().api(recorder)
        fact = build_assertion(
            tenant_id=TENANT,
            subject_id="party:acme",
            predicate="REPRESENTS",
            object_id="party:beta",
            epistemic_class=EpistemicClass.EXTRACTED_MODEL,
            method="llm:m1",
            confidence=0.7,
            matter_id="matter-1",
            source_locator=SourceLocator(
                document_id="doc-1",
                page=1,
                quote="Acme acts for Beta",
                char_start=0,
                char_end=18,
            ),
        )
        services.review_queue.stage(ctx(), [fact], job_id="ingest")

        client.post(f"/api/tenants/{TENANT}/assertions/{fact.assertion_id}/approve")

        assert recorder.calls == []


class TestTheSynonymProvider:
    """What reaches table selection when a question uses the firm's own word for a table.

    Keyed by `full_name`, never by graph id. `approved_synonyms` returns subject ids because that is
    what the edge carries, and a mapping handed to `relevant_tables` in that shape matches nothing at
    all: the id is built here with `table_node_id` rather than parsed anywhere, which is the rule
    `catalog_overlay` states.
    """

    def rows(self, review_state: str = "APPROVED", full_name: str = FULL_NAME) -> list[dict]:
        return [
            {
                "subject_id": table_node_id(SOURCE, full_name),
                "name": "credit lines",
                "review_state": review_state,
            }
        ]

    def provider(self, graph: Any) -> tuple[Any, Any]:
        _, services = TestTheTableDetailContract().api(graph)
        return services.catalog_synonyms(), services

    def test_the_mapping_is_keyed_by_the_table_name(self):
        synonyms_for, _ = self.provider(RouteGraph(synonyms=self.rows()))
        assert synonyms_for(ctx()) == {FULL_NAME: ["credit lines"]}

    def test_no_graph_id_survives_into_the_query_layer(self):
        synonyms_for, _ = self.provider(RouteGraph(synonyms=self.rows()))
        assert not [k for k in synonyms_for(ctx()) if k.startswith("table:")]

    def test_a_synonym_for_a_table_this_firm_has_not_scanned_is_dropped(self):
        synonyms_for, _ = self.provider(RouteGraph(synonyms=self.rows(full_name="fin.elsewhere")))
        assert synonyms_for(ctx()) == {}

    def test_a_pending_synonym_is_not_offered(self):
        """An unapproved guess must not widen what a question matches."""
        synonyms_for, _ = self.provider(RouteGraph(synonyms=self.rows(review_state="PENDING")))
        assert synonyms_for(ctx()) == {}

    def test_the_graph_is_read_once_however_many_lanes_ask(self):
        """`Planner` asks in the catalog part and again in the SQL part, so an uncached provider
        costs two round trips for one composed answer."""
        graph = RouteGraph(synonyms=self.rows())
        synonyms_for, _ = self.provider(graph)

        synonyms_for(ctx())
        synonyms_for(ctx())

        assert graph.reads == ["synonyms"]

    def test_no_graph_means_no_provider_rather_than_an_error(self):
        synonyms_for, _ = self.provider(None)
        assert synonyms_for is None

    def test_the_resolver_and_the_planner_both_get_one(self):
        _, services = self.provider(RouteGraph(synonyms=self.rows()))

        assert services.build_resolver(TENANT)._synonyms_for is not None
        assert services.build_planner(TENANT)._synonyms_for is not None
