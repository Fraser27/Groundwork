"""Tier 2: the graph chooses what gets read next.

Tier 3 lets similarity pick the documents and asks the graph to ground them. This is the
inversion, and the property that makes it worth having a direction at all: **every passage in a
graph-first answer is in a document some verified fact came out of.** So the tests here assert on
the request that went to the vector store rather than on what came back -- a lane that retrieved
across the corpus and then filtered would pass an assertion about its results and still have made
the wrong claim about them.

The second property is the dangerous one: **an empty landing zone retrieves nothing, never
everything.** A graph outage that quietly degraded this into tier 3 would return passages labelled
as though a fact had vouched for them, which is worse than returning none.

No AWS. Every collaborator is injected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from src.documents.embed import Embedder, InMemoryVectorStore, VectorRecord
from src.governance import GovernanceSettings
from src.graph.scope import AuthContext
from src.query.graph_first import MAX_GRAPH_TABLES, GraphFirstLane
from src.query.vector_search import VectorSearch

TENANT = "demo-firm"


@pytest.fixture
def ctx() -> AuthContext:
    return AuthContext(tenant_id=TENANT, user_id="alice")


def fact(aid: str, *, document_id: str | None = None, subject: str = "party:acme") -> dict:
    row: dict[str, Any] = {"assertion_id": aid, "subject_id": subject, "object_id": "party:beta"}
    if document_id is not None:
        row["source"] = {"document_id": document_id, "quote": "..."}
    return row


class FakeGraph:
    """Records what it was asked, which is what most of these assert on."""

    def __init__(
        self,
        hits: list[dict] | None = None,
        walked: list[dict] | None = None,
        catalog_nodes: list[str] | None = None,
    ) -> None:
        self.hits = hits or []
        self.walked = walked or []
        self.catalog_nodes = catalog_nodes or []
        self.searched_at: float | None = None
        self.expanded_from: list[str] | None = None

    def search(self, ctx: AuthContext, question: str, *, min_confidence: float = 0.8) -> list[dict]:
        self.searched_at = min_confidence
        return self.hits

    def expand(self, ctx: AuthContext, seeds: list[str], **kw: Any) -> list[dict]:
        self.expanded_from = list(seeds)
        return self.walked

    def catalog_search(self, ctx: AuthContext, question: str, **kw: Any) -> list[str]:
        return self.catalog_nodes


class FakeVectors:
    def __init__(self, passages: list[dict] | None = None) -> None:
        self.passages = passages or []
        self.calls: list[dict[str, Any]] = []

    def search(self, ctx: AuthContext, question: str, **kw: Any) -> list[dict]:
        self.calls.append(kw)
        return self.passages


@dataclass
class FakeColumn:
    name: str


@dataclass
class FakeTable:
    full_name: str
    description: str = ""
    columns: list[FakeColumn] = field(default_factory=list)


class FakeCatalog:
    def __init__(self, tables: list[FakeTable] | None = None, broken: bool = False) -> None:
        self._tables = tables or []
        self.broken = broken

    def tables(self, tenant_id: str) -> list[FakeTable]:
        if self.broken:
            raise RuntimeError("catalog store unreachable")
        return self._tables


@dataclass
class FakeGenerated:
    sql: str = "SELECT 1"
    tables_offered: list[str] = field(default_factory=list)


@dataclass
class FakeSqlResult:
    generated: FakeGenerated = field(default_factory=FakeGenerated)
    rows: dict | None = None
    error: str | None = None
    error_code: str | None = None


_DEFAULT = FakeSqlResult()


class FakeSqlLane:
    def __init__(self, result: FakeSqlResult | None = _DEFAULT) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    def run(self, question: str, **kw: Any) -> FakeSqlResult | None:
        self.calls.append(kw)
        return self.result


def lane(**over: Any) -> GraphFirstLane:
    kw: dict[str, Any] = {"graph_reader": FakeGraph()}
    kw.update(over)
    return GraphFirstLane(**kw)


TABLE_NODE = "table:glue:legal_db.matters"
COLUMN_NODE = "column:glue:legal_db.matters.fees"


class TestTheGraphChoosesTheDocuments:
    def test_retrieval_is_confined_to_the_documents_a_fact_came_from(self, ctx):
        """The claim the tier makes. Asserted on the request, because a lane that searched the
        corpus and filtered afterwards would look identical in its results and be entitled to
        none of the claim."""
        vectors = FakeVectors([{"document_id": "d1"}])
        lane(
            graph_reader=FakeGraph(hits=[fact("a1", document_id="d1")]), vector_search=vectors
        ).run(ctx, "who represents acme", GovernanceSettings())

        assert vectors.calls[0]["seed_documents"] == frozenset({"d1"})

    def test_a_fact_with_no_source_document_seeds_nothing(self, ctx):
        """An inferred fact has a proof tree rather than a page, so there is no document to read."""
        vectors = FakeVectors()
        result = lane(graph_reader=FakeGraph(hits=[fact("a1")]), vector_search=vectors).run(
            ctx, "q", GovernanceSettings()
        )

        assert vectors.calls == [], "the vector store was searched with no landing zone"
        assert "documents" not in result.landed

    def test_a_document_reached_by_the_walk_is_still_in_the_landing_zone(self, ctx):
        """Two hops out is still a document a verified fact came from, which is the whole claim.
        Seeding only from the term match would make reachability depend on which fact happened to
        contain the question's words."""
        vectors = FakeVectors([{"document_id": "d2"}])
        graph = FakeGraph(
            hits=[fact("a1", document_id="d1")], walked=[fact("a2", document_id="d2")]
        )
        lane(graph_reader=graph, vector_search=vectors).run(ctx, "q", GovernanceSettings())

        assert vectors.calls[0]["seed_documents"] == frozenset({"d1", "d2"})

    def test_the_walk_is_seeded_from_the_matched_entities(self, ctx):
        """The inversion in one assertion: tier 3 expands from retrieved passages, this expands
        from the entities the term match found, before anything has been retrieved."""
        graph = FakeGraph(hits=[fact("a1", subject="party:calder")])
        lane(graph_reader=graph).run(ctx, "calder", GovernanceSettings())

        assert graph.expanded_from == ["party:calder", "party:beta"]

    def test_a_walked_fact_already_matched_is_not_repeated(self, ctx):
        graph = FakeGraph(hits=[fact("a1")], walked=[fact("a1"), fact("a2")])
        result = lane(graph_reader=graph).run(ctx, "q", GovernanceSettings())

        assert [f["assertion_id"] for f in result.facts] == ["a1", "a2"]

    def test_the_term_search_is_held_to_the_trust_floor(self, ctx):
        graph = FakeGraph()
        settings = GovernanceSettings(min_confidence_floor=0.9)
        lane(graph_reader=graph).run(ctx, "q", settings)

        assert graph.searched_at == 0.9

    def test_no_graph_match_costs_the_passages_too(self, ctx):
        """Not a bug, and the reason the direction is a tenant-level choice: a document nothing has
        been extracted from yet is unreachable here however well it matches."""
        vectors = FakeVectors([{"document_id": "d1"}])
        result = lane(vector_search=vectors).run(ctx, "q", GovernanceSettings())

        assert vectors.calls == []
        assert result.empty


class TestAnEmptyLandingZoneRetrievesNothing:
    """The failure that would matter most: degrading into tier 3 while still claiming the graph
    vouched for the passages."""

    def test_vector_search_refuses_an_empty_seed_set(self, ctx):
        search = VectorSearch(Embedder(InMemoryVectorStore(), bedrock_factory=lambda: None))
        assert search.search(ctx, "q", seed_documents=frozenset()) == []

    def test_the_embedder_refuses_an_empty_seed_set_rather_than_widening(self, ctx):
        """Refused rather than treated as "no restriction". An empty set reaching the store as
        `None` is one missing guard away from an unfiltered search labelled as a graph-first one."""
        embedder = Embedder(InMemoryVectorStore(), bedrock_factory=lambda: None)
        with pytest.raises(ValueError, match="empty"):
            embedder.search(ctx, "q", seed_documents=frozenset())

    def test_the_store_returns_only_seeded_documents(self, ctx):
        store = InMemoryVectorStore()
        store.upsert("i", [_record("v1", "d1"), _record("v2", "d2")])

        hits = store.search("i", [1.0, 0.0], seed_documents=frozenset({"d1"}))

        assert [h.record.document_id for h in hits] == ["d1"]

    def test_a_seed_cannot_reach_past_the_matter_wall(self, ctx):
        """`seed_documents` narrows relevance, never authorization. A graph-first question must not
        become a way to read a walled matter, and the wall is applied whatever the seeds say."""
        store = InMemoryVectorStore()
        store.upsert("i", [_record("v1", "d1", matter_id="M-walled")])

        hits = store.search(
            "i",
            [1.0, 0.0],
            seed_documents=frozenset({"d1"}),
            matter_denylist=frozenset({"M-walled"}),
        )

        assert hits == []


def _record(vector_id: str, document_id: str, *, matter_id: str | None = None) -> VectorRecord:
    return VectorRecord(
        vector_id=vector_id,
        tenant_id=TENANT,
        document_id=document_id,
        page=1,
        char_start=0,
        char_end=3,
        text="...",
        embedding=(1.0, 0.0),
        model_id="test",
        matter_id=matter_id,
    )


class TestWhereItLanded:
    """ "No passage matched" and "no document was reached" are different facts about the same
    corpus, and a reader cannot tell them apart from an empty list."""

    def test_landing_is_reported_in_order(self, ctx):
        graph = FakeGraph(hits=[fact("a1", document_id="d1")], catalog_nodes=[TABLE_NODE])
        result = lane(
            graph_reader=graph,
            vector_search=FakeVectors([{"document_id": "d1"}]),
            catalog=FakeCatalog([FakeTable("legal_db.matters")]),
        ).run(ctx, "q", GovernanceSettings())

        assert result.landed == ["facts", "documents", "tables"]

    def test_nothing_matched_lands_nowhere(self, ctx):
        result = lane().run(ctx, "q", GovernanceSettings())
        assert result.landed == []

    def test_documents_reached_but_nothing_quotable_is_a_note(self, ctx):
        """The distinction the note exists for: the graph did its job and the documents held no
        close-enough passage, which is not the same as the graph reaching nothing."""
        result = lane(
            graph_reader=FakeGraph(hits=[fact("a1", document_id="d1")]),
            vector_search=FakeVectors([]),
        ).run(ctx, "q", GovernanceSettings())

        assert "documents" in result.landed
        assert any("no passage" in note for note in result.notes)

    def test_facts_alone_are_not_empty(self, ctx):
        result = lane(graph_reader=FakeGraph(hits=[fact("a1")])).run(ctx, "q", GovernanceSettings())
        assert result.empty is False


class TestTheTablesTheGraphNamed:
    def test_a_matched_table_node_resolves_against_the_catalog(self, ctx):
        table = FakeTable("legal_db.matters")
        result = lane(
            graph_reader=FakeGraph(catalog_nodes=[TABLE_NODE]), catalog=FakeCatalog([table])
        ).run(ctx, "how many matters", GovernanceSettings())

        assert result.tables == [table]

    def test_a_matched_column_counts_for_its_table(self, ctx):
        """ "Which columns exist" is not a question. The table holding the matched column is the
        thing a query runs against."""
        table = FakeTable("legal_db.matters")
        result = lane(
            graph_reader=FakeGraph(catalog_nodes=[COLUMN_NODE]), catalog=FakeCatalog([table])
        ).run(ctx, "fees", GovernanceSettings())

        assert result.tables == [table]

    def test_a_table_no_longer_catalogued_is_dropped(self, ctx):
        """Offering a generator a table the firewall's allowlist will not contain produces a query
        that cannot run, which reads to a user as the warehouse being empty."""
        result = lane(
            graph_reader=FakeGraph(catalog_nodes=[TABLE_NODE]), catalog=FakeCatalog([])
        ).run(ctx, "q", GovernanceSettings())

        assert result.tables == []

    def test_the_same_table_matched_twice_is_offered_once(self, ctx):
        table = FakeTable("legal_db.matters")
        result = lane(
            graph_reader=FakeGraph(catalog_nodes=[TABLE_NODE, COLUMN_NODE]),
            catalog=FakeCatalog([table]),
        ).run(ctx, "q", GovernanceSettings())

        assert result.tables == [table]

    def test_the_table_list_is_capped(self, ctx):
        """A long list means the term match was generic, not that the question was about twelve
        tables, and the generator's prompt would fill with schema the question only brushed."""
        tables = [FakeTable(f"legal_db.t{i}") for i in range(MAX_GRAPH_TABLES + 3)]
        nodes = [f"table:glue:{t.full_name}" for t in tables]
        result = lane(graph_reader=FakeGraph(catalog_nodes=nodes), catalog=FakeCatalog(tables)).run(
            ctx, "q", GovernanceSettings()
        )

        assert len(result.tables) == MAX_GRAPH_TABLES

    def test_no_catalog_means_no_table_lane(self, ctx):
        result = lane(graph_reader=FakeGraph(catalog_nodes=[TABLE_NODE])).run(
            ctx, "q", GovernanceSettings()
        )
        assert result.tables == []

    def test_a_catalog_that_cannot_be_read_is_a_note_not_a_failure(self, ctx):
        result = lane(
            graph_reader=FakeGraph(hits=[fact("a1")], catalog_nodes=[TABLE_NODE]),
            catalog=FakeCatalog(broken=True),
        ).run(ctx, "q", GovernanceSettings())

        assert result.facts, "a broken catalog took the facts down with it"
        assert any("catalog could not be read" in note for note in result.notes)

    def test_a_graph_that_cannot_search_the_catalog_is_not_an_error(self, ctx):
        class SearchOnly:
            def search(self, ctx, question, **kw):
                return []

        result = lane(
            graph_reader=SearchOnly(), catalog=FakeCatalog([FakeTable("legal_db.matters")])
        ).run(ctx, "q", GovernanceSettings())

        assert result.tables == []


class TestTheGeneratedQuery:
    def test_the_graphs_tables_are_passed_as_candidates(self, ctx):
        """`candidates=` rather than letting the lane re-derive them by word overlap. The graph
        reached these through DECLARED edges and approved synonyms, and overlap would drop exactly
        the table a synonym was approved to rescue: nothing in "turnover" matches `revenue`."""
        table = FakeTable("legal_db.revenue")
        sql = FakeSqlLane()
        lane(
            graph_reader=FakeGraph(catalog_nodes=["table:glue:legal_db.revenue"]),
            catalog=FakeCatalog([table]),
            sql_lane=sql,
        ).run(ctx, "what was our turnover", GovernanceSettings())

        assert sql.calls[0]["candidates"] == [table]

    def test_the_synonyms_the_caller_holds_reach_the_generator(self, ctx):
        sql = FakeSqlLane()
        lane(
            graph_reader=FakeGraph(catalog_nodes=[TABLE_NODE]),
            catalog=FakeCatalog([FakeTable("legal_db.matters")]),
            sql_lane=sql,
        ).run(ctx, "q", GovernanceSettings(), synonyms={"turnover": ["revenue"]})

        assert sql.calls[0]["synonyms"] == {"turnover": ["revenue"]}

    def test_no_query_could_be_written_is_a_note(self, ctx):
        sql = FakeSqlLane(result=None)
        result = lane(
            graph_reader=FakeGraph(catalog_nodes=[TABLE_NODE]),
            catalog=FakeCatalog([FakeTable("legal_db.matters")]),
            sql_lane=sql,
        ).run(ctx, "q", GovernanceSettings())

        assert result.generated is None
        assert any("legal_db.matters" in note for note in result.notes)

    def test_the_kill_switch_removes_the_query_and_keeps_everything_above_it(self, ctx):
        """The switch turns off AI-written queries, not the tier. Refusing the whole lane would
        take verified facts down with an ungoverned capability."""
        sql = FakeSqlLane()
        result = lane(
            graph_reader=FakeGraph(hits=[fact("a1")], catalog_nodes=[TABLE_NODE]),
            catalog=FakeCatalog([FakeTable("legal_db.matters")]),
            sql_lane=sql,
        ).run(ctx, "q", GovernanceSettings(), sql_allowed=False)

        assert sql.calls == []
        assert result.facts
        assert any("turned off" in note for note in result.notes)

    def test_no_sql_lane_at_all_still_names_the_tables(self, ctx):
        result = lane(
            graph_reader=FakeGraph(catalog_nodes=[TABLE_NODE]),
            catalog=FakeCatalog([FakeTable("legal_db.matters")]),
        ).run(ctx, "q", GovernanceSettings())

        assert result.landed == ["tables"]
        assert result.generated is None


class TestTheResultShape:
    def test_the_keys_are_the_ones_the_ui_narrows_on(self, ctx):
        """Pinned because nothing else can pin it. This payload is tier 2's whole `answer`, and the
        UI reads it by checking for a key -- `GraphFirstAnswer` in `ui/src/api.ts`. When this shape
        arrived, `asHits` was still looking for tier 3's `related` and returned nothing for every
        graph-first answer, with a clean `tsc` throughout. Renaming a key here must fail here."""
        payload = (
            lane(graph_reader=FakeGraph(hits=[fact("a1")]))
            .run(ctx, "q", GovernanceSettings())
            .to_dict()
        )

        assert set(payload) == {"facts", "passages", "tables", "landed"}

    def test_generated_is_absent_rather_than_null_when_nothing_was_written(self, ctx):
        """The UI reads `generated` to decide whether an answer contains model-written SQL, so a
        present-but-null key is the difference between "governed" and "check this query"."""
        payload = (
            lane(graph_reader=FakeGraph(hits=[fact("a1")]))
            .run(ctx, "q", GovernanceSettings())
            .to_dict()
        )

        assert "generated" not in payload

    def test_a_table_is_flattened_to_what_a_reader_needs(self, ctx):
        table = FakeTable("legal_db.matters", "Open matters", [FakeColumn("matter_id")])
        payload = (
            lane(graph_reader=FakeGraph(catalog_nodes=[TABLE_NODE]), catalog=FakeCatalog([table]))
            .run(ctx, "q", GovernanceSettings())
            .to_dict()
        )

        assert payload["tables"] == [
            {
                "full_name": "legal_db.matters",
                "description": "Open matters",
                "columns": ["matter_id"],
            }
        ]

    def test_the_generated_query_carries_its_error(self, ctx):
        """A hallucinated column errors at Athena, and reporting that as no rows would read as
        "no data" -- the silent empty the whole path exists to make impossible."""
        sql = FakeSqlLane(FakeSqlResult(error="column not found", error_code="INVALID"))
        payload = (
            lane(
                graph_reader=FakeGraph(catalog_nodes=[TABLE_NODE]),
                catalog=FakeCatalog([FakeTable("legal_db.matters")]),
                sql_lane=sql,
            )
            .run(ctx, "q", GovernanceSettings())
            .to_dict()
        )

        assert payload["generated"]["error"] == "column not found"
        assert payload["generated"]["error_code"] == "INVALID"
