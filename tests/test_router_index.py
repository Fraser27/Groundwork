"""The routing index, against a fake client shaped like the real one.

The assertions are about the *request body*, because that is the only place the properties that
make this safe are visible.

**One index, one mapping.** Metrics, entities and tables share it and are separated by a `kind`
keyword. The router compares layers, and the comparison only means anything because one query
vector hit one index.

**The kind filter is a pre-filter.** A tier the administrator forbade must never be queried, or
its items appear in the trace and disclose exactly what the tenant forbade the system to use.

**The matter wall matches `OpenSearchVectorStore.search` exactly**, including that a record with
no matter is tenant-wide and that the denylist is `must_not`. An entity's routing label is a
subject name, so a dropped denial names a screened party before the ethical wall has run.

**Scores come back unconverted.** `cosinesimil` reports `1 / (2 - cos)`; `router_scoring` owns the
conversion, and a second place that also converted would double it.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.documents.opensearch_store import DELETE_PAGE_SIZE, VECTOR_FIELD, VectorStoreError
from src.graph.scope import AuthContext
from src.query.router_index import (
    KIND_ENTITY,
    KIND_METRIC,
    KIND_TABLE,
    RoutingIndex,
    RoutingRecord,
    routing_index_for_tenant,
    routing_index_name,
)

INDEX = "tenant-demo-firm-routing"


class FakeIndices:
    def __init__(self, existing: set[str]) -> None:
        self.existing = existing
        self.created: list[dict[str, Any]] = []

    def exists(self, index: str) -> bool:
        return index in self.existing

    def create(self, index: str, body: dict[str, Any]) -> dict[str, Any]:
        self.created.append({"index": index, "body": body})
        self.existing.add(index)
        return {"acknowledged": True}


class FakeOpenSearch:
    def __init__(
        self, *, hits: list[dict[str, Any]] | None = None, existing: set[str] | None = None
    ) -> None:
        self.indices = FakeIndices(existing or set())
        self.hits = hits or []
        self.searches: list[dict[str, Any]] = []
        self.bulks: list[list[dict[str, Any]]] = []
        self.delete_hits: list[dict[str, Any]] = []
        self.bulk_errors = False

    def bulk(self, body: list[dict[str, Any]], **kw: Any) -> dict[str, Any]:
        self.bulks.append(body)
        if self.bulk_errors:
            return {
                "errors": True,
                "items": [{"index": {"error": {"type": "mapper_parsing_exception"}}}],
            }
        return {"errors": False, "items": []}

    def search(self, index: str, body: dict[str, Any], **kw: Any) -> dict[str, Any]:
        self.searches.append({"index": index, "body": body})
        if "_source" in body and body["_source"] is False:
            page, self.delete_hits = self.delete_hits, []
            return {"hits": {"hits": page}}
        return {"hits": {"hits": self.hits}}

    def count(self, index: str, **kw: Any) -> dict[str, Any]:
        return {"count": 0}


def index(client: FakeOpenSearch, **kw: Any) -> RoutingIndex:
    return RoutingIndex(endpoint="https://x.aoss.amazonaws.com", client=client, **kw)


def record(
    vector_id: str = "demo-firm:metric:fees_billed",
    *,
    kind: str = KIND_METRIC,
    item_id: str = "fees_billed",
    matter_id: str | None = None,
) -> RoutingRecord:
    return RoutingRecord(
        vector_id=vector_id,
        tenant_id="demo-firm",
        kind=kind,
        item_id=item_id,
        label="Fees billed",
        text="Fees billed. invoiced. amounts billed. Value invoiced to clients in a period.",
        embedding=(0.1, 0.2, 0.3),
        model_id="amazon.titan-embed-text-v2:0",
        matter_id=matter_id,
        detail={"expression": "SUM(billed_value)", "source_table": "legal_ops.invoices"},
    )


def _knn(client: FakeOpenSearch) -> dict[str, Any]:
    return client.searches[-1]["body"]["query"]["knn"][VECTOR_FIELD]


def _filter(client: FakeOpenSearch) -> dict[str, Any]:
    return _knn(client).get("filter", {}).get("bool", {})


class TestTheIndexName:
    def test_it_is_the_cluster_key_plus_routing(self):
        ctx = AuthContext(user_id="alice", tenant_id="demo-firm")
        assert routing_index_name(ctx) == INDEX

    def test_the_admin_form_agrees_with_the_scoped_form(self):
        """A reset runs from an admin route with no request scope. If these two ever disagreed a
        reset would clear an index nobody reads and leave the live one populated."""
        ctx = AuthContext(user_id="alice", tenant_id="demo-firm")
        assert routing_index_for_tenant("demo-firm") == routing_index_name(ctx)


class TestTheMapping:
    def test_it_carries_kind_as_a_keyword(self):
        """One index for three layers. Without `kind` the scores are still comparable but a
        forbidden tier cannot be filtered out, which is the whole disclosure problem."""
        client = FakeOpenSearch()
        index(client).upsert(INDEX, [record()])

        props = client.indices.created[0]["body"]["mappings"]["properties"]
        assert props["kind"] == {"type": "keyword"}

    def test_every_routable_field_is_mapped(self):
        client = FakeOpenSearch()
        index(client).upsert(INDEX, [record()])

        props = client.indices.created[0]["body"]["mappings"]["properties"]
        for name in ("item_id", "label", "text", "tenant_id", "matter_id", "model_id"):
            assert name in props, name

    def test_detail_is_stored_but_not_indexed(self):
        """Trace payload only. Indexed, a metric's SQL expression would become searchable text and
        a column name would influence routing scores."""
        client = FakeOpenSearch()
        index(client).upsert(INDEX, [record()])

        props = client.indices.created[0]["body"]["mappings"]["properties"]
        assert props["detail"] == {"type": "object", "enabled": False}

    def test_the_vector_is_cosine_hnsw_with_no_engine(self):
        """`engine` is rejected outright by a NextGen collection -- "Field parameter 'engine' is
        not supported" -- and cosine matches the chunk index so the two score alike."""
        client = FakeOpenSearch()
        index(client).upsert(INDEX, [record()])

        vector = client.indices.created[0]["body"]["mappings"]["properties"][VECTOR_FIELD]
        assert vector["type"] == "knn_vector"
        assert vector["method"]["space_type"] == "cosinesimil"
        assert "engine" not in vector["method"]

    def test_the_declared_dimension_is_the_configured_one(self):
        client = FakeOpenSearch()
        index(client, dimensions=512).upsert(INDEX, [record()])

        props = client.indices.created[0]["body"]["mappings"]["properties"]
        assert props[VECTOR_FIELD]["dimension"] == 512

    def test_an_existing_index_is_not_recreated(self):
        client = FakeOpenSearch(existing={INDEX})
        index(client).upsert(INDEX, [record()])
        assert client.indices.created == []


class TestWriting:
    def test_upsert_is_idempotent_by_vector_id(self):
        """A rebuild must converge. An accumulating routing index inflates a layer's hit count on
        every rebuild and skews the router toward whichever layer was rebuilt most."""
        client = FakeOpenSearch()
        idx = index(client)
        idx.upsert(INDEX, [record()])
        idx.upsert(INDEX, [record()])

        ids = [body[0]["index"]["_id"] for body in client.bulks]
        assert ids == ["demo-firm:metric:fees_billed", "demo-firm:metric:fees_billed"]

    def test_a_record_writes_its_kind_and_its_trace_detail(self):
        client = FakeOpenSearch()
        index(client).upsert(INDEX, [record()])

        _, doc = client.bulks[0]
        assert doc["kind"] == KIND_METRIC
        assert doc["item_id"] == "fees_billed"
        assert doc["detail"]["expression"] == "SUM(billed_value)"
        assert doc[VECTOR_FIELD] == [0.1, 0.2, 0.3]

    def test_three_kinds_go_to_one_index(self):
        """Not three indexes. Scores from different kinds have to be comparable, and they are
        comparable precisely because one query vector hit one index."""
        client = FakeOpenSearch()
        written = index(client).upsert(
            INDEX,
            [
                record("a", kind=KIND_METRIC, item_id="fees_billed"),
                record("b", kind=KIND_TABLE, item_id="legal_ops.invoices"),
                record("c", kind=KIND_ENTITY, item_id="party:meridian"),
            ],
        )

        assert written == 3
        targets = {action["index"]["_index"] for action in client.bulks[0][::2]}
        assert targets == {INDEX}

    def test_writing_nothing_is_not_an_error(self):
        client = FakeOpenSearch()
        assert index(client).upsert(INDEX, []) == 0
        assert client.bulks == []

    def test_a_partial_write_is_raised_rather_than_counted(self):
        client = FakeOpenSearch()
        client.bulk_errors = True
        with pytest.raises(VectorStoreError, match="rejected"):
            index(client).upsert(INDEX, [record()])

    def test_a_write_does_not_request_a_refresh(self):
        """Serverless rejects a refresh policy: refresh timing is the service's to decide."""
        client = FakeOpenSearch()
        seen: list[dict[str, Any]] = []

        def strict_bulk(body: list[dict[str, Any]], **kw: Any) -> dict[str, Any]:
            assert "refresh" not in kw
            seen.append({"body": body})
            return {"errors": False, "items": []}

        client.bulk = strict_bulk  # type: ignore[method-assign]
        index(client).upsert(INDEX, [record()])
        assert len(seen) == 1


class TestTheKindFilter:
    def test_kinds_become_a_terms_clause(self):
        """The router passes only the permitted kinds. A tier the administrator forbade must not
        be queried at all -- its items would otherwise reach the trace."""
        client = FakeOpenSearch()
        index(client).search(INDEX, [0.1], kinds=frozenset({KIND_METRIC, KIND_TABLE}))

        assert {"terms": {"kind": [KIND_METRIC, KIND_TABLE]}} in _filter(client)["must"]

    def test_no_kinds_means_no_kind_clause(self):
        """None is "every layer". A clause would narrow it, and a router that could not ask about
        everything could not fall back to trying everything."""
        client = FakeOpenSearch()
        index(client).search(INDEX, [0.1])

        assert "filter" not in _knn(client)

    def test_an_empty_kind_set_still_filters(self):
        """Distinct from None. An administrator who forbade every tier must get nothing, not
        everything -- the difference between the two is the whole disclosure risk."""
        client = FakeOpenSearch()
        index(client).search(INDEX, [0.1], kinds=frozenset())

        assert {"terms": {"kind": []}} in _filter(client)["must"]

    def test_the_kind_filter_is_inside_the_knn_clause(self):
        """A pre-filter, so top_k is computed over the permitted layers. Filtered afterwards, a
        forbidden layer would consume slots and the permitted ones would come back short."""
        client = FakeOpenSearch()
        index(client).search(INDEX, [0.1], top_k=5, kinds=frozenset({KIND_METRIC}))

        assert "filter" in _knn(client)
        assert client.searches[-1]["body"]["size"] == 5
        assert _knn(client)["k"] == 5


class TestTheMatterWallMirrorsTheChunkStore:
    def test_an_allowlist_also_admits_records_with_no_matter(self):
        """A routing description of a metric or a table is tenant-wide, mirroring `edge_scope`:
        matter_id IS NULL OR matter_id IN allowlist."""
        client = FakeOpenSearch()
        index(client).search(INDEX, [0.1], matter_allowlist=frozenset({"M-1"}))

        should = _filter(client)["must"][0]["bool"]["should"]
        assert {"bool": {"must_not": {"exists": {"field": "matter_id"}}}} in should
        assert {"terms": {"matter_id": ["M-1"]}} in should

    def test_a_denylist_becomes_must_not(self):
        """A denial beats a permission. An entity's routing label is a subject name, so a screened
        party would otherwise be named before the ethical wall runs."""
        client = FakeOpenSearch()
        index(client).search(
            INDEX,
            [0.1],
            matter_allowlist=frozenset({"M-1", "M-2"}),
            matter_denylist=frozenset({"M-2"}),
        )

        assert _filter(client)["must_not"] == [{"terms": {"matter_id": ["M-2"]}}]

    def test_an_admin_with_no_allowlist_gets_no_allowlist_clause(self):
        client = FakeOpenSearch()
        index(client).search(INDEX, [0.1], matter_allowlist=None)
        assert "filter" not in _knn(client)

    def test_an_admin_is_still_subject_to_a_screen(self):
        client = FakeOpenSearch()
        index(client).search(
            INDEX, [0.1], matter_allowlist=None, matter_denylist=frozenset({"M-9"})
        )

        clause = _filter(client)
        assert "must" not in clause
        assert clause["must_not"] == [{"terms": {"matter_id": ["M-9"]}}]

    def test_the_kind_filter_and_the_matter_wall_compose(self):
        """Both, not either. A forbidden tier and a screened matter are independent refusals."""
        client = FakeOpenSearch()
        index(client).search(
            INDEX,
            [0.1],
            kinds=frozenset({KIND_ENTITY}),
            matter_allowlist=frozenset({"M-1"}),
            matter_denylist=frozenset({"M-3"}),
        )

        clause = _filter(client)
        assert {"terms": {"kind": [KIND_ENTITY]}} in clause["must"]
        assert len(clause["must"]) == 2
        assert clause["must_not"] == [{"terms": {"matter_id": ["M-3"]}}]

    def test_the_clause_shape_matches_the_chunk_store(self):
        """Asserted against the real thing, not a copy of it. The two walls diverging silently is
        how one half of retrieval ends up scoped and the other half not."""
        from src.documents.opensearch_store import OpenSearchVectorStore

        chunks = FakeOpenSearch()
        OpenSearchVectorStore(endpoint="x", client=chunks).search(
            "chunks",
            [0.1],
            matter_allowlist=frozenset({"M-1"}),
            matter_denylist=frozenset({"M-2"}),
        )
        routing = FakeOpenSearch()
        index(routing).search(
            INDEX,
            [0.1],
            matter_allowlist=frozenset({"M-1"}),
            matter_denylist=frozenset({"M-2"}),
        )

        assert _filter(routing) == _filter(chunks)


class TestReading:
    def test_a_hit_carries_its_kind_item_and_label(self):
        client = FakeOpenSearch(
            hits=[
                {
                    "_id": "demo-firm:entity:party:meridian",
                    "_score": 0.82,
                    "_source": {
                        "tenant_id": "demo-firm",
                        "kind": KIND_ENTITY,
                        "item_id": "party:meridian",
                        "label": "meridian",
                        "text": "meridian. Party. Represents",
                        "model_id": "titan",
                        "matter_id": "M-1",
                        "detail": {"layer": "domain"},
                    },
                }
            ]
        )
        hits = index(client).search(INDEX, [0.1])

        assert len(hits) == 1
        assert hits[0].record.kind == KIND_ENTITY
        assert hits[0].record.item_id == "party:meridian"
        assert hits[0].record.label == "meridian"
        assert hits[0].record.detail == {"layer": "domain"}

    def test_the_score_comes_back_unconverted(self):
        """`router_scoring.cosine_of` owns the conversion. Converting here as well would apply it
        twice, and 0.82 would silently become a different threshold than an admin set."""
        client = FakeOpenSearch(
            hits=[{"_id": "a", "_score": 0.82, "_source": {"kind": KIND_METRIC}}]
        )
        assert index(client).search(INDEX, [0.1])[0].raw_score == pytest.approx(0.82)

    def test_the_embedding_is_not_returned(self):
        client = FakeOpenSearch(
            hits=[{"_id": "a", "_score": 0.5, "_source": {"kind": KIND_METRIC}}]
        )
        assert index(client).search(INDEX, [0.1])[0].record.embedding == ()

    def test_a_missing_index_is_an_empty_result(self):
        """A tenant whose router has never been built has no index. The router reads an empty
        result as "nothing looked relevant" and tries every tier, which is the honest fallback."""

        class NoIndex(FakeOpenSearch):
            def search(self, index: str, body: dict[str, Any], **kw: Any) -> dict[str, Any]:
                raise RuntimeError("index_not_found_exception")

        assert index(NoIndex()).search(INDEX, [0.1]) == []

    def test_a_real_failure_is_raised(self):
        """A refused query must not look like a tenant with no router: silently routing every
        question to every tier because of a 403 is a cost nobody would notice."""

        class Broken(FakeOpenSearch):
            def search(self, index: str, body: dict[str, Any], **kw: Any) -> dict[str, Any]:
                raise RuntimeError("403 Forbidden")

        with pytest.raises(VectorStoreError):
            index(Broken()).search(INDEX, [0.1])


class TestDeleting:
    """Search-then-bulk, paged. Serverless has no `_delete_by_query`."""

    def _with_hits(self, ids: list[str]) -> FakeOpenSearch:
        client = FakeOpenSearch(existing={INDEX})
        client.delete_hits = [{"_id": i} for i in ids]
        return client

    def test_a_kind_is_deleted_by_term_on_kind(self):
        client = self._with_hits(["a", "b"])
        assert index(client).delete_kind(INDEX, KIND_METRIC) == 2

        assert client.searches[-1]["body"]["query"] == {"term": {"kind": KIND_METRIC}}
        assert client.bulks[-1] == [
            {"delete": {"_index": INDEX, "_id": "a"}},
            {"delete": {"_index": INDEX, "_id": "b"}},
        ]

    def test_deleting_one_kind_leaves_the_others(self):
        """A metric reindex must not take the entity layer with it, or a rebuild of one layer
        would silently unroute the other two."""
        client = self._with_hits(["a"])
        index(client).delete_kind(INDEX, KIND_METRIC)

        assert client.searches[-1]["body"]["query"]["term"]["kind"] == KIND_METRIC

    def test_a_tenant_reset_targets_the_tenant_and_keeps_the_mapping(self):
        client = self._with_hits(["a"])
        index(client).delete_tenant(INDEX, "demo-firm")

        assert client.searches[-1]["body"]["query"] == {"term": {"tenant_id": "demo-firm"}}
        assert client.indices.created == []

    def test_the_id_search_does_not_fetch_the_documents(self):
        client = self._with_hits(["a"])
        index(client).delete_kind(INDEX, KIND_METRIC)
        assert client.searches[-1]["body"]["_source"] is False

    def test_a_full_page_is_followed_by_another_round(self):
        """A firm with thousands of entities spans pages. Stopping after the first would leave
        descriptions behind for entities the graph no longer holds."""
        client = FakeOpenSearch(existing={INDEX})
        client.delete_hits = [{"_id": f"e{i}"} for i in range(DELETE_PAGE_SIZE)]

        assert index(client).delete_kind(INDEX, KIND_ENTITY) == DELETE_PAGE_SIZE
        assert len(client.searches) == 2

    def test_deleting_from_a_missing_index_is_zero(self):
        class NoIndex(FakeOpenSearch):
            def search(self, index: str, body: dict[str, Any], **kw: Any):
                raise RuntimeError("index_not_found_exception")

        assert index(NoIndex()).delete_kind(INDEX, KIND_METRIC) == 0

    def test_nothing_matching_deletes_nothing(self):
        client = FakeOpenSearch(existing={INDEX})
        assert index(client).delete_kind(INDEX, KIND_METRIC) == 0
        assert client.bulks == []


class TestItReusesTheServerlessConstraints:
    def test_the_constants_are_imported_rather_than_redefined(self):
        """One file expresses each Serverless constraint. Two copies drift, and the second one to
        be edited costs a deploy to discover."""
        import src.documents.opensearch_store as chunks
        import src.query.router_index as routing

        assert routing.VECTOR_FIELD is chunks.VECTOR_FIELD
        assert routing.DELETE_PAGE_SIZE is chunks.DELETE_PAGE_SIZE
        assert routing.REQUEST_TIMEOUT_SECONDS is chunks.REQUEST_TIMEOUT_SECONDS
        assert routing.VectorStoreError is chunks.VectorStoreError

    def test_it_signs_through_the_same_client_builder(self):
        """`_client` pairs `RequestsAWSV4SignerAuth` with `RequestsHttpConnection`. A second
        builder here would eventually pair the urllib3 signer instead and send unsigned requests
        that OpenSearch answers with 401, which points at authentication rather than the client."""
        import src.documents.opensearch_store as chunks
        import src.query.router_index as routing

        assert routing._client is chunks._client
