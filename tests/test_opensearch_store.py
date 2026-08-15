"""The OpenSearch vector store, against a fake client shaped like the real one.

The interesting assertions are about the *query*, not the round trip. Two properties decide
whether this store is safe, and both are visible only in the request body:

**The matter wall is a pre-filter inside the kNN clause.** OpenSearch applies `filter` during
the search, so top_k is computed over documents this caller may read. Filtering afterwards
would silently return fewer than top_k whenever a screen applied, and a lawyer would read that
shortfall as "nothing relevant" rather than "you cannot see it".

**A denial beats a permission.** The denylist is `must_not`, so a screened matter is refused
even when it is also assigned — matching `edge_scope`, where the same rule holds.

A record with no matter is tenant-wide, because tenancy is already enforced physically by the
index name. That mirrors `edge_scope`'s `matter_id IS NULL OR matter_id IN allowlist`.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.documents.embed import VectorRecord
from src.documents.opensearch_store import (
    VECTOR_FIELD,
    OpenSearchVectorStore,
    VectorStoreError,
)

INDEX = "t-demo-firm-chunks"


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
    """Enough of opensearch-py to exercise the store, recording what was sent."""

    def __init__(
        self, *, hits: list[dict[str, Any]] | None = None, existing: set[str] | None = None
    ):
        self.indices = FakeIndices(existing or set())
        self.hits = hits or []
        self.searches: list[dict[str, Any]] = []
        self.bulks: list[list[dict[str, Any]]] = []
        self.deletes: list[dict[str, Any]] = []
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
        return {"hits": {"hits": self.hits}}

    def delete_by_query(self, index: str, body: dict[str, Any], **kw: Any) -> dict[str, Any]:
        self.deletes.append({"index": index, "body": body})
        return {"deleted": 2}

    def count(self, index: str, **kw: Any) -> dict[str, Any]:
        if index not in self.indices.existing:
            raise RuntimeError("index_not_found_exception")
        return {"count": 42}


def store(client: FakeOpenSearch) -> OpenSearchVectorStore:
    return OpenSearchVectorStore(endpoint="https://x.aoss.amazonaws.com", client=client)


def record(vector_id: str = "c1", *, matter_id: str | None = "M-1") -> VectorRecord:
    return VectorRecord(
        vector_id=vector_id,
        tenant_id="demo-firm",
        document_id="doc-1",
        page=3,
        char_start=10,
        char_end=90,
        text="Meridian holds 18 per cent of Calder Shipping AG",
        embedding=(0.1, 0.2, 0.3),
        model_id="amazon.titan-embed-text-v2:0",
        matter_id=matter_id,
    )


def _knn(client: FakeOpenSearch) -> dict[str, Any]:
    return client.searches[-1]["body"]["query"]["knn"][VECTOR_FIELD]


class TestWriting:
    def test_records_are_written_with_the_chunk_id_as_the_document_id(self):
        """Re-embedding must overwrite. The store is a derived index, so a rebuild has to
        converge rather than accumulate duplicates of every chunk."""
        client = FakeOpenSearch()
        assert store(client).upsert(INDEX, [record("chunk-7")]) == 1

        action, doc = client.bulks[0]
        assert action["index"]["_id"] == "chunk-7"
        assert action["index"]["_index"] == INDEX
        assert doc["page"] == 3
        assert doc[VECTOR_FIELD] == [0.1, 0.2, 0.3]

    def test_the_index_is_created_on_first_use(self):
        """Lazily, because the index name contains the tenant, so the set is not known until
        a tenant exists."""
        client = FakeOpenSearch()
        store(client).upsert(INDEX, [record()])

        created = client.indices.created[0]
        assert created["index"] == INDEX
        assert created["body"]["mappings"]["properties"][VECTOR_FIELD]["type"] == "knn_vector"

    def test_cosine_is_the_similarity(self):
        """Matches `_cosine` in the in-memory store, so a local result and a deployed one rank
        the same way. With l2 a retrieval bug could not be reproduced locally."""
        client = FakeOpenSearch()
        store(client).upsert(INDEX, [record()])

        method = client.indices.created[0]["body"]["mappings"]["properties"][VECTOR_FIELD]["method"]
        assert method["space_type"] == "cosinesimil"

    def test_the_declared_dimension_is_the_configured_one(self):
        client = FakeOpenSearch()
        OpenSearchVectorStore(endpoint="x", client=client, dimensions=512).upsert(INDEX, [record()])
        props = client.indices.created[0]["body"]["mappings"]["properties"]
        assert props[VECTOR_FIELD]["dimension"] == 512

    def test_an_existing_index_is_not_recreated(self):
        client = FakeOpenSearch(existing={INDEX})
        store(client).upsert(INDEX, [record()])
        assert client.indices.created == []

    def test_writing_nothing_is_not_an_error(self):
        client = FakeOpenSearch()
        assert store(client).upsert(INDEX, []) == 0
        assert client.bulks == []

    def test_a_partial_write_is_raised_rather_than_counted(self):
        """A half-written index disagrees with the graph about what is searchable, and the
        reason is what makes that diagnosable."""
        client = FakeOpenSearch()
        client.bulk_errors = True
        with pytest.raises(VectorStoreError, match="rejected"):
            store(client).upsert(INDEX, [record()])


class TestTheMatterWallIsAPreFilter:
    def test_the_filter_lives_inside_the_knn_clause(self):
        """Not applied afterwards. A post-filter returns fewer than top_k when a screen bites,
        and the shortfall reads as irrelevance rather than refusal."""
        client = FakeOpenSearch()
        store(client).search(INDEX, [0.1, 0.2], top_k=5, matter_allowlist=frozenset({"M-1"}))

        assert "filter" in _knn(client)

    def test_an_allowlist_also_admits_records_with_no_matter(self):
        """Tenant-wide records are readable by anyone in the tenant, matching `edge_scope`:
        matter_id IS NULL OR matter_id IN allowlist."""
        client = FakeOpenSearch()
        store(client).search(INDEX, [0.1], matter_allowlist=frozenset({"M-1"}))

        should = _knn(client)["filter"]["bool"]["must"][0]["bool"]["should"]
        assert {"bool": {"must_not": {"exists": {"field": "matter_id"}}}} in should
        assert {"terms": {"matter_id": ["M-1"]}} in should

    def test_a_denylist_becomes_must_not(self):
        """A denial beats a permission, as everywhere else. An ethical screen is refused even
        when the same matter is assigned."""
        client = FakeOpenSearch()
        store(client).search(
            INDEX,
            [0.1],
            matter_allowlist=frozenset({"M-1", "M-2"}),
            matter_denylist=frozenset({"M-2"}),
        )

        clause = _knn(client)["filter"]["bool"]
        assert clause["must_not"] == [{"terms": {"matter_id": ["M-2"]}}]

    def test_an_admin_with_no_allowlist_gets_no_allowlist_clause(self):
        """`matter_allowlist=None` means "every matter", so adding a clause would narrow it."""
        client = FakeOpenSearch()
        store(client).search(INDEX, [0.1], matter_allowlist=None)
        assert "filter" not in _knn(client)

    def test_an_admin_is_still_subject_to_a_screen(self):
        """A screen beats the administrator role. An admin who can read through an ethical
        wall is not screened."""
        client = FakeOpenSearch()
        store(client).search(
            INDEX, [0.1], matter_allowlist=None, matter_denylist=frozenset({"M-9"})
        )

        clause = _knn(client)["filter"]["bool"]
        assert "must" not in clause
        assert clause["must_not"] == [{"terms": {"matter_id": ["M-9"]}}]

    def test_top_k_reaches_both_the_query_and_the_knn_clause(self):
        client = FakeOpenSearch()
        store(client).search(INDEX, [0.1], top_k=7)
        assert client.searches[-1]["body"]["size"] == 7
        assert _knn(client)["k"] == 7


class TestReading:
    def test_a_hit_becomes_a_record_with_its_citation(self):
        client = FakeOpenSearch(
            hits=[
                {
                    "_id": "chunk-7",
                    "_score": 0.82,
                    "_source": {
                        "tenant_id": "demo-firm",
                        "document_id": "doc-1",
                        "page": 4,
                        "char_start": 5,
                        "char_end": 40,
                        "text": "the Adverse Party",
                        "model_id": "titan",
                        "matter_id": "M-1",
                    },
                }
            ]
        )
        hits = store(client).search(INDEX, [0.1])

        assert len(hits) == 1
        assert hits[0].score == pytest.approx(0.82)
        assert hits[0].record.page == 4
        assert hits[0].record.text == "the Adverse Party"

    def test_the_embedding_is_not_returned(self):
        """Megabytes across a page of hits, and no caller reads it. The citation is the page
        and the text."""
        client = FakeOpenSearch(
            hits=[{"_id": "c1", "_score": 0.5, "_source": {"tenant_id": "t", "page": 1}}]
        )
        assert store(client).search(INDEX, [0.1])[0].record.embedding == ()

    def test_a_missing_index_is_an_empty_result(self):
        """A tenant who has uploaded nothing has no index. That is a normal state."""

        class NoIndex(FakeOpenSearch):
            def search(self, index: str, body: dict[str, Any], **kw: Any) -> dict[str, Any]:
                raise RuntimeError("index_not_found_exception")

        assert store(NoIndex()).search(INDEX, [0.1]) == []

    def test_a_real_failure_is_raised(self):
        """Distinct from a missing index: a cluster that refused the query must not look like
        a tenant with no documents."""

        class Broken(FakeOpenSearch):
            def search(self, index: str, body: dict[str, Any], **kw: Any) -> dict[str, Any]:
                raise RuntimeError("403 Forbidden")

        with pytest.raises(VectorStoreError):
            store(Broken()).search(INDEX, [0.1])


class TestDeleting:
    def test_a_document_is_deleted_by_query(self):
        client = FakeOpenSearch()
        assert store(client).delete_document(INDEX, "doc-1") == 2
        assert client.deletes[0]["body"]["query"] == {"term": {"document_id": "doc-1"}}

    def test_a_tenant_reset_keeps_the_mapping(self):
        """Deletes by query rather than dropping the index, so the next upload does not race
        to recreate it."""
        client = FakeOpenSearch()
        store(client).delete_tenant(INDEX, "demo-firm")
        assert client.deletes[0]["body"]["query"] == {"term": {"tenant_id": "demo-firm"}}

    def test_deleting_from_a_missing_index_is_zero(self):
        class NoIndex(FakeOpenSearch):
            def delete_by_query(self, index: str, body: dict[str, Any], **kw: Any):
                raise RuntimeError("index_not_found_exception")

        assert store(NoIndex()).delete_document(INDEX, "doc-1") == 0


class TestCounting:
    def test_it_reports_how_many_vectors_an_index_holds(self):
        client = FakeOpenSearch(existing={INDEX})
        assert store(client).count(INDEX) == 42

    def test_a_missing_index_counts_zero(self):
        assert store(FakeOpenSearch()).count(INDEX) == 0


class TestItSatisfiesTheProtocol:
    def test_it_is_interchangeable_with_the_in_memory_store(self):
        """The reference implementation is the contract. A method the real store lacks would
        fail only in production, on whichever path first called it."""
        from src.documents.embed import InMemoryVectorStore

        reference = {
            name
            for name in dir(InMemoryVectorStore)
            if not name.startswith("_") and callable(getattr(InMemoryVectorStore, name))
        }
        real = set(dir(OpenSearchVectorStore))
        assert reference <= real, f"missing: {sorted(reference - real)}"


class TestWiring:
    def test_an_endpoint_selects_opensearch(self):
        """The bug this closes: deps built the in-memory store even with an endpoint set, so a
        deployed system lost its embeddings on every deploy while the collection billed."""
        from src.api.deps import _build_vector_store
        from src.config import LexGraphConfig, VectorConfig

        cfg = LexGraphConfig(vector=VectorConfig(endpoint="https://x.aoss.amazonaws.com"))
        assert isinstance(_build_vector_store(cfg), OpenSearchVectorStore)

    def test_no_endpoint_stays_in_memory(self):
        from src.api.deps import _build_vector_store
        from src.config import LexGraphConfig, VectorConfig
        from src.documents.embed import InMemoryVectorStore

        cfg = LexGraphConfig(vector=VectorConfig(endpoint=""))
        assert isinstance(_build_vector_store(cfg), InMemoryVectorStore)
