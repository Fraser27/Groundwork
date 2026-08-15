"""Embeddings in OpenSearch Serverless, so they survive a restart.

Satisfies the same `VectorStore` protocol as `InMemoryVectorStore`, which stays the reference
implementation. The collection was provisioned, its VPC endpoint existed and `VECTOR_ENDPOINT`
was passed to the container, but nothing was ever written against it: `deps.py` built the
in-memory store unconditionally, so a firm's embeddings lived in one Fargate task's memory and
vanished on every deploy while the collection sat idle and billed.

**Scoping is physical, not a filter.** One index per tenant, named by `index_name()` off
`cluster_key()`. An index-level mistake fails closed with "index not found"; a dropped filter
predicate returns another firm's documents. The matter wall *is* a filter — it changes per
request and cannot be baked into an index name — so it is applied inside the query rather than
after it, because filtering a truncated top-k in Python silently drops hits a screened user
should have seen.

**kNN with a pre-filter.** OpenSearch applies `filter` inside the k-NN search rather than
after, so top_k is computed over the documents this caller may read. Post-filtering would
return fewer than top_k results whenever a screen applied, and the shortfall would look like
"the document isn't relevant" rather than "you cannot see it".

Requests are signed with SigV4. There is no password: the task role is the credential, the
same arrangement as Neptune, so there is nothing to rotate or leak.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from src.documents.embed import SearchHit, VectorRecord

logger = logging.getLogger(__name__)

#: The vector field. `knn_vector` is the OpenSearch type; the name is ours.
VECTOR_FIELD = "embedding"

#: How many ids to delete per round. Deletes are a search plus a bulk rather than one call,
#: because Serverless has no `_delete_by_query`, so a reset is inherently paged.
DELETE_PAGE_SIZE = 500

#: HNSW with cosine similarity, matching `_cosine` in the in-memory store so a local result and
#: a deployed one rank the same way. `l2` would rank differently for identical embeddings, which
#: would make a local reproduction of a retrieval bug impossible.
_INDEX_SETTINGS: dict[str, Any] = {
    "settings": {"index": {"knn": True}},
    "mappings": {
        "properties": {
            VECTOR_FIELD: {
                "type": "knn_vector",
                "method": {"name": "hnsw", "space_type": "cosinesimil", "engine": "faiss"},
            },
            "tenant_id": {"type": "keyword"},
            "document_id": {"type": "keyword"},
            "matter_id": {"type": "keyword"},
            "model_id": {"type": "keyword"},
            "page": {"type": "integer"},
            "char_start": {"type": "integer"},
            "char_end": {"type": "integer"},
            # Verbatim document text. Indexed as `text` so keyword search works over the same
            # documents, which is what makes the hybrid lane possible without a second store.
            "text": {"type": "text"},
        }
    },
}


class OpenSearchLike(Protocol):
    """The slice of `opensearch-py` this module uses, so tests need no cluster."""

    def index(self, **kwargs: Any) -> dict[str, Any]: ...
    def search(self, **kwargs: Any) -> dict[str, Any]: ...
    def bulk(self, **kwargs: Any) -> dict[str, Any]: ...
    def count(self, **kwargs: Any) -> dict[str, Any]: ...

    @property
    def indices(self) -> Any: ...


class VectorStoreError(RuntimeError):
    """The vector store could not be reached or refused a request."""


def _client(endpoint: str, region: str) -> Any:
    """A SigV4-signed OpenSearch client.

    Imported lazily so the module is importable without `opensearch-py`, which keeps the
    dependency off any path that does not use vectors.
    """
    try:
        import boto3

        # `RequestsAWSV4SignerAuth`, not the bare `AWSV4SignerAuth`. In opensearch-py 3.x the
        # unqualified name is the *urllib3* signer, and pairing it with RequestsHttpConnection
        # silently sends the request unsigned -- which OpenSearch answers with 401, not the 403
        # that would point at a policy. The signer and the connection class have to match.
        from opensearchpy import OpenSearch, RequestsAWSV4SignerAuth, RequestsHttpConnection
    except ImportError as e:  # pragma: no cover - dependency presence, not logic
        raise VectorStoreError(
            "opensearch-py is required for the OpenSearch vector store; "
            "install it or leave VECTOR_ENDPOINT unset to run without vector search"
        ) from e

    host = endpoint.replace("https://", "").replace("http://", "").rstrip("/")
    credentials = boto3.Session().get_credentials()
    # `aoss`, not `es`: OpenSearch Serverless is a distinct signing service name, and using
    # `es` produces a 403 whose message does not mention the service name at all.
    auth = RequestsAWSV4SignerAuth(credentials, region, "aoss")
    return OpenSearch(
        hosts=[{"host": host, "port": 443}],
        http_auth=auth,
        use_ssl=True,
        verify_certs=True,
        connection_class=RequestsHttpConnection,
        pool_maxsize=20,
    )


@dataclass
class OpenSearchVectorStore:
    """Durable vector storage, one index per tenant."""

    endpoint: str
    region: str = "us-east-1"
    dimensions: int = 1024
    client: Any | None = None
    _ensured: set[str] = field(default_factory=set)

    def _os(self) -> Any:
        if self.client is None:
            self.client = _client(self.endpoint, self.region)
        return self.client

    def _ensure_index(self, index: str) -> None:
        """Create the index on first use, idempotently.

        Created lazily rather than provisioned up front because the index name contains the
        tenant, so the set is not known until a tenant exists. `_ensured` avoids an existence
        check per write; a concurrent creation loses the race harmlessly, since
        `resource_already_exists_exception` is the expected outcome.
        """
        if index in self._ensured:
            return
        client = self._os()
        try:
            if not client.indices.exists(index=index):
                body = {
                    "settings": _INDEX_SETTINGS["settings"],
                    "mappings": {
                        "properties": {
                            **_INDEX_SETTINGS["mappings"]["properties"],
                            VECTOR_FIELD: {
                                **_INDEX_SETTINGS["mappings"]["properties"][VECTOR_FIELD],
                                "dimension": self.dimensions,
                            },
                        }
                    },
                }
                client.indices.create(index=index, body=body)
                logger.info("created vector index %s (dim=%d)", index, self.dimensions)
        except Exception as e:
            if "resource_already_exists" not in str(e):
                raise VectorStoreError(f"could not create index {index}: {e}") from e
        self._ensured.add(index)

    def upsert(self, index: str, records: Sequence[VectorRecord]) -> int:
        """Write records, overwriting by `vector_id`.

        The id is the chunk id, so re-embedding a document converges instead of accumulating —
        the store is a derived index and a rebuild must be idempotent.
        """
        if not records:
            return 0
        self._ensure_index(index)
        client = self._os()

        actions: list[dict[str, Any]] = []
        for record in records:
            actions.append({"index": {"_index": index, "_id": record.vector_id}})
            actions.append(
                {
                    VECTOR_FIELD: list(record.embedding),
                    "tenant_id": record.tenant_id,
                    "document_id": record.document_id,
                    "matter_id": record.matter_id,
                    "model_id": record.model_id,
                    "page": record.page,
                    "char_start": record.char_start,
                    "char_end": record.char_end,
                    "text": record.text,
                }
            )

        try:
            response = client.bulk(body=actions, refresh=True)
        except Exception as e:
            raise VectorStoreError(f"could not write {len(records)} vectors: {e}") from e

        if response.get("errors"):
            # Named rather than counted: a partial write leaves the index disagreeing with the
            # graph about what is searchable, and the reason is what makes that diagnosable.
            failed = [
                item["index"].get("error")
                for item in response.get("items", [])
                if item.get("index", {}).get("error")
            ]
            raise VectorStoreError(f"{len(failed)} vectors rejected: {failed[:3]}")
        return len(records)

    def search(
        self,
        index: str,
        query: Sequence[float],
        *,
        top_k: int = 10,
        matter_allowlist: frozenset[str] | None = None,
        matter_denylist: frozenset[str] = frozenset(),
    ) -> list[SearchHit]:
        """kNN search with the matter wall applied inside the query.

        The filter is a pre-filter: OpenSearch applies it during the search, so top_k is
        computed over what this caller may read. Filtering afterwards would return fewer than
        top_k whenever a screen applied, and the shortfall reads as irrelevance rather than
        refusal.

        A missing index is an empty result, not an error: a tenant who has uploaded nothing has
        no index, and that is a normal state rather than a failure.
        """
        # Mirrors `edge_scope`: a record with no matter is tenant-wide, so it is readable by
        # anyone in the tenant. Allowlist and denylist are separate clauses because a denial
        # must beat a permission -- a screened matter is refused even when it is also assigned.
        must: list[dict[str, Any]] = []
        must_not: list[dict[str, Any]] = []
        if matter_allowlist is not None:
            must.append(
                {
                    "bool": {
                        "should": [
                            {"bool": {"must_not": {"exists": {"field": "matter_id"}}}},
                            {"terms": {"matter_id": sorted(matter_allowlist)}},
                        ],
                        "minimum_should_match": 1,
                    }
                }
            )
        if matter_denylist:
            must_not.append({"terms": {"matter_id": sorted(matter_denylist)}})

        knn: dict[str, Any] = {"vector": list(query), "k": top_k}
        if must or must_not:
            clause: dict[str, Any] = {}
            if must:
                clause["must"] = must
            if must_not:
                clause["must_not"] = must_not
            knn["filter"] = {"bool": clause}

        body = {"size": top_k, "query": {"knn": {VECTOR_FIELD: knn}}}

        try:
            response = self._os().search(index=index, body=body)
        except Exception as e:
            if "index_not_found" in str(e):
                return []
            raise VectorStoreError(f"vector search failed on {index}: {e}") from e

        return [self._to_hit(hit) for hit in response.get("hits", {}).get("hits", [])]

    def _to_hit(self, hit: dict[str, Any]) -> SearchHit:
        source = hit.get("_source", {})
        record = VectorRecord(
            vector_id=str(hit.get("_id", "")),
            tenant_id=str(source.get("tenant_id", "")),
            document_id=str(source.get("document_id", "")),
            page=int(source.get("page") or 0),
            char_start=int(source.get("char_start") or 0),
            char_end=int(source.get("char_end") or 0),
            text=str(source.get("text", "")),
            # Not returned: the vector is megabytes across a page of hits and no caller reads
            # it. The citation is the page and the text.
            embedding=(),
            model_id=str(source.get("model_id", "")),
            matter_id=source.get("matter_id"),
        )
        return SearchHit(record=record, score=float(hit.get("_score") or 0.0))

    def count(self, index: str) -> int:
        """How many vectors an index holds. Part of the reference store's surface, so it is
        implemented here too: a method the real store lacks fails only in production, on
        whichever path first happens to call it."""
        try:
            response = self._os().count(index=index)
        except Exception as e:
            if "index_not_found" in str(e):
                return 0
            raise VectorStoreError(f"could not count {index}: {e}") from e
        return int(response.get("count") or 0)

    def delete_document(self, index: str, document_id: str) -> int:
        return self._delete_by(index, {"term": {"document_id": document_id}})

    def delete_tenant(self, index: str, tenant_id: str) -> int:
        """Drop a tenant's vectors. Used by the reset surface.

        Deletes the documents rather than the index, so the mapping survives and the next upload
        does not race to recreate it.
        """
        return self._delete_by(index, {"term": {"tenant_id": tenant_id}})

    def _delete_by(self, index: str, query: dict[str, Any]) -> int:
        """Search for matching ids, then bulk-delete them.

        Not `_delete_by_query`, which OpenSearch *Serverless* does not support -- it is absent
        from the supported-operations table and 404s. Only `DELETE <index>/_doc/<id>` and
        `_bulk` are available, both under `aoss:WriteDocument`, so a delete-by-predicate has to
        be assembled from a search and a bulk.

        Paged, because a reset on a large tenant deletes more than one page of hits and a single
        search would silently leave the remainder behind -- vectors for documents the graph no
        longer knows about, which is worse than a slow reset.
        """
        client = self._os()
        deleted = 0
        while True:
            try:
                found = client.search(
                    index=index,
                    body={"query": query, "size": DELETE_PAGE_SIZE, "_source": False},
                )
            except Exception as e:
                if "index_not_found" in str(e):
                    return deleted
                raise VectorStoreError(f"could not find documents to delete in {index}: {e}") from e

            ids = [hit["_id"] for hit in found.get("hits", {}).get("hits", [])]
            if not ids:
                return deleted

            actions = [{"delete": {"_index": index, "_id": doc_id}} for doc_id in ids]
            try:
                client.bulk(body=actions, refresh=True)
            except Exception as e:
                raise VectorStoreError(
                    f"could not delete {len(ids)} vectors from {index}: {e}"
                ) from e
            deleted += len(ids)

            if len(ids) < DELETE_PAGE_SIZE:
                return deleted
