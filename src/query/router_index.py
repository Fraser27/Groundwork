"""The routable descriptions a question is matched against to choose which tiers to run.

A sibling of `OpenSearchVectorStore`, not a subclass of it. That class's surface is chunk-shaped
-- `delete_document`, `relabel_matter`, a record carrying a page and a char span -- and it is the
matter wall for document retrieval. Generalising it into a field-mapping abstraction would put
indirection underneath the one thing standing between two firms' documents, which is the wrong
place for it. The Serverless constraints are *imported* from it instead, so they stay expressed
once: no `engine` in the knn mapping, no `_delete_by_query`, and a requests-flavoured SigV4
signer paired with `RequestsHttpConnection`.

**One index, not three.** Metrics, graph entities and catalog tables share it and are told apart
by a `kind` keyword. The router compares layers against each other, and scores are only
comparable because one query vector hit one index with one mapping -- three indexes would make
"metrics scored higher than tables" a statement about two unrelated number ranges.

**Scores come back raw.** `cosinesimil` reports `1 / (2 - cos)`, and converting it is
`router_scoring`'s job. Converting here too would mean two places that both believe they own the
conversion, and the second one to be edited would be silently wrong.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from src.documents.opensearch_store import (
    DELETE_PAGE_SIZE,
    REQUEST_TIMEOUT_SECONDS,
    VECTOR_FIELD,
    OpenSearchLike,
    VectorStoreError,
    _client,
)
from src.graph.scope import AuthContext

logger = logging.getLogger(__name__)

__all__ = [
    "DELETE_PAGE_SIZE",
    "KIND_ENTITY",
    "KIND_METRIC",
    "KIND_TABLE",
    "REQUEST_TIMEOUT_SECONDS",
    "ROUTING_KINDS",
    "OpenSearchLike",
    "RoutingHit",
    "RoutingIndex",
    "RoutingRecord",
    "VectorStoreError",
    "routing_index_for_tenant",
    "routing_index_name",
]

KIND_METRIC = "metric"
KIND_ENTITY = "entity"
KIND_TABLE = "table"

#: The closed set of routable kinds, for the same reason entity kinds are closed: a kind nothing
#: recognises is a record that can never be filtered in or out, so it would leak into every trace.
ROUTING_KINDS = frozenset({KIND_METRIC, KIND_ENTITY, KIND_TABLE})

#: HNSW with cosine, matching the chunk index so a routing score and a retrieval score mean the
#: same thing. No `engine`: a NextGen collection rejects the parameter with a 400.
_INDEX_SETTINGS: dict[str, Any] = {
    "settings": {"index": {"knn": True}},
    "mappings": {
        "properties": {
            VECTOR_FIELD: {
                "type": "knn_vector",
                "method": {"name": "hnsw", "space_type": "cosinesimil"},
            },
            "kind": {"type": "keyword"},
            "item_id": {"type": "keyword"},
            "label": {"type": "keyword"},
            # The embedded prose. `text` rather than `keyword` so a future keyword lane can run
            # over the same descriptions without a second index.
            "text": {"type": "text"},
            "tenant_id": {"type": "keyword"},
            "matter_id": {"type": "keyword"},
            "model_id": {"type": "keyword"},
            # Trace payload only -- expression, columns, ontology layer. Nothing searches it, and
            # leaving it enabled would map a metric's SQL expression as searchable text and let a
            # column name in one tenant's warehouse influence scoring.
            "detail": {"type": "object", "enabled": False},
        }
    },
}


def routing_index_name(ctx: AuthContext) -> str:
    """One routing index per tenant, named off `cluster_key()` like the chunk index."""
    return f"{ctx.cluster_key()}-routing"


def routing_index_for_tenant(tenant_id: str) -> str:
    """The same name from a bare tenant id, for an admin path with no request scope.
    `cluster_key()` is `tenant-<id>`, so the two agree by construction."""
    return f"tenant-{tenant_id}-routing"


@dataclass(frozen=True)
class RoutingRecord:
    """One routable thing: a metric, a graph entity, or a catalog table.

    `vector_id` is derived from the kind and the item, so a rebuild converges. This is a derived
    index and a reindex that accumulated would inflate every layer's hit count over time.
    """

    vector_id: str
    tenant_id: str
    kind: str
    item_id: str
    label: str
    text: str
    embedding: tuple[float, ...]
    model_id: str
    matter_id: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RoutingHit:
    """A hit and its **unconverted** OpenSearch score.

    Named `raw_score` rather than `score` so a caller cannot mistake it for a cosine. With
    `cosinesimil` the engine reports `1 / (2 - cos)`; `router_scoring.cosine_of` converts it.
    """

    record: RoutingRecord
    raw_score: float


@dataclass
class RoutingIndex:
    """Durable routing descriptions, one index per tenant."""

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
        """Create on first use, idempotently. The name carries the tenant, so the set of indexes
        is not known until a tenant exists."""
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
                logger.info("created routing index %s (dim=%d)", index, self.dimensions)
        except Exception as e:
            if "resource_already_exists" not in str(e):
                raise VectorStoreError(f"could not create index {index}: {e}") from e
        self._ensured.add(index)

    def upsert(self, index: str, records: Sequence[RoutingRecord]) -> int:
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
                    "kind": record.kind,
                    "item_id": record.item_id,
                    "label": record.label,
                    "text": record.text,
                    "tenant_id": record.tenant_id,
                    "matter_id": record.matter_id,
                    "model_id": record.model_id,
                    "detail": dict(record.detail),
                }
            )

        try:
            # No `refresh`: Serverless decides refresh timing and rejects the parameter. A
            # rebuilt description becoming routable a second later is invisible to an operator
            # who is still reading the rebuild report.
            response = client.bulk(body=actions)
        except Exception as e:
            raise VectorStoreError(f"could not write {len(records)} routing records: {e}") from e

        if response.get("errors"):
            failed = [
                item["index"].get("error")
                for item in response.get("items", [])
                if item.get("index", {}).get("error")
            ]
            raise VectorStoreError(f"{len(failed)} routing records rejected: {failed[:3]}")
        return len(records)

    def search(
        self,
        index: str,
        query_vector: Sequence[float],
        *,
        top_k: int = 10,
        kinds: frozenset[str] | Sequence[str] | None = None,
        matter_allowlist: frozenset[str] | None = None,
        matter_denylist: frozenset[str] = frozenset(),
    ) -> list[RoutingHit]:
        """kNN over every kind at once, with the kind filter and the matter wall as pre-filters.

        `kinds` is how a forbidden tier stays out of the trace. An administrator who disabled a
        tier has forbidden the system to use its data, and a routing hit is a metric name or a
        subject name -- so a dropped-but-shown layer would disclose exactly what was forbidden.

        The matter clauses mirror `OpenSearchVectorStore.search`, which mirrors `edge_scope`: a
        record with no `matter_id` is tenant-wide, and the denylist is `must_not` so a screen
        beats an assignment. An entity's routing label is a subject name, so a missing denial
        here names a screened party before the ethical wall has run.

        A missing index is an empty result: a tenant whose router has never been built has no
        index, and the router degrades to trying every tier rather than erroring.
        """
        must: list[dict[str, Any]] = []
        must_not: list[dict[str, Any]] = []

        if kinds is not None:
            must.append({"terms": {"kind": sorted(kinds)}})

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

        knn: dict[str, Any] = {"vector": list(query_vector), "k": top_k}
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
            raise VectorStoreError(f"routing search failed on {index}: {e}") from e

        return [_to_hit(hit) for hit in response.get("hits", {}).get("hits", [])]

    def delete_kind(self, index: str, kind: str) -> int:
        """Drop one layer. A reindex deletes then writes, so a removed metric stops routing."""
        return self._delete_by(index, {"term": {"kind": kind}})

    def delete_tenant(self, index: str, tenant_id: str) -> int:
        """Drop a tenant's routing records. Deletes documents rather than the index, so the
        mapping survives and the next rebuild does not race to recreate it."""
        return self._delete_by(index, {"term": {"tenant_id": tenant_id}})

    def _delete_by(self, index: str, query: dict[str, Any]) -> int:
        """Search for ids, then bulk-delete them, paged.

        Not `_delete_by_query`: Serverless does not support it. Paged because a reindex of a
        large tenant's entities spans more than one page, and stopping after the first would
        leave descriptions behind for entities the graph no longer holds -- which routes a
        question toward a tier that then finds nothing.
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
                raise VectorStoreError(f"could not find routing records in {index}: {e}") from e

            ids = [hit["_id"] for hit in found.get("hits", {}).get("hits", [])]
            if not ids:
                return deleted

            actions = [{"delete": {"_index": index, "_id": doc_id}} for doc_id in ids]
            try:
                client.bulk(body=actions)
            except Exception as e:
                raise VectorStoreError(
                    f"could not delete {len(ids)} routing records from {index}: {e}"
                ) from e
            deleted += len(ids)

            if len(ids) < DELETE_PAGE_SIZE:
                return deleted


def _to_hit(hit: dict[str, Any]) -> RoutingHit:
    source = hit.get("_source", {})
    record = RoutingRecord(
        vector_id=str(hit.get("_id", "")),
        tenant_id=str(source.get("tenant_id", "")),
        kind=str(source.get("kind", "")),
        item_id=str(source.get("item_id", "")),
        label=str(source.get("label", "")),
        text=str(source.get("text", "")),
        # Not returned: a page of 1024-float vectors is megabytes and the router reads the label.
        embedding=(),
        model_id=str(source.get("model_id", "")),
        matter_id=source.get("matter_id"),
        detail=dict(source.get("detail") or {}),
    )
    return RoutingHit(record=record, raw_score=float(hit.get("_score") or 0.0))
