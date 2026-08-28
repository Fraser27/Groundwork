"""Embed chunks via Bedrock and write them to a vector store.

Verbatim text is searchable as soon as it is embedded — it does **not** wait for
review. The gate protects the graph, not the search index: showing a lawyer a
paragraph that exists in a document they uploaded is not a claim about the world,
whereas asserting that one holding undercuts another is.

Every vector carries the chunk's page and char offsets in its metadata, so a retrieval
hit resolves to a highlight in the original PDF without a second lookup. Tenant id is
in the metadata *and* in the index name, because a vector store filter is the same
class of single-point-of-failure that `scope.py` exists to prevent for the graph.

The store is an interface with an in-memory implementation. Production is OpenSearch;
tests need neither AWS nor a cluster.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from src.documents.models import Chunk
from src.graph.scope import AuthContext

logger = logging.getLogger(__name__)

TITAN_V2 = "amazon.titan-embed-text-v2:0"
COHERE_V3 = "cohere.embed-english-v3"

#: Bedrock's embedding endpoints take one document per call; batching is client-side.
#: Kept modest so a failure retries a small slice rather than the whole document.
DEFAULT_BATCH_SIZE = 16


class BedrockLike(Protocol):
    def invoke_model(self, **kwargs: Any) -> dict[str, Any]: ...


class EmbeddingFailed(RuntimeError):
    pass


@dataclass(frozen=True)
class VectorRecord:
    """A chunk's embedding plus everything needed to cite it.

    `vector_id` is the chunk id, so re-embedding overwrites rather than duplicating —
    the store is a derived index and a rebuild must converge, not accumulate.
    """

    vector_id: str
    tenant_id: str
    document_id: str
    page: int
    char_start: int
    char_end: int
    text: str
    embedding: tuple[float, ...]
    model_id: str
    matter_id: str | None = None

    def to_metadata(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "matter_id": self.matter_id,
            "document_id": self.document_id,
            "page": self.page,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "model_id": self.model_id,
        }


@dataclass(frozen=True)
class SearchHit:
    record: VectorRecord
    score: float


class VectorStore(Protocol):
    def upsert(self, index: str, records: Sequence[VectorRecord]) -> int: ...
    def search(
        self,
        index: str,
        query: Sequence[float],
        *,
        top_k: int,
        matter_allowlist: frozenset[str] | None,
        matter_denylist: frozenset[str],
        seed_documents: frozenset[str] | None = None,
    ) -> list[SearchHit]: ...
    def delete_document(self, index: str, document_id: str) -> int: ...
    def delete_tenant(self, index: str, tenant_id: str) -> int: ...
    def relabel_matter(self, index: str, document_id: str, matter_id: str | None) -> int: ...


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return 0.0 if na == 0 or nb == 0 else dot / (na * nb)


@dataclass
class InMemoryVectorStore:
    """Reference implementation. Enforces the same scoping as the real store."""

    _indexes: dict[str, dict[str, VectorRecord]] = field(default_factory=dict)

    def upsert(self, index: str, records: Sequence[VectorRecord]) -> int:
        bucket = self._indexes.setdefault(index, {})
        for record in records:
            bucket[record.vector_id] = record
        return len(records)

    def search(
        self,
        index: str,
        query: Sequence[float],
        *,
        top_k: int = 10,
        matter_allowlist: frozenset[str] | None = None,
        matter_denylist: frozenset[str] = frozenset(),
        seed_documents: frozenset[str] | None = None,
    ) -> list[SearchHit]:
        hits = []
        for record in self._indexes.get(index, {}).values():
            if seed_documents is not None and record.document_id not in seed_documents:
                continue
            if record.matter_id is not None:
                if record.matter_id in matter_denylist:
                    continue
                if matter_allowlist is not None and record.matter_id not in matter_allowlist:
                    continue
            hits.append(SearchHit(record=record, score=_cosine(query, record.embedding)))
        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:top_k]

    def delete_document(self, index: str, document_id: str) -> int:
        bucket = self._indexes.get(index, {})
        doomed = [vid for vid, r in bucket.items() if r.document_id == document_id]
        for vid in doomed:
            del bucket[vid]
        return len(doomed)

    def delete_tenant(self, index: str, tenant_id: str) -> int:
        bucket = self._indexes.get(index, {})
        doomed = [vid for vid, r in bucket.items() if r.tenant_id == tenant_id]
        for vid in doomed:
            del bucket[vid]
        return len(doomed)

    def relabel_matter(self, index: str, document_id: str, matter_id: str | None) -> int:
        bucket = self._indexes.get(index, {})
        moved = 0
        for vid, record in list(bucket.items()):
            if record.document_id == document_id and record.matter_id != matter_id:
                bucket[vid] = replace(record, matter_id=matter_id)
                moved += 1
        return moved

    def count(self, index: str) -> int:
        return len(self._indexes.get(index, {}))


def index_name(ctx: AuthContext) -> str:
    """One index per tenant, named off `cluster_key()`.

    Physical separation rather than a filter, for the same reason S3 keys are
    tenant-prefixed: an index-level mistake fails closed with "index not found",
    whereas a dropped filter predicate returns another firm's documents.
    """
    return f"{ctx.cluster_key()}-chunks"


class Embedder:
    def __init__(
        self,
        store: VectorStore,
        *,
        bedrock: BedrockLike | None = None,
        bedrock_factory: Callable[[], BedrockLike] | None = None,
        model_id: str = TITAN_V2,
        dimensions: int = 1024,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        self.store = store
        self.model_id = model_id
        self.dimensions = dimensions
        self.batch_size = batch_size
        self._bedrock = bedrock
        self._bedrock_factory = bedrock_factory

    @property
    def bedrock(self) -> BedrockLike:
        if self._bedrock is None:
            factory = self._bedrock_factory
            if factory is None:
                import boto3

                factory = lambda: boto3.client("bedrock-runtime")  # noqa: E731
            self._bedrock = factory()
        return self._bedrock

    def _request_body(self, text: str) -> str:
        if self.model_id.startswith("cohere."):
            return json.dumps({"texts": [text], "input_type": "search_document"})
        return json.dumps({"inputText": text, "dimensions": self.dimensions, "normalize": True})

    def _read_embedding(self, payload: dict[str, Any]) -> list[float]:
        if "embedding" in payload:
            return list(payload["embedding"])
        if payload.get("embeddings"):
            return list(payload["embeddings"][0])
        raise EmbeddingFailed(f"no embedding in response keys {sorted(payload)}")

    def embed_text(self, text: str) -> list[float]:
        try:
            response = self.bedrock.invoke_model(
                modelId=self.model_id, body=self._request_body(text)
            )
            payload = json.loads(response["body"].read())
        except EmbeddingFailed:
            raise
        except Exception as e:
            raise EmbeddingFailed(f"bedrock embed failed: {e}") from e
        return self._read_embedding(payload)

    def embed_query(self, text: str) -> list[float]:
        """Same model as ingestion, a query embedded by a different model is noise."""
        return self.embed_text(text)

    def embed_chunks(self, ctx: AuthContext, chunks: Sequence[Chunk]) -> list[VectorRecord]:
        records: list[VectorRecord] = []
        for chunk in chunks:
            if chunk.tenant_id != ctx.tenant_id:
                raise EmbeddingFailed(
                    f"chunk {chunk.chunk_id} belongs to {chunk.tenant_id}, not {ctx.tenant_id}"
                )
            records.append(
                VectorRecord(
                    vector_id=chunk.chunk_id,
                    tenant_id=chunk.tenant_id,
                    matter_id=chunk.matter_id,
                    document_id=chunk.document_id,
                    page=chunk.page,
                    char_start=chunk.char_start,
                    char_end=chunk.char_end,
                    text=chunk.text,
                    embedding=tuple(self.embed_text(chunk.text)),
                    model_id=self.model_id,
                )
            )
        return records

    def embed_and_store(self, ctx: AuthContext, chunks: Sequence[Chunk]) -> int:
        index = index_name(ctx)
        written = 0
        for start in range(0, len(chunks), self.batch_size):
            batch = chunks[start : start + self.batch_size]
            written += self.store.upsert(index, self.embed_chunks(ctx, batch))
        logger.info("embedded %d chunks into %s", written, index)
        return written

    def drop_tenant(self, tenant_id: str) -> int:
        """Empty a tenant's vector index, returning how many chunks went.

        Takes a plain tenant id rather than an `AuthContext` because a reset runs from an
        admin route, not from the requesting user's scope. The index name is derived the
        same way `index_name` derives it, so this cannot reach another tenant's index.
        """
        return self.store.delete_tenant(f"tenant-{tenant_id}-chunks", tenant_id)

    def forget_document(self, ctx: AuthContext, document_id: str) -> int:
        """Drop one document's chunks, returning how many went.

        Deleted outright rather than superseded, unlike an assertion. A vector is not a claim
        about anything -- it is a derived index entry with no provenance to preserve -- so there
        is no such thing as an audit trail of an embedding. Replaying the document rebuilds it.

        Takes a `ctx` rather than a bare tenant id because this runs on a request path, and the
        index name has to come from the caller's own scope.
        """
        dropped = self.store.delete_document(index_name(ctx), document_id)
        logger.info("dropped %d vectors for %s", dropped, document_id)
        return dropped

    def search(
        self,
        ctx: AuthContext,
        query: str,
        *,
        top_k: int = 10,
        seed_documents: frozenset[str] | None = None,
    ) -> list[SearchHit]:
        """Nearest chunks this caller may read, optionally restricted to named documents.

        `seed_documents` narrows *relevance*, never authorization: the matter wall below is applied
        whatever it says, so a seed can only ever shrink the candidate set the wall already allows.
        The graph-first tier uses it to read only the documents its verified facts came from. None
        means no restriction; an empty set is a caller error and is refused rather than treated as
        one, because silently searching everything would turn a graph-first question into a
        vector-first one and label the passages as though the graph had vouched for them.
        """
        if seed_documents is not None and not seed_documents:
            raise ValueError("seed_documents is empty; pass None to search every document")
        return self.store.search(
            index_name(ctx),
            self.embed_query(query),
            top_k=top_k,
            matter_allowlist=ctx.matter_allowlist,
            matter_denylist=ctx.matter_denylist,
            seed_documents=seed_documents,
        )
