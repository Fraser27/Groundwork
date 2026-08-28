"""Adapter between the embedder and the resolver.

`Embedder.search` returns `SearchHit` objects; the resolver wants plain dicts with a
`document_id`, so it can hand them to `GraphReader.expand` and turn them into
citations. This is that translation, kept as its own object rather than pushed into
either side — the embedder should not know about resolution tiers, and the resolver
should not know about vector records.

It also absorbs failure. An embedding call needs Bedrock, and Bedrock is a network
hop that can be missing credentials or simply down. When that happens the retrieval lane
returns nothing and the tier answers from what else it has, rather than 500: a degraded
answer beats an error page.
"""

from __future__ import annotations

import logging
from typing import Any

from src.documents.embed import Embedder
from src.graph.scope import AuthContext

logger = logging.getLogger(__name__)


class VectorSearch:
    def __init__(self, embedder: Embedder) -> None:
        self._embedder = embedder

    def search(
        self,
        ctx: AuthContext,
        question: str,
        *,
        top_k: int = 20,
        seed_documents: frozenset[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Nearest passages, optionally only within documents the graph named.

        An empty `seed_documents` returns nothing rather than everything. The graph-first tier
        calls this only when its facts named documents, so an empty set means the graph landed on
        no document at all -- and answering that with an unfiltered search would silently
        substitute a vector-first result for a graph-first one.
        """
        if seed_documents is not None and not seed_documents:
            return []
        try:
            hits = self._embedder.search(ctx, question, top_k=top_k, seed_documents=seed_documents)
        except Exception as e:
            # Includes a missing Bedrock credential, which is the common local case.
            logger.info("vector search unavailable, skipping tier 3: %s", e)
            return []

        return [
            {
                "document_id": h.record.document_id,
                "chunk_id": h.record.vector_id,
                "page": h.record.page,
                "char_start": h.record.char_start,
                "char_end": h.record.char_end,
                "text": h.record.text,
                "matter_id": h.record.matter_id,
                "score": h.score,
            }
            for h in hits
        ]
