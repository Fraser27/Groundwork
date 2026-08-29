"""Tier 2: start at the graph, then query only what the graph landed on.

Tier 3 goes the other way. It retrieves passages by similarity, walks the graph out of them, and
grounds what it found. That is the right shape when the documents are the fuller source, and the
wrong shape when the graph is: similarity picks the documents, so the graph never gets to say
which documents were worth reading, and a question whose answer is a *figure* reaches the
warehouse only because its words happened to overlap a table name.

So this runs the same three stores in the opposite order:

    1. **Match the graph, lexically.** `GraphReader.search` for facts, `catalog_search` for the
       tables the question's words reach through DECLARED schema edges and approved synonyms.
    2. **Walk out.** From the entities that matched, along trusted edges only.
    3. **Land.** The facts name source documents; the catalog nodes name tables. Those two sets
       are the *only* things queried next.
    4. **Retrieve inside the landing zone.** kNN over those documents and no others.
    5. **Query the tables the graph named.** Model-written SQL, restricted to them.

What that buys, and the reason the direction is a tenant-level choice rather than a heuristic:
every passage in a graph-first answer is in a document some verified fact came out of. A reader
can ask "why was I shown this page" and the answer is an assertion id, not a cosine. The cost is
symmetric and real: a document nothing has been extracted from yet is unreachable here, however
well it matches. That is why `allowed_tiers` picks a direction instead of running both.

**Nothing here is allowed to widen.** An empty landing zone means the lane retrieves nothing, and
never that it retrieves everything -- see `VectorSearch.search`. A graph outage must not silently
degrade this into tier 3 while the answer still claims the graph vouched for its passages.

**No model runs before step 4.** Steps 1 to 3 are lexical matching and a trusted-edge walk, so
which documents were read is reproducible for the same question and the same graph. A model
writes SQL in step 5 and nothing else, which is what makes `generated` the one thing on the
result that stops the answer being governed.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from src.governance import GovernanceSettings
from src.graph.scope import AuthContext
from src.query.paths import chains

logger = logging.getLogger(__name__)

#: How many catalogued tables one question may point a generated query at. Small: the graph
#: pointed at these by name, so a long list means the term match was generic rather than that the
#: question was about twelve tables, and `sql_generation.MAX_TABLES` would then be filled with
#: schema the question only brushed.
MAX_GRAPH_TABLES = 6

#: How many documents the landing zone may hold. A well-connected party appears in hundreds, and
#: `terms` filters on `document_id` so the clause is sent to OpenSearch verbatim.
MAX_SEED_DOCUMENTS = 50


@dataclass
class GraphFirstResult:
    """What the graph found, and what querying its landing zone returned.

    `landed` is the load-bearing field. "The graph matched nothing" and "the graph matched facts
    but none of them came from a document" are different answers, and only the first is an absence
    -- so the caller reports where the traversal reached rather than leaving a reader to infer it
    from which lists are empty.
    """

    facts: list[dict[str, Any]] = field(default_factory=list)
    passages: list[dict[str, Any]] = field(default_factory=list)
    tables: list[Any] = field(default_factory=list)
    generated: Any | None = None
    landed: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    """Why a leg produced nothing, when the reason is a failure rather than an absence."""

    @property
    def empty(self) -> bool:
        return not (self.facts or self.passages or self.tables or self.generated)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "facts": self.facts,
            "passages": self.passages,
            "tables": [
                {
                    "full_name": str(getattr(t, "full_name", "")),
                    "description": getattr(t, "description", ""),
                    "columns": [getattr(c, "name", "") for c in getattr(t, "columns", ())],
                }
                for t in self.tables
            ],
            "landed": list(self.landed),
            # Derived here rather than held as a field: `facts` is appended to after construction
            # (the walk's edges join the term matches), so a chain set computed earlier would
            # describe a fact list that no longer exists.
            "paths": chains(self.facts),
        }
        if self.generated is not None:
            out["generated"] = {
                "sql": self.generated.generated.sql,
                "tables_offered": list(self.generated.generated.tables_offered),
                "rows": self.generated.rows,
                "error": self.generated.error,
                "error_code": self.generated.error_code,
            }
        return out


def _source_documents(facts: Sequence[dict[str, Any]]) -> frozenset[str]:
    """Documents the matched facts came out of.

    From `source.document_id`, which is the assertion's own provenance, never a name read out of
    a passage. Same rule as `graph_reader.passage_seeds` and for the same reason: a join on
    something a model produced would make which documents an answer read unreproducible.
    """
    found: set[str] = set()
    for fact in facts:
        source = fact.get("source")
        document_id = source.get("document_id") if isinstance(source, dict) else None
        if isinstance(document_id, str) and document_id:
            found.add(document_id)
    return frozenset(sorted(found)[:MAX_SEED_DOCUMENTS])


def _entity_seeds(facts: Sequence[dict[str, Any]]) -> list[str]:
    """The endpoints of the matched facts, as a frontier for the walk."""
    seeds: list[str] = []
    for fact in facts:
        for key in ("subject_id", "object_id"):
            value = fact.get(key)
            if isinstance(value, str) and value:
                seeds.append(value)
    return seeds


class GraphFirstLane:
    """Runs tier 2. Constructed from collaborators the caller already holds.

    Not injected as a service: `Resolver` and `Planner` are both handed a graph reader, a vector
    search, a catalog and a `SqlLane` already, and a lane assembled from those cannot disagree
    with the one the other endpoint assembled. Two endpoints disagreeing about what a question
    found is a bug class this repo keeps hitting.
    """

    def __init__(
        self,
        *,
        graph_reader: Any,
        vector_search: Any | None = None,
        catalog: Any | None = None,
        sql_lane: Any | None = None,
    ) -> None:
        self._graph = graph_reader
        self._vectors = vector_search
        self._catalog = catalog
        self._sql = sql_lane

    def run(
        self,
        ctx: AuthContext,
        question: str,
        settings: GovernanceSettings,
        *,
        sql_allowed: bool = True,
        synonyms: Mapping[str, Sequence[str]] | None = None,
    ) -> GraphFirstResult:
        """Traverse, then query what the traversal landed on.

        `sql_allowed` is the caller's kill-switch check rather than this lane's. `Resolver` and
        `Planner` both already record a refused ungoverned query for the Governance screen, and a
        second copy of that policy here is how two of them come to disagree about what was
        refused.
        """
        result = GraphFirstResult()
        floor = settings.min_confidence_floor

        facts = self._graph.search(ctx, question, min_confidence=floor)
        result.facts = list(facts)
        if facts:
            result.landed.append("facts")
            result.facts.extend(self._walked(ctx, facts, settings))

        # Every fact, walked ones included: a document reached two hops out is still a document
        # some verified fact came from, which is the whole claim this lane makes about its
        # passages.
        documents = _source_documents(result.facts)
        if documents:
            result.landed.append("documents")
            result.passages = self._retrieve(ctx, question, settings, documents, result)

        result.tables = self._tables(ctx, question, settings, result)
        if result.tables:
            result.landed.append("tables")
            if sql_allowed:
                result.generated = self._generate(question, result, synonyms)
            else:
                # Named rather than silent. The switch removes this leg, and the facts and
                # passages above it still stand.
                result.notes.append(
                    "The tables the graph pointed at were not queried, because AI-written queries "
                    "are turned off for this tenant."
                )
        return result

    def _walked(
        self, ctx: AuthContext, facts: Sequence[dict[str, Any]], settings: GovernanceSettings
    ) -> list[dict[str, Any]]:
        """Trusted edges around the entities that matched, deduplicated against the match.

        Seeded from the matched entities, not from retrieved passages -- that is the whole
        inversion. `expand` is the same method tier 3 calls, so a fact two hops from a matched
        party and a fact two hops from a retrieved page are ranked and capped identically.
        """
        expand = getattr(self._graph, "expand", None)
        seeds = _entity_seeds(facts)
        if expand is None or not seeds:
            return []
        edges = expand(
            ctx,
            seeds,
            depth=settings.graph_expand_depth,
            min_confidence=settings.min_confidence_floor,
            limit=settings.graph_expand_limit,
        )
        seen = {f.get("assertion_id") for f in facts}
        return [e for e in edges if e.get("assertion_id") not in seen]

    def _retrieve(
        self,
        ctx: AuthContext,
        question: str,
        settings: GovernanceSettings,
        documents: frozenset[str],
        result: GraphFirstResult,
    ) -> list[dict[str, Any]]:
        """The nearest passages *inside* the landing zone.

        Still a similarity search, and still `vector_top_k` of them -- what changed is the
        candidate set. Ranking inside a set the graph chose is a different claim from ranking
        across the corpus, and it is the claim this tier is entitled to make.
        """
        if self._vectors is None:
            return []
        passages = self._vectors.search(
            ctx, question, top_k=settings.vector_top_k, seed_documents=documents
        )
        if not passages:
            result.notes.append(
                f"{len(documents)} document{'' if len(documents) == 1 else 's'} were reached "
                "through the graph, but no passage in them matched the question closely enough to "
                "quote."
            )
        return passages

    def _tables(
        self,
        ctx: AuthContext,
        question: str,
        settings: GovernanceSettings,
        result: GraphFirstResult,
    ) -> list[Any]:
        """Catalogued tables the question's words reach through the graph.

        Two steps, and the second is what keeps this honest: the graph gives node ids, and those
        are resolved against `catalog.tables` rather than parsed into a table name. A node whose
        table is no longer catalogued is dropped, because offering a generator a table the
        firewall's allowlist will not contain produces a query that cannot run.
        """
        if self._catalog is None:
            return []
        search = getattr(self._graph, "catalog_search", None)
        if search is None:
            return []
        node_ids = search(ctx, question, min_confidence=settings.min_confidence_floor)
        if not node_ids:
            return []

        try:
            catalogued = self._catalog.tables(ctx.tenant_id)
        except Exception as e:  # noqa: BLE001
            logger.debug("graph-first table lane unavailable, no catalog: %s", e)
            result.notes.append(
                "The graph pointed at catalogued tables, but the catalog could not be read, so "
                "no query was written over them."
            )
            return []

        from src.discovery.glue_scanner import parse_catalog_node_id

        by_name = {str(getattr(t, "full_name", "")): t for t in catalogued}
        out: list[Any] = []
        for node_id in node_ids:
            ref = parse_catalog_node_id(node_id)
            # A column node counts for its table: "which columns exist" is not a question, and the
            # table holding the matched column is the thing a query runs against.
            if ref is None or not ref.table:
                continue
            table = by_name.get(f"{ref.database}.{ref.table}")
            if table is not None and table not in out:
                out.append(table)
            if len(out) >= MAX_GRAPH_TABLES:
                break
        return out

    def _generate(
        self,
        question: str,
        result: GraphFirstResult,
        synonyms: Mapping[str, Sequence[str]] | None,
    ) -> Any | None:
        """A query over the tables the graph named, and only those.

        `candidates=` rather than letting the lane re-derive them: the graph reached these through
        DECLARED edges and approved synonyms, and word overlap would drop exactly the table a
        synonym was approved to rescue.
        """
        if self._sql is None:
            return None
        generated = self._sql.run(
            question, tables=result.tables, synonyms=synonyms, candidates=result.tables
        )
        if generated is None:
            result.notes.append(
                "The graph pointed at "
                + ", ".join(str(getattr(t, "full_name", "")) for t in result.tables)
                + ", but no query could be written over them for this question."
            )
        return generated
