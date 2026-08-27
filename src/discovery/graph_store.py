"""Writing catalog nodes to the graph, and reading approved descriptions back.

The write half closes a gap that was invisible: `scan_catalog` and `enrich_tables` both return
`nodes`, and nothing wrote them. See `src/graph/catalog_queries.py` for what that cost.

The read half is the other end of enrichment. A model proposes a description, a human approves it,
and this is what puts the approved text in front of the query planner -- via
`src/discovery/catalog_overlay.py`, which merges it onto the scanned schema rather than into it.
Glue stays authoritative for the shape of a table; the graph is authoritative for what it means.

Since the nodes are written, the graph is also the durable copy of the scan itself, which is what
`source_rows`/`table_rows`/`column_rows` are for: the catalog cache is process-local, so a second
task or a redeploy would otherwise report "no scan has been run" over a graph holding every table.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from src.graph import catalog_queries as q
from src.graph.catalog_queries import UPSERT_BATCH_SIZE
from src.graph.scope import AuthContext, edge_scope, node_scope

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DescriptionText:
    """One description, with enough provenance for the overlay to rank it."""

    text: str
    epistemic_class: str
    confidence: float
    method: str

    @property
    def is_human(self) -> bool:
        """DECLARED means a person typed it, and a person outranks a model."""
        return self.epistemic_class == "DECLARED"


def _scalar_props(props: Mapping[str, Any]) -> dict[str, Any]:
    """Props with every value a scalar, or a refusal.

    Neptune rejects list- and map-valued properties where Neo4j accepts them, so a nested value
    would pass locally and fail only once deployed. Refused at the boundary rather than coerced,
    because silently flattening a list is how a column's type becomes the string "['a', 'b']".
    """
    bad = [k for k, v in props.items() if isinstance(v, list | dict | tuple | set)]
    if bad:
        raise ValueError(f"catalog node props must be scalars; {sorted(bad)} are not")
    return dict(props)


class CatalogGraphStore:
    """Catalog nodes in the graph. Injectable client, so tests need no database."""

    def __init__(self, graph: Any) -> None:
        self.graph = graph

    def persist(self, nodes: Sequence[Any]) -> int:
        """Write these nodes, grouped by label. Returns how many were written.

        Grouped because `write_batch` prepends one `UNWIND`, so one statement serves one label.
        Chunked because a wide table is hundreds of column nodes and an unbounded transaction is
        what `schema.DELETE_BATCH_SIZE` exists to avoid.
        """
        by_label: dict[str, list[dict[str, Any]]] = {}
        for node in nodes:
            labels = tuple(getattr(node, "labels", ()) or ())
            if not labels:
                # A node with no label would merge as a bare :Entity and be indistinguishable from
                # an assertion endpoint. Skipped loudly rather than written as a mystery node.
                logger.warning("catalog node %s has no label, not written", node.node_id)
                continue
            item = {
                "tenant_id": str(node.props.get("tenant_id", "")),
                "node_id": node.node_id,
                "props": _scalar_props(node.props),
            }
            if not item["tenant_id"]:
                raise ValueError(f"catalog node {node.node_id} carries no tenant_id")
            by_label.setdefault(labels[0], []).append(item)

        written = 0
        for label, batch in by_label.items():
            cypher = q.upsert_node(label)
            for start in range(0, len(batch), UPSERT_BATCH_SIZE):
                chunk = batch[start : start + UPSERT_BATCH_SIZE]
                self.graph.write_batch(cypher, chunk)
                written += len(chunk)
        if written:
            logger.info("wrote %d catalog nodes across %d labels", written, len(by_label))
        return written

    def approved_descriptions(self, ctx: AuthContext) -> dict[str, DescriptionText]:
        """Approved description per subject id, best one only.

        Two `DESCRIBED_AS` edges from one subject are legitimate: a model reading a schema and a
        person correcting it are distinct claims with distinct provenance, and `upsert_edge` keeps
        both on purpose. So the ranking is here, in one place, rather than something being
        superseded at write time.
        """
        rows = self.graph.read_scoped(q.APPROVED_DESCRIPTIONS, edge_scope(ctx))
        best: dict[str, DescriptionText] = {}
        for row in rows:
            subject = str(row.get("subject_id") or "")
            text = str(row.get("text") or "").strip()
            if not subject or not text:
                continue
            found = DescriptionText(
                text=text,
                epistemic_class=str(row.get("epistemic_class") or ""),
                confidence=float(row.get("confidence") or 0.0),
                method=str(row.get("method") or ""),
            )
            if _outranks(found, best.get(subject)):
                best[subject] = found
        return best

    def source_rows(self, ctx: AuthContext) -> list[dict[str, Any]]:
        """The tenant's sources as plain rows.

        Rows rather than `SourceRecord`, here and below: `catalog_store` imports the scanner and
        this module, so building its types here would close a cycle. `catalog_hydrate` owns that
        assembly.
        """
        return list(self.graph.read_scoped(q.SOURCES_FOR_TENANT, node_scope(ctx, node_var="s")))

    def table_rows(self, ctx: AuthContext) -> list[dict[str, Any]]:
        return list(self.graph.read_scoped(q.TABLES_FOR_TENANT, edge_scope(ctx)))

    def column_rows(self, ctx: AuthContext) -> list[dict[str, Any]]:
        return list(self.graph.read_scoped(q.COLUMNS_FOR_TENANT, edge_scope(ctx)))

    def approved_synonyms(self, ctx: AuthContext) -> dict[str, list[str]]:
        """Approved synonyms per subject id, in a stable order."""
        rows = self.graph.read_scoped(q.APPROVED_SYNONYMS, edge_scope(ctx))
        out: dict[str, set[str]] = {}
        for row in rows:
            subject = str(row.get("subject_id") or "")
            name = str(row.get("name") or "").strip()
            if subject and name:
                out.setdefault(subject, set()).add(name)
        return {k: sorted(v) for k, v in out.items()}

    def approved_topics(self, ctx: AuthContext) -> dict[str, list[str]]:
        """Approved topics per table `full_name`, in a stable order.

        Keyed by `full_name` rather than by subject id, unlike the synonyms: the traversal already
        reaches the `:Table` node, and that name is what a caller asking about one table holds.
        """
        rows = self.graph.read_scoped(q.APPROVED_TOPICS, edge_scope(ctx))
        out: dict[str, set[str]] = {}
        for row in rows:
            full_name = str(row.get("full_name") or "")
            name = str(row.get("name") or "").strip()
            if full_name and name:
                out.setdefault(full_name, set()).add(name)
        return {k: sorted(v) for k, v in out.items()}


def _outranks(candidate: DescriptionText, incumbent: DescriptionText | None) -> bool:
    """Whether `candidate` should be read instead of `incumbent`.

    A person's description beats a model's whatever the numbers say. Within a class, higher
    confidence then lower method alphabetically, so the winner does not depend on row order.
    """
    if incumbent is None:
        return True
    if candidate.is_human != incumbent.is_human:
        return candidate.is_human
    if candidate.confidence != incumbent.confidence:
        return candidate.confidence > incumbent.confidence
    return candidate.method < incumbent.method
