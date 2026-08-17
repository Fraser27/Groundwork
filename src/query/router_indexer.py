"""Populating the routing index: what each layer offers, in words a question can match.

The problem this exists to fix. Tier 1 matched a question against metric names by keyword, so
"what did we invoice last quarter" missed `fees_billed` unless somebody had hand-maintained
"invoiced" as a synonym -- and a governance miss is not a neutral outcome, it silently downgrades
the question to the tier where a model writes SQL. Embedding a description of each routable thing
moves that from vocabulary maintenance to similarity.

What gets embedded is chosen per kind, and each choice is a judgement about what a question
actually says:

- **metric** -- name, synonyms, definition. Not the expression: `SUM(billed_value)` is how the
  answer is computed, not what it means, and nobody phrases a question in SQL.
- **table** -- name, database, description, column names. Column names carry most of the signal
  for a warehouse table whose description is empty, which is most of them.
- **entity** -- label, its declared entity-kind label, and the predicate labels it participates
  in. The predicates are what distinguish two entities with similar names: a party that
  `REPRESENTS` reads differently from one that is `ADVERSE_TO`.

Only **approved** metrics are indexed. An unapproved draft must not route a question, for the
same reason it must not answer one -- routing toward tier 1 on the strength of a draft would end
in "no metric matched" after the decision had already been made.

Entity kinds come from `Ontology.entity_kind_of`, never from splitting an id. Guessing the kind
from a prefix is how `court:` came to exist alongside a declared `Court`, and an entity whose kind
the pack does not declare is left out rather than filed under a guess: an undeclared kind is drift
to surface, and a routing record would hide it behind a plausible label.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from src.graph.scope import AuthContext, edge_scope
from src.ontology.loader import Ontology
from src.query.router_index import (
    KIND_ENTITY,
    KIND_METRIC,
    KIND_TABLE,
    RoutingIndex,
    RoutingRecord,
    routing_index_name,
)

logger = logging.getLogger(__name__)

#: How many distinct entities to describe per tenant. Entities are the one unbounded layer -- a
#: firm with a million parties would otherwise pay a Bedrock call each -- and the router only
#: needs enough of them to tell "this question is about a subject in the graph" from "this
#: question is about a number in the warehouse".
DEFAULT_ENTITY_LIMIT = 2000

#: Predicate labels per entity. Bounded because a hub entity like a matter participates in every
#: predicate the pack declares, and a description listing all of them is a description of the
#: ontology rather than of that entity -- it would match every question equally well.
_MAX_PREDICATES_PER_ENTITY = 12


@dataclass
class RoutingRebuildReport:
    """What a rebuild wrote, per layer, plus what it could not.

    Errors are per layer rather than fatal: a graph that is down must not stop the metric and
    table layers being routable, because the alternative is a router with no index at all, which
    degrades to running every tier on every question.
    """

    metrics_indexed: int = 0
    tables_indexed: int = 0
    entities_indexed: int = 0
    entities_skipped: int = 0
    """Entities whose kind the ontology pack does not declare. Reported rather than swallowed:
    a rising number here is vocabulary drift, which is exactly what the closed kind list exists
    to make visible."""

    errors: list[str] = field(default_factory=list)

    @property
    def total_indexed(self) -> int:
        return self.metrics_indexed + self.tables_indexed + self.entities_indexed

    def to_dict(self) -> dict[str, Any]:
        return {
            "metrics_indexed": self.metrics_indexed,
            "tables_indexed": self.tables_indexed,
            "entities_indexed": self.entities_indexed,
            "entities_skipped": self.entities_skipped,
            "total_indexed": self.total_indexed,
            "errors": self.errors,
            "note": (
                "Routing descriptions only -- no metric definition, table or fact was changed. "
                "Approved metrics only, so a draft cannot route a question."
            ),
        }


def _vector_id(kind: str, tenant_id: str, item_id: str) -> str:
    """Deterministic, so a reindex overwrites. An accumulating routing index would inflate a
    layer's hit count on every rebuild and skew the router toward whichever layer was rebuilt
    most often."""
    return f"{tenant_id}:{kind}:{item_id}"


def metric_text(metric: Any) -> str:
    parts = [str(getattr(metric, "name", "") or "")]
    parts.extend(str(s) for s in (getattr(metric, "synonyms", ()) or ()))
    definition = str(getattr(metric, "definition", "") or "")
    if definition:
        parts.append(definition)
    return ". ".join(p for p in parts if p)


def table_text(table: Any) -> str:
    name = str(getattr(table, "name", "") or "")
    database = str(getattr(table, "database", "") or "")
    parts = [f"{name} in {database}" if database else name]
    description = str(getattr(table, "description", "") or "")
    if description:
        parts.append(description)
    columns = [str(c.name) for c in (getattr(table, "columns", ()) or ()) if getattr(c, "name", "")]
    if columns:
        parts.append("Columns: " + ", ".join(columns))
    return ". ".join(p for p in parts if p)


def entity_text(label: str, kind_label: str, predicate_labels: list[str]) -> str:
    parts = [label]
    if kind_label:
        parts.append(kind_label)
    if predicate_labels:
        parts.append(", ".join(predicate_labels))
    return ". ".join(p for p in parts if p)


def _entity_label(entity_id: str) -> str:
    """The readable half of a `kind:slug` id. Never used to derive the *kind* -- that comes
    from the ontology, because a prefix is a claim and the vocabulary is the authority."""
    _, _, rest = entity_id.partition(":")
    return (rest or entity_id).replace("-", " ").replace("_", " ")


class RouterIndexer:
    """Builds the routing descriptions. Every dependency is injected, so a test needs no AWS.

    `graph`, `metric_store` and `catalog` are all optional at construction and each layer
    degrades on its own. A tenant with no graph should still route on metrics and tables.
    """

    def __init__(
        self,
        index: RoutingIndex,
        *,
        embedder: Any,
        ontology: Ontology,
        catalog: Any | None = None,
        graph: Any | None = None,
        metric_store: Any | None = None,
        entity_limit: int = DEFAULT_ENTITY_LIMIT,
    ) -> None:
        self.index = index
        self.embedder = embedder
        self.ontology = ontology
        self.catalog = catalog
        self.graph = graph
        self.metric_store = metric_store
        self.entity_limit = entity_limit

    def _model_id(self) -> str:
        return str(getattr(self.embedder, "model_id", "") or "")

    def _write(self, index: str, kind: str, records: list[RoutingRecord]) -> int:
        """Delete the layer, then write it.

        Deleted first rather than upserted over: an upsert leaves behind a metric that has since
        been deprecated or a table dropped from Glue, and a routing hit on something that no
        longer exists sends the question to a tier that will find nothing.
        """
        self.index.delete_kind(index, kind)
        return self.index.upsert(index, records)

    def reindex_metrics(self, ctx: AuthContext) -> int:
        if self.metric_store is None:
            return 0
        metrics = self.metric_store.list_metrics(ctx.tenant_id, approved_only=True)
        records: list[RoutingRecord] = []
        for metric in metrics:
            text = metric_text(metric)
            if not text:
                continue
            records.append(
                RoutingRecord(
                    vector_id=_vector_id(KIND_METRIC, ctx.tenant_id, metric.metric_id),
                    tenant_id=ctx.tenant_id,
                    kind=KIND_METRIC,
                    item_id=metric.metric_id,
                    label=metric.name or metric.metric_id,
                    text=text,
                    embedding=tuple(self.embedder.embed_text(text)),
                    model_id=self._model_id(),
                    detail={
                        "expression": metric.expression,
                        "source_table": metric.source_table,
                    },
                )
            )
        return self._write(routing_index_name(ctx), KIND_METRIC, records)

    def reindex_tables(self, ctx: AuthContext) -> int:
        if self.catalog is None:
            return 0
        records: list[RoutingRecord] = []
        for table in self.catalog.tables(ctx.tenant_id):
            text = table_text(table)
            if not text:
                continue
            records.append(
                RoutingRecord(
                    vector_id=_vector_id(KIND_TABLE, ctx.tenant_id, table.full_name),
                    tenant_id=ctx.tenant_id,
                    kind=KIND_TABLE,
                    item_id=table.full_name,
                    label=table.full_name,
                    text=text,
                    embedding=tuple(self.embedder.embed_text(text)),
                    model_id=self._model_id(),
                    detail={"columns": [c.name for c in table.columns]},
                )
            )
        return self._write(routing_index_name(ctx), KIND_TABLE, records)

    def reindex_entities(self, ctx: AuthContext) -> tuple[int, int]:
        """Returns (indexed, skipped). Skipped means the pack does not declare the id's kind."""
        if self.graph is None:
            return 0, 0

        records: list[RoutingRecord] = []
        skipped = 0
        for entity_id, predicates in self._entities(ctx):
            kind = self.ontology.entity_kind_of(entity_id)
            if kind is None:
                # Not indexed and not guessed. An undeclared kind is drift to surface, and a
                # routable description would hide it behind a label that looks fine.
                skipped += 1
                continue

            kind_label = ""
            for entity_def in self.ontology.entities.values():
                if entity_def.slug == kind:
                    kind_label = entity_def.label
                    break

            labels = [
                self.ontology.predicates[p].label
                for p in predicates
                if p in self.ontology.predicates
            ][:_MAX_PREDICATES_PER_ENTITY]

            label = _entity_label(entity_id)
            text = entity_text(label, kind_label, labels)
            records.append(
                RoutingRecord(
                    vector_id=_vector_id(KIND_ENTITY, ctx.tenant_id, entity_id),
                    tenant_id=ctx.tenant_id,
                    kind=KIND_ENTITY,
                    item_id=entity_id,
                    label=label,
                    text=text,
                    embedding=tuple(self.embedder.embed_text(text)),
                    model_id=self._model_id(),
                    detail={"layer": self.ontology.layer_of(entity_id)},
                )
            )

        written = self._write(routing_index_name(ctx), KIND_ENTITY, records)
        if skipped:
            logger.info(
                "%d entities for %s claim a kind the %s pack does not declare",
                skipped,
                ctx.tenant_id,
                self.ontology.domain,
            )
        return written, skipped

    def _entities(self, ctx: AuthContext) -> list[tuple[str, list[str]]]:
        """Distinct entity ids with their predicates, tenant-scoped.

        The Cypher lives in `src/graph/assertion_queries.py` and the scope comes from
        `edge_scope`, so this cannot express an unscoped read. Trust filters are relaxed the way
        `GraphAssertionStore` relaxes them for an operator read: an entity named only in an
        unreviewed claim is still a subject the graph knows about, and hiding it from the router
        would route those questions away from the tier that holds them.
        """
        from src.graph import assertion_queries as q
        from src.graph.assertions import EpistemicClass

        admin = AuthContext(
            user_id="system",
            tenant_id=ctx.tenant_id,
            matter_allowlist=ctx.matter_allowlist,
            matter_denylist=ctx.matter_denylist,
            include_suggestions=True,
        )
        scope = edge_scope(
            admin,
            edge_var="a",
            min_confidence=0.0,
            trusted_classes=frozenset(EpistemicClass),
            include_pending=True,
        )
        # `LIST_ENTITY_IDS` aggregates, so the limit bounds distinct entities rather than
        # assertions -- which is the thing that costs a Bedrock call each.
        rows = self.graph.read_scoped(q.LIST_ENTITY_IDS, scope, {"limit": self.entity_limit})

        out: list[tuple[str, list[str]]] = []
        for row in rows:
            entity_id = str(row.get("entity_id") or "")
            if not entity_id:
                continue
            predicates = [str(p) for p in (row.get("predicates") or []) if p]
            out.append((entity_id, predicates))
        return out

    def rebuild(
        self,
        ctx: AuthContext,
        *,
        metrics: bool = True,
        tables: bool = True,
        entities: bool = True,
    ) -> RoutingRebuildReport:
        """Rebuild the selected layers, reporting per-layer failures rather than raising.

        A rebuild with one broken layer is still worth having: the router treats a missing layer
        as "nothing looked relevant here" and falls back to trying it anyway, whereas a failed
        rebuild leaves the previous descriptions in place with no indication they are stale.
        """
        report = RoutingRebuildReport()

        if metrics:
            try:
                report.metrics_indexed = self.reindex_metrics(ctx)
            except Exception as e:  # noqa: BLE001
                report.errors.append(f"metrics: {e}")
                logger.warning("routing reindex of metrics failed for %s: %s", ctx.tenant_id, e)

        if tables:
            try:
                report.tables_indexed = self.reindex_tables(ctx)
            except Exception as e:  # noqa: BLE001
                report.errors.append(f"tables: {e}")
                logger.warning("routing reindex of tables failed for %s: %s", ctx.tenant_id, e)

        if entities:
            try:
                report.entities_indexed, report.entities_skipped = self.reindex_entities(ctx)
            except Exception as e:  # noqa: BLE001
                report.errors.append(f"entities: {e}")
                logger.warning("routing reindex of entities failed for %s: %s", ctx.tenant_id, e)

        logger.info(
            "rebuilt routing index for %s: %d metrics, %d tables, %d entities",
            ctx.tenant_id,
            report.metrics_indexed,
            report.tables_indexed,
            report.entities_indexed,
        )
        return report

    def drop_tenant(self, tenant_id: str) -> int:
        """Empty a tenant's routing index. Takes a bare tenant id because a reset runs from an
        admin route rather than the requesting user's scope, mirroring `Embedder.drop_tenant`."""
        from src.query.router_index import routing_index_for_tenant

        return self.index.delete_tenant(routing_index_for_tenant(tenant_id), tenant_id)
