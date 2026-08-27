"""Rebuild the catalog cache from the graph.

`CatalogStore` is process-local, and `record_scan` was its only writer. So a redeploy, a second
Fargate task, or the MCP sidecar (a separate process with its own `Services`) reported "no catalogue
scan has been run" while the graph held every `:Table` and `:Column` the scan wrote. This is the
read back.

Separate module rather than a method on either side: `catalog_store` owns the UI-shaped types and
`graph_store` must not import them, or the two would form a cycle. Assembly belongs to neither, so
it lives here.

No Cypher and no client: it reads through `CatalogGraphStore`, which reads through
`GraphClient.read_scoped`, so the tenant filter is not something this module can forget.
"""

from __future__ import annotations

import logging
from typing import Any

from src.discovery.catalog_store import CatalogColumn, CatalogStore, CatalogTable, SourceRecord
from src.discovery.graph_store import CatalogGraphStore
from src.graph.scope import AuthContext

logger = logging.getLogger(__name__)


def hydrate(store: CatalogStore, graph_store: CatalogGraphStore, ctx: AuthContext) -> int:
    """Load this tenant's catalog from the graph into `store`. Returns tables loaded.

    Reads happen before the store is touched, so no lock is held across a graph call.
    """
    table_rows = graph_store.table_rows(ctx)
    column_rows = graph_store.column_rows(ctx)
    source_rows = graph_store.source_rows(ctx)

    tables = build_tables(table_rows, column_rows)
    sources = build_sources(source_rows, tables)
    loaded = store.hydrate(ctx.tenant_id, tables, sources)
    logger.info(
        "hydrated catalog for %s from the graph: %d tables, %d sources",
        ctx.tenant_id,
        loaded,
        len(sources),
    )
    return loaded


def hydrate_once(store: CatalogStore, graph_store: CatalogGraphStore, ctx: AuthContext) -> int:
    """Hydrate unless this tenant already was. Returns 0 when there was nothing to do.

    Cheap enough to call on every read path, which is the point: the alternative is each caller
    remembering, and the one that forgets is the one that shows an empty Tables page.
    """
    if store.is_hydrated(ctx.tenant_id):
        return 0
    return hydrate(store, graph_store, ctx)


def build_tables(
    table_rows: list[dict[str, Any]], column_rows: list[dict[str, Any]]
) -> list[CatalogTable]:
    """Graph rows to UI-shaped tables, columns grouped onto their parent."""
    columns: dict[str, list[CatalogColumn]] = {}
    for row in column_rows:
        full_name = _text(row.get("full_name"))
        name = _text(row.get("name"))
        if not full_name or not name:
            continue
        columns.setdefault(full_name, []).append(
            CatalogColumn(
                name=name,
                data_type=_text(row.get("data_type")),
                description=_text(row.get("description")),
                is_partition=bool(row.get("is_partition")),
                is_primary_key=bool(row.get("is_primary_key")),
            )
        )

    out: list[CatalogTable] = []
    seen: set[str] = set()
    for row in table_rows:
        full_name = _text(row.get("full_name"))
        # One table can hold several HAS_TABLE edges, one per scan generation, since the assertion
        # id is part of the MERGE key. The rows are ordered, so the first wins and the rest are
        # duplicates rather than second tables.
        if not full_name or full_name in seen:
            continue
        seen.add(full_name)
        out.append(
            CatalogTable(
                full_name=full_name,
                name=_text(row.get("name")),
                database=_text(row.get("database")),
                source_id=_text(row.get("source_id")),
                description=_text(row.get("description")),
                catalog_type=_text(row.get("catalog_type")),
                location=_text(row.get("location")),
                columns=tuple(columns.get(full_name, [])),
                scanned_at=_text(row.get("scanned_at")) or None,
            )
        )
    return out


def build_sources(
    source_rows: list[dict[str, Any]], tables: list[CatalogTable]
) -> list[SourceRecord]:
    """Graph rows to source records, with the table count the cache would have held.

    `last_scanned_at` falls back to the newest `HAS_TABLE` assertion for a `DataSource` node
    written before that property existed. It stays empty when there is nothing to fall back to,
    because an invented timestamp would report a scan that may never have run.
    """
    counts: dict[str, int] = {}
    newest: dict[str, str] = {}
    for table in tables:
        counts[table.source_id] = counts.get(table.source_id, 0) + 1
        if table.scanned_at and table.scanned_at > newest.get(table.source_id, ""):
            newest[table.source_id] = table.scanned_at

    out: list[SourceRecord] = []
    for row in source_rows:
        source_id = _text(row.get("source_id"))
        if not source_id:
            continue
        out.append(
            SourceRecord(
                source_id=source_id,
                name=source_id,
                kind=_text(row.get("type")).upper() or "GLUE",
                table_count=counts.get(source_id, 0),
                last_scanned_at=_text(row.get("last_scanned_at")) or newest.get(source_id) or None,
            )
        )
    return out


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()
