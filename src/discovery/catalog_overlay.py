"""Approved descriptions, merged onto the scanned schema at read time.

Three sources can describe a column, and they are not equally authoritative:

1. a `DECLARED` description in the graph, which a person typed
2. the Glue `Comment`, which `CatalogTable` already carries
3. an approved `EXTRACTED_MODEL` description in the graph, which a model proposed

A person beats the upstream comment deliberately: somebody edits a description precisely because
the comment was wrong or absent, and demoting their edit below it would make the edit look broken.
A model does not beat it, because a comment somebody wrote in Glue is still a human statement.

Merged rather than written back. `CatalogStore` is an explicit cache whose `record_scan` replaces a
source's tables wholesale, so a description stored there would be destroyed by the next scan. Glue
stays authoritative for shape, the graph for meaning, and nothing has to be kept in step.

Ids are **built, never parsed**. A `full_name` contains dots, so splitting `column:src:db.t.col`
back apart is ambiguous, and a parser here would be a second definition of an id format that
`glue_scanner` already owns.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from src.discovery.catalog_store import CatalogTable
from src.discovery.glue_scanner import column_node_id, table_node_id
from src.discovery.graph_store import DescriptionText

logger = logging.getLogger(__name__)

#: Where a description a reader is looking at came from. Reported so a model's guess and a
#: colleague's correction are never rendered as the same kind of statement.
SOURCE_NONE = ""
SOURCE_GLUE = "glue"
SOURCE_HUMAN = "human"
SOURCE_MODEL = "model"


def source_of(existing: str, found: DescriptionText | None) -> str:
    """Which of the three sources wins for one subject."""
    if found is not None and found.is_human:
        return SOURCE_HUMAN
    if existing.strip():
        return SOURCE_GLUE
    if found is not None:
        return SOURCE_MODEL
    return SOURCE_NONE


def _resolve(existing: str, found: DescriptionText | None) -> str:
    source = source_of(existing, found)
    if source in (SOURCE_HUMAN, SOURCE_MODEL) and found is not None:
        return found.text
    return existing


def sources_for(
    table: CatalogTable,
    descriptions: Mapping[str, DescriptionText],
) -> dict[str, str]:
    """Where each description on this table came from, keyed by column name, `""` for the table.

    Reported alongside the text rather than recomputed by the caller. A page that showed a model's
    guess and a colleague's correction identically would be hiding the one distinction a reader
    needs in order to know what to double-check.
    """
    out = {
        "": source_of(
            table.description, descriptions.get(table_node_id(table.source_id, table.full_name))
        )
    }
    for column in table.columns:
        node_id = column_node_id(table.source_id, table.full_name, column.name)
        out[column.name] = source_of(column.description, descriptions.get(node_id))
    return out


def overlay_tables(
    tables: Sequence[CatalogTable],
    descriptions: Mapping[str, DescriptionText],
) -> list[CatalogTable]:
    """These tables with graph descriptions applied. Frozen dataclasses, so copies."""
    if not descriptions:
        return list(tables)

    out: list[CatalogTable] = []
    for table in tables:
        table_id = table_node_id(table.source_id, table.full_name)
        columns = tuple(
            replace(
                column,
                description=_resolve(
                    column.description,
                    descriptions.get(column_node_id(table.source_id, table.full_name, column.name)),
                ),
            )
            for column in table.columns
        )
        out.append(
            replace(
                table,
                description=_resolve(table.description, descriptions.get(table_id)),
                columns=columns,
            )
        )
    return out


class EnrichedCatalog:
    """A `CatalogStore` with approved descriptions layered on.

    Satisfies the slice of the store that readers use (`tables`, `table`), so `Resolver`,
    `Planner` and the Tables page can be pointed at this instead with no other change. That
    matters: the schema a reader is shown and the schema the SQL generator was given have to be
    the same text, or the page is not showing what the model saw.
    """

    def __init__(self, catalog: Any, store: Any, ctx_for: Any) -> None:
        self._catalog = catalog
        self._store = store
        self._ctx_for = ctx_for
        """Builds an `AuthContext` for a tenant id. The overlay read is scoped like any other."""

    def tables(self, tenant_id: str) -> list[CatalogTable]:
        base = self._catalog.tables(tenant_id)
        return overlay_tables(base, self._descriptions(tenant_id))

    def table(self, tenant_id: str, full_name: str) -> CatalogTable | None:
        found = self._catalog.table(tenant_id, full_name)
        if found is None:
            return None
        overlaid = overlay_tables([found], self._descriptions(tenant_id))
        return overlaid[0] if overlaid else found

    def with_sources(self, tenant_id: str, full_name: str) -> tuple[CatalogTable, dict[str, str]]:
        """One table and where each of its descriptions came from, in a single graph read.

        The pair, because a caller that fetched the text and then asked separately where it came
        from would do two reads that could disagree.
        """
        found = self._catalog.table(tenant_id, full_name)
        if found is None:
            raise KeyError(full_name)
        descriptions = self._descriptions(tenant_id)
        return overlay_tables([found], descriptions)[0], sources_for(found, descriptions)

    def _descriptions(self, tenant_id: str) -> dict[str, DescriptionText]:
        """Approved descriptions, or none if the graph will not answer.

        Logged at warning rather than debug: degrading here means approved descriptions stop
        reaching the SQL prompt, and the generated query gets worse with no other signal.
        """
        if self._store is None:
            return {}
        try:
            return self._store.approved_descriptions(self._ctx_for(tenant_id))
        except Exception as e:  # noqa: BLE001
            logger.warning("catalog descriptions unavailable for %s: %s", tenant_id, e)
            return {}

    def __getattr__(self, name: str) -> Any:
        """Anything else falls through to the real store, which owns writes and source records."""
        return getattr(self._catalog, name)
