"""What the last catalog scan found.

A scan reads Glue and produces `Table` and `Column` nodes plus DECLARED assertions. Those
go to the graph, but the UI needs to answer "what tables do we know about" without a graph
round trip on every page load, and the API needs somewhere to record *when* a source was
last scanned. That is what this holds.

Deliberately a cache, not a source of truth. Glue is authoritative for schemas and S3 is
authoritative for documents; everything here can be rebuilt by scanning again. So it is
in-memory by default and losing it costs a re-scan, which is the same posture as the
vector index.

Scoped by tenant throughout. Two firms may both have a `warehouse.matters` table and they
are different tables.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime

from src.discovery.glue_scanner import ScanResult

logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class CatalogColumn:
    name: str
    data_type: str
    description: str = ""
    is_partition: bool = False
    is_primary_key: bool = False


@dataclass(frozen=True)
class CatalogTable:
    """A scanned table, as the UI shows it.

    Richer than `TableSchema`, which carries only what the metric compiler needs
    (name → type, plus primary keys). The catalog scan also produces a database, a
    description and partition flags on its graph nodes, and losing those would make the
    Tables page less informative than the scan that fed it.
    """

    full_name: str
    name: str
    database: str
    source_id: str
    description: str = ""
    catalog_type: str = ""
    location: str = ""
    columns: tuple[CatalogColumn, ...] = ()
    scanned_at: str | None = None


def _tables_from_scan(result: ScanResult, *, source_id: str) -> list[CatalogTable]:
    """Rebuild UI-shaped tables from the scan's graph nodes.

    The nodes are the richer record — `ScanResult.tables` is the compiler's narrower view,
    carrying only name → type and primary keys — so everything here is read from the nodes
    the scan already built rather than recomputed.
    """
    columns: dict[str, list[CatalogColumn]] = {}
    for node in result.nodes:
        if "Column" not in node.labels:
            continue
        full_name = str(node.props.get("table", ""))
        if not full_name:
            continue
        columns.setdefault(full_name, []).append(
            CatalogColumn(
                name=str(node.props.get("name", "")),
                data_type=str(node.props.get("data_type", "")),
                description=str(node.props.get("description", "")),
                is_partition=bool(node.props.get("is_partition", False)),
                is_primary_key=bool(node.props.get("is_primary_key", False)),
            )
        )

    out: list[CatalogTable] = []
    for node in result.nodes:
        if "Table" not in node.labels:
            continue
        full_name = str(node.props.get("full_name", ""))
        out.append(
            CatalogTable(
                full_name=full_name,
                name=str(node.props.get("name", "")),
                database=str(node.props.get("database", "")),
                source_id=source_id,
                description=str(node.props.get("description", "")),
                catalog_type=str(node.props.get("catalog_type", "")),
                location=str(node.props.get("location", "")),
                columns=tuple(columns.get(full_name, [])),
            )
        )
    return out


@dataclass
class SourceRecord:
    """A configured data source and the outcome of its last scan."""

    source_id: str
    name: str
    kind: str = "GLUE"
    database: str | None = None
    region: str | None = None
    table_count: int = 0
    last_scanned_at: str | None = None
    errors: list[str] = field(default_factory=list)

    @property
    def status(self) -> str:
        """What an operator needs to know at a glance.

        `PARTIAL` is its own state rather than folded into `ERROR`: one inaccessible
        database out of six is a permissions problem to fix, not a broken source, and the
        tables that did scan are usable meanwhile.
        """
        if self.last_scanned_at is None:
            return "NOT_SCANNED"
        if self.errors and self.table_count == 0:
            return "ERROR"
        if self.errors:
            return "PARTIAL"
        return "CONNECTED"


class CatalogStore:
    def __init__(self) -> None:
        self._tables: dict[str, dict[str, CatalogTable]] = {}
        self._sources: dict[str, dict[str, SourceRecord]] = {}
        # Scans run in a background task while pages read; both touch this.
        self._lock = threading.Lock()

    def record_scan(
        self,
        tenant_id: str,
        *,
        source_id: str,
        result: ScanResult,
        name: str = "",
        region: str = "",
    ) -> SourceRecord:
        """Replace what we know about a source with what this scan found.

        Replace, not merge: a table dropped in Glue must disappear here too, or the UI
        keeps offering a table that no longer exists.
        """
        at = _now()
        scanned = _tables_from_scan(result, source_id=source_id)
        with self._lock:
            tables = self._tables.setdefault(tenant_id, {})
            # Drop this source's previous tables before adding the new set, leaving other
            # sources untouched. A table dropped in Glue must disappear here too.
            for full_name in [fn for fn, t in tables.items() if t.source_id == source_id]:
                del tables[full_name]
            for table in scanned:
                tables[table.full_name] = replace(table, scanned_at=at)

            record = SourceRecord(
                source_id=source_id,
                name=name or source_id,
                database=None,
                region=region or None,
                table_count=len(result.tables),
                last_scanned_at=at,
                errors=list(result.errors),
            )
            self._sources.setdefault(tenant_id, {})[source_id] = record

        logger.info(
            "recorded scan of %s for %s: %d tables, %d errors",
            source_id,
            tenant_id,
            len(result.tables),
            len(result.errors),
        )
        return record

    def tables(self, tenant_id: str) -> list[CatalogTable]:
        with self._lock:
            return sorted(self._tables.get(tenant_id, {}).values(), key=lambda t: t.full_name)

    def table(self, tenant_id: str, full_name: str) -> CatalogTable | None:
        with self._lock:
            return self._tables.get(tenant_id, {}).get(full_name)

    def sources(self, tenant_id: str) -> list[SourceRecord]:
        with self._lock:
            return sorted(self._sources.get(tenant_id, {}).values(), key=lambda s: s.source_id)

    def register_source(
        self, tenant_id: str, source_id: str, *, name: str = "", region: str = ""
    ) -> SourceRecord:
        """Make a configured-but-unscanned source visible.

        Without this a source appears only once it has been scanned, which is backwards:
        an operator needs to see that a source is configured in order to press Scan.
        """
        with self._lock:
            sources = self._sources.setdefault(tenant_id, {})
            if source_id not in sources:
                sources[source_id] = SourceRecord(
                    source_id=source_id, name=name or source_id, region=region or None
                )
            return sources[source_id]

    def clear(self, tenant_id: str) -> None:
        """Forget everything for one tenant. Used by the reset action."""
        with self._lock:
            self._tables.pop(tenant_id, None)
            self._sources.pop(tenant_id, None)
