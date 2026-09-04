"""What the last catalog scan found.

A scan reads Glue and produces `Table` and `Column` nodes plus DECLARED assertions. Those
go to the graph, but the UI needs to answer "what tables do we know about" without a graph
round trip on every page load, and the API needs somewhere to record *when* a source was
last scanned. That is what this holds.

Deliberately a cache, not a source of truth. Glue is authoritative for schemas and S3 is
authoritative for documents; everything here can be rebuilt.

It used to be rebuilt only by re-scanning, and this docstring said so. That was written when
the scan's `Table` and `Column` nodes were being discarded, so the graph held nothing to reload
from. They are persisted now, which makes the graph the durable copy: losing this cache costs
one scoped read, not an operator noticing and pressing Scan. `catalog_hydrate` does that read,
because a process-local cache is lost on every redeploy and is empty in the MCP sidecar from the
moment it starts.

Scoped by tenant throughout. Two firms may both have a `warehouse.matters` table and they
are different tables.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime

from src.discovery.glue_scanner import ScanResult

logger = logging.getLogger(__name__)

#: How long an *empty* tenant stays settled before the graph is read again. Shorter than
#: `GRAPH_RECONNECT_COOLDOWN_SECONDS` because this costs three fast queries rather than a connect
#: timeout, and short enough that someone who scans and then asks a question does not have to ask
#: twice.
EMPTY_RETRY_SECONDS = 15.0


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
        self._hydrated: set[str] = set()
        self._empty_until: dict[str, float] = {}
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
            self._settle(tenant_id)

        logger.info(
            "recorded scan of %s for %s: %d tables, %d errors",
            source_id,
            tenant_id,
            len(result.tables),
            len(result.errors),
        )
        return record

    def _settle(self, tenant_id: str) -> None:
        """Record that this tenant has been reconciled. Caller holds the lock.

        A tenant that ended up with **tables** is settled for good: the graph cannot take them
        away, only a scan or a reset can. A tenant that ended up empty is settled only for
        `EMPTY_RETRY_SECONDS`, because empty is the one state something outside this process can
        change -- a scan in the sibling process, or the graph coming up after this read.

        Permanently settling the empty case is what this replaces, and it cost an afternoon: the
        MCP sidecar read a genuinely empty `anycorp` once, marked it complete, and then refused
        every table of a scan run four minutes later in the API process. `SQLFirewall`'s allowlist
        is built from this cache, so the symptom was `unauthorized tables` on a catalog the graph
        held correctly all along -- which reads as a permissions problem and is not one.

        One rule in one place, so `record_scan` and `hydrate` cannot disagree about it. A scan
        that found nothing is as retryable as a read that found nothing, and for the same reason.
        """
        if self._tables.get(tenant_id):
            self._hydrated.add(tenant_id)
            self._empty_until.pop(tenant_id, None)
        else:
            self._hydrated.discard(tenant_id)
            self._empty_until[tenant_id] = time.monotonic() + EMPTY_RETRY_SECONDS

    def is_hydrated(self, tenant_id: str) -> bool:
        """Whether this tenant has been reconciled against the graph in this process.

        False and empty are different states, and conflating them is the bug: an empty cache
        means either "never scanned" or "scanned by a process that has since been replaced",
        and only the graph can say which.

        An empty tenant answers True only inside its cooldown, so a never-scanned tenant costs one
        graph read per `EMPTY_RETRY_SECONDS` rather than one per request -- and, unlike before, not
        one per process lifetime. See `_settle`.
        """
        with self._lock:
            if tenant_id in self._hydrated:
                return True
            return time.monotonic() < self._empty_until.get(tenant_id, 0.0)

    def hydrate(
        self,
        tenant_id: str,
        tables: Sequence[CatalogTable],
        sources: Sequence[SourceRecord],
    ) -> int:
        """Fill from the graph without overwriting anything a live scan produced.

        Not `record_scan`: that replaces a source's tables wholesale, which is right for a scan
        (a table dropped in Glue must disappear) and wrong here, where the graph may be behind a
        scan running concurrently in this process. So an existing entry always wins, and a source
        already carrying a scan timestamp is left alone rather than reverted to the graph's.
        """
        loaded = 0
        with self._lock:
            tables_for_tenant = self._tables.setdefault(tenant_id, {})
            for table in tables:
                if table.full_name and table.full_name not in tables_for_tenant:
                    tables_for_tenant[table.full_name] = table
                    loaded += 1

            sources_for_tenant = self._sources.setdefault(tenant_id, {})
            for source in sources:
                found = sources_for_tenant.get(source.source_id)
                if found is None or found.last_scanned_at is None:
                    sources_for_tenant[source.source_id] = source

            self._settle(tenant_id)
        return loaded

    def tables(self, tenant_id: str) -> list[CatalogTable]:
        with self._lock:
            return sorted(self._tables.get(tenant_id, {}).values(), key=lambda t: t.full_name)

    def table(self, tenant_id: str, full_name: str) -> CatalogTable | None:
        with self._lock:
            return self._tables.get(tenant_id, {}).get(full_name)

    def with_sources(self, tenant_id: str, full_name: str) -> tuple[CatalogTable, dict[str, str]]:
        """One table, and where each description came from. Raises `KeyError` when unknown.

        Part of this class rather than only of `EnrichedCatalog` because both have to answer it:
        with no graph reachable, `Services.enriched_catalog` hands back this store directly, and a
        route calling a method only the wrapper has would 500 on exactly the deployment that has
        least to fall back on. Every description here came from the scan, so the answer is `glue`
        or nothing.
        """
        from src.discovery.catalog_overlay import sources_for

        found = self.table(tenant_id, full_name)
        if found is None:
            raise KeyError(full_name)
        return found, sources_for(found, {})

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
            # Leaving the flag set would make a reset permanent: the next read would consider the
            # tenant already loaded and never look at the graph again. The cooldown goes too, so a
            # reset is followed by a read rather than by up to `EMPTY_RETRY_SECONDS` of staleness.
            self._hydrated.discard(tenant_id)
            self._empty_until.pop(tenant_id, None)
