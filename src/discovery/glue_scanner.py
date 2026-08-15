"""Scan the AWS Glue Data Catalog into graph assertions.

A catalog is a system of record. When Glue says `matters.opened_date` is a
timestamp, that is not an inference or an extraction — it is a declaration, so every
edge here is `DECLARED` and auto-asserts. This is the one pipeline in LexGraph that
needs no review queue, and saying why is worth more than the code that does it.

Two things this module deliberately does *not* do:

**It does not write Cypher.** No module outside `src/graph/` may. The scanner
returns nodes and assertions; a graph writer persists them. That also means the
scanner is testable against a fake Glue client with no database anywhere.

**It does not move data.** Only metadata enters the graph. Rows stay in S3 and are
queried in place at read time (`src/executors/athena.py`), so a structured source
being catalogued is never a copy of privileged material.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from src.graph.assertions import Assertion, EpistemicClass, SourceLocator, build_assertion
from src.metrics.models import StaticCatalog, TableSchema

logger = logging.getLogger(__name__)

METHOD = "glue:catalog_scan"
"""Versioned by the scan strategy, not the catalog contents. If the scanner ever
starts deriving something (inferring keys, normalising types) this becomes @v2 and
the old assertions are superseded rather than mixed with the new ones."""

#: A catalog declaration is true by definition — there is nothing to be uncertain
#: about. Anything less than 1.0 here would be false modesty that then trips the
#: retrieval confidence floor.
DECLARED_CONFIDENCE = 1.0

#: Structural predicates. Descriptive, not governing: a catalog edge naming the
#: wrong column produces a broken query, not a missed conflict check, and new
#: catalog shapes should not require an ontology release.
HAS_TABLE = "HAS_TABLE"
HAS_COLUMN = "HAS_COLUMN"
PARTITIONED_BY = "PARTITIONED_BY"

#: Glue table parameter naming the identifier column(s), comma-separated. Glue has
#: no first-class primary key, but this convention is common enough to honour — and
#: the compiler's fan-out detection is silent without it.
_PRIMARY_KEY_PARAM = "primary_key"


@dataclass(frozen=True)
class CatalogNode:
    node_id: str
    labels: tuple[str, ...]
    props: Mapping[str, Any]


@dataclass
class ScanResult:
    nodes: list[CatalogNode] = field(default_factory=list)
    assertions: list[Assertion] = field(default_factory=list)
    tables: list[TableSchema] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    """Databases or tables that could not be read. A partial scan is still useful,
    so one inaccessible database does not abort the rest."""

    def extend(self, other: ScanResult) -> None:
        self.nodes.extend(other.nodes)
        self.assertions.extend(other.assertions)
        self.tables.extend(other.tables)
        self.errors.extend(other.errors)

    def schema_catalog(self) -> StaticCatalog:
        """The scan, in the shape the metric compiler resolves names against."""
        return StaticCatalog({t.full_name: t for t in self.tables})

    def allowed_tables(self) -> set[str]:
        """Everything catalogued, the firewall's allowlist for this source."""
        return {t.full_name for t in self.tables}


def table_node_id(source_id: str, full_name: str) -> str:
    return f"table:{source_id}:{full_name}"


def column_node_id(source_id: str, full_name: str, column: str) -> str:
    return f"column:{source_id}:{full_name}.{column}"


def source_node_id(source_id: str) -> str:
    return f"source:{source_id}"


def scan_catalog(
    glue_client,
    *,
    tenant_id: str,
    source_id: str,
    databases: Iterable[str] | None = None,
) -> ScanResult:
    """Scan Glue into nodes and DECLARED assertions.

    `databases=None` discovers every database the caller's credentials can see, which
    is the right default for onboarding: an operator should not have to enumerate a
    catalog they are trying to explore.
    """
    result = ScanResult()
    result.nodes.append(
        CatalogNode(
            node_id=source_node_id(source_id),
            labels=("DataSource",),
            props={"source_id": source_id, "tenant_id": tenant_id, "type": "glue"},
        )
    )

    if databases is None:
        databases, discovery_errors = _list_databases(glue_client)
        result.errors.extend(discovery_errors)

    for database in databases:
        result.extend(
            _scan_database(glue_client, database, tenant_id=tenant_id, source_id=source_id)
        )

    logger.info(
        "glue scan of %s: %d tables, %d assertions, %d errors",
        source_id,
        len(result.tables),
        len(result.assertions),
        len(result.errors),
    )
    return result


def _list_databases(glue_client) -> tuple[list[str], list[str]]:
    names: list[str] = []
    try:
        for page in glue_client.get_paginator("get_databases").paginate():
            names.extend(db["Name"] for db in page.get("DatabaseList", []))
    except Exception as e:
        return names, [f"could not list Glue databases: {e}"]
    return names, []


def _scan_database(
    glue_client, database: str, *, tenant_id: str, source_id: str
) -> ScanResult:
    result = ScanResult()
    try:
        pages = list(glue_client.get_paginator("get_tables").paginate(DatabaseName=database))
    except Exception as e:
        logger.warning("skipping Glue database %s: %s", database, e)
        return ScanResult(errors=[f"database {database!r}: {e}"])

    for page in pages:
        for raw in page.get("TableList", []):
            try:
                result.extend(
                    _scan_table(raw, database, tenant_id=tenant_id, source_id=source_id)
                )
            except (KeyError, TypeError) as e:
                # A malformed table entry is a defect in one table, not the database.
                name = raw.get("Name", "<unnamed>") if isinstance(raw, dict) else "<unnamed>"
                result.errors.append(f"table {database}.{name}: {e}")
    return result


def _scan_table(
    raw: Mapping[str, Any], database: str, *, tenant_id: str, source_id: str
) -> ScanResult:
    name = raw["Name"]
    full_name = f"{database}.{name}"
    params = raw.get("Parameters") or {}
    primary_keys = _parse_primary_keys(params)
    # World time for the schema: Glue's UpdateTime is when this shape became true,
    # which is a different question from when we scanned it (recorded_at).
    valid_from = _iso(raw.get("UpdateTime") or raw.get("CreateTime"))

    locator = SourceLocator(source_id=source_id, table=full_name)
    result = ScanResult()

    result.nodes.append(
        CatalogNode(
            node_id=table_node_id(source_id, full_name),
            labels=("Table",),
            props={
                "tenant_id": tenant_id,
                "source_id": source_id,
                "full_name": full_name,
                "database": database,
                "name": name,
                "description": raw.get("Description", ""),
                "catalog_type": _catalog_type(params),
                "location": (raw.get("StorageDescriptor") or {}).get("Location", ""),
            },
        )
    )
    result.assertions.append(
        build_assertion(
            tenant_id=tenant_id,
            subject_id=source_node_id(source_id),
            predicate=HAS_TABLE,
            object_id=table_node_id(source_id, full_name),
            epistemic_class=EpistemicClass.DECLARED,
            method=METHOD,
            confidence=DECLARED_CONFIDENCE,
            source_locator=locator,
            valid_from=valid_from,
        )
    )

    columns: dict[str, str] = {}
    for column, is_partition in _iter_columns(raw):
        col_name = column["Name"]
        data_type = (column.get("Type") or "string").lower()
        columns[col_name] = data_type

        result.nodes.append(
            CatalogNode(
                node_id=column_node_id(source_id, full_name, col_name),
                labels=("Column",),
                props={
                    "tenant_id": tenant_id,
                    "source_id": source_id,
                    "table": full_name,
                    "name": col_name,
                    "data_type": data_type,
                    "description": column.get("Comment", ""),
                    "is_partition": is_partition,
                    "is_primary_key": col_name in primary_keys,
                },
            )
        )
        col_locator = SourceLocator(source_id=source_id, table=full_name, column=col_name)
        for predicate in (HAS_COLUMN, *((PARTITIONED_BY,) if is_partition else ())):
            result.assertions.append(
                build_assertion(
                    tenant_id=tenant_id,
                    subject_id=table_node_id(source_id, full_name),
                    predicate=predicate,
                    object_id=column_node_id(source_id, full_name, col_name),
                    epistemic_class=EpistemicClass.DECLARED,
                    method=METHOD,
                    confidence=DECLARED_CONFIDENCE,
                    source_locator=col_locator,
                    valid_from=valid_from,
                )
            )

    result.tables.append(
        TableSchema(
            full_name=full_name,
            columns=columns,
            primary_keys=frozenset(k for k in primary_keys if k in columns),
        )
    )
    return result


def _iter_columns(raw: Mapping[str, Any]) -> Iterable[tuple[Mapping[str, Any], bool]]:
    """Storage columns then partition keys.

    Glue keeps partition keys out of StorageDescriptor.Columns, but they are queryable
    columns and — being calendar parts more often than not — the compiler needs to
    know about them to stop a `month` partition bypassing a declared time grain.
    """
    for column in (raw.get("StorageDescriptor") or {}).get("Columns") or []:
        if column.get("Name"):
            yield column, False
    for column in raw.get("PartitionKeys") or []:
        if column.get("Name"):
            yield column, True


def _catalog_type(params: Mapping[str, Any]) -> str:
    if (params.get("table_type") or "").upper() == "ICEBERG":
        return "iceberg"
    if "iceberg" in (params.get("metadata_location") or "").lower():
        return "iceberg"
    return "glue"


def _parse_primary_keys(params: Mapping[str, Any]) -> frozenset[str]:
    raw = params.get(_PRIMARY_KEY_PARAM) or ""
    return frozenset(part.strip() for part in raw.split(",") if part.strip())


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    isoformat = getattr(value, "isoformat", None)
    return isoformat() if callable(isoformat) else str(value)
