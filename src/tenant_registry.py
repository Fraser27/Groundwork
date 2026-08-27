"""Which tenants exist, as a record rather than an inference.

A tenant used to exist only implicitly: a `USER#{sub}` row in the same table named it, so
"which tenants are there" could only be answered by reading every user and collecting the
distinct values. That works until a tenant has no users -- which is exactly the state a
newly created one is in for the moment between the two writes, and the state a
half-deleted one is in for good.

Same table as `tenant_directory`, because the two answer halves of one question and a
second table would be a second thing to keep consistent. Sharing it is safe because the
key attribute holds an *entity* key: `USER#{sub}` there, `TENANT#{id}` here.

**Listing is a query, not a scan.** Every record carries the same value in the index
attribute, so all tenants sit in one partition of the existing `TenantIndex` and reading
them is a single query. The alternative was a table scan, which this codebase refuses on
principle -- and adding a second GSI would replace the table, which changes the ARN
`GroundworkApp` imports.

**A deleted tenant is tombstoned, not removed.** The row stays with `deleted_at` set. This
is not an audit trail -- the tenant delete destroys those deliberately -- it is a namespace
ledger, and it answers the one question that survives the data: was this id used before?
An id silently reused is how one firm's people end up looking at a graph built by another.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from src.constants import DEFAULT_ONTOLOGY_PACK
from src.tenant_directory import KEY_ATTR, TENANT_GSI, TENANT_GSI_ATTR

logger = logging.getLogger(__name__)

TENANT_PREFIX = "TENANT#"

#: The index value every tenant record shares, so listing them is one partition read. Not a
#: real tenant id -- `#` cannot appear in one (`scope.is_valid_tenant_id`), so this can never
#: collide with a `users_for_tenant` query.
REGISTRY_PARTITION = "#REGISTRY"


def tenant_key(tenant_id: str) -> str:
    return f"{TENANT_PREFIX}{tenant_id}"


def _now() -> str:
    return datetime.now(UTC).isoformat()


class TableLike(Protocol):
    """The slice of a boto3 DynamoDB `Table` this module uses. No scan, deliberately."""

    def get_item(self, **kwargs: Any) -> dict[str, Any]: ...
    def put_item(self, **kwargs: Any) -> dict[str, Any]: ...
    def query(self, **kwargs: Any) -> dict[str, Any]: ...


@dataclass(frozen=True)
class TenantRecord:
    tenant_id: str
    name: str = ""
    ontology_domain: str = DEFAULT_ONTOLOGY_PACK
    created_at: str = ""
    created_by: str = ""

    deleted_at: str | None = None
    """Set when the tenant was deleted. The row survives so the id is not silently reusable."""

    deleted_by: str | None = None

    @property
    def is_live(self) -> bool:
        return self.deleted_at is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "name": self.name or self.tenant_id,
            "ontology_domain": self.ontology_domain,
            "created_at": self.created_at,
            "created_by": self.created_by,
            "deleted_at": self.deleted_at,
            "deleted_by": self.deleted_by,
            "is_live": self.is_live,
        }


def _to_record(item: dict[str, Any]) -> TenantRecord:
    return TenantRecord(
        tenant_id=str(item.get("registry_tenant_id", "")),
        name=str(item.get("name", "")),
        ontology_domain=str(item.get("ontology_domain") or DEFAULT_ONTOLOGY_PACK),
        created_at=str(item.get("created_at", "")),
        created_by=str(item.get("created_by", "")),
        deleted_at=item.get("deleted_at") or None,
        deleted_by=item.get("deleted_by") or None,
    )


class TenantRegistry:
    """Tenant records in DynamoDB. boto3 is injectable, so tests need no AWS."""

    def __init__(
        self,
        table_name: str = "",
        *,
        table: TableLike | None = None,
        table_factory: Callable[[], TableLike] | None = None,
        index_name: str = TENANT_GSI,
    ) -> None:
        self.table_name = table_name
        self.index_name = index_name
        self._table = table
        self._table_factory = table_factory

    @property
    def table(self) -> TableLike:
        if self._table is None:
            factory = self._table_factory
            if factory is None:
                import boto3

                name = self.table_name
                factory = lambda: boto3.resource("dynamodb").Table(name)
            self._table = factory()
        return self._table

    def get(self, tenant_id: str) -> TenantRecord | None:
        got = self.table.get_item(Key={KEY_ATTR: tenant_key(tenant_id)})
        item = got.get("Item")
        return _to_record(item) if item else None

    def put(self, record: TenantRecord) -> TenantRecord:
        """Write a record. Used for creation and for tombstoning, so it is a plain put."""
        item: dict[str, Any] = {
            KEY_ATTR: tenant_key(record.tenant_id),
            TENANT_GSI_ATTR: REGISTRY_PARTITION,
            # The tenant id under its own name, because the key attribute holds an entity key
            # and `TENANT#` would have to be stripped off every read.
            "registry_tenant_id": record.tenant_id,
            "name": record.name,
            "ontology_domain": record.ontology_domain,
            "created_at": record.created_at or _now(),
            "created_by": record.created_by,
        }
        if record.deleted_at:
            item["deleted_at"] = record.deleted_at
            item["deleted_by"] = record.deleted_by or ""
        self.table.put_item(Item=item)
        return record

    def list(self, *, include_deleted: bool = False) -> list[TenantRecord]:
        """Every tenant, one partition query, paged.

        Paged rather than capped: a truncated tenant list would hide a tenant from the only
        screen that can delete it.
        """
        records: list[TenantRecord] = []
        start: dict[str, Any] | None = None
        while True:
            page = self.table.query(
                IndexName=self.index_name,
                KeyConditionExpression="#t = :partition",
                ExpressionAttributeNames={"#t": TENANT_GSI_ATTR},
                ExpressionAttributeValues={":partition": REGISTRY_PARTITION},
                **({"ExclusiveStartKey": start} if start else {}),
            )
            records.extend(_to_record(i) for i in page.get("Items", []))
            start = page.get("LastEvaluatedKey")
            if not start:
                break
        live = [r for r in records if include_deleted or r.is_live]
        return sorted(live, key=lambda r: r.tenant_id)

    def tombstone(self, tenant_id: str, *, actor: str) -> TenantRecord | None:
        """Mark a tenant deleted, keeping the row. None when there was no record."""
        existing = self.get(tenant_id)
        if existing is None:
            return None
        record = TenantRecord(
            tenant_id=existing.tenant_id,
            name=existing.name,
            ontology_domain=existing.ontology_domain,
            created_at=existing.created_at,
            created_by=existing.created_by,
            deleted_at=_now(),
            deleted_by=actor,
        )
        self.put(record)
        logger.info("tombstoned tenant %s by %s", tenant_id, actor)
        return record


class InMemoryTenantRegistry:
    """Reference implementation for tests and local development."""

    def __init__(self) -> None:
        self._records: dict[str, TenantRecord] = {}

    def get(self, tenant_id: str) -> TenantRecord | None:
        return self._records.get(tenant_id)

    def put(self, record: TenantRecord) -> TenantRecord:
        stored = (
            record
            if record.created_at
            else TenantRecord(
                tenant_id=record.tenant_id,
                name=record.name,
                ontology_domain=record.ontology_domain,
                created_at=_now(),
                created_by=record.created_by,
                deleted_at=record.deleted_at,
                deleted_by=record.deleted_by,
            )
        )
        self._records[record.tenant_id] = stored
        return stored

    def list(self, *, include_deleted: bool = False) -> list[TenantRecord]:
        return sorted(
            (r for r in self._records.values() if include_deleted or r.is_live),
            key=lambda r: r.tenant_id,
        )

    def tombstone(self, tenant_id: str, *, actor: str) -> TenantRecord | None:
        existing = self._records.get(tenant_id)
        if existing is None:
            return None
        return self.put(
            TenantRecord(
                tenant_id=existing.tenant_id,
                name=existing.name,
                ontology_domain=existing.ontology_domain,
                created_at=existing.created_at,
                created_by=existing.created_by,
                deleted_at=_now(),
                deleted_by=actor,
            )
        )
