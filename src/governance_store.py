"""Governance settings in DynamoDB, so an Admin change survives a deploy.

The settings were a per-process dict, which meant an administrator could lower the trust floor
or turn on the ungoverned-query kill switch, see it applied, and silently lose it on the next
deploy. For a kill switch that is worse than a cosmetic bug: the firm believes ungoverned
questions are being refused while they are quietly being answered again.

Shares the tenant table using the same entity-prefixed single-key pattern as
`tenant_directory`, under `GOVERNANCE#{tenant_id}`. No new table: the settings are one small
item read per request, which is what an existing key-value table is for.

Two encoding details, both forced by DynamoDB rather than chosen:

- `allowed_tiers` is a frozenset, which DynamoDB cannot store, so it is written as a sorted
  list and read back as a frozenset. Same class of constraint as Neptune refusing list-valued
  properties.
- Floats go in as `Decimal`. DynamoDB has no float type, and boto3 raises rather than rounding
  silently — which is the right behaviour for a confidence floor.

Reads are cached in-process with a short TTL, because `settings_for` is on the path of every
request and a governance change is rare. The TTL is what bounds how long a stale kill switch
can stay stale.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from decimal import Decimal
from typing import Any, Protocol

from src.governance import GovernanceError, GovernanceSettings

logger = logging.getLogger(__name__)

#: Entity-prefixed key in the shared tenant table. The attribute is named `tenant_id` because
#: that is the deployed table's partition key and it cannot be renamed in place; the *value* is
#: an entity key. See `tenant_directory` for the same reasoning.
KEY_ATTR = "tenant_id"
GOVERNANCE_PREFIX = "GOVERNANCE#"

#: Short enough that turning on the kill switch takes effect in seconds across tasks, long
#: enough that a burst of requests is one read. A governance change is rare; a request is not.
CACHE_TTL_SECONDS = 30


class TableLike(Protocol):
    def get_item(self, **kwargs: Any) -> dict[str, Any]: ...
    def put_item(self, **kwargs: Any) -> dict[str, Any]: ...
    def delete_item(self, **kwargs: Any) -> dict[str, Any]: ...


def governance_key(tenant_id: str) -> str:
    return f"{GOVERNANCE_PREFIX}{tenant_id}"


def _encode(value: Any) -> Any:
    """Make a settings value storable.

    `float` -> `Decimal` because DynamoDB has no float type. `frozenset` -> sorted list because
    it has no set-of-numbers type that round-trips cleanly through boto3's resource API.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return Decimal(str(value))
    if isinstance(value, frozenset | set):
        return sorted(value)
    return value


def _decode(name: str, value: Any, template: Any) -> Any:
    """Restore a stored value to the type the dataclass declares.

    Driven by the current default rather than by a hardcoded field list, so a new setting needs
    no change here — and a type that stops matching its default is a bug worth failing on
    rather than papering over.
    """
    if isinstance(template, bool):
        return bool(value)
    if isinstance(template, frozenset):
        stored = frozenset(int(v) for v in value or ())
        # Narrowed to what the default permits. A tenant whose row was written while a fourth
        # tier existed still has `[1,2,3,4]` in DynamoDB, and passing it through would let a
        # persisted setting resurrect a tier the code no longer has -- `Tier(4)` then raises deep
        # inside the resolver. Values the current build does not recognise are dropped, not kept.
        return frozenset(stored & template) if template else stored
    if isinstance(template, float):
        return float(value)
    if isinstance(template, int):
        return int(value)
    return value


class GovernanceStore:
    """Reads and writes per-tenant governance settings."""

    def __init__(
        self,
        table_name: str = "",
        *,
        table: TableLike | None = None,
        table_factory: Callable[[], TableLike] | None = None,
        ttl_seconds: int = CACHE_TTL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.table_name = table_name
        self.ttl_seconds = ttl_seconds
        self._table = table
        self._table_factory = table_factory
        self._clock = clock
        self._cache: dict[str, tuple[GovernanceSettings, float]] = {}
        self._lock = threading.Lock()

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

    def get(self, tenant_id: str) -> GovernanceSettings:
        """Stored settings, or the environment defaults when a tenant has never changed any.

        A read failure returns defaults rather than raising. The alternative is an API that
        cannot answer a question because it cannot read a *policy* about answering questions,
        and the defaults are the safe end: the floor is on, the caps are on.
        """
        cached = self._cached(tenant_id)
        if cached is not None:
            return cached

        try:
            got = self.table.get_item(Key={KEY_ATTR: governance_key(tenant_id)})
        except Exception as e:
            logger.warning("could not read governance for %s (%s), using defaults", tenant_id, e)
            return GovernanceSettings.from_env()

        item = got.get("Item") or {}
        settings = self._from_item(item) if item.get("settings_present") else None
        if settings is None:
            settings = GovernanceSettings.from_env()

        self._remember(tenant_id, settings)
        return settings

    def put(self, tenant_id: str, settings: GovernanceSettings) -> GovernanceSettings:
        """Persist settings, validating first.

        Validated here as well as in `apply` because this is a public entry point: a caller
        writing a hand-built settings object must not be able to store a combination the
        domain refuses, or the next read would return something `apply` would have rejected.
        """
        settings.validate()

        item: dict[str, Any] = {
            KEY_ATTR: governance_key(tenant_id),
            "tenant": tenant_id,
            # Distinguishes "this tenant stored settings" from an item that exists for another
            # reason. Absent means fall back to env defaults rather than to a partial record.
            "settings_present": True,
        }
        for key, value in settings.to_dict().items():
            item[key] = _encode(value)

        self.table.put_item(Item=item)
        self._remember(tenant_id, settings)
        logger.info("governance settings stored for %s by %s", tenant_id, settings.updated_by)
        return settings

    def forget(self, tenant_id: str) -> None:
        """Drop the cached copy, so a change made by another task is picked up now."""
        with self._lock:
            self._cache.pop(tenant_id, None)

    def delete(self, tenant_id: str) -> None:
        """Remove stored settings, reverting the tenant to environment defaults."""
        self.table.delete_item(Key={KEY_ATTR: governance_key(tenant_id)})
        self.forget(tenant_id)

    def _from_item(self, item: dict[str, Any]) -> GovernanceSettings | None:
        defaults = GovernanceSettings.from_env()
        template = defaults.to_dict()
        kwargs: dict[str, Any] = {}
        for name, default in template.items():
            if name in item:
                kwargs[name] = _decode(name, item[name], default)
        try:
            return GovernanceSettings(**kwargs)
        except (TypeError, ValueError, GovernanceError) as e:
            # A stored item that no longer constructs -- a field renamed, or a value that has
            # since become invalid. Defaults are the safe answer, and the warning is how
            # somebody finds out rather than wondering why their setting reverted.
            logger.warning("stored governance settings are unusable (%s), using defaults", e)
            return None

    def _cached(self, tenant_id: str) -> GovernanceSettings | None:
        with self._lock:
            hit = self._cache.get(tenant_id)
            if hit is None:
                return None
            settings, expires_at = hit
            if self._clock() >= expires_at:
                del self._cache[tenant_id]
                return None
            return settings

    def _remember(self, tenant_id: str, settings: GovernanceSettings) -> None:
        with self._lock:
            self._cache[tenant_id] = (settings, self._clock() + self.ttl_seconds)


class InMemoryGovernanceStore:
    """Reference store for tests and local dev. Same interface, no table."""

    def __init__(self) -> None:
        self._settings: dict[str, GovernanceSettings] = {}

    def get(self, tenant_id: str) -> GovernanceSettings:
        if tenant_id not in self._settings:
            self._settings[tenant_id] = GovernanceSettings.from_env()
        return self._settings[tenant_id]

    def put(self, tenant_id: str, settings: GovernanceSettings) -> GovernanceSettings:
        settings.validate()
        self._settings[tenant_id] = settings
        return settings

    def forget(self, tenant_id: str) -> None:
        pass

    def delete(self, tenant_id: str) -> None:
        self._settings.pop(tenant_id, None)
