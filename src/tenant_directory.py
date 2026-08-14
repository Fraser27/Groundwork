"""Which tenant a user belongs to.

This exists because of a hard Cognito constraint. `TokenVerifier` requires an **access**
token — accepting an id token would accept a token minted for a different purpose — but
Cognito puts custom attributes such as `custom:tenant_id` only on the **id** token. So the
tenant binding cannot travel in the token this API accepts, and something else has to hold
it.

The split is deliberate about which facts live where:

**Identity comes from the token.** `sub` and `cognito:groups` are Cognito-signed and are
never read from here — a user's role is still whatever the verified JWT says.

**Tenant comes from this table.** Server-side only, so a caller cannot assert their own
tenant, which is the one boundary the whole system rests on.

Keyed on `sub`, not email. An email address is mutable and can be reassigned to a
different person; `sub` is immutable for the life of the user. Email is stored alongside
as an attribute so an operator can find a record by the identifier they actually know.

Lookups are cached in-process with a short TTL. Tenant membership changes about never,
and the alternative is a DynamoDB read on the authentication path of every request. The
cache is per-process and in-memory on purpose: anything client-visible would let a user
choose their own tenant.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from typing import Any, Protocol

logger = logging.getLogger(__name__)

#: Entity-prefixed keys in a single-key table.
#:
#: The attribute is named `tenant_id` because that is the deployed table's partition key
#: and it cannot be renamed in place — DynamoDB replaces the table, which changes the ARN
#: that `LexGraphApp` imports. So the *name* is historical while the *value* is an entity
#: key, and the `USER#` prefix is what keeps user records from colliding with anything else
#: stored here later.
KEY_ATTR = "tenant_id"
USER_PREFIX = "USER#"

#: Short enough that revoking access is measured in seconds, long enough that a burst of
#: requests from one user is a single read.
CACHE_TTL_SECONDS = 300


class TableLike(Protocol):
    """The slice of a boto3 DynamoDB `Table` this module uses. No scan, deliberately."""

    def get_item(self, **kwargs: Any) -> dict[str, Any]: ...
    def put_item(self, **kwargs: Any) -> dict[str, Any]: ...
    def query(self, **kwargs: Any) -> dict[str, Any]: ...


class UnknownUser(LookupError):
    """No tenant binding for this subject.

    Raised rather than defaulted. A user with no tenant must not silently land in
    somebody else's — the safe answer is to refuse, and the fix is an operator creating
    the record.
    """


def user_key(sub: str) -> str:
    return f"{USER_PREFIX}{sub}"


class TenantDirectory:
    """Resolves a Cognito `sub` to a tenant id."""

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
        self._cache: dict[str, tuple[str, float]] = {}
        # Requests are served on a thread pool, so the cache is touched concurrently.
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

    def tenant_for(self, sub: str) -> str:
        cached = self._cached(sub)
        if cached is not None:
            return cached

        got = self.table.get_item(Key={KEY_ATTR: user_key(sub)})
        item = got.get("Item")
        # `tenant` rather than `tenant_id`: the key attribute already owns that name.
        tenant_id = (item or {}).get("tenant")
        if not tenant_id:
            raise UnknownUser(f"no tenant binding for subject {sub!r}")

        with self._lock:
            self._cache[sub] = (str(tenant_id), self._clock() + self.ttl_seconds)
        return str(tenant_id)

    def put_user(self, sub: str, tenant_id: str, *, email: str = "") -> None:
        """Bind a user to a tenant. Called by admin tooling, never by a request handler."""
        item: dict[str, Any] = {
            KEY_ATTR: user_key(sub),
            "sub": sub,
            "tenant": tenant_id,
        }
        if email:
            # Stored for operator lookup only. Not indexed: adding a GSI would replace the
            # table, and finding a record by email is a rare, manual act.
            item["email"] = email.lower()
        self.table.put_item(Item=item)
        self.forget(sub)
        logger.info("bound subject %s to tenant %s", sub, tenant_id)

    def forget(self, sub: str) -> None:
        """Drop a cached binding, so a correction takes effect without waiting for TTL."""
        with self._lock:
            self._cache.pop(sub, None)

    def _cached(self, sub: str) -> str | None:
        with self._lock:
            hit = self._cache.get(sub)
            if hit is None:
                return None
            tenant_id, expires_at = hit
            if self._clock() >= expires_at:
                del self._cache[sub]
                return None
            return tenant_id


class StaticTenantDirectory:
    """In-memory directory for local dev and tests.

    Not a null object: it still refuses an unknown subject, so the closed-by-default
    behaviour is the same one the tests exercise.
    """

    def __init__(self, bindings: dict[str, str] | None = None) -> None:
        self._bindings = dict(bindings or {})

    def tenant_for(self, sub: str) -> str:
        tenant_id = self._bindings.get(sub)
        if not tenant_id:
            raise UnknownUser(f"no tenant binding for subject {sub!r}")
        return tenant_id

    def put_user(self, sub: str, tenant_id: str, *, email: str = "") -> None:
        self._bindings[sub] = tenant_id

    def forget(self, sub: str) -> None:
        self._bindings.pop(sub, None)
