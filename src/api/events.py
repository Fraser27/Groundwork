"""In-process pub/sub for ingest progress, behind the websocket endpoint.

**This is per-process, and that is a real limit.** A client connected to task A never
sees an event published on task B, so with `appDesiredCount` above 1 the websocket
becomes best-effort. That is why the UI also polls: the poll is the correctness
guarantee and the websocket is the latency improvement. Making this cross-task means
shared pub/sub, which is deliberately not here yet.

Subscriptions are per tenant, and delivery is filtered on the tenant a subscriber
authenticated as — a progress event names a document id, so a broadcast would leak the
fact that another firm is ingesting something.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from collections import defaultdict
from typing import Any

logger = logging.getLogger(__name__)

#: Per-subscriber queue depth. A slow reader is dropped rather than allowed to grow
#: unboundedly: this is progress reporting, and stale progress is worthless anyway.
QUEUE_MAXSIZE = 64


class _Subscriber:
    """A queue plus the loop it belongs to.

    The loop is captured at subscribe time because `publish` is called from the ingest
    worker *thread*, and `asyncio.Queue` is not thread-safe. Handing the put back to the
    owning loop via `call_soon_threadsafe` is the only correct way across that boundary —
    calling `put_nowait` directly from the thread corrupts the queue's internal state
    instead of raising, which is the kind of bug that shows up as a hung websocket weeks
    later.
    """

    __slots__ = ("loop", "queue")

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self.queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=QUEUE_MAXSIZE)
        self.loop = loop

    def offer(self, event: dict[str, Any]) -> None:
        def put() -> None:
            try:
                self.queue.put_nowait(event)
            except asyncio.QueueFull:
                logger.debug("dropping progress event for a slow subscriber")

        try:
            self.loop.call_soon_threadsafe(put)
        except RuntimeError:
            # The loop is closed: the subscriber's connection is already gone.
            pass


class EventHub:
    def __init__(self) -> None:
        self._subscribers: dict[str, set[_Subscriber]] = defaultdict(set)
        # A plain threading lock, not asyncio.Lock: this is touched from both the loop
        # and the ingest worker thread.
        self._lock = threading.Lock()

    def subscribe(self, tenant_id: str) -> _Subscriber:
        sub = _Subscriber(asyncio.get_running_loop())
        with self._lock:
            self._subscribers[tenant_id].add(sub)
        return sub

    def unsubscribe(self, tenant_id: str, sub: _Subscriber) -> None:
        with self._lock:
            self._subscribers[tenant_id].discard(sub)
            if not self._subscribers[tenant_id]:
                del self._subscribers[tenant_id]

    def publish(self, tenant_id: str, event: dict[str, Any]) -> None:
        """Fan out to this tenant's subscribers. Safe to call from a worker thread, and
        never raises — an ingest must not fail because of a websocket."""
        with self._lock:
            subs = list(self._subscribers.get(tenant_id, ()))
        for sub in subs:
            sub.offer(event)

    def subscriber_count(self, tenant_id: str) -> int:
        with self._lock:
            return len(self._subscribers.get(tenant_id, ()))


_hub = EventHub()


def get_event_hub() -> EventHub:
    return _hub
