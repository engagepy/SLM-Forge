"""A tiny thread-safe pub/sub bus bridging worker threads to SSE clients.

Topics: `job:{id}` (logs, metrics, status), `jobs` (status changes for any job),
`project:{id}` (agent events and proposals).
"""

import asyncio
import threading
from collections import defaultdict
from contextlib import asynccontextmanager


class Bus:
    def __init__(self) -> None:
        self._subs: dict[str, set[tuple[asyncio.AbstractEventLoop, asyncio.Queue]]] = defaultdict(set)
        self._lock = threading.Lock()

    def publish(self, topic: str, event: dict) -> None:
        with self._lock:
            subs = list(self._subs.get(topic, ()))
        for loop, q in subs:
            try:
                loop.call_soon_threadsafe(q.put_nowait, event)
            except RuntimeError:  # loop closed; the subscriber is gone
                pass

    @asynccontextmanager
    async def subscribe(self, topic: str, maxsize: int = 1000):
        entry = (asyncio.get_running_loop(), asyncio.Queue(maxsize=maxsize))
        with self._lock:
            self._subs[topic].add(entry)
        try:
            yield entry[1]
        finally:
            with self._lock:
                self._subs[topic].discard(entry)


bus = Bus()
