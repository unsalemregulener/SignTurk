"""In-memory sliding-window limits for login and registration attempts."""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque


class SlidingWindowLimiter:
    """At most `limit` events per key within `window` seconds."""

    def __init__(self, limit: int, window: float):
        self.limit = limit
        self.window = window
        self._events = defaultdict(deque)
        self._lock = threading.Lock()

    def _recent(self, key, now: float) -> int:
        events = self._events.get(key)
        if not events:
            return 0
        while events and now - events[0] > self.window:
            events.popleft()
        if not events:
            del self._events[key]
            return 0
        return len(events)

    def blocked(self, key) -> bool:
        with self._lock:
            return self._recent(key, time.monotonic()) >= self.limit

    def hit(self, key) -> None:
        with self._lock:
            now = time.monotonic()
            self._recent(key, now)
            self._events[key].append(now)

    def reset(self, key) -> None:
        with self._lock:
            self._events.pop(key, None)
