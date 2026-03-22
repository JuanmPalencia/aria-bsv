"""Simple in-memory sliding-window rate limiter for the ARIA Portal."""

from __future__ import annotations

import time
from collections import defaultdict
from threading import Lock

_WINDOW_SECONDS = 60.0
_MAX_REQUESTS = 60  # per IP per window


class _RateLimiter:
    def __init__(self, max_requests: int = _MAX_REQUESTS, window: float = _WINDOW_SECONDS) -> None:
        self._max = max_requests
        self._window = window
        self._calls: dict[str, list[float]] = defaultdict(list)
        self._lock = Lock()

    def is_allowed(self, client_ip: str) -> bool:
        now = time.monotonic()
        cutoff = now - self._window
        with self._lock:
            timestamps = [t for t in self._calls[client_ip] if t > cutoff]
            if len(timestamps) >= self._max:
                self._calls[client_ip] = timestamps
                return False
            timestamps.append(now)
            self._calls[client_ip] = timestamps
            return True


_limiter = _RateLimiter()


def get_limiter() -> _RateLimiter:
    return _limiter
