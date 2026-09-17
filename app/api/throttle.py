"""A ceiling on how often one caller can ask for an inference.

Counted in this process, in memory. That is a real limitation and it is stated
rather than hidden: run two uvicorn workers and each keeps its own count, so the
effective limit doubles. It is still worth having - it stops one misbehaving
client from occupying the model with a loop - and for anything stricter the
right place is the reverse proxy, which sees every worker. The README's nginx
example is where that belongs.

A fixed window rather than a token bucket: the thing being protected is a
several-second CPU-bound inference, so the precision of a bucket buys nothing
and a dict of two integers cannot leak.
"""

from __future__ import annotations

import time
from collections import defaultdict
from threading import Lock

from app.core.errors import AppError


class RateLimitedError(AppError):
    code = "rate_limited"
    status_code = 429


class FixedWindowLimiter:
    """Counts requests per caller inside one wall-clock minute."""

    def __init__(self, limit: int, window_seconds: int = 60) -> None:
        self._limit = limit
        self._window = window_seconds
        self._lock = Lock()
        #: caller -> (window start, count so far)
        self._hits: dict[str, tuple[float, int]] = defaultdict(lambda: (0.0, 0))

    def hit(self, caller: str) -> None:
        """Record one request, or refuse it.

        Raises:
            RateLimitedError: when this caller is over the limit for this window.
        """
        if self._limit <= 0:
            return

        now = time.monotonic()

        with self._lock:
            started, count = self._hits[caller]

            if now - started >= self._window:
                self._hits[caller] = (now, 1)
                return

            if count >= self._limit:
                retry_after = int(self._window - (now - started)) + 1
                raise RateLimitedError(
                    f"Too many requests. Try again in {retry_after}s.",
                    details={"retry_after_seconds": retry_after, "limit": self._limit},
                )

            self._hits[caller] = (started, count + 1)

    def reset(self) -> None:
        """Forget every count. For tests, and for a key that was just rotated."""
        with self._lock:
            self._hits.clear()
