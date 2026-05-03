"""
Correlation ID middleware.

Generates or propagates an `x-correlation-id` header on every request and
injects it into the logging context so every log line emitted during that
request carries the same ID.

ID format:  req_{8-char hex}_{unix_seconds}_{4-digit counter}
Example:    req_a3f2b1c9_1746268496_0042

The 8-char hex comes from os.urandom(4) — cryptographically random, not
sequential, so IDs from different server instances don't collide.
The unix timestamp makes IDs roughly sortable by time.
The counter disambiguates multiple requests arriving in the same second.

If the client sends an `x-correlation-id` header (e.g. from a frontend that
already generated one), that value is used as-is so the ID is consistent
across the full request chain.
"""

from __future__ import annotations

import itertools
import os
import time

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.logging_config import correlation_id_var

_counter = itertools.count(1)
_HEADER = "x-correlation-id"


def _generate_id() -> str:
    """Generate a unique, human-readable correlation ID."""
    rand_hex = os.urandom(4).hex()          # 8 hex chars, cryptographically random
    ts = int(time.time())                   # unix seconds — sortable
    seq = next(_counter) % 10_000          # 4-digit counter, wraps at 9999
    return f"req_{rand_hex}_{ts}_{seq:04d}"


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """
    Attach a correlation ID to every request/response.

    1. Read `x-correlation-id` from the incoming request headers.
       If absent, generate a new one.
    2. Set the ID in the `correlation_id_var` ContextVar so all log records
       emitted during this request automatically include it.
    3. Add the ID to the response headers so the client can reference it
       when reporting errors.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        # Honour a client-supplied ID (e.g. from a frontend or API gateway).
        cid = request.headers.get(_HEADER) or _generate_id()

        # Inject into the async context — visible to all loggers in this task.
        token = correlation_id_var.set(cid)
        try:
            response: Response = await call_next(request)
        finally:
            # Always restore the context var, even on exception.
            correlation_id_var.reset(token)

        # Echo the ID back in the response so clients can correlate errors.
        response.headers[_HEADER] = cid
        return response
