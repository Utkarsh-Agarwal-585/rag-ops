"""
Structured JSON logging configuration.

Sets up a JSON formatter for all log records so every line is machine-parseable.
Each log record automatically includes the correlation_id from the current
request context (via a contextvars.ContextVar), making it trivial to filter
all logs for a single request in any log aggregation tool.

JSON log line example:
{
  "timestamp": "2025-05-03T12:34:56.789Z",
  "level": "INFO",
  "logger": "app.services.ingestion.ingestor",
  "message": "Ingestion complete for 'system-design.pdf': 132 text chunk(s)",
  "correlation_id": "req_a3f2b1c9_1746268496_0042",
  "module": "ingestor",
  "line": 87
}
"""

from __future__ import annotations

import json
import logging
import sys
import time
from contextvars import ContextVar

# ---------------------------------------------------------------------------
# Correlation ID context — set per-request by the middleware
# ---------------------------------------------------------------------------

# Holds the correlation ID for the currently executing async task.
# ContextVar is safe for concurrent async requests — each request gets its
# own copy of the variable without interfering with others.
correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="-")


# ---------------------------------------------------------------------------
# JSON log formatter
# ---------------------------------------------------------------------------

class JsonFormatter(logging.Formatter):
    """
    Formats log records as single-line JSON objects.

    Fields emitted:
        timestamp      — ISO-8601 UTC
        level          — DEBUG / INFO / WARNING / ERROR / CRITICAL
        logger         — dotted logger name (e.g. app.routes.query)
        message        — the formatted log message
        correlation_id — request correlation ID from context var
        module         — source file stem
        line           — line number
        exc_info       — exception traceback (only when an exception is attached)
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "timestamp": self._utc_iso(record.created),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": correlation_id_var.get("-"),
            "module": record.module,
            "line": record.lineno,
        }

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False)

    @staticmethod
    def _utc_iso(created: float) -> str:
        """Convert a log record's created timestamp to ISO-8601 UTC string."""
        t = time.gmtime(created)
        ms = int((created % 1) * 1000)
        return (
            f"{t.tm_year:04d}-{t.tm_mon:02d}-{t.tm_mday:02d}T"
            f"{t.tm_hour:02d}:{t.tm_min:02d}:{t.tm_sec:02d}.{ms:03d}Z"
        )


# ---------------------------------------------------------------------------
# Setup function — call once at application startup
# ---------------------------------------------------------------------------

def configure_logging(level: str = "INFO") -> None:
    """
    Replace the root logger's handlers with a single JSON stdout handler.

    Call this before creating the FastAPI app so all loggers (including
    uvicorn's) emit structured JSON.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Silence noisy third-party loggers that aren't useful in production.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("faiss").setLevel(logging.WARNING)
