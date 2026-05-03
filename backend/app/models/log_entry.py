from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class LogEntry(BaseModel):
    """
    Structured representation of a single parsed log line.

    Fields
    ------
    timestamp : ISO-8601 string (or syslog date) if one could be parsed.
    level     : Normalised log level (INFO, ERROR, DEBUG, …) or None.
    message   : The human-readable part of the log line.
    metadata  : Key-value pairs extracted from the message (e.g. user_id=42).
    """

    timestamp: str | None = None
    level: str | None = None
    message: str
    metadata: dict[str, Any] = {}
