"""
Log parsing module.

Responsibilities
----------------
* Detect whether a file's content looks like a log file (heuristic).
* Parse individual log lines into structured LogEntry objects, supporting:
    - ISO-8601 / RFC-3339 datetime + level + message
    - Python logging format  (timestamp - module - LEVEL - message)
    - Syslog format          (Mon DD HH:MM:SS hostname process: message)
    - Bare level prefix      (ERROR: message)
    - JSON-structured logs   ({"level": "...", "message": "...", ...})
    - Unrecognised lines     (stored as plain messages)
* Handle multi-line entries (e.g. Java stack traces) by merging continuation
  lines into the preceding entry.
"""

from __future__ import annotations

import json
import re

from app.models.log_entry import LogEntry

# ---------------------------------------------------------------------------
# Shared level alternation used across multiple patterns
# ---------------------------------------------------------------------------
_LEVELS = r"TRACE|DEBUG|INFO|NOTICE|WARNING|WARN|ERROR|CRITICAL|FATAL|SEVERE"

# ---------------------------------------------------------------------------
# Compiled regex patterns — ordered from most-specific to least-specific
# ---------------------------------------------------------------------------
_PATTERNS: list[re.Pattern[str]] = [
    # ISO-8601 / space-separated datetime  +  optional bracket  +  level  +  message
    # Matches:
    #   "2024-01-15 10:30:45,123 INFO  User logged in"
    #   "2024-01-15T10:30:45Z [ERROR] Disk full"
    re.compile(
        r"^(?P<timestamp>"
        r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?"
        r"(?:Z|[+-]\d{2}:?\d{2})?"
        r")"
        r"\s+\[?(?P<level>" + _LEVELS + r")\]?"
        r"\s+(?P<message>.+)$",
        re.IGNORECASE,
    ),
    # Python logging: "2024-01-15 10:30:45,123 - module.name - ERROR - msg"
    re.compile(
        r"^(?P<timestamp>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:[.,]\d+)?)"
        r"\s+-\s+[\w.]+"
        r"\s+-\s+(?P<level>" + _LEVELS + r")"
        r"\s+-\s+(?P<message>.+)$",
        re.IGNORECASE,
    ),
    # Syslog: "Jan 15 10:30:45 hostname process[pid]: message"
    re.compile(
        r"^(?P<timestamp>"
        r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}"
        r")"
        r"\s+\S+"       # hostname
        r"\s+\S+:"      # process[pid]:
        r"\s+(?P<message>.+)$",
        re.IGNORECASE,
    ),
    # Bare level prefix: "ERROR: something" or "WARN something happened"
    re.compile(
        r"^(?P<level>" + _LEVELS + r")[:\s]+(?P<message>.+)$",
        re.IGNORECASE,
    ),
]


# ---------------------------------------------------------------------------
# Helper: extract key=value pairs from a log message
# ---------------------------------------------------------------------------

def _extract_kv_pairs(text: str) -> dict[str, str]:
    """
    Pull structured key=value or key="value" pairs out of a free-text message.

    Example:  'user_id=42 action="login" status=ok'
    Returns:  {'user_id': '42', 'action': 'login', 'status': 'ok'}
    """
    pattern = re.compile(r'(\w+)=(?:"([^"]*)"|\'([^\']*)\'|(\S+))')
    result: dict[str, str] = {}
    for m in pattern.finditer(text):
        key = m.group(1)
        # Take whichever capture group matched (double-quoted, single-quoted, or bare)
        value = m.group(2) or m.group(3) or m.group(4) or ""
        result[key] = value
    return result


# ---------------------------------------------------------------------------
# Core: parse a single log line
# ---------------------------------------------------------------------------

def _parse_line(line: str) -> LogEntry | None:
    """
    Attempt to parse *line* into a LogEntry.  Returns None for blank lines.

    Tries, in order:
      1. JSON object
      2. Known regex patterns
      3. Fallback plain message
    """
    line = line.strip()
    if not line:
        return None

    # ── 1. JSON log line ────────────────────────────────────────────────────
    if line.startswith("{"):
        try:
            data: dict = json.loads(line)
            return LogEntry(
                timestamp=(
                    data.get("timestamp")
                    or data.get("time")
                    or data.get("@timestamp")
                ),
                level=(
                    (data.get("level") or data.get("severity") or "").upper() or None
                ),
                message=str(data.get("message") or data.get("msg") or data),
                metadata={
                    k: v
                    for k, v in data.items()
                    if k
                    not in {
                        "timestamp",
                        "time",
                        "@timestamp",
                        "level",
                        "severity",
                        "message",
                        "msg",
                    }
                },
            )
        except json.JSONDecodeError:
            pass  # fall through to regex matching

    # ── 2. Regex patterns ───────────────────────────────────────────────────
    for pattern in _PATTERNS:
        m = pattern.match(line)
        if m:
            groups = m.groupdict()
            msg = groups.get("message", line).strip()
            return LogEntry(
                timestamp=groups.get("timestamp"),
                level=(groups.get("level") or "").upper() or None,
                message=msg,
                metadata=_extract_kv_pairs(msg),
            )

    # ── 3. Unrecognised — store as plain message ─────────────────────────────
    return LogEntry(message=line, metadata=_extract_kv_pairs(line))


# ---------------------------------------------------------------------------
# Public: parse full log text
# ---------------------------------------------------------------------------

def parse_log_text(text: str) -> list[LogEntry]:
    """
    Parse an entire log file into a list of LogEntry objects.

    Multi-line entries (Java stack traces, Python tracebacks, etc.) are handled
    by buffering lines and flushing the buffer whenever a new log-entry header
    is detected.  Continuation lines are appended to the current entry's message
    so that the full stack trace is preserved.
    """
    entries: list[LogEntry] = []
    pending: list[str] = []

    def _flush() -> None:
        if not pending:
            return
        entry = _parse_line("\n".join(pending))
        if entry:
            entries.append(entry)
        pending.clear()

    for raw_line in text.splitlines():
        stripped = raw_line.strip()

        if not stripped:
            # Blank line — treat as entry separator
            _flush()
            continue

        is_new_entry = stripped.startswith("{") or any(
            p.match(stripped) for p in _PATTERNS
        )

        if is_new_entry and pending:
            _flush()

        pending.append(raw_line)

    _flush()
    return entries


# ---------------------------------------------------------------------------
# Public: heuristic log-file detector
# ---------------------------------------------------------------------------

def is_log_content(filename: str, content: str) -> bool:
    """
    Decide whether *content* looks like a log file.

    A file is classified as a log when either:
      * Its extension is ".log", OR
      * At least 30 % of its first 20 non-empty lines match a known pattern.
    """
    if filename.lower().endswith(".log"):
        return True

    sample = [line for line in content.splitlines() if line.strip()][:20]
    if not sample:
        return False

    matched = sum(
        1
        for line in sample
        if line.strip().startswith("{")
        or any(p.match(line.strip()) for p in _PATTERNS)
    )
    return (matched / len(sample)) >= 0.3
