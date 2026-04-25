"""
Log chunking module.

Strategy
--------
Raw log entries are rarely useful one-by-one for a RAG system.  Instead we
build *summaries* that answer questions like:

  "How many VISA retry_failed errors occurred in the last 10 minutes?"

Grouping algorithm
------------------
1. Compute a **message signature** for every entry by stripping dynamic tokens
   (numbers, UUIDs, IPs, quoted strings) so that semantically identical messages
   that differ only in IDs / values map to the same key.

2. Cluster entries by  (level, message_signature).

3. Within each cluster, if timestamps are available, further subdivide into
   fixed-width time windows (default: 10 minutes).

4. Each resulting bucket becomes one Chunk whose *content* is a human-readable
   summary:

     [ERROR] 120 log entries between 2024-01-15 10:00:00 and 2024-01-15 10:10:00
     Most common: "retry_failed for VISA" (87 occurrences)
     Other patterns: retry_failed for MASTERCARD; retry_failed for AMEX
     Metadata fields: card_type, merchant_id, attempt
"""

from __future__ import annotations

import re
import uuid
from collections import defaultdict
from datetime import datetime, timedelta

from app.config import LOG_TIME_WINDOW_MINUTES
from app.models.chunk import Chunk
from app.models.log_entry import LogEntry

try:
    from dateutil import parser as _dateutil_parser

    def _parse_ts(ts_str: str | None) -> datetime | None:
        if not ts_str:
            return None
        try:
            return _dateutil_parser.parse(ts_str)
        except Exception:
            return None

except ImportError:
    # Minimal fallback when python-dateutil is absent
    _FALLBACK_FMTS = (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S,%f",
        "%Y-%m-%dT%H:%M:%SZ",
    )

    def _parse_ts(ts_str: str | None) -> datetime | None:  # type: ignore[misc]
        if not ts_str:
            return None
        for fmt in _FALLBACK_FMTS:
            try:
                return datetime.strptime(ts_str[: len(fmt)], fmt)
            except ValueError:
                continue
        return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _message_signature(message: str) -> str:
    """
    Derive a stable, normalised grouping key from *message* by replacing
    high-cardinality tokens with generic placeholders.

    This ensures that these two lines hash to the same key:
      "retry_failed for VISA user_id=1234 attempt=3"
      "retry_failed for VISA user_id=5678 attempt=7"
    """
    sig = message
    # UUID / GUID
    sig = re.sub(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
        "<uuid>",
        sig,
        flags=re.IGNORECASE,
    )
    # IPv4 addresses
    sig = re.sub(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", "<ip>", sig)
    # Quoted strings
    sig = re.sub(r'"[^"]*"', "<str>", sig)
    sig = re.sub(r"'[^']*'", "<str>", sig)
    # Standalone numbers (including hex)
    sig = re.sub(r"\b(?:0x[0-9a-fA-F]+|\d+)\b", "<n>", sig)
    return sig.strip().lower()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def chunk_logs(entries: list[LogEntry], source: str) -> list[Chunk]:
    """
    Convert a list of structured LogEntry objects into summarised Chunks.

    Parameters
    ----------
    entries : Parsed log entries (from log_parser.parse_log_text).
    source  : Original filename for provenance tracking.
    """
    if not entries:
        return []

    # Step 1 — group by (level, normalised message signature)
    groups: dict[str, list[LogEntry]] = defaultdict(list)
    for entry in entries:
        level = (entry.level or "UNKNOWN").upper()
        sig = _message_signature(entry.message)
        groups[f"{level}::{sig}"].append(entry)

    # Step 2 — convert each group into one-or-more time-windowed chunks
    chunks: list[Chunk] = []
    for group_key, group_entries in groups.items():
        level = group_key.split("::", 1)[0]
        chunks.extend(_entries_to_chunks(group_entries, level, source))

    return chunks


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _entries_to_chunks(
    entries: list[LogEntry],
    level: str,
    source: str,
) -> list[Chunk]:
    """
    Split a single message-group into time-window sub-chunks.
    If no timestamps are available the whole group becomes one chunk.
    """
    timestamped = [(e, _parse_ts(e.timestamp)) for e in entries]
    has_ts = any(ts is not None for _, ts in timestamped)

    if not has_ts:
        return [_build_chunk(entries, level, source, window=None)]

    # Sort entries that have parseable timestamps
    with_ts: list[tuple[LogEntry, datetime]] = sorted(
        [(e, ts) for e, ts in timestamped if ts is not None],
        key=lambda x: x[1],
    )
    without_ts: list[LogEntry] = [e for e, ts in timestamped if ts is None]

    window_size = timedelta(minutes=LOG_TIME_WINDOW_MINUTES)
    window_start: datetime = with_ts[0][1]
    current_bucket: list[LogEntry] = []
    chunks: list[Chunk] = []

    for entry, ts in with_ts:
        if ts >= window_start + window_size:
            # Current window is complete — flush it
            chunks.append(
                _build_chunk(
                    current_bucket,
                    level,
                    source,
                    window=(window_start, ts - timedelta(seconds=1)),
                )
            )
            window_start = ts
            current_bucket = []
        current_bucket.append(entry)

    # Flush the last (possibly partial) window
    if current_bucket:
        chunks.append(
            _build_chunk(
                current_bucket,
                level,
                source,
                window=(window_start, with_ts[-1][1]),
            )
        )

    # Attach timestamp-less entries to the last chunk to avoid losing them
    if without_ts:
        if chunks:
            last = chunks[-1]
            extra = _build_chunk(without_ts, level, source, window=None)
            chunks[-1] = Chunk(
                id=last.id,
                content=last.content + "\n" + extra.content,
                type="log",
                source=source,
                metadata={**last.metadata, "has_untimed_entries": True},
            )
        else:
            chunks.append(_build_chunk(without_ts, level, source, window=None))

    return chunks


def _build_chunk(
    entries: list[LogEntry],
    level: str,
    source: str,
    window: tuple[datetime, datetime] | None,
) -> Chunk:
    """
    Summarise a bucket of related log entries into a single human-readable Chunk.

    Example content
    ---------------
    [ERROR] 120 log entries between 2024-01-15 10:00:00 and 2024-01-15 10:10:00
    Most common: "retry_failed for VISA" (87 occurrences)
    Other patterns: retry_failed for MASTERCARD; retry_failed for AMEX
    Metadata fields: attempt, card_type, merchant_id
    """
    count = len(entries)

    # Frequency table of exact messages
    freq: dict[str, int] = defaultdict(int)
    for e in entries:
        freq[e.message] += 1
    top_msg, top_count = max(freq.items(), key=lambda x: x[1])

    # Up to 4 other distinct messages for context
    others = [m for m in list(freq) if m != top_msg][:4]

    # Human-readable time description
    if window:
        fmt = "%Y-%m-%d %H:%M:%S"
        time_desc = f"between {window[0].strftime(fmt)} and {window[1].strftime(fmt)}"
    else:
        time_desc = "(no timestamp)"

    lines = [
        f"[{level}] {count} log entr{'y' if count == 1 else 'ies'} {time_desc}",
        f'Most common: "{top_msg}" ({top_count} occurrence{"s" if top_count != 1 else ""})',
    ]
    if others:
        lines.append(f"Other patterns: {'; '.join(others)}")

    # Surface any structured metadata keys present across entries
    meta_keys = sorted({k for e in entries for k in e.metadata})
    if meta_keys:
        lines.append(f"Metadata fields: {', '.join(meta_keys)}")

    chunk_meta: dict = {
        "level": level,
        "count": count,
        "top_message": top_msg,
        "unique_messages": len(freq),
    }
    if window:
        chunk_meta["window_start"] = window[0].isoformat()
        chunk_meta["window_end"] = window[1].isoformat()

    return Chunk(
        id=str(uuid.uuid4()),
        content="\n".join(lines),
        type="log",
        source=source,
        metadata=chunk_meta,
    )
