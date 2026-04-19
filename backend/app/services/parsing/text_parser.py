from __future__ import annotations

import logging

from app.utils.text_utils import normalize_text

logger = logging.getLogger(__name__)


def parse_text(file_path: str) -> str:
    """
    Read a plain-text or log file and return normalised content.

    UTF-8 is assumed; invalid byte sequences are replaced rather than raising
    an exception so that slightly malformed files still get ingested.
    """
    with open(file_path, "r", encoding="utf-8", errors="replace") as fh:
        raw = fh.read()

    normalized = normalize_text(raw)
    logger.info(
        "Text file parsed: %d chars → %d chars after normalisation.",
        len(raw),
        len(normalized),
    )
    return normalized
