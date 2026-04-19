"""
Document chunking module.

Strategy
--------
Split the word list of a document into overlapping windows:

  [ w0 … w399 ]             chunk 0
       [ w350 … w749 ]      chunk 1   (50-word overlap with chunk 0)
              [ w700 … … ]  chunk 2   …

This ensures that sentences / concepts near a chunk boundary are present in
*both* adjacent chunks, reducing the chance that a retrieval query falls into
a gap between chunks.
"""

from __future__ import annotations

import logging
import uuid

from app.config import DOC_CHUNK_SIZE_WORDS, DOC_OVERLAP_WORDS
from app.models.chunk import Chunk

logger = logging.getLogger(__name__)


def chunk_document(text: str, source: str) -> list[Chunk]:
    """
    Split *text* into overlapping word-based chunks and return them as a list
    of Chunk objects.

    Parameters
    ----------
    text   : Normalised document text.
    source : Original filename (used as provenance in each Chunk).
    """
    words = text.split()
    if not words:
        return []

    step = DOC_CHUNK_SIZE_WORDS - DOC_OVERLAP_WORDS  # advance per iteration
    chunks: list[Chunk] = []

    for idx, start in enumerate(range(0, len(words), step)):
        end = min(start + DOC_CHUNK_SIZE_WORDS, len(words))
        chunk_words = words[start:end]

        chunks.append(
            Chunk(
                id=str(uuid.uuid4()),
                content=" ".join(chunk_words),
                type="doc",
                source=source,
                metadata={
                    "chunk_index": idx,
                    "word_count": len(chunk_words),
                    "start_word": start,
                    "end_word": end,
                },
            )
        )

        # Stop once the last word has been included
        if end == len(words):
            break

    logger.info(
        "Doc chunked '%s': %d words → %d chunks (size=%d, overlap=%d).",
        source,
        len(words),
        len(chunks),
        DOC_CHUNK_SIZE_WORDS,
        DOC_OVERLAP_WORDS,
    )
    return chunks
