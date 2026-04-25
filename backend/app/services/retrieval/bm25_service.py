"""
BM25 keyword search service.

Builds a BM25 index over chunk content for sparse retrieval.
The index is rebuilt whenever new chunks are added.
"""

from __future__ import annotations

import logging
import re

from rank_bm25 import BM25Okapi

from app.models.chunk import Chunk

logger = logging.getLogger(__name__)

_bm25: BM25Okapi | None = None
_corpus_ids: list[str] = []
_corpus_tokens: list[list[str]] = []


def _tokenize(text: str) -> list[str]:
    """Lowercase tokenizer that strips punctuation so 'concurrency?' matches 'concurrency'."""
    return re.findall(r'\b\w+\b', text.lower())


def build_index(chunks: list[Chunk]) -> int:
    """
    (Re)build the BM25 index from the given chunks.

    Returns the corpus size.
    """
    global _bm25, _corpus_ids, _corpus_tokens

    _corpus_ids.clear()
    _corpus_tokens.clear()

    for c in chunks:
        _corpus_ids.append(c.id)
        _corpus_tokens.append(_tokenize(c.content))

    if _corpus_tokens:
        _bm25 = BM25Okapi(_corpus_tokens)
    else:
        _bm25 = None

    logger.info("BM25 index built with %d document(s).", len(_corpus_ids))
    return len(_corpus_ids)


def search(query: str, top_k: int = 20) -> list[tuple[str, float]]:
    """
    Search the BM25 index.

    Returns a list of (chunk_id, score) tuples sorted by descending score.
    """
    if _bm25 is None or not _corpus_ids:
        return []

    tokens = _tokenize(query)
    scores = _bm25.get_scores(tokens)

    # Pair with IDs and sort descending.
    paired = list(zip(_corpus_ids, scores))
    paired.sort(key=lambda x: x[1], reverse=True)

    return [(cid, float(s)) for cid, s in paired[:top_k]]


def clear_index() -> None:
    """Reset the BM25 index."""
    global _bm25
    _bm25 = None
    _corpus_ids.clear()
    _corpus_tokens.clear()
