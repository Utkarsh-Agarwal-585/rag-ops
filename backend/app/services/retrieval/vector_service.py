"""
FAISS vector store.

Maintains an in-memory FAISS index alongside a parallel list of chunk IDs
so we can map search results back to the memory store.
"""

from __future__ import annotations

import logging

import faiss
import numpy as np

from app.config import EMBEDDING_DIMENSION
from app.models.chunk import Chunk

logger = logging.getLogger(__name__)

# Process-scoped state.
_index: faiss.IndexFlatIP | None = None
_chunk_ids: list[str] = []


def _ensure_index() -> faiss.IndexFlatIP:
    """Create the FAISS index on first use."""
    global _index
    if _index is None:
        # Inner-product on L2-normalised vectors == cosine similarity.
        _index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
        logger.info("FAISS index created (dim=%d).", EMBEDDING_DIMENSION)
    return _index


def add_embeddings(chunks: list[Chunk]) -> int:
    """
    Add chunk embeddings to the FAISS index.

    Returns the number of vectors added.
    """
    idx = _ensure_index()
    vectors = []
    ids = []

    for c in chunks:
        if c.embedding is not None:
            vectors.append(c.embedding)
            ids.append(c.id)

    if not vectors:
        return 0

    matrix = np.array(vectors, dtype=np.float32)
    idx.add(matrix)
    _chunk_ids.extend(ids)

    logger.info("Added %d vector(s) to FAISS (total: %d).", len(ids), idx.ntotal)
    return len(ids)


def search(query_embedding: list[float], top_k: int = 20) -> list[tuple[str, float]]:
    """
    Search the FAISS index for the *top_k* nearest chunks.

    Returns a list of (chunk_id, score) tuples sorted by descending score.
    """
    idx = _ensure_index()
    if idx.ntotal == 0:
        return []

    q = np.array([query_embedding], dtype=np.float32)
    k = min(top_k, idx.ntotal)
    scores, indices = idx.search(q, k)

    results = []
    for score, i in zip(scores[0], indices[0]):
        if i < 0 or i >= len(_chunk_ids):
            continue
        results.append((_chunk_ids[i], float(score)))

    return results


def clear_index() -> None:
    """Reset the FAISS index.  Useful for tests."""
    global _index
    _index = None
    _chunk_ids.clear()
