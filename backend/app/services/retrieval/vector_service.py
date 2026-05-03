"""
FAISS vector store.

Maintains an in-memory FAISS IndexFlatIP alongside a parallel list of chunk IDs
so search results can be mapped back to the memory store.

Why IndexFlatIP?
    Inner product on L2-normalised vectors is mathematically equivalent to
    cosine similarity.  All embeddings from embedding_service.py are already
    L2-normalised (normalize_embeddings=True), so scores returned by search()
    are cosine similarities in [0, 1].
"""

from __future__ import annotations

import logging

import faiss
import numpy as np

from app.config import EMBEDDING_DIMENSION
from app.models.chunk import Chunk

logger = logging.getLogger(__name__)

# Process-scoped state — reset on server restart (persistence_service handles disk I/O).
_index: faiss.IndexFlatIP | None = None
_chunk_ids: list[str] = []  # parallel list: _chunk_ids[i] is the chunk ID for FAISS row i


def _ensure_index() -> faiss.IndexFlatIP:
    """
    Lazily create the FAISS index on first use.

    Using lazy initialisation means the app starts instantly even if no
    documents have been uploaded yet.
    """
    global _index
    if _index is None:
        _index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
        logger.info("FAISS index created (dim=%d).", EMBEDDING_DIMENSION)
    return _index


def add_embeddings(chunks: list[Chunk]) -> int:
    """
    Add chunk embeddings to the FAISS index.

    Only chunks that have a non-None embedding are indexed.  Chunks without
    embeddings (e.g. if the embedding step failed) are silently skipped.

    Returns the number of vectors actually added.
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

    The query vector must be L2-normalised (same as the stored vectors) so
    that inner product equals cosine similarity.

    Returns a list of (chunk_id, cosine_score) tuples sorted by descending score.
    Invalid FAISS indices (value -1) are filtered out.
    """
    idx = _ensure_index()
    if idx.ntotal == 0:
        return []

    q = np.array([query_embedding], dtype=np.float32)
    k = min(top_k, idx.ntotal)
    scores, indices = idx.search(q, k)

    results = []
    for score, i in zip(scores[0], indices[0]):
        # FAISS returns -1 for padding when fewer than k results exist.
        if i < 0 or i >= len(_chunk_ids):
            continue
        results.append((_chunk_ids[i], float(score)))

    return results


def remove_ids(ids_to_remove: set[str]) -> int:
    """
    Remove vectors for the given chunk IDs from the FAISS index.

    Algorithm:
        IndexFlatIP does not support in-place deletion.  Instead we:
        1. Identify which row positions to keep (O(n) scan of _chunk_ids).
        2. Read all vectors back out with reconstruct_n() (O(n)).
        3. Build a new index with only the kept vectors (O(k)).
        4. Replace the global _index and _chunk_ids.

    This is O(n) in the total number of indexed vectors, which is acceptable
    for the in-memory MVP scale (< 10,000 chunks).

    Returns the number of vectors removed.
    """
    global _index, _chunk_ids

    if not ids_to_remove or not _chunk_ids:
        return 0

    idx = _ensure_index()
    if idx.ntotal == 0:
        return 0

    # Identify which positions to keep.
    keep_positions = [i for i, cid in enumerate(_chunk_ids) if cid not in ids_to_remove]
    removed_count = len(_chunk_ids) - len(keep_positions)

    if removed_count == 0:
        return 0

    # Read all vectors, filter to kept positions, rebuild index.
    all_vectors = idx.reconstruct_n(0, idx.ntotal)  # shape: (ntotal, dim)
    kept_vectors = all_vectors[keep_positions]
    kept_ids = [_chunk_ids[i] for i in keep_positions]

    _index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
    if len(kept_vectors) > 0:
        _index.add(kept_vectors)

    _chunk_ids.clear()
    _chunk_ids.extend(kept_ids)

    logger.info("FAISS: removed %d vector(s), %d remaining.", removed_count, _index.ntotal)
    return removed_count


def clear_index() -> None:
    """Reset the FAISS index and chunk ID list.  Useful for unit tests."""
    global _index
    _index = None
    _chunk_ids.clear()
