"""
In-memory chunk store.

This module acts as a simple, process-scoped database.  It is intentionally
kept as a thin façade so that swapping it out for a real vector store (Chroma,
Pinecone, pgvector …) in a future iteration requires changing only this file.

Internal layout
---------------
_chunks : list[Chunk]
    Ordered list — preserves insertion order for paginated listing.

_index : dict[str, Chunk]
    ID-keyed dict for O(1) lookup by chunk ID.

_source_index : dict[str, set[str]]
    Maps base source filename → set of chunk IDs from that source.
    Used for fast cross-batch deduplication on re-upload: instead of scanning
    the full _chunks list, we look up the source key in O(1) and get back all
    chunk IDs that belong to it.
"""

from __future__ import annotations

from app.models.chunk import Chunk

_chunks: list[Chunk] = []
_index: dict[str, Chunk] = {}
_source_index: dict[str, set[str]] = {}  # base_source → {chunk_id, ...}


def _base_source(source: str) -> str:
    """
    Strip the '#page=N' suffix from a chunk source string.

    Image chunks carry a source like 'document.pdf#page=6' while text chunks
    use 'document.pdf'.  Normalising to the base filename lets both share the
    same _source_index key so a single source_exists() / remove_chunks_by_source()
    call covers all chunk types from the same document.
    """
    return source.split("#")[0]


# ---------------------------------------------------------------------------
# Write operations
# ---------------------------------------------------------------------------

def store_chunks(chunks: list[Chunk]) -> None:
    """
    Append *chunks* to the in-memory store.

    Updates all three internal structures:
    - _chunks  : ordered list for pagination
    - _index   : ID → Chunk dict for O(1) lookup
    - _source_index : base_source → {chunk_id} for fast eviction on re-upload
    """
    for chunk in chunks:
        _chunks.append(chunk)
        _index[chunk.id] = chunk
        base = _base_source(chunk.source)
        _source_index.setdefault(base, set()).add(chunk.id)


def remove_chunks_by_source(source: str) -> list[str]:
    """
    Remove all chunks whose base source matches *source*.

    Algorithm:
    1. Look up the source key in _source_index (O(1)).
    2. Rebuild _chunks without the removed IDs (O(n) but preserves order).
    3. Delete each ID from _index (O(k) where k = removed count).

    Returns the list of removed chunk IDs so callers can also clean up
    the FAISS and BM25 indexes.
    """
    base = _base_source(source)
    ids_to_remove = _source_index.pop(base, set())
    if not ids_to_remove:
        return []

    # Rebuild _chunks list without the removed IDs (preserves insertion order).
    global _chunks
    _chunks = [c for c in _chunks if c.id not in ids_to_remove]

    for cid in ids_to_remove:
        _index.pop(cid, None)

    return list(ids_to_remove)


def source_exists(source: str) -> bool:
    """Return True if any chunks from *source* are already in the store."""
    return _base_source(source) in _source_index


# ---------------------------------------------------------------------------
# Read operations
# ---------------------------------------------------------------------------

def get_chunk_by_id(chunk_id: str) -> Chunk | None:
    """Return the chunk with the given ID, or None if it does not exist."""
    return _index.get(chunk_id)


def get_all_chunks() -> list[Chunk]:
    """Return all stored chunks in insertion order."""
    return list(_chunks)


def get_chunks_by_source(source: str) -> list[Chunk]:
    """
    Return all chunks that originate from *source*.

    Uses an exact match on the full source string (including '#page=N' suffix
    for image chunks).  To match all chunks from a document regardless of type,
    use get_all_chunks() and filter by _base_source().
    """
    return [c for c in _chunks if c.source == source]


# ---------------------------------------------------------------------------
# Maintenance
# ---------------------------------------------------------------------------

def clear_store() -> None:
    """Wipe all stored chunks and reset all indexes.  Useful for unit tests."""
    _chunks.clear()
    _index.clear()
    _source_index.clear()


def get_store_stats() -> dict:
    """Return aggregate statistics about the current store contents."""
    return {
        "total_chunks": len(_chunks),
        "doc_chunks": sum(1 for c in _chunks if c.type == "doc"),
        "log_chunks": sum(1 for c in _chunks if c.type == "log"),
        "image_chunks": sum(1 for c in _chunks if c.type == "image"),
        "sources": sorted({c.source for c in _chunks}),
    }
