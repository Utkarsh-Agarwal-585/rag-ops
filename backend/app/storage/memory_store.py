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
"""

from __future__ import annotations

from app.models.chunk import Chunk

_chunks: list[Chunk] = []
_index: dict[str, Chunk] = {}


# ---------------------------------------------------------------------------
# Write operations
# ---------------------------------------------------------------------------

def store_chunks(chunks: list[Chunk]) -> None:
    """Append *chunks* to the in-memory store."""
    for chunk in chunks:
        _chunks.append(chunk)
        _index[chunk.id] = chunk


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
    """Return all chunks that originate from *source* (exact filename match)."""
    return [c for c in _chunks if c.source == source]


# ---------------------------------------------------------------------------
# Maintenance
# ---------------------------------------------------------------------------

def clear_store() -> None:
    """Wipe all stored chunks.  Useful for unit tests."""
    _chunks.clear()
    _index.clear()


def get_store_stats() -> dict:
    """Return aggregate statistics about the current store contents."""
    return {
        "total_chunks": len(_chunks),
        "doc_chunks": sum(1 for c in _chunks if c.type == "doc"),
        "log_chunks": sum(1 for c in _chunks if c.type == "log"),
        "image_chunks": sum(1 for c in _chunks if c.type == "image"),
        "sources": sorted({c.source for c in _chunks}),
    }
