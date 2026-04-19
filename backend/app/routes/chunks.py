"""
Chunks inspection routes — GET /api/v1/chunks  &  GET /api/v1/chunks/stats

These endpoints expose the in-memory chunk store for debugging and validation.
They are not part of the retrieval (query) flow — that will be added in a
future iteration once embeddings and a vector store are integrated.
"""

from __future__ import annotations

from fastapi import APIRouter, Query

from app.storage.memory_store import get_all_chunks, get_chunks_by_source, get_store_stats

router = APIRouter()


@router.get(
    "/chunks",
    summary="List stored chunks",
    response_description="Paginated list of chunks",
)
async def list_chunks(
    source: str | None = Query(
        None,
        description="Filter by source filename (exact match)",
    ),
    limit: int = Query(100, ge=1, le=1000, description="Maximum chunks to return"),
    offset: int = Query(0, ge=0, description="Number of chunks to skip"),
) -> dict:
    """
    Return a paginated list of all chunks currently held in memory.

    Optionally filter by the *source* filename that was used during upload.
    """
    all_chunks = get_chunks_by_source(source) if source else get_all_chunks()
    page = all_chunks[offset : offset + limit]

    return {
        "total": len(all_chunks),
        "offset": offset,
        "limit": limit,
        "chunks": [c.model_dump() for c in page],
    }


@router.get(
    "/chunks/stats",
    summary="Store statistics",
    response_description="Aggregate counts and source list",
)
async def store_stats() -> dict:
    """Return aggregate statistics about the current in-memory chunk store."""
    return get_store_stats()


@router.get(
    "/chunks/index-stats",
    summary="Retrieval index statistics",
    response_description="FAISS and BM25 index state",
)
async def index_stats() -> dict:
    """
    Return the current state of the FAISS vector index and BM25 keyword index.

    Useful for debugging whether embeddings were generated and indexed
    after file uploads.
    """
    from app.services.retrieval import vector_service, bm25_service

    faiss_index = vector_service._index
    faiss_total = faiss_index.ntotal if faiss_index is not None else 0
    faiss_dim = faiss_index.d if faiss_index is not None else 0

    bm25_corpus_size = len(bm25_service._corpus_ids)
    bm25_ready = bm25_service._bm25 is not None

    return {
        "faiss": {
            "initialized": faiss_index is not None,
            "total_vectors": faiss_total,
            "dimension": faiss_dim,
            "chunk_ids_tracked": len(vector_service._chunk_ids),
        },
        "bm25": {
            "initialized": bm25_ready,
            "corpus_size": bm25_corpus_size,
        },
    }


@router.get(
    "/chunks/cache-stats",
    summary="Query cache statistics",
    response_description="Cache hit/miss and TTL info",
)
async def cache_stats() -> dict:
    """Return statistics about the in-memory query response cache."""
    from app.services.retrieval.cache_service import get_cache_stats
    return get_cache_stats()
