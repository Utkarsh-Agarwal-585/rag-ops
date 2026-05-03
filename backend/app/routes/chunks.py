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


@router.get(
    "/documents",
    summary="List all uploaded documents",
    response_description="Unique documents with chunk counts",
)
async def list_documents() -> dict:
    """
    Return a deduplicated list of all documents currently in the store,
    with per-document chunk counts broken down by type.

    Each entry uses the base filename (without '#page=N' suffix) so image
    chunks and text chunks from the same PDF are grouped together.
    """
    from app.storage.memory_store import get_all_chunks

    all_chunks = get_all_chunks()

    # Aggregate per base-source document.
    docs: dict[str, dict] = {}
    for chunk in all_chunks:
        base = chunk.source.split("#")[0]
        if base not in docs:
            docs[base] = {"name": base, "doc": 0, "log": 0, "image": 0, "total": 0}
        docs[base][chunk.type] += 1
        docs[base]["total"] += 1

    return {
        "total_documents": len(docs),
        "documents": sorted(docs.values(), key=lambda d: d["name"]),
    }


@router.delete(
    "/documents/{filename}",
    summary="Delete a document and all its associated data",
    response_description="Deletion summary with per-step status",
)
async def delete_document(filename: str) -> dict:
    """
    Delete a document from the store, FAISS index, BM25 index, and disk.

    Steps (each is attempted independently — a failure in one does not
    prevent the remaining steps from running):
      1. Remove chunks from memory store
      2. Remove vectors from FAISS
      3. Rebuild BM25 over remaining chunks
      4. Delete image subdirectory from disk (images + caption sidecars)
      5. Persist updated state to disk

    Returns a structured response with per-step results and any warnings,
    so the caller knows exactly what succeeded and what did not.
    """
    from app.storage.memory_store import (
        get_all_chunks,
        remove_chunks_by_source,
        source_exists,
    )
    from app.services.retrieval.vector_service import remove_ids
    from app.services.retrieval.bm25_service import build_index
    from app.services.retrieval.persistence_service import save_all
    from app.services.parsing.image_extractor import _safe_stem
    from app.config import IMAGES_DIR
    import shutil
    import os

    if not source_exists(filename):
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Document '{filename}' not found.")

    warnings: list[str] = []

    # ── Step 1: Remove chunks from memory store ──────────────────────────────
    removed_ids = remove_chunks_by_source(filename)
    chunks_removed = len(removed_ids)

    # ── Step 2: Remove vectors from FAISS ───────────────────────────────────
    faiss_removed = 0
    try:
        faiss_removed = remove_ids(set(removed_ids))
    except Exception as exc:
        warnings.append(f"FAISS cleanup partial: {exc}")

    # ── Step 3: Rebuild BM25 over remaining chunks ───────────────────────────
    try:
        build_index(get_all_chunks())
    except Exception as exc:
        warnings.append(f"BM25 rebuild failed: {exc}")

    # ── Step 4: Delete image subdirectory from disk ──────────────────────────
    # Images live under storage/images/<stem>/ — same stem used during upload.
    images_deleted = 0
    stem = _safe_stem(filename)
    doc_image_dir = os.path.join(IMAGES_DIR, stem)

    if os.path.isdir(doc_image_dir):
        try:
            # Count files before deletion for the response summary.
            images_deleted = sum(
                1 for f in os.listdir(doc_image_dir)
                if not f.endswith(".caption.txt")
            )
            shutil.rmtree(doc_image_dir)
        except PermissionError as exc:
            warnings.append(
                f"Image directory could not be deleted (permission denied): {exc}"
            )
        except Exception as exc:
            warnings.append(f"Image directory deletion failed: {exc}")

    # ── Step 5: Persist updated state to disk ────────────────────────────────
    persisted = False
    try:
        save_all()
        persisted = True
    except Exception as exc:
        warnings.append(f"Persistence failed — restart will reload old state: {exc}")

    return {
        "deleted": filename,
        "chunks_removed": chunks_removed,
        "faiss_vectors_removed": faiss_removed,
        "images_deleted": images_deleted,
        "persisted": persisted,
        "warnings": warnings,
    }
