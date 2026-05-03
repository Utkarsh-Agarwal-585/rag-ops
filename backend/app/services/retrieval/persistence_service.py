"""
Persistence service — save and restore the full retrieval state to disk.

Saves on every upload (after indexing) and loads on server startup.
This eliminates the need to re-upload documents after a server restart.

Files written to backend/storage/index/:
    chunks.pkl      — memory store (_chunks list + _index dict + _source_index)
    faiss.index     — FAISS IndexFlatIP binary
    bm25.pkl        — BM25 corpus (token lists + chunk IDs)
"""

from __future__ import annotations

import logging
import os
import pickle
from pathlib import Path

logger = logging.getLogger(__name__)

# Persist alongside the images directory under backend/storage/
_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent.parent  # backend/
PERSIST_DIR: Path = _BACKEND_DIR / "storage" / "index"

_CHUNKS_FILE = PERSIST_DIR / "chunks.pkl"
_FAISS_FILE = str(PERSIST_DIR / "faiss.index")
_BM25_FILE = PERSIST_DIR / "bm25.pkl"


def _ensure_dir() -> None:
    """Create the persistence directory if it does not already exist."""
    PERSIST_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_all() -> None:
    """
    Persist the full retrieval state to disk.

    Called after every successful upload so the state is always current.
    Failures are logged but never raised — a save failure must not break
    the upload response.
    """
    try:
        _ensure_dir()
        _save_chunks()
        _save_faiss()
        _save_bm25()
        logger.info("Retrieval state persisted to %s.", PERSIST_DIR)
    except Exception as exc:
        logger.error("Failed to persist retrieval state: %s", exc)


def _save_chunks() -> None:
    """
    Pickle the three memory store data structures to chunks.pkl.

    Saves _chunks (ordered list), _index (ID dict), and _source_index
    (source → chunk ID set) so the full store can be restored exactly.
    """
    from app.storage import memory_store
    payload = {
        "chunks": memory_store._chunks,
        "index": memory_store._index,
        "source_index": memory_store._source_index,
    }
    with open(_CHUNKS_FILE, "wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    logger.debug("Chunks saved: %d chunk(s).", len(memory_store._chunks))


def _save_faiss() -> None:
    """
    Write the FAISS index to disk using faiss.write_index().

    Also pickles the _chunk_ids list that maps FAISS row positions back to
    chunk IDs — without this mapping, search results cannot be resolved.
    Skips saving if the index is empty (nothing to persist).
    """
    import faiss
    from app.services.retrieval import vector_service

    idx = vector_service._index
    if idx is None or idx.ntotal == 0:
        logger.debug("FAISS index empty — skipping save.")
        return

    faiss.write_index(idx, _FAISS_FILE)

    # Also pickle the chunk ID list that maps FAISS positions → chunk IDs.
    ids_file = PERSIST_DIR / "faiss_ids.pkl"
    with open(ids_file, "wb") as fh:
        pickle.dump(vector_service._chunk_ids, fh, protocol=pickle.HIGHEST_PROTOCOL)

    logger.debug("FAISS index saved: %d vector(s).", idx.ntotal)


def _save_bm25() -> None:
    """
    Pickle the BM25 corpus to bm25.pkl.

    Saves the token lists and chunk ID list.  The BM25Okapi object itself is
    not pickled — it is cheap to reconstruct from the token lists on load.
    """
    from app.services.retrieval import bm25_service
    payload = {
        "corpus_ids": bm25_service._corpus_ids,
        "corpus_tokens": bm25_service._corpus_tokens,
    }
    with open(_BM25_FILE, "wb") as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
    logger.debug("BM25 corpus saved: %d document(s).", len(bm25_service._corpus_ids))


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_all() -> bool:
    """
    Restore the full retrieval state from disk on server startup.

    Returns True if state was successfully loaded, False if no saved state
    exists (first run) or if loading fails.
    """
    if not _CHUNKS_FILE.exists():
        logger.info("No persisted state found — starting fresh.")
        return False

    try:
        _load_chunks()
        _load_faiss()
        _load_bm25()
        logger.info("Retrieval state restored from %s.", PERSIST_DIR)
        return True
    except Exception as exc:
        logger.error(
            "Failed to restore persisted state: %s. Starting fresh.", exc
        )
        return False


def _load_chunks() -> None:
    """
    Restore the memory store from chunks.pkl.

    Clears the existing store first to avoid duplicates, then extends all
    three data structures from the pickled payload.  Handles missing
    source_index key gracefully for files saved before that field was added.
    """
    from app.storage import memory_store

    with open(_CHUNKS_FILE, "rb") as fh:
        payload = pickle.load(fh)

    memory_store._chunks.clear()
    memory_store._index.clear()
    memory_store._source_index.clear()

    memory_store._chunks.extend(payload["chunks"])
    memory_store._index.update(payload["index"])
    memory_store._source_index.update(payload.get("source_index", {}))

    logger.debug("Chunks restored: %d chunk(s).", len(memory_store._chunks))


def _load_faiss() -> None:
    """
    Restore the FAISS index and chunk ID mapping from disk.

    Both faiss.index and faiss_ids.pkl must exist; if either is missing the
    function returns early without modifying the in-memory index.
    """
    import faiss
    from app.services.retrieval import vector_service

    faiss_path = _FAISS_FILE
    ids_path = PERSIST_DIR / "faiss_ids.pkl"

    if not os.path.exists(faiss_path) or not ids_path.exists():
        logger.debug("No FAISS index file found — skipping.")
        return

    vector_service._index = faiss.read_index(faiss_path)

    with open(ids_path, "rb") as fh:
        ids = pickle.load(fh)

    vector_service._chunk_ids.clear()
    vector_service._chunk_ids.extend(ids)

    logger.debug("FAISS index restored: %d vector(s).", vector_service._index.ntotal)


def _load_bm25() -> None:
    """
    Restore the BM25 corpus from bm25.pkl and rebuild the BM25Okapi object.

    The BM25Okapi object is not stored directly — it is reconstructed from
    the token lists, which is fast and avoids pickle compatibility issues
    across rank-bm25 versions.
    """
    from app.services.retrieval import bm25_service
    from rank_bm25 import BM25Okapi

    if not _BM25_FILE.exists():
        logger.debug("No BM25 file found — skipping.")
        return

    with open(_BM25_FILE, "rb") as fh:
        payload = pickle.load(fh)

    bm25_service._corpus_ids.clear()
    bm25_service._corpus_tokens.clear()

    bm25_service._corpus_ids.extend(payload["corpus_ids"])
    bm25_service._corpus_tokens.extend(payload["corpus_tokens"])

    if bm25_service._corpus_tokens:
        bm25_service._bm25 = BM25Okapi(bm25_service._corpus_tokens)

    logger.debug("BM25 corpus restored: %d document(s).", len(bm25_service._corpus_ids))
