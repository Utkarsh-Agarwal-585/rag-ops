"""
Hybrid retrieval engine.

Combines FAISS vector search and BM25 keyword search with configurable
weights, then re-ranks to produce the final top-N chunks.
"""

from __future__ import annotations

import logging

import re

from app.config import (
    HYBRID_BM25_WEIGHT,
    HYBRID_VECTOR_WEIGHT,
    RERANK_TOP_N,
    RETRIEVAL_TOP_K,
)
from app.models.chunk import Chunk
from app.services.retrieval import bm25_service, vector_service
from app.services.retrieval.embedding_service import generate_embedding
from app.storage.memory_store import get_all_chunks, get_chunk_by_id

logger = logging.getLogger(__name__)

# Queries containing these terms signal the user wants visual content.
_IMAGE_KEYWORDS = re.compile(
    r"\b(image|diagram|figure|picture|illustration|chart|screenshot|"
    r"show\s+me|visual|architecture\s+diagram)\b",
    re.IGNORECASE,
)


def _normalize_scores(results: list[tuple[str, float]]) -> dict[str, float]:
    """Min-max normalize scores to [0, 1]."""
    if not results:
        return {}
    scores = [s for _, s in results]
    lo, hi = min(scores), max(scores)
    span = hi - lo if hi != lo else 1.0
    return {cid: (s - lo) / span for cid, s in results}


def retrieve(query: str, top_n: int | None = None) -> list[dict]:
    """
    Run hybrid retrieval for *query* and return the top-N ranked results.

    Each result dict contains:
        chunk   : The Chunk object
        score   : Combined hybrid score (0–1)
    """
    top_n = top_n or RERANK_TOP_N

    # 1. Embed the query.
    query_vec = generate_embedding(query)

    # 2. Retrieve candidates from both engines.
    vec_results = vector_service.search(query_vec, top_k=RETRIEVAL_TOP_K)
    bm25_results = bm25_service.search(query, top_k=RETRIEVAL_TOP_K)

    # 3. Normalize scores to [0, 1].
    vec_scores = _normalize_scores(vec_results)
    bm25_scores = _normalize_scores(bm25_results)

    # 4. Combine with weighted sum.
    all_ids = set(vec_scores.keys()) | set(bm25_scores.keys())
    combined: list[tuple[str, float]] = []

    for cid in all_ids:
        vs = vec_scores.get(cid, 0.0)
        bs = bm25_scores.get(cid, 0.0)
        score = HYBRID_VECTOR_WEIGHT * vs + HYBRID_BM25_WEIGHT * bs
        combined.append((cid, score))

    # 5. Re-rank by final score descending.
    combined.sort(key=lambda x: x[1], reverse=True)

    # 6. Resolve chunk objects and return top-N.
    results = []
    for cid, score in combined[:top_n]:
        chunk = get_chunk_by_id(cid)
        if chunk is not None:
            results.append({"chunk": chunk, "score": round(score, 4)})

    # 7. If the query asks for images/diagrams, ensure relevant image chunks
    #    are included by cross-referencing figure mentions in the top text
    #    chunks with image chunks from the same pages.
    if _IMAGE_KEYWORDS.search(query):
        existing_ids = {r["chunk"].id for r in results}
        injected = _inject_relevant_images(results, existing_ids, query_vec)
        if injected:
            results.extend(injected)
            logger.info("Injected %d image chunk(s) for visual query.", len(injected))

    logger.info(
        "Hybrid retrieval for '%.60s…': %d candidates → %d results.",
        query,
        len(all_ids),
        len(results),
    )
    return results


# ---------------------------------------------------------------------------
# Image injection helpers
# ---------------------------------------------------------------------------

# Matches "Figure 1-1", "Figure 15-3", "fig. 2", etc. in text chunks.
_FIGURE_REF = re.compile(r"(?:figure|fig\.?)\s*(\d+[\-\.]\d+|\d+)", re.IGNORECASE)


def _inject_relevant_images(
    results: list[dict],
    existing_ids: set[str],
    query_vec: list[float],
) -> list[dict]:
    """
    Find image chunks relevant to the current result set.

    Strategy (in priority order):
    1. Extract figure references (e.g. "Figure 1-1") from the top text chunks,
       then find image chunks whose captions mention the same figure.
    2. Find image chunks from the same source pages as the top text chunks.
    3. Fall back to embedding similarity against all image chunks.
    """
    all_chunks = get_all_chunks()
    image_chunks = [c for c in all_chunks if c.type == "image"]
    if not image_chunks:
        return []

    injected: list[dict] = []

    # --- Strategy 1: Figure reference matching ---
    figure_refs: set[str] = set()
    for r in results:
        if r["chunk"].type != "image":
            for m in _FIGURE_REF.finditer(r["chunk"].content):
                figure_refs.add(m.group(1).lower())

    if figure_refs:
        for ic in image_chunks:
            if ic.id in existing_ids:
                continue
            caption_refs = {m.group(1).lower() for m in _FIGURE_REF.finditer(ic.content)}
            if caption_refs & figure_refs:
                injected.append({"chunk": ic, "score": 0.95})
                existing_ids.add(ic.id)
                if len(injected) >= 3:
                    return injected

    # --- Strategy 2: Same source/page matching (skip page 1 — usually a cover) ---
    if len(injected) < 2:
        source_bases = set()
        for r in results:
            src = r["chunk"].source.split("#")[0]
            source_bases.add(src)

        for ic in image_chunks:
            if ic.id in existing_ids:
                continue
            # Skip page 1 images — almost always a book/document cover.
            if ic.metadata.get("page", 0) == 1:
                continue
            ic_base = ic.source.split("#")[0]
            if ic_base in source_bases:
                injected.append({"chunk": ic, "score": 0.5})
                existing_ids.add(ic.id)
                if len(injected) >= 3:
                    return injected

    # --- Strategy 3: Embedding similarity fallback (skip page 1 covers) ---
    if not injected:
        import numpy as np
        embedded_images = [
            c for c in image_chunks
            if c.embedding and c.id not in existing_ids
            and c.metadata.get("page", 0) != 1
        ]
        if embedded_images:
            q_vec = np.array(query_vec, dtype=np.float32)
            scored = []
            for ic in embedded_images:
                ic_vec = np.array(ic.embedding, dtype=np.float32)
                sim = float(np.dot(q_vec, ic_vec))
                scored.append((ic, sim))
            scored.sort(key=lambda x: x[1], reverse=True)
            for ic, sim in scored[:2]:
                injected.append({"chunk": ic, "score": round(sim, 4)})

    return injected
