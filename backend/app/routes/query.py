"""
Query route — POST /api/v1/query

Accepts a natural-language question, retrieves relevant chunks via hybrid
search, sends them to an LLM, and returns the answer with sources.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.config import RETRIEVAL_BM25_RELEVANCE_THRESHOLD, RETRIEVAL_RELEVANCE_THRESHOLD
from app.services.retrieval.cache_service import get_cache, set_cache
from app.services.retrieval.llm_service import (
    _CHITCHAT_SYSTEM_PROMPT,
    build_chitchat_prompt,
    build_prompt,
    call_llm,
    classify_intent,
    enrich_query,
)
from app.services.retrieval.retrieval_service import retrieve

logger = logging.getLogger(__name__)

router = APIRouter()

_MAX_HISTORY_TURNS = 5  # Max conversation turns sent from UI


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(..., min_length=1, max_length=4000)


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    api_key: str = Field(..., min_length=1)
    provider: str = Field(default="gemini", pattern="^(gemini|openai)$")
    include_sources: bool = Field(default=False)
    history: list[HistoryMessage] = Field(default_factory=list, max_length=_MAX_HISTORY_TURNS * 2)

    model_config = {"json_schema_extra": {"examples": [{"query": "...", "api_key": "***", "provider": "gemini"}]}}


@router.post(
    "/query",
    summary="Ask a question over ingested documents",
    response_description="LLM answer with source chunks",
)
async def query_documents(req: QueryRequest) -> JSONResponse:
    """
    Run hybrid retrieval, build a prompt from the top chunks,
    call the LLM, and return the answer with sources.
    """
    # Sanitize history — cap to last N turns, strip whitespace.
    history = [
        {"role": m.role, "content": m.content.strip()}
        for m in req.history[-_MAX_HISTORY_TURNS * 2:]
        if m.content.strip()
    ]

    # 0. Check cache — key includes a digest of recent history.
    cache_key_suffix = _history_digest(history)
    cached = get_cache(req.query + cache_key_suffix, req.provider)
    if cached is not None:
        body = {"answer": cached["answer"]}
        if req.include_sources:
            body["sources"] = cached.get("sources", [])
        return JSONResponse(status_code=200, content=body)

    # 1. Classify intent — chitchat and capability questions bypass retrieval.
    intent = classify_intent(req.query)
    if intent in ("chitchat", "capability"):
        chitchat_prompt = build_chitchat_prompt(req.query, history)
        try:
            answer = await asyncio.to_thread(
                call_llm, chitchat_prompt, req.api_key, req.provider,
                system_prompt=_CHITCHAT_SYSTEM_PROMPT,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        set_cache(req.query + cache_key_suffix, req.provider, {"answer": answer, "sources": []})
        return JSONResponse(status_code=200, content={"answer": answer, "sources": []})

    # 2. Enrich the query with conversation context for better retrieval.
    #    e.g. "can you give an image?" → "image of single server setup"
    enriched_query = enrich_query(req.query, history)
    logger.info("Query enriched: '%s' → '%s'", req.query[:60], enriched_query[:60])

    # 3. Retrieve relevant chunks using the enriched query.
    try:
        results, max_raw_faiss, max_raw_bm25 = await asyncio.to_thread(retrieve, enriched_query)
    except Exception as exc:
        logger.error("Retrieval failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Retrieval error: {exc}") from exc

    # 3a. No documents uploaded yet — tell the user warmly, no LLM call.
    if not results:
        return JSONResponse(
            status_code=200,
            content={
                "answer": (
                    "It looks like you haven't uploaded any documents yet! "
                    "Go ahead and upload a PDF, TXT, or LOG file using the panel on the left, "
                    "and I'll be ready to answer questions based on its content."
                ),
                "sources": [],
            },
        )

    # 3b. Off-topic gate — OR logic across two independent relevance signals:
    #
    #   FAISS cosine ≥ 0.30  →  semantically close to the document domain
    #   BM25 score  ≥ 1.0    →  a meaningful domain keyword appears in the docs
    #                            (e.g. "concurrency" in a Lambda doc scores ~2–5;
    #                             noise words like "aws" score ~0.05 due to low IDF)
    #
    # Passing either signal is enough.  This handles phrased questions like
    # "can you tell me merits of provisioned concurrency?" whose FAISS score is
    # diluted by framing words, but whose BM25 score fires on "concurrency".
    # Both signals fail only for truly off-topic queries ("quick sort", "Fargate"
    # not in the doc), where neither semantic nor keyword overlap exists.
    is_on_topic = (
        max_raw_faiss >= RETRIEVAL_RELEVANCE_THRESHOLD
        or max_raw_bm25 >= RETRIEVAL_BM25_RELEVANCE_THRESHOLD
    )
    if not is_on_topic:
        logger.info(
            "Query '%.60s…' off-topic (faiss=%.3f, bm25=%.2f) — skipping LLM.",
            req.query, max_raw_faiss, max_raw_bm25,
        )
        return JSONResponse(
            status_code=200,
            content={
                "answer": (
                    "That topic doesn't appear to be covered in your uploaded documents. "
                    "Try asking something related to the content you've uploaded, "
                    "or upload a document on this subject to get detailed answers."
                ),
                "sources": [],
            },
        )

    # 4. Build prompt with history context and call LLM.
    prompt = build_prompt(req.query, results, history=history)

    try:
        answer = await asyncio.to_thread(call_llm, prompt, req.api_key, req.provider)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    # 5. Build sources list (always computed, conditionally returned).
    sources = []
    for item in results:
        chunk = item["chunk"]
        source_entry = {
            "content": chunk.content,
            "source": chunk.source,
            "type": chunk.type,
        }
        if chunk.type == "image":
            source_entry["image_path"] = chunk.metadata.get("image_path")
        sources.append(source_entry)

    # 6. Cache the full response.
    set_cache(req.query + cache_key_suffix, req.provider, {"answer": answer, "sources": sources})

    # 7. Return answer; include sources only if the client asked for them.
    response_body: dict = {"answer": answer}
    if req.include_sources:
        response_body["sources"] = sources

    return JSONResponse(status_code=200, content=response_body)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_n: int = Field(default=5, ge=1, le=50)


@router.post(
    "/search",
    summary="Search chunks without calling an LLM",
    response_description="Ranked chunks from hybrid retrieval",
)
async def search_chunks(req: SearchRequest) -> JSONResponse:
    """
    Run hybrid retrieval (FAISS + BM25) and return the top-N ranked chunks
    directly — no LLM key required.  Useful for testing retrieval quality,
    debugging, and building UIs that show search results before generating
    an answer.
    """
    try:
        results, _, _ = await asyncio.to_thread(retrieve, req.query, req.top_n)
    except Exception as exc:
        logger.error("Search failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Search error: {exc}") from exc

    sources = []
    for item in results:
        chunk = item["chunk"]
        entry = {
            "id": chunk.id,
            "content": chunk.content,
            "source": chunk.source,
            "type": chunk.type,
            "score": item["score"],
            "metadata": chunk.metadata,
        }
        if chunk.type == "image":
            entry["image_path"] = chunk.metadata.get("image_path")
        sources.append(entry)

    return JSONResponse(
        status_code=200,
        content={
            "query": req.query,
            "total_results": len(sources),
            "results": sources,
        },
    )


def _history_digest(history: list[dict]) -> str:
    """
    Produce a short string digest of the last 2 turns of history for use
    as a cache key suffix.  Prevents stale cache hits when context changes.
    """
    if not history:
        return ""
    recent = history[-4:]  # last 2 user+assistant pairs
    combined = "|".join(f"{m['role']}:{m['content'][:50]}" for m in recent)
    import hashlib
    return ":" + hashlib.md5(combined.encode()).hexdigest()[:8]
