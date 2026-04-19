"""
Ingestion orchestrator.

This module is the single entry point for the ingestion pipeline.  Given a
file path and its original filename it:

  1. Validates the extension.
  2. Dispatches to the appropriate text parser (PDF / plain text).
  3. Decides whether the content is a log file or a regular document.
  4. Chunks the text content via the matching strategy.
  5. For PDFs: additionally extracts images, generates Gemini captions,
     and creates image chunks — merged with the text chunks.
  6. Persists all chunks to the in-memory store.
  7. Returns the full list of created chunks to the caller.

CPU-bound and blocking work (PDF parsing, image extraction, Gemini API calls)
is offloaded to a thread pool via asyncio.to_thread so the FastAPI event loop
is never blocked.

Image pipeline error handling
------------------------------
* Extraction failures per-image are swallowed inside image_extractor.py.
* Caption failures are swallowed inside gemini_captioner.py (fallback used).s
* The overall image pipeline runs in a try/except so a total failure cannot
  abort the text pipeline that already succeeded.
"""

from __future__ import annotations

import asyncio
import logging
import os

from app.config import (
    ALLOWED_EXTENSIONS,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    IMAGES_DIR,
    MAX_IMAGES_PER_PDF,
    MIN_IMAGE_SIZE_BYTES,
)
from app.models.chunk import Chunk
from app.services.captioning.gemini_captioner import FALLBACK_CAPTION, generate_image_caption
from app.services.chunking.doc_chunker import chunk_document
from app.services.chunking.log_chunker import chunk_logs
from app.services.parsing.image_extractor import extract_images_from_pdf, save_caption
from app.services.parsing.log_parser import is_log_content, parse_log_text
from app.services.parsing.pdf_parser import parse_pdf
from app.services.parsing.text_parser import parse_text
from app.storage.memory_store import store_chunks

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def ingest_file(file_path: str, original_filename: str, *, api_key: str = "", provider: str = "gemini") -> list[Chunk]:
    """
    Run the full ingestion pipeline for *file_path*.

    Parameters
    ----------
    file_path          : Absolute path to the temporary file on disk.
    original_filename  : The name supplied by the uploader (used for extension
                         detection and stored as the chunk's *source* field).
    api_key            : Optional API key from the request.  Falls back
                         to the GEMINI_API_KEY env var if empty.
    provider           : "gemini" or "openai" — used for image captioning.

    Returns
    -------
    The list of Chunk objects that were created and stored (text + image chunks).

    Raises
    ------
    ValueError  : Extension not supported, or file is empty / unreadable.
    """
    ext = os.path.splitext(original_filename)[1].lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file extension '{ext}'. "
            f"Allowed: {sorted(ALLOWED_EXTENSIONS)}"
        )

    # ── Step 1: Parse raw text ───────────────────────────────────────────────
    # Run in a thread to avoid blocking the event loop during I/O + CPU work.
    if ext == ".pdf":
        raw_text: str = await asyncio.to_thread(parse_pdf, file_path)
    else:
        raw_text = await asyncio.to_thread(parse_text, file_path)

    if not raw_text.strip():
        raise ValueError(
            "File appears to be empty or no text could be extracted. "
            "If this is a scanned PDF, OCR is not yet supported."
        )

    # ── Step 2: Route to the appropriate text-chunking pipeline ─────────────
    if ext == ".log" or is_log_content(original_filename, raw_text):
        log_entries = parse_log_text(raw_text)
        text_chunks: list[Chunk] = chunk_logs(log_entries, source=original_filename)

        # Graceful fallback: treat as plain text if log parser yields nothing.
        if not text_chunks:
            text_chunks = chunk_document(raw_text, source=original_filename)
    else:
        text_chunks = chunk_document(raw_text, source=original_filename)

    # ── Step 3: Image pipeline (PDFs only) ───────────────────────────────────
    image_chunks: list[Chunk] = []
    if ext == ".pdf":
        image_chunks = await _run_image_pipeline(file_path, original_filename, api_key=api_key, provider=provider)

    # ── Step 4: Merge, persist, return ───────────────────────────────────────
    all_chunks = text_chunks + image_chunks
    store_chunks(all_chunks)

    # ── Step 5: Embed, deduplicate, and index for retrieval ──────────────────
    all_chunks = await _run_retrieval_indexing(all_chunks)

    logger.info(
        "Ingestion complete for '%s': %d text chunk(s), %d image chunk(s)",
        original_filename,
        len(text_chunks),
        len(image_chunks),
    )
    return all_chunks


# ---------------------------------------------------------------------------
# Retrieval indexing pipeline (internal)
# ---------------------------------------------------------------------------

async def _run_retrieval_indexing(chunks: list[Chunk]) -> list[Chunk]:
    """
    Embed chunks, deduplicate near-duplicates, and add to FAISS + BM25 indexes.

    Runs in a try/except so a failure here never breaks the ingestion pipeline.
    The chunks are already persisted in memory_store before this runs.
    """
    try:
        from app.services.retrieval.embedding_service import batch_embeddings
        from app.services.retrieval.deduplication_service import deduplicate_chunks
        from app.services.retrieval.vector_service import add_embeddings
        from app.services.retrieval.bm25_service import build_index
        from app.storage.memory_store import get_all_chunks

        # Embed all new chunks (CPU-bound model inference).
        embedded = await asyncio.to_thread(batch_embeddings, chunks)

        # Deduplicate across the new batch.
        deduped = await asyncio.to_thread(deduplicate_chunks, embedded)

        # Add to FAISS vector index.
        await asyncio.to_thread(add_embeddings, deduped)

        # Rebuild BM25 over the full store (cheap for in-memory sizes).
        all_chunks = get_all_chunks()
        await asyncio.to_thread(build_index, all_chunks)

        logger.info(
            "Retrieval indexing: %d chunks embedded, %d after dedup, indexes updated.",
            len(chunks),
            len(deduped),
        )
        return deduped

    except Exception as exc:
        logger.error(
            "Retrieval indexing failed: %s. Chunks are stored but not indexed.",
            exc,
        )
        return chunks


# ---------------------------------------------------------------------------
# Image pipeline (internal)
# ---------------------------------------------------------------------------

async def _run_image_pipeline(
    file_path: str,
    original_filename: str,
    *,
    api_key: str = "",
    provider: str = "gemini",
) -> list[Chunk]:
    """
    Extract images from a PDF, caption each one with Gemini, and return a
    list of image Chunks.

    All errors are caught internally so a failure here can never propagate
    upward and break the text ingestion that already succeeded.
    """
    try:
        # ── Extract images (CPU-bound, run in thread) ────────────────────────
        extracted: list[dict] = await asyncio.to_thread(
            extract_images_from_pdf,
            file_path,
            original_filename,
            IMAGES_DIR,
            MAX_IMAGES_PER_PDF,
            MIN_IMAGE_SIZE_BYTES,
        )

        if not extracted:
            return []

        logger.info(
            "PDF '%s': %d image(s) extracted, generating captions…",
            original_filename,
            len(extracted),
        )

        # ── Caption each image (network call, run in thread) ─────────────────
        image_chunks: list[Chunk] = []
        captions_generated = 0
        captions_reused = 0

        for img_data in extracted:
            effective_key = api_key or GEMINI_API_KEY

            if img_data["saved_caption"]:
                # Real caption already persisted from a previous upload — reuse it.
                caption = img_data["saved_caption"]
                captions_reused += 1
                logger.debug(
                    "Image '%s' — reusing saved caption from sidecar.",
                    img_data["filename"],
                )
            elif img_data["already_exists"] and not img_data["saved_caption"]:
                # Image on disk but no sidecar (e.g. uploaded before this feature
                # was added). Re-read the file bytes and call Gemini to generate
                # and persist the caption now.
                try:
                    with open(img_data["full_path"], "rb") as fh:
                        img_bytes = fh.read()
                except OSError:
                    img_bytes = None

                if img_bytes:
                    caption = await asyncio.to_thread(
                        generate_image_caption,
                        img_bytes,
                        effective_key,
                        GEMINI_MODEL,
                        provider,
                    )
                else:
                    caption = FALLBACK_CAPTION
            else:
                # Brand new image — call Gemini with the freshly extracted bytes.
                caption = await asyncio.to_thread(
                    generate_image_caption,
                    img_data["bytes"],
                    effective_key,
                    GEMINI_MODEL,
                    provider,
                )

            if caption != FALLBACK_CAPTION:
                captions_generated += 1
                # Persist caption as sidecar so re-uploads skip the API call.
                save_caption(img_data["full_path"], caption)

            chunk = Chunk(
                content=caption,
                type="image",
                source=f"{original_filename}#page={img_data['page']}",
                metadata={
                    "page": img_data["page"],
                    "image_index": img_data["index"],
                    "image_path": img_data["image_path"],
                    "filename": img_data["filename"],
                },
            )
            image_chunks.append(chunk)

        logger.info(
            "PDF '%s': %d/%d caption(s) generated, %d reused from disk.",
            original_filename,
            captions_generated,
            len(extracted),
            captions_reused,
        )
        return image_chunks

    except Exception as exc:
        # Log but never raise — text chunks must still be stored.
        logger.error(
            "Image pipeline failed for '%s': %s. Continuing without images.",
            original_filename,
            exc,
        )
        return []
