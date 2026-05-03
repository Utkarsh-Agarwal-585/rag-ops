"""
Upload route — POST /api/v1/upload

Accepts a single file (.pdf, .txt, .log), runs it through the ingestion
pipeline, and returns the number of chunks created.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.config import ALLOWED_EXTENSIONS, MAX_UPLOAD_SIZE_BYTES
from app.services.ingestion.ingestor import ingest_file
from app.utils.file_utils import cleanup_temp_file, save_upload_file, validate_extension

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/upload",
    summary="Upload a document or log file",
    response_description="Ingestion result with chunk count",
)
async def upload_file(
    file: UploadFile = File(...),
    api_key: str = Form(default=""),
    provider: str = Form(default="gemini"),
) -> JSONResponse:
    """
    Upload a **.pdf**, **.txt**, or **.log** file.

    The file is:
    1. Validated (extension + size).
    2. Saved to a temporary location.
    3. Parsed and chunked by the ingestion pipeline.
    4. Chunks are stored in memory for retrieval.

    Returns the total number of chunks created from the file.
    """
    # ── Validate extension ───────────────────────────────────────────────────
    if not validate_extension(file.filename, ALLOWED_EXTENSIONS):
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type for '{file.filename}'. "
                f"Allowed extensions: {sorted(ALLOWED_EXTENSIONS)}"
            ),
        )

    # ── Read content & validate size before touching the filesystem ──────────
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if len(content) > MAX_UPLOAD_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"File too large ({size_mb:.1f} MB). "
                f"Maximum allowed size is {MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)} MB."
            ),
        )

    logger.info(
        "Upload received: '%s' (%.2f MB) via provider='%s'.",
        file.filename,
        size_mb,
        provider,
    )

    tmp_path: str | None = None
    try:
        # ── Persist to temp file ─────────────────────────────────────────────
        tmp_path = await save_upload_file(file.filename or "upload", content)

        # ── Run ingestion pipeline ───────────────────────────────────────────
        chunks = await ingest_file(tmp_path, file.filename or "upload", api_key=api_key, provider=provider)

        # Build a per-type breakdown so the caller can see exactly what was
        # produced without having to call GET /chunks separately.
        doc_chunks   = [c for c in chunks if c.type == "doc"]
        log_chunks   = [c for c in chunks if c.type == "log"]
        image_chunks = [c for c in chunks if c.type == "image"]

        return JSONResponse(
            status_code=200,
            content={
                "message": "File processed successfully",
                "source": file.filename,
                "chunks_created": len(chunks),
                "breakdown": {
                    "doc":   len(doc_chunks),
                    "log":   len(log_chunks),
                    "image": len(image_chunks),
                },
                # Surface every image chunk so the UI can immediately render
                # the diagrams and their captions without a second API call.
                "images": [
                    {
                        "page":       c.metadata.get("page"),
                        "image_path": c.metadata.get("image_path"),
                        "caption":    c.content,
                    }
                    for c in image_chunks
                ],
            },
        )

    except ValueError as exc:
        # Validation errors from the ingestion layer (empty file, bad format …)
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Ingestion failed unexpectedly: {exc}",
        ) from exc

    finally:
        # Always clean up the temp file — even on error
        if tmp_path:
            cleanup_temp_file(tmp_path)
