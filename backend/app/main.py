"""
FastAPI application factory.

Registers all routers, global middleware, and static file mounts.  Run with:

    uvicorn app.main:app --reload          (development)
    uvicorn app.main:app --host 0.0.0.0    (production)

Static files
------------
Extracted PDF images are served from:
    GET /storage/images/<filename>

The directory is created on startup if it does not exist.
"""

from __future__ import annotations

import os

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import IMAGES_DIR, IMAGES_URL_PREFIX
from app.logging_config import configure_logging
from app.middleware.correlation import CorrelationIdMiddleware
from app.routes import chunks, upload
from app.routes import query as query_route

# Configure JSON structured logging before anything else so all startup
# messages (including persistence load) are captured in structured format.
configure_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load persisted retrieval state on startup."""
    from app.services.retrieval.persistence_service import load_all
    load_all()
    yield
    # Shutdown: nothing extra needed — state is saved after every upload.


# Ensure the images directory exists before mounting it as static files.
# StaticFiles raises a RuntimeError at startup if the directory is absent.
os.makedirs(IMAGES_DIR, exist_ok=True)

app = FastAPI(
    title="RAG Ingestion API",
    description=(
        "Production-grade document and log ingestion pipeline for RAG systems. "
        "Supports PDF (text + diagrams), plain-text, and structured log files."
    ),
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------

# CorrelationIdMiddleware must be added FIRST so the correlation ID is set
# before any other middleware or route handler runs.
app.add_middleware(CorrelationIdMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Tighten this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Static files — extracted PDF images
# Served at:  GET /storage/images/<filename>
# Stored in:  backend/storage/images/<filename>
# ---------------------------------------------------------------------------

app.mount(
    IMAGES_URL_PREFIX,
    StaticFiles(directory=IMAGES_DIR),
    name="images",
)

# ---------------------------------------------------------------------------
# Routers
# ---------------------------------------------------------------------------

app.include_router(upload.router, prefix="/api/v1", tags=["Ingestion"])
app.include_router(chunks.router, prefix="/api/v1", tags=["Chunks"])
app.include_router(query_route.router, prefix="/api/v1", tags=["Query"])


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Health"], summary="Liveness probe")
async def health_check() -> dict:
    """Returns 200 OK when the service is up."""
    return {"status": "healthy", "version": "1.1.0"}
