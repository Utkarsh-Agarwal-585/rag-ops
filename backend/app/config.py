# ---------------------------------------------------------------------------
# Application-wide configuration constants.
# Override these values via environment variables in a future iteration.
# ---------------------------------------------------------------------------

from __future__ import annotations

import os
from pathlib import Path

# Resolve the backend/ root directory at import time so all path constants
# are absolute and correct regardless of the working directory.
_BACKEND_DIR: Path = Path(__file__).resolve().parent.parent  # backend/

# ── Upload limits ────────────────────────────────────────────────────────────
MAX_UPLOAD_SIZE_BYTES: int = 50 * 1024 * 1024  # 50 MB
ALLOWED_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".txt", ".log"})

# ── Document chunking ────────────────────────────────────────────────────────
DOC_CHUNK_SIZE_WORDS: int = 400   # Target words per chunk
DOC_OVERLAP_WORDS: int = 50       # Overlap between successive chunks

# ── Log chunking ─────────────────────────────────────────────────────────────
LOG_TIME_WINDOW_MINUTES: int = 10  # Width of each time-window bucket

# ── Image extraction (PDF) ───────────────────────────────────────────────────
# Absolute path to the directory where extracted images are saved.
IMAGES_DIR: str = str(_BACKEND_DIR / "storage" / "images")

# URL prefix under which FastAPI will serve the images statically.
IMAGES_URL_PREFIX: str = "/storage/images"

# Hard cap on images extracted per PDF. Set high to extract all meaningful
# images from a single document. Increase further if needed.
MAX_IMAGES_PER_PDF: int = 20

# Images smaller than this are assumed to be decorative icons/logos.
MIN_IMAGE_SIZE_BYTES: int = 2_000  # 2 KB

# ── Gemini captioning ────────────────────────────────────────────────────────
# Set GEMINI_API_KEY in the environment before starting the server.
# If unset, image chunks will carry a fallback caption but ingestion still works.
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# ── Retrieval pipeline ───────────────────────────────────────────────────────
EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION: int = 384  # Output dim for all-MiniLM-L6-v2

DEDUP_SIMILARITY_THRESHOLD: float = 0.85  # Cosine sim above this → merge

HYBRID_VECTOR_WEIGHT: float = 0.6   # Weight for FAISS vector score
HYBRID_BM25_WEIGHT: float = 0.4     # Weight for BM25 keyword score
RETRIEVAL_TOP_K: int = 20           # Candidates from each retriever
RERANK_TOP_N: int = 5               # Final chunks returned after re-ranking

# Minimum raw FAISS cosine similarity (0–1) between the query embedding and the
# best-matching chunk.  Only one of the two relevance signals needs to pass.
RETRIEVAL_RELEVANCE_THRESHOLD: float = 0.30

# Minimum raw BM25 score for the best-matching chunk.  BM25 > 1.0 means a
# meaningful domain keyword from the query appears in the documents (e.g.
# "concurrency" in a Lambda doc).  Values near 0 are noise from high-frequency
# words like "aws" that appear everywhere with near-zero IDF weight.
# Only one of the two signals needs to pass for the LLM to be called.
RETRIEVAL_BM25_RELEVANCE_THRESHOLD: float = 1.0

# ── LLM query service ────────────────────────────────────────────────────────
# OpenAI model used for both chat completions and image captioning.
OPENAI_MODEL: str = "gpt-4o-mini"

# Temperature for LLM responses.  Lower = more deterministic / factual.
# 0.2 is a good default for RAG — factual answers with minimal hallucination.
OPENAI_TEMPERATURE: float = 0.2

# OpenAI vision model used for image captioning (same as chat model for gpt-4o-mini).
OPENAI_VISION_MODEL: str = "gpt-4o-mini"
