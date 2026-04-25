"""
Embedding service — sentence-transformers wrapper.

Loads the model lazily on first call so the app starts fast even if
retrieval is never used in a given session.
"""

from __future__ import annotations

import logging

import numpy as np

from app.config import EMBEDDING_MODEL
from app.models.chunk import Chunk

logger = logging.getLogger(__name__)

_model = None


def _get_model():
    """Lazy-load the sentence-transformers model."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        logger.info("Loading embedding model '%s'…", EMBEDDING_MODEL)
        _model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("Embedding model loaded.")
    return _model


def generate_embedding(text: str) -> list[float]:
    """Return a normalised embedding vector for a single text string."""
    model = _get_model()
    vec = model.encode(text, normalize_embeddings=True)
    return vec.tolist()


def batch_embeddings(chunks: list[Chunk]) -> list[Chunk]:
    """
    Generate embeddings for every chunk and attach them in-place.

    Uses batch encoding for efficiency.  Each chunk's *content* field is
    the text that gets embedded — this works for doc, log, and image chunks
    alike (image chunks contain their caption as content).
    """
    if not chunks:
        return chunks

    model = _get_model()
    texts = [c.content for c in chunks]
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    for chunk, vec in zip(chunks, vectors):
        chunk.embedding = vec.tolist()

    logger.info("Generated embeddings for %d chunk(s).", len(chunks))
    return chunks