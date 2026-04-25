"""
Semantic deduplication layer.

Compares chunk embeddings via cosine similarity and merges near-duplicates
(similarity > threshold).  Preserves all metadata — especially image_path
for image chunks.
"""

from __future__ import annotations

import logging
import uuid

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from app.config import DEDUP_SIMILARITY_THRESHOLD
from app.models.chunk import Chunk

logger = logging.getLogger(__name__)


def deduplicate_chunks(chunks: list[Chunk]) -> list[Chunk]:
    """
    Merge chunks whose embeddings exceed the similarity threshold.

    Returns a new list with duplicates merged.  The merged chunk keeps
    the longer content, unions the sources, and preserves all image_path
    values from the originals.
    """
    if len(chunks) < 2:
        return chunks

    # Build the embedding matrix — skip chunks without embeddings.
    embedded = [c for c in chunks if c.embedding is not None]
    unembedded = [c for c in chunks if c.embedding is None]

    if len(embedded) < 2:
        return chunks

    matrix = np.array([c.embedding for c in embedded], dtype=np.float32)
    sim = cosine_similarity(matrix)

    # Union-find to cluster duplicates.
    n = len(embedded)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(n):
        for j in range(i + 1, n):
            if sim[i][j] >= DEDUP_SIMILARITY_THRESHOLD:
                union(i, j)

    # Group by cluster root.
    clusters: dict[int, list[int]] = {}
    for i in range(n):
        root = find(i)
        clusters.setdefault(root, []).append(i)

    merged: list[Chunk] = []
    merge_count = 0

    for indices in clusters.values():
        if len(indices) == 1:
            merged.append(embedded[indices[0]])
            continue

        group = [embedded[i] for i in indices]
        merged.append(_merge_group(group))
        merge_count += len(indices) - 1

    if merge_count:
        logger.info(
            "Deduplication: merged %d chunk(s) into %d cluster(s) "
            "(threshold=%.2f).",
            merge_count,
            len(merged),
            DEDUP_SIMILARITY_THRESHOLD,
        )

    return merged + unembedded


def _merge_group(group: list[Chunk]) -> Chunk:
    """Merge a cluster of near-duplicate chunks into one."""
    # Pick the chunk with the longest content as the base.
    group.sort(key=lambda c: len(c.content), reverse=True)
    base = group[0]

    # Collect all unique sources.
    sources = sorted({c.source for c in group})

    # Preserve all image_paths from image chunks.
    image_paths = []
    for c in group:
        path = c.metadata.get("image_path")
        if path and path not in image_paths:
            image_paths.append(path)

    # Build merged metadata.
    meta = dict(base.metadata)
    meta["merged_sources"] = sources
    meta["merged_count"] = len(group)
    if image_paths:
        meta["image_paths"] = image_paths
        # Keep the first image_path for backward compat.
        if "image_path" not in meta:
            meta["image_path"] = image_paths[0]

    return Chunk(
        id=str(uuid.uuid4()),
        content=base.content,
        type=base.type,
        source="; ".join(sources),
        metadata=meta,
        embedding=base.embedding,
    )
