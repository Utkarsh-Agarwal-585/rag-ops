from __future__ import annotations

import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """
    A single unit of ingested content ready for storage and retrieval.

    Fields
    ------
    id       : Unique identifier (UUID4 by default).
    content  : The actual text that will be embedded/retrieved.
    type     : "doc"   — sliding-window text chunk from a document.
               "log"   — time-windowed log summary chunk.
               "image" — caption generated from a diagram/image in a PDF.
    source   : Original filename (plus "#page=N" suffix for image chunks).
    metadata : Arbitrary key-value bag.
               Image chunks carry "page" and "image_path" keys.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    type: Literal["doc", "log", "image"]
    source: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] | None = Field(default=None, exclude=True)
