from __future__ import annotations

import logging

import pdfplumber

from app.utils.text_utils import normalize_text

logger = logging.getLogger(__name__)


def parse_pdf(file_path: str) -> str:
    """
    Extract all text from a PDF using pdfplumber and return normalised content.

    Pages are joined with a double newline to preserve the logical structure
    of the document (e.g. page breaks ≈ paragraph breaks for chunking).
    Pages that yield no text (e.g. image-only scans) are silently skipped.
    """
    pages: list[str] = []
    skipped = 0

    with pdfplumber.open(file_path) as pdf:
        total_pages = len(pdf.pages)
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text and page_text.strip():
                pages.append(page_text)
            else:
                skipped += 1

    logger.info(
        "PDF parsed: %d/%d pages extracted, %d skipped (image-only or empty).",
        len(pages),
        total_pages,
        skipped,
    )

    raw = "\n\n".join(pages)
    return normalize_text(raw)
