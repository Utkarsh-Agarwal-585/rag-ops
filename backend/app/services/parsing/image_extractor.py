"""
PDF image extraction module.

Uses PyMuPDF (fitz) to pull raster images out of each page of a PDF,
filter out icons/logos that are too small to be meaningful, save the
survivors to the local images directory, and return structured metadata
for the captioning and chunking stages.

Limits
------
* Maximum images per PDF  : controlled by MAX_IMAGES_PER_PDF in config.
* Minimum image size      : controlled by MIN_IMAGE_SIZE_BYTES in config.
  Tiny images (< ~5 KB) are almost always decorative icons and are skipped.

Logged statistics
-----------------
  - Total images found in the PDF
  - Images saved
  - Images skipped (too small, extraction error, save error)
"""

from __future__ import annotations

import logging
import os
import re

logger = logging.getLogger(__name__)


def caption_path_for(image_full_path: str) -> str:
    """Return the sidecar .txt path for a given image file path."""
    return os.path.splitext(image_full_path)[0] + ".caption.txt"


def load_saved_caption(image_full_path: str) -> str | None:
    """
    Return the saved caption for an image, or None if no sidecar exists.
    The sidecar file lives next to the image: <stem>.caption.txt
    """
    cap_path = caption_path_for(image_full_path)
    if os.path.exists(cap_path):
        try:
            with open(cap_path, "r", encoding="utf-8") as fh:
                caption = fh.read().strip()
                return caption if caption else None
        except OSError:
            return None
    return None


def save_caption(image_full_path: str, caption: str) -> None:
    """Persist a caption as a sidecar .txt file next to the image."""
    cap_path = caption_path_for(image_full_path)
    try:
        with open(cap_path, "w", encoding="utf-8") as fh:
            fh.write(caption)
    except OSError as exc:
        logger.warning("Could not save caption sidecar '%s': %s", cap_path, exc)


def _safe_stem(filename: str) -> str:
    """Strip directory and extension, then replace non-alphanumeric chars."""
    stem = os.path.splitext(os.path.basename(filename))[0]
    return re.sub(r"[^\w-]", "_", stem)


def extract_images_from_pdf(
    file_path: str,
    source_name: str,
    images_dir: str,
    max_images: int = 5,
    min_size_bytes: int = 5_000,
) -> list[dict]:
    """
    Extract up to *max_images* images from a PDF and save them to *images_dir*.

    Parameters
    ----------
    file_path      : Absolute path to the PDF file.
    source_name    : Original filename (used to name the saved images).
    images_dir     : Directory where extracted images will be written.
    max_images     : Hard cap on how many images to extract (MVP limit).
    min_size_bytes : Images smaller than this are assumed to be icons/logos
                     and are skipped.

    Returns
    -------
    List of dicts, one per extracted image:
    {
        "page"       : int   — 1-based page number,
        "index"      : int   — 1-based image index within that page,
        "filename"   : str   — basename of the saved file,
        "image_path" : str   — URL-style path  ("/storage/images/<filename>"),
        "bytes"      : bytes — raw image bytes (used downstream for captioning),
    }
    The "bytes" key is intentionally kept so the caller can pass the data
    directly to the captioner without re-reading the file.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.error(
            "PyMuPDF is not installed. Run: pip install PyMuPDF. "
            "Image extraction disabled."
        )
        return []

    os.makedirs(images_dir, exist_ok=True)
    stem = _safe_stem(source_name)

    # ── Per-document subdirectory ────────────────────────────────────────────
    # All images and caption sidecars for this PDF live under:
    #   <images_dir>/<stem>/
    # e.g. storage/images/system_design/ for system_design.pdf
    doc_dir = os.path.join(images_dir, stem)
    os.makedirs(doc_dir, exist_ok=True)

    try:
        doc = fitz.open(file_path)
    except Exception as exc:
        logger.error("Failed to open '%s' for image extraction: %s", file_path, exc)
        return []

    results: list[dict] = []
    total_found = 0
    total_skipped = 0

    for page_idx in range(len(doc)):
        if len(results) >= max_images:
            break

        page = doc[page_idx]
        page_num = page_idx + 1
        image_list = page.get_images(full=True)
        total_found += len(image_list)

        for img_idx, img_info in enumerate(image_list):
            if len(results) >= max_images:
                break

            xref = img_info[0]

            # ── Extract raw bytes ────────────────────────────────────────────
            try:
                base_image = doc.extract_image(xref)
            except Exception as exc:
                logger.warning(
                    "Page %d img %d: extraction failed (xref=%d): %s",
                    page_num,
                    img_idx + 1,
                    xref,
                    exc,
                )
                total_skipped += 1
                continue

            image_bytes: bytes = base_image["image"]
            image_ext: str = base_image.get("ext", "png")

            # ── Skip tiny decorative images ──────────────────────────────────
            if len(image_bytes) < min_size_bytes:
                logger.debug(
                    "Page %d img %d skipped — size %d B below threshold %d B",
                    page_num,
                    img_idx + 1,
                    len(image_bytes),
                    min_size_bytes,
                )
                total_skipped += 1
                continue

            # ── Stable filename keyed on xref (content-addressable) ──────────
            # xref is PyMuPDF's internal image reference ID — unique per image
            # within the PDF and stable across repeated extractions of the same
            # file. Using it instead of a loop counter means the same image
            # always maps to the same filename, so we can safely skip re-saving
            # and re-captioning when the file already exists on disk.
            filename = f"page{page_num}_xref{xref}.{image_ext}"
            full_path = os.path.join(doc_dir, filename)
            # URL path served by FastAPI StaticFiles:
            #   /storage/images/<stem>/page{n}_xref{xref}.<ext>
            image_url_path = f"/storage/images/{stem}/{filename}"

            already_exists = os.path.exists(full_path)
            saved_caption: str | None = load_saved_caption(full_path) if already_exists else None

            if already_exists:
                logger.debug(
                    "Page %d xref %d — image already on disk (caption=%s), skipping save.",
                    page_num,
                    xref,
                    "yes" if saved_caption else "no",
                )
            else:
                # ── Save to disk ─────────────────────────────────────────────
                try:
                    with open(full_path, "wb") as fh:
                        fh.write(image_bytes)
                except OSError as exc:
                    logger.error("Failed to save image '%s': %s", full_path, exc)
                    total_skipped += 1
                    continue

            results.append(
                {
                    "page": page_num,
                    "index": img_idx + 1,
                    "filename": filename,
                    "image_path": image_url_path,
                    "full_path": full_path,
                    # bytes=None when image already existed — captioner skips API call
                    "bytes": image_bytes if not already_exists else None,
                    "already_exists": already_exists,
                    # Non-None only when sidecar .caption.txt exists on disk
                    "saved_caption": saved_caption,
                }
            )

    doc.close()

    total_reused = sum(1 for r in results if r["already_exists"])
    total_new    = len(results) - total_reused

    logger.info(
        "Image extraction complete for '%s': found=%d  saved=%d  reused=%d  skipped=%d",
        source_name,
        total_found,
        total_new,
        total_reused,
        total_skipped,
    )
    return results
