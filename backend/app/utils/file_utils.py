from __future__ import annotations

import os
import tempfile

import aiofiles


async def save_upload_file(filename: str, content: bytes) -> str:
    """
    Persist raw upload bytes to a named temporary file.

    The original file extension is preserved so that downstream parsers can
    detect the file type from the path alone if needed.

    Returns the absolute path to the temporary file.
    """
    suffix = os.path.splitext(filename or "")[1]
    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)  # close the OS-level fd; aiofiles will reopen it

    async with aiofiles.open(tmp_path, "wb") as fh:
        await fh.write(content)

    return tmp_path


def validate_extension(filename: str | None, allowed: frozenset[str]) -> bool:
    """Return True if *filename* carries an extension present in *allowed*."""
    if not filename:
        return False
    return os.path.splitext(filename)[1].lower() in allowed


def cleanup_temp_file(path: str) -> None:
    """Delete a temporary file, silently ignoring errors (already deleted, etc.)."""
    try:
        os.unlink(path)
    except OSError:
        pass
