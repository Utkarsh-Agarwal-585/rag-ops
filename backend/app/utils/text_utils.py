from __future__ import annotations

import re


def normalize_text(text: str) -> str:
    """
    Clean up raw extracted text so downstream chunkers get consistent input.

    Steps
    -----
    1. Unify line endings (CRLF / CR → LF).
    2. Collapse repeated spaces and tabs within each line to a single space.
    3. Strip trailing whitespace from every line.
    4. Collapse three or more consecutive blank lines down to two.
    5. Strip leading / trailing whitespace from the whole document.
    """
    # Normalise line endings
    text = re.sub(r"\r\n|\r", "\n", text)

    # Collapse intra-line whitespace, strip trailing spaces per line
    lines = [re.sub(r"[ \t]+", " ", line).rstrip() for line in text.split("\n")]
    text = "\n".join(lines)

    # Remove excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()
