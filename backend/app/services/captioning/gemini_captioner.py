"""
Multimodal image captioning service.

Supports both Gemini and OpenAI vision models.  The provider is selected
per-request so users can use whichever API key they have.

Error handling
--------------
* If the API key is missing the function returns FALLBACK_CAPTION immediately.
* On API failure the call is retried **once** after a short pause.
* After both attempts fail FALLBACK_CAPTION is returned so the overall
  ingestion pipeline never crashes because of a captioning error.
"""

from __future__ import annotations

import base64
import io
import logging
import time

import requests as http_requests

logger = logging.getLogger(__name__)

CAPTION_PROMPT: str = (
    "Describe this diagram or image clearly so it can be understood without "
    "seeing it. Focus on flows, components, labels, and relationships. "
    "Be concise but complete."
)

FALLBACK_CAPTION: str = "Diagram present but description unavailable."

# Retry configuration
_MAX_RETRIES: int = 4          # Total attempts (1 original + 3 retries)
_BASE_DELAY: float = 2.0       # Initial backoff delay in seconds
_MAX_DELAY: float = 60.0       # Cap on backoff delay
_RATE_LIMIT_CODES: frozenset[int] = frozenset({429, 503})


def _backoff_delay(attempt: int) -> float:
    """
    Exponential backoff: 2s, 4s, 8s, … capped at 60s.
    attempt is 1-indexed (attempt=1 → first retry after initial failure).
    """
    return min(_BASE_DELAY * (2 ** (attempt - 1)), _MAX_DELAY)


def generate_image_caption(
    image_bytes: bytes,
    api_key: str,
    model_name: str = "gemini-2.5-flash",
    provider: str = "gemini",
) -> str:
    """
    Generate a text caption for *image_bytes* using the specified provider.

    Dispatches to the appropriate provider implementation.  Both providers
    use the same CAPTION_PROMPT and the same retry/backoff configuration.

    Parameters
    ----------
    image_bytes : Raw bytes of the image (PNG, JPEG, WebP, etc.).
    api_key     : API key for the chosen provider.  Empty string → FALLBACK_CAPTION.
    model_name  : Gemini model identifier.  Ignored for OpenAI (uses OPENAI_VISION_MODEL).
    provider    : "gemini" or "openai".

    Returns
    -------
    A descriptive caption string, or FALLBACK_CAPTION on persistent failure.
    """
    if not api_key:
        logger.warning("No API key provided — skipping captioning.")
        return FALLBACK_CAPTION

    provider = provider.lower().strip()
    if provider == "openai":
        return _caption_openai(image_bytes, api_key)
    return _caption_gemini(image_bytes, api_key, model_name)

def _caption_gemini(image_bytes: bytes, api_key: str, model_name: str) -> str:
    """
    Caption an image via the Gemini vision API with exponential backoff.

    Decodes the raw bytes into a PIL image (normalising to RGB if needed),
    then calls the Gemini generateContent endpoint.  On 429/503 errors the
    call is retried with exponential backoff up to _MAX_RETRIES attempts.
    """
    try:
        from google import genai
        import PIL.Image
    except ImportError as exc:
        logger.error("Missing dependency for Gemini captioning (%s).", exc)
        return FALLBACK_CAPTION

    try:
        pil_image = PIL.Image.open(io.BytesIO(image_bytes))
        if pil_image.mode not in ("RGB", "L"):
            pil_image = pil_image.convert("RGB")
    except Exception as exc:
        logger.error("Cannot decode image bytes: %s", exc)
        return FALLBACK_CAPTION

    client = genai.Client(api_key=api_key)

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[CAPTION_PROMPT, pil_image],
            )
            caption = (response.text or "").strip()
            if caption:
                logger.debug("Gemini caption (attempt %d): %.80s…", attempt, caption)
                return caption
            logger.warning("Gemini attempt %d returned empty caption.", attempt)

        except Exception as exc:
            exc_str = str(exc)
            # Detect rate limit / server overload from the exception message.
            is_rate_limited = any(str(c) in exc_str for c in _RATE_LIMIT_CODES)

            if attempt < _MAX_RETRIES:
                delay = _backoff_delay(attempt)
                if is_rate_limited:
                    logger.warning(
                        "Gemini rate-limited (attempt %d/%d) — backing off %.1fs: %s",
                        attempt, _MAX_RETRIES, delay, exc,
                    )
                else:
                    logger.warning(
                        "Gemini attempt %d/%d failed — retrying in %.1fs: %s",
                        attempt, _MAX_RETRIES, delay, exc,
                    )
                time.sleep(delay)
            else:
                logger.error(
                    "Gemini captioning failed after %d attempts: %s", _MAX_RETRIES, exc
                )

    return FALLBACK_CAPTION


def _caption_openai(image_bytes: bytes, api_key: str) -> str:
    """
    Caption an image via the OpenAI vision API with exponential backoff.

    Encodes the raw bytes as base64 and sends them as an inline image_url
    to the chat completions endpoint.  HTTPError status codes are inspected
    to distinguish rate limits (429/503) from other errors.  Non-HTTP errors
    (network timeouts, etc.) are also retried with the same backoff schedule.
    """
    from app.config import OPENAI_VISION_MODEL
    b64 = base64.b64encode(image_bytes).decode("utf-8")

    payload = {
        "model": OPENAI_VISION_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": CAPTION_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    },
                ],
            }
        ],
        "max_tokens": 500,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = http_requests.post(
                "https://api.openai.com/v1/chat/completions",
                json=payload,
                headers=headers,
                timeout=30,
            )
            resp.raise_for_status()
            caption = (
                resp.json()
                .get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
            )
            if caption:
                logger.debug("OpenAI caption (attempt %d): %.80s…", attempt, caption)
                return caption
            logger.warning("OpenAI attempt %d returned empty caption.", attempt)

        except http_requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else 0
            if attempt < _MAX_RETRIES:
                delay = _backoff_delay(attempt)
                if status in _RATE_LIMIT_CODES:
                    logger.warning(
                        "OpenAI rate-limited %d (attempt %d/%d) — backing off %.1fs.",
                        status, attempt, _MAX_RETRIES, delay,
                    )
                else:
                    logger.warning(
                        "OpenAI HTTP %d (attempt %d/%d) — retrying in %.1fs.",
                        status, attempt, _MAX_RETRIES, delay,
                    )
                time.sleep(delay)
            else:
                logger.error("OpenAI captioning failed after %d attempts: %s", _MAX_RETRIES, exc)

        except Exception as exc:
            if attempt < _MAX_RETRIES:
                delay = _backoff_delay(attempt)
                logger.warning(
                    "OpenAI attempt %d/%d failed — retrying in %.1fs: %s",
                    attempt, _MAX_RETRIES, delay, exc,
                )
                time.sleep(delay)
            else:
                logger.error("OpenAI captioning failed after %d attempts: %s", _MAX_RETRIES, exc)

    return FALLBACK_CAPTION
