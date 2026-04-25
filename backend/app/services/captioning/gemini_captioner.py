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

_MAX_RETRIES: int = 2
_RETRY_DELAY_SECONDS: float = 1.5


def generate_image_caption(
    image_bytes: bytes,
    api_key: str,
    model_name: str = "gemini-2.5-flash",
    provider: str = "gemini",
) -> str:
    """
    Generate a text caption for *image_bytes* using the specified provider.

    Parameters
    ----------
    image_bytes : Raw bytes of the image (PNG, JPEG, WebP, etc.).
    api_key     : API key for the chosen provider.
    model_name  : Model identifier (ignored for OpenAI — uses gpt-4o-mini).
    provider    : "gemini" or "openai".
    """
    if not api_key:
        logger.warning("No API key provided — skipping captioning.")
        return FALLBACK_CAPTION

    provider = provider.lower().strip()
    if provider == "openai":
        return _caption_openai(image_bytes, api_key)
    return _caption_gemini(image_bytes, api_key, model_name)

def _caption_gemini(image_bytes: bytes, api_key: str, model_name: str) -> str:
    """Caption via Gemini vision."""
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
            logger.warning("Gemini attempt %d/%d failed: %s", attempt, _MAX_RETRIES, exc)
        if attempt < _MAX_RETRIES:
            time.sleep(_RETRY_DELAY_SECONDS)

    logger.error("All Gemini caption attempts failed.")
    return FALLBACK_CAPTION


def _caption_openai(image_bytes: bytes, api_key: str) -> str:
    """Caption via OpenAI GPT-4o-mini vision."""
    b64 = base64.b64encode(image_bytes).decode("utf-8")

    payload = {
        "model": "gpt-4o-mini",
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
        except Exception as exc:
            logger.warning("OpenAI attempt %d/%d failed: %s", attempt, _MAX_RETRIES, exc)
        if attempt < _MAX_RETRIES:
            time.sleep(_RETRY_DELAY_SECONDS)

    logger.error("All OpenAI caption attempts failed.")
    return FALLBACK_CAPTION
