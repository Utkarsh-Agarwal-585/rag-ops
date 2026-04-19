"""
LLM integration service — BYOK (Bring Your Own Key).

Supports Gemini and OpenAI as providers.  API keys are passed per-request
and never stored.
"""

from __future__ import annotations

import logging

import requests

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are an assistant. Answer using the provided context.\n\n"
    "RULES:\n"
    "1. If the user asks about a topic and the context contains relevant text, "
    "answer using that text.\n"
    "2. Some context chunks are of type [IMAGE] — these contain captions describing "
    "diagrams or figures extracted from documents. Each image chunk includes an "
    "image_path (e.g. /storage/images/filename.png).\n"
    "3. When the user asks for an image, diagram, or figure, look through the [IMAGE] "
    "chunks in the context. Pick the one whose caption best matches the topic. "
    "DO NOT pick book covers, title pages, or decorative images — only pick diagrams "
    "that show architecture, flows, or technical concepts.\n"
    "4. When you pick an image, include ONLY its image_path in your response "
    "using this exact format: [Image: <image_path>]\n"
    "5. If a text chunk mentions a figure (e.g. 'Figure 1-1 shows the single server "
    "setup') and an [IMAGE] chunk's caption mentions the same figure number, that is "
    "the correct image to reference.\n"
    "6. Only say 'I don't know' if the context has absolutely no relevant information."
)

_GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
_OPENAI_URL = "https://api.openai.com/v1/chat/completions"


def enrich_query(query: str, history: list[dict]) -> str:
    """
    Rewrite a short follow-up query into a standalone query using conversation
    history.  This ensures retrieval finds the right chunks even when the user
    asks a context-dependent question like "can you give an image?".

    Uses simple heuristics — no extra LLM call needed for most cases.
    """
    if not history:
        return query

    query_lower = query.lower().strip()

    # Detect follow-up patterns that need context injection.
    followup_patterns = (
        "can you", "give me", "show me", "what about", "tell me more",
        "explain more", "more detail", "give an image", "give image",
        "show image", "show diagram", "give diagram", "provide image",
    )
    is_followup = (
        len(query.split()) <= 10
        or any(query_lower.startswith(p) for p in followup_patterns)
        or query_lower in ("yes", "no", "why", "how", "what", "when", "where")
    )

    if not is_followup:
        return query

    # Extract topic from the last user message in history.
    last_user_msgs = [m["content"] for m in history if m["role"] == "user"]
    if not last_user_msgs:
        return query

    last_topic = last_user_msgs[-1][:120]  # cap to avoid token bloat

    # Combine: keep the user's intent but add the topic context.
    enriched = f"{query} (context: {last_topic})"
    return enriched


def build_prompt(query: str, context_chunks: list[dict], *, history: list[dict] | None = None) -> str:
    """
    Assemble the final prompt from retrieved chunks, conversation history,
    and the user query.

    Each chunk dict must have at least a 'content' key.  Image chunks
    will have their caption included as text context.
    """
    context_parts: list[str] = []
    for i, item in enumerate(context_chunks, 1):
        chunk = item["chunk"]
        prefix = f"[{chunk.type.upper()}]" if hasattr(chunk, "type") else ""
        source = chunk.source if hasattr(chunk, "source") else "unknown"

        # For image chunks, include the image_path so the LLM can reference it.
        image_line = ""
        if hasattr(chunk, "type") and chunk.type == "image":
            img_path = chunk.metadata.get("image_path", "")
            if img_path:
                image_line = f"\nimage_path: {img_path}"

        context_parts.append(
            f"--- Chunk {i} {prefix} (source: {source}) ---\n{chunk.content}{image_line}"
        )

    context_block = "\n\n".join(context_parts)

    # Build conversation history block (last 5 turns max).
    history_block = ""
    if history:
        turns = []
        for m in history[-10:]:  # last 5 user+assistant pairs
            role_label = "User" if m["role"] == "user" else "Assistant"
            turns.append(f"{role_label}: {m['content']}")
        history_block = (
            "\n\nConversation History (most recent last):\n"
            + "\n".join(turns)
        )

    return (
        f"{_SYSTEM_PROMPT}"
        f"{history_block}\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question:\n{query}\n\n"
        f"Answer:"
    )


def call_llm(
    prompt: str,
    api_key: str,
    provider: str = "gemini",
) -> str:
    """
    Send the prompt to the chosen LLM provider and return the response text.

    Parameters
    ----------
    prompt   : The fully assembled prompt (context + question).
    api_key  : User-supplied API key — never stored.
    provider : "gemini" or "openai".

    Returns
    -------
    The LLM's answer as a plain string.

    Raises
    ------
    ValueError  : Unknown provider.
    RuntimeError: API call failed.
    """
    provider = provider.lower().strip()

    if provider == "gemini":
        return _call_gemini(prompt, api_key)
    elif provider == "openai":
        return _call_openai(prompt, api_key)
    else:
        raise ValueError(f"Unsupported LLM provider: '{provider}'. Use 'gemini' or 'openai'.")


def _call_gemini(prompt: str, api_key: str) -> str:
    """Call the Gemini generateContent REST endpoint."""
    from app.config import GEMINI_MODEL
    url = _GEMINI_URL.format(model=GEMINI_MODEL)
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
    }
    try:
        resp = requests.post(
            url,
            json=payload,
            params={"key": api_key},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return (
            data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "")
            .strip()
        )
    except requests.RequestException as exc:
        logger.error("Gemini API call failed: %s", exc)
        raise RuntimeError(f"Gemini API error: {exc}") from exc


def _call_openai(prompt: str, api_key: str) -> str:
    """Call the OpenAI chat completions endpoint."""
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    try:
        resp = requests.post(
            _OPENAI_URL,
            json=payload,
            headers=headers,
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        return (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
    except requests.RequestException as exc:
        logger.error("OpenAI API call failed: %s", exc)
        raise RuntimeError(f"OpenAI API error: {exc}") from exc
