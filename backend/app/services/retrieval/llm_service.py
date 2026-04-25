"""
LLM integration service — BYOK (Bring Your Own Key).

Supports Gemini and OpenAI as providers.  API keys are passed per-request
and never stored.
"""

from __future__ import annotations

import logging
import re
from typing import Literal

import requests

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are RAG Assistant, a warm and knowledgeable AI. Answer questions using "
    "the provided context from the user's uploaded documents.\n\n"
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
    "6. If the retrieved context is related to the document domain but doesn't "
    "specifically answer the question, respond in 1–2 sentences only: acknowledge "
    "what the documents cover, note what is missing, and optionally add one short "
    "factual hint that is directly relevant to the document's domain. Do NOT provide "
    "full explanations, tutorials, step-by-step answers, or any information about "
    "topics that are outside the uploaded documents' scope.\n"
    "7. If the question asks about MULTIPLE topics and only SOME of them are covered "
    "in the uploaded documents, you MUST: (a) explicitly state upfront which topic(s) "
    "are NOT found in the uploaded documents, then (b) answer ONLY the topic(s) that "
    "ARE covered by the document context. Never silently answer topics that are not "
    "in the documents. For example: 'The uploaded documents do not contain information "
    "about EC2. For Lambda, the documents explain...'"
)

_CHITCHAT_SYSTEM_PROMPT = (
    "You are RAG Assistant, a warm and knowledgeable AI. The user has already "
    "uploaded their documents and is now in the Q&A session.\n\n"
    "The user has sent a conversational message — a greeting, pleasantry, or "
    "question about your capabilities. Respond naturally and warmly. "
    "If it's a greeting, reply briefly and let them know you're ready to answer "
    "questions about their uploaded documents. "
    "If they ask what you can do, explain concisely: ask questions about the content "
    "of their uploaded files, get cited answers, and view extracted diagrams. "
    "Keep your response short (2–4 sentences max)."
)


_GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
_OPENAI_URL = "https://api.openai.com/v1/chat/completions"

# Matches queries that are purely greetings, pleasantries, thanks, or farewells.
_CHITCHAT_RE = re.compile(
    r"^\s*("
    r"(?:hi+|hello+|hey+)\s*(?:there|everyone|folks|all|friend|buddy|guys)?\s*[,!]?\s*(?:how\s+are\s+you(?:\s+doing)?)?|"
    r"howdy|greetings?|good\s+(?:morning|afternoon|evening|night|day)|"
    r"what'?s\s+up|sup\b|"
    r"how\s+are\s+you(?:\s+doing)?|how'?s\s+it\s+going|how\s+do\s+you\s+do|"
    r"thank(?:s|\s+you)(?:\s+(?:so\s+)?much)?|cheers\b|appreciate\s+(?:it|that)|"
    r"bye+\b|goodbye\b|see\s+you\b|take\s+care\b"
    r")\s*[!?.,]*\s*$",
    re.IGNORECASE,
)

# Matches questions about the assistant's identity or capabilities.
_CAPABILITY_RE = re.compile(
    r"what\s+can\s+you\s+do|what\s+do\s+you\s+do|"
    r"what\s+are\s+you(?:\s+for|\s+capable\s+of)?|who\s+are\s+you|"
    r"what\s+is\s+this(?:\s+(?:tool|app|assistant|system))?|"
    r"how\s+(?:do|can)\s+you\s+help|tell\s+me\s+about\s+yourself|"
    r"how\s+does\s+this\s+work|how\s+do\s+you\s+work|"
    r"(?:your\s+)?capabilities\b|what\s+(?:can\s+you|do\s+you)\s+help\s+with|"
    r"^\s*help\s*[!?.]?\s*$",
    re.IGNORECASE,
)


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


def classify_intent(query: str) -> Literal["chitchat", "capability", "rag"]:
    """
    Classify a query before hitting the retrieval pipeline.

    chitchat   — greetings, pleasantries, thanks, farewells
    capability — questions about what the assistant does
    rag        — everything else; use the full retrieval pipeline
    """
    stripped = query.strip()
    if _CAPABILITY_RE.search(stripped):
        return "capability"
    if _CHITCHAT_RE.match(stripped):
        return "chitchat"
    return "rag"


def build_chitchat_prompt(query: str, history: list[dict]) -> str:
    """Assemble a prompt for chitchat/capability queries — no document context."""
    history_block = ""
    if history:
        turns = [
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in history[-6:]
        ]
        history_block = "\n\nConversation History:\n" + "\n".join(turns)
    return f"{_CHITCHAT_SYSTEM_PROMPT}{history_block}\n\nUser: {query}\n\nAssistant:"



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
    *,
    system_prompt: str | None = None,
) -> str:
    """
    Send the prompt to the chosen LLM provider and return the response text.

    Parameters
    ----------
    prompt        : The fully assembled prompt (context + question).
    api_key       : User-supplied API key — never stored.
    provider      : "gemini" or "openai".
    system_prompt : Override the default RAG system prompt. Used by the
                    chitchat path to swap in a conversational persona.

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
        return _call_gemini(prompt, api_key, system_prompt=system_prompt)
    elif provider == "openai":
        return _call_openai(prompt, api_key, system_prompt=system_prompt)
    else:
        raise ValueError(f"Unsupported LLM provider: '{provider}'. Use 'gemini' or 'openai'.")


def _call_gemini(prompt: str, api_key: str, *, system_prompt: str | None = None) -> str:
    """Call the Gemini generateContent REST endpoint.

    For Gemini the system instruction is already embedded in the prompt string
    by build_prompt() / build_chitchat_prompt(), so system_prompt is accepted
    for API consistency but not used separately here.
    """
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


def _call_openai(prompt: str, api_key: str, *, system_prompt: str | None = None) -> str:
    """Call the OpenAI chat completions endpoint."""
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": system_prompt or _SYSTEM_PROMPT},
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
