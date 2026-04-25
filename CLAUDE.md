# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Commands

### Backend

```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Development (auto-reload)
uvicorn app.main:app --reload

# Production
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

Optional environment variables:
```bash
export GEMINI_API_KEY=your_key_here
export GEMINI_MODEL=gemini-2.5-flash   # default
```

There are no tests yet. The interactive API docs are at `http://localhost:8000/docs`.

### Frontend

```bash
cd frontend
npm install
npm start   # http://localhost:3000
```

---

## Architecture

### Two-phase pipeline

Every user interaction follows one of two paths:

**Upload** (`POST /api/v1/upload`) → `ingestor.py` → parse → chunk → embed → deduplicate → FAISS + BM25 index → `memory_store.py`

**Query** (`POST /api/v1/query`) → cache check → intent classify → (chitchat/capability: LLM with chitchat prompt, skip retrieval) → query enrichment → hybrid retrieval → relevance gate → LLM call → cache write → response

### Ingestion pipeline (`services/ingestion/ingestor.py`)

`ingestor.py` is the single orchestrator — it is the only place routes call into. It coordinates in order:
1. **Parse** — `pdf_parser.py` (pdfplumber) or `text_parser.py`; `log_parser.py` detects structured logs
2. **Chunk** — `doc_chunker.py` (400-word sliding window, 50-word overlap) or `log_chunker.py` (10-min time windows)
3. **Image pipeline** (PDFs only) — `image_extractor.py` (PyMuPDF) → `gemini_captioner.py` → captions persisted as `.caption.txt` sidecars under `backend/storage/images/<doc_stem>/`
4. **Persist** — `memory_store.store_chunks()` runs before indexing so chunks are always stored even if indexing fails
5. **Index** — `embedding_service.py` (all-MiniLM-L6-v2, 384-dim) → `deduplication_service.py` (cosine sim ≥ 0.85 union-find) → `vector_service.py` (FAISS) + `bm25_service.py`

Image pipeline and retrieval indexing failures are caught internally and never abort text ingestion.

### Query pipeline (`routes/query.py` + `services/retrieval/`)

**Intent classification** (`llm_service.classify_intent`) runs before retrieval on every query:
- `chitchat` — greetings/pleasantries: calls LLM with `_CHITCHAT_SYSTEM_PROMPT`, no retrieval
- `capability` — "what can you do?" questions: same chitchat path
- `rag` — everything else: full retrieval pipeline

**Query enrichment** (`llm_service.enrich_query`): short or follow-up queries (≤10 words, or starting with "show me", "can you", etc.) are rewritten to include the last user message as context, so retrieval finds the right chunks even for context-dependent questions like "give me an image".

**Relevance gate** (OR logic): after retrieval, the raw FAISS cosine similarity and raw BM25 score are captured *before* min-max normalization (which would destroy absolute scale). An LLM call is made only if:
```
max_raw_faiss >= 0.30  OR  max_raw_bm25 >= 1.0
```
- FAISS threshold 0.30: semantic closeness to the document domain
- BM25 threshold 1.0: a meaningful domain keyword appears in the corpus (noise words like "aws" score ~0.05 due to low IDF; domain terms like "concurrency" score 2–5)
- OR logic handles phrased questions where framing dilutes FAISS but BM25 picks up the keyword

If neither passes, a static "not in your documents" message is returned without calling the LLM.

**Hybrid retrieval** (`retrieval_service.retrieve`): `0.6 × FAISS + 0.4 × BM25`, both min-max normalized to [0, 1] before combining. Returns `(results, max_raw_faiss, max_raw_bm25)` — the raw scores are needed by the relevance gate in the route. Default top 5 chunks (`RERANK_TOP_N`). For image/diagram queries, additional image chunks are injected via figure-reference matching → same-page matching → embedding similarity fallback.

**BM25 tokenizer** (`bm25_service._tokenize`): uses `re.findall(r'\b\w+\b', text.lower())` — strips punctuation so "concurrency?" matches "concurrency" in the corpus.

### Storage

- **`memory_store.py`** is a deliberate thin façade (a list + dict in process memory). All chunks and indexes are lost on server restart. It is the **only file to change** when swapping to a real vector DB.
- **Images on disk** (`backend/storage/images/`) survive restarts. Caption sidecars (`.caption.txt`) make re-uploads idempotent — no Gemini call is made if the sidecar already exists.

### Core data model

`Chunk` (in `models/chunk.py`) is the universal unit across ingestion, storage, and retrieval:
- `type`: `"doc"` | `"log"` | `"image"`
- `source`: filename, or `filename#page=N` for image chunks
- `metadata`: image chunks carry `"page"` and `"image_path"` keys
- `embedding`: excluded from serialization (`exclude=True`), lives only in memory

### LLM prompt rules (`llm_service._SYSTEM_PROMPT`)

The RAG system prompt has 7 rules. Rules 6 and 7 control edge-case behavior:
- **Rule 6:** If context is domain-related but doesn't specifically answer, reply in 1–2 sentences only with one short factual hint. No full explanations or tutorials.
- **Rule 7:** If a question spans multiple topics and only some are in the docs, explicitly state which topics are NOT found, then answer only what IS covered. Never silently answer out-of-scope topics.

### Configuration

All tunable constants are in `app/config.py` — chunk sizes, overlap, weights, thresholds, image limits, directory paths. No magic numbers anywhere else. Key retrieval thresholds:
- `RETRIEVAL_RELEVANCE_THRESHOLD = 0.30` — min raw FAISS cosine similarity
- `RETRIEVAL_BM25_RELEVANCE_THRESHOLD = 1.0` — min raw BM25 score for keyword gate

### Key patterns

- `from __future__ import annotations` in every module
- All blocking I/O and CPU work runs via `asyncio.to_thread()` — route handlers are `async def` but never block the event loop directly
- Logging via `logging.getLogger(__name__)` throughout
- API keys are BYOK (Bring Your Own Key) — passed per-request via form field or JSON body, never stored or logged

### Frontend

`App.js` owns all state: `apiKey`, `provider`, `messages[]`, `uploaded`. Chat is disabled until a file is successfully uploaded. History is sent client-side on every request (no server-side session). The `include_sources` flag on `/query` must be `true` to receive source cards; the frontend currently does not set this flag.
