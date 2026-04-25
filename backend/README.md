# RAG Assistant — Backend

Full-stack Retrieval-Augmented Generation backend built with **FastAPI**. Handles document ingestion, hybrid retrieval, intent classification, and LLM-grounded Q&A.

---

## Tech Stack

| Component | Library |
|-----------|---------|
| Web framework | FastAPI + Uvicorn |
| PDF text extraction | pdfplumber |
| PDF image extraction | PyMuPDF (fitz) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector search | faiss-cpu (IndexFlatIP) |
| Keyword search | rank-bm25 (BM25Okapi) |
| LLM | Gemini / OpenAI (BYOK) |
| Async file I/O | aiofiles |
| Data validation | Pydantic v2 |

---

## Architecture

```
backend/
└── app/
    ├── main.py                  # FastAPI app factory, CORS, static mounts
    ├── config.py                # All tunable constants (single source of truth)
    ├── routes/
    │   ├── upload.py            # POST /api/v1/upload
    │   ├── chunks.py            # GET  /api/v1/chunks[/stats/index-stats/cache-stats]
    │   └── query.py             # POST /api/v1/query + /search
    ├── services/
    │   ├── ingestion/
    │   │   └── ingestor.py      # Pipeline orchestrator (single entry point)
    │   ├── parsing/
    │   │   ├── text_parser.py   # .txt / .log → plain text
    │   │   ├── pdf_parser.py    # .pdf → plain text (pdfplumber)
    │   │   ├── log_parser.py    # plain text → list[LogEntry]
    │   │   └── image_extractor.py  # PDF image extraction (PyMuPDF)
    │   ├── chunking/
    │   │   ├── doc_chunker.py   # Sliding-window word chunker (400w, 50w overlap)
    │   │   └── log_chunker.py   # Time-window log summariser (10-min buckets)
    │   ├── captioning/
    │   │   └── gemini_captioner.py  # Gemini / OpenAI vision captioning
    │   └── retrieval/
    │       ├── embedding_service.py     # all-MiniLM-L6-v2, 384-dim
    │       ├── deduplication_service.py # Union-find dedup at cosine sim >= 0.85
    │       ├── vector_service.py        # FAISS IndexFlatIP
    │       ├── bm25_service.py          # BM25Okapi with regex tokenizer
    │       ├── retrieval_service.py     # Hybrid scoring + image injection
    │       ├── llm_service.py           # Intent classify + prompt builder + LLM calls
    │       └── cache_service.py         # In-memory TTL query cache (15 min)
    ├── models/
    │   ├── chunk.py             # Chunk — core unit across ingestion + retrieval
    │   └── log_entry.py         # LogEntry — parsed log line
    ├── storage/
    │   └── memory_store.py      # In-memory list + dict (designed to be swapped)
    └── utils/
        ├── text_utils.py        # Text normalisation helpers
        └── file_utils.py        # Async temp-file helpers
```

---

## Setup

```bash
cd backend
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Optional environment variables:

```bash
export GEMINI_API_KEY=your_key_here   # used for image captioning
export GEMINI_MODEL=gemini-2.5-flash  # default
```

---

## Running

```bash
# Development (auto-reload)
uvicorn app.main:app --reload

# Production
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

Interactive docs: **http://localhost:8000/docs**

---

## Query Pipeline

Every `POST /api/v1/query` follows this flow:

```
1. Cache check (query + history digest + provider)
2. Intent classification (classify_intent)
   ├── chitchat / capability → LLM with _CHITCHAT_SYSTEM_PROMPT (no retrieval)
   └── rag → continue below
3. Query enrichment (enrich_query) — rewrites short follow-ups with last topic
4. Hybrid retrieval (FAISS + BM25, top 20 each → normalize → 0.6×FAISS + 0.4×BM25 → top 5)
5. Relevance gate (OR logic on raw pre-normalization scores):
   max_raw_faiss >= 0.30  OR  max_raw_bm25 >= 1.0
   └── Both fail → static "not in documents" reply, no LLM call
6. LLM call (build_prompt → call_llm)
7. Cache write + return response
```

---

## API Reference

### `POST /api/v1/upload`

```bash
curl -X POST http://localhost:8000/api/v1/upload \
     -F "file=@./sample.pdf" \
     -F "api_key=your_key" \
     -F "provider=gemini"
```

**Response:**
```json
{
  "message": "File processed successfully",
  "source": "sample.pdf",
  "chunks_created": 145,
  "breakdown": { "doc": 132, "log": 0, "image": 13 },
  "images": [{ "page": 5, "image_path": "/storage/images/sample/page5_xref34.png", "caption": "..." }]
}
```

---

### `POST /api/v1/query`

```json
{
  "query": "What is load balancing?",
  "api_key": "your_key",
  "provider": "gemini",
  "include_sources": true,
  "history": [
    {"role": "user", "content": "explain caching"},
    {"role": "assistant", "content": "Caching stores..."}
  ]
}
```

---

### `POST /api/v1/search`

Retrieve chunks without calling an LLM (no API key needed):

```json
{ "query": "load balancer", "top_n": 5 }
```

---

### Other Endpoints

| Endpoint | Purpose |
|---|---|
| GET /api/v1/chunks | Paginated chunk listing (?source=, ?limit=, ?offset=) |
| GET /api/v1/chunks/stats | Aggregate counts by type + source list |
| GET /api/v1/chunks/index-stats | FAISS + BM25 index state |
| GET /api/v1/chunks/cache-stats | Query cache hit/miss stats |
| GET /health | Liveness probe |

---

## Configuration

Key constants in `app/config.py`:

| Constant | Default | Purpose |
|---|---|---|
| `MAX_UPLOAD_SIZE_BYTES` | 50 MB | Upload size limit |
| `DOC_CHUNK_SIZE_WORDS` | 400 | Words per text chunk |
| `DOC_OVERLAP_WORDS` | 50 | Overlap between chunks |
| `HYBRID_VECTOR_WEIGHT` | 0.6 | FAISS score weight |
| `HYBRID_BM25_WEIGHT` | 0.4 | BM25 score weight |
| `RETRIEVAL_TOP_K` | 20 | Candidates from each retriever |
| `RERANK_TOP_N` | 5 | Final chunks after re-ranking |
| `RETRIEVAL_RELEVANCE_THRESHOLD` | 0.30 | Min raw FAISS cosine for LLM call |
| `RETRIEVAL_BM25_RELEVANCE_THRESHOLD` | 1.0 | Min raw BM25 score for LLM call |
| `DEDUP_SIMILARITY_THRESHOLD` | 0.85 | Cosine sim above this → merge chunks |
| `MAX_IMAGES_PER_PDF` | 20 | Image extraction cap per PDF |
