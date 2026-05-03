# RAG Assistant

A full-stack Retrieval-Augmented Generation system. Upload documents, ask questions, get answers grounded in your content with source citations and diagram rendering.

---

## Project Structure

```
rag-project/
├── README.md                   <- you are here
├── HLD.md                      <- High Level Design
├── .gitignore
├── backend/                    <- FastAPI Python backend
│   ├── ARCHITECTURE.md         <- Detailed technical architecture
│   ├── requirements.txt
│   └── app/
│       ├── main.py             <- FastAPI app factory, middleware, static mounts
│       ├── config.py           <- All tunable constants (single source of truth)
│       ├── models/
│       │   ├── chunk.py        <- Chunk — core unit of ingested content
│       │   └── log_entry.py    <- LogEntry — parsed log line
│       ├── routes/
│       │   ├── upload.py       <- POST /api/v1/upload
│       │   ├── chunks.py       <- GET /api/v1/chunks, /stats, /index-stats, /cache-stats
│       │   └── query.py        <- POST /api/v1/query, /search
│       ├── services/
│       │   ├── ingestion/
│       │   │   └── ingestor.py          <- Pipeline orchestrator (single entry point)
│       │   ├── parsing/
│       │   │   ├── pdf_parser.py        <- PDF text extraction (pdfplumber)
│       │   │   ├── text_parser.py       <- Plain text / log file reader
│       │   │   ├── log_parser.py        <- Structured log parser (regex + JSON)
│       │   │   └── image_extractor.py   <- PDF image extraction (PyMuPDF)
│       │   ├── chunking/
│       │   │   ├── doc_chunker.py       <- Sliding-window word chunker
│       │   │   └── log_chunker.py       <- Time-window log summariser
│       │   ├── captioning/
│       │   │   └── gemini_captioner.py  <- Gemini / OpenAI vision captioning (exp. backoff)
│       │   └── retrieval/
│       │       ├── embedding_service.py     <- sentence-transformers embeddings
│       │       ├── deduplication_service.py <- Cosine-sim dedup (union-find)
│       │       ├── vector_service.py        <- FAISS in-memory index
│       │       ├── bm25_service.py          <- BM25 keyword index
│       │       ├── retrieval_service.py     <- Hybrid scoring + image injection
│       │       ├── llm_service.py           <- Prompt builder + BYOK LLM calls
│       │       ├── cache_service.py         <- In-memory TTL query cache
│       │       └── persistence_service.py   <- Save/load FAISS + BM25 + chunks to disk
│       ├── middleware/
│       │   └── correlation.py       <- x-correlation-id generation + propagation
│       ├── logging_config.py        <- JSON structured logging + correlation ID context
│       ├── storage/
│       │   └── memory_store.py      <- In-memory chunk store (swappable facade)
│       └── utils/
│           ├── text_utils.py        <- Text normalisation helpers
│           └── file_utils.py        <- Async temp-file helpers
├── frontend/                   <- React UI
│   ├── package.json
│   └── src/
│       ├── App.js              <- Root component, state management
│       ├── index.js            <- React entry point
│       ├── styles.css          <- All styles
│       └── components/
│           ├── Upload.js       <- File upload with full-screen loader
│           ├── UploadLoader.js <- Modal overlay loader (portal)
│           ├── Chat.js         <- Chat panel + input bar
│           ├── DocList.js      <- Collapsible sidebar list of uploaded docs with delete buttons
│           └── Message.js      <- Message bubble + source cards + image rendering
└── backend/storage/
    └── images/                 <- Extracted PDF images (per-doc subdirectories)
        └── <doc_stem>/
            ├── page5_xref34.png
            └── page5_xref34.caption.txt
```

---

## Quick Start

### Prerequisites

| Tool | Version |
|---|---|
| Python | 3.10+ |
| Node.js | 18+ |
| npm | 9+ |

You will need an API key from **Google AI Studio** (Gemini) or **OpenAI** to use the query and image captioning features.

---

### 1. Backend Setup

> **Note:** `.venv` is not committed to the repo (excluded by `.gitignore`). You must create it locally after cloning.

```bash
cd backend

# Create the virtual environment
python -m venv .venv

# Activate it
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows (cmd)
# .venv\Scripts\Activate.ps1     # Windows (PowerShell)

# Install dependencies
pip install -r requirements.txt
```

Optional — set a default Gemini key for image captioning (can also be provided per-request via the UI):

```bash
export GEMINI_API_KEY=your_key_here
export GEMINI_MODEL=gemini-2.5-flash   # optional, this is the default
```

Start the server:

```bash
# Development (auto-reload)
uvicorn app.main:app --reload

# Production
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

Backend runs at **http://localhost:8000**  
Interactive API docs at **http://localhost:8000/docs**

---

### 2. Frontend Setup

> **Note:** `node_modules` is not committed to the repo (excluded by `.gitignore`). Run `npm install` after cloning.

```bash
cd frontend
npm install
npm start
```

Frontend runs at **http://localhost:3000**

---

### 3. Using the App

1. Enter your **API key** and select a **provider** (Gemini or OpenAI) in the sidebar
2. Choose a `.pdf`, `.txt`, or `.log` file and click **Upload**
3. Wait for the upload loader to complete — the chat unlocks automatically
4. Ask questions in the chat input
5. Answers appear with source cards below; PDF diagrams render as images

---

## API Reference

### POST /api/v1/upload

Upload a document for ingestion.

```bash
curl -X POST http://localhost:8000/api/v1/upload \
  -F "file=@./document.pdf" \
  -F "api_key=your_key" \
  -F "provider=gemini"
```

**Response:**
```json
{
  "message": "File processed successfully",
  "source": "document.pdf",
  "chunks_created": 145,
  "breakdown": { "doc": 132, "log": 0, "image": 13 },
  "images": [
    { "page": 5, "image_path": "/storage/images/document/page5_xref34.png", "caption": "..." }
  ]
}
```

---

### POST /api/v1/query

Ask a question with optional conversation history.

```json
{
  "query": "What is a CDN?",
  "api_key": "your_key",
  "provider": "gemini",
  "include_sources": true,
  "history": [
    {"role": "user", "content": "explain caching"},
    {"role": "assistant", "content": "Caching stores frequently accessed data..."}
  ]
}
```

**Response:**
```json
{
  "answer": "A CDN (Content Delivery Network) is...",
  "sources": [
    { "content": "...", "source": "document.pdf", "type": "doc" },
    { "content": "...", "source": "document.pdf#page=5", "type": "image",
      "image_path": "/storage/images/document/page5_xref34.png" }
  ]
}
```

---

### POST /api/v1/search

Retrieve chunks without calling an LLM (no API key needed).

```json
{ "query": "load balancer", "top_n": 5 }
```

---

### Other Endpoints

| Endpoint | Purpose |
|---|---|
| GET /api/v1/documents | List all uploaded documents with chunk counts per type |
| DELETE /api/v1/documents/{filename} | Delete a document and all its data (chunks, vectors, BM25, images) |
| GET /api/v1/chunks | Paginated chunk listing (?source=, ?limit=, ?offset=) |
| GET /api/v1/chunks/stats | Aggregate counts by type + source list |
| GET /api/v1/chunks/index-stats | FAISS + BM25 index state |
| GET /api/v1/chunks/cache-stats | Query cache hit/miss stats |
| GET /storage/images/doc/file | Serve extracted PDF images |
| GET /health | Liveness probe |

---

## Configuration

All backend constants are in [`backend/app/config.py`](backend/app/config.py).

| Constant | Default | Purpose |
|---|---|---|
| MAX_UPLOAD_SIZE_BYTES | 50 MB | Upload size limit |
| DOC_CHUNK_SIZE_WORDS | 400 | Words per text chunk |
| DOC_OVERLAP_WORDS | 50 | Overlap between chunks |
| MAX_IMAGES_PER_PDF | 20 | Image extraction cap per PDF |
| MIN_IMAGE_SIZE_BYTES | 2 KB | Skip decorative icons below this |
| RERANK_TOP_N | 5 | Final chunks returned per query |
| HYBRID_VECTOR_WEIGHT | 0.6 | FAISS score weight in hybrid search |
| HYBRID_BM25_WEIGHT | 0.4 | BM25 score weight in hybrid search |
| DEDUP_SIMILARITY_THRESHOLD | 0.85 | Cosine sim above this merges chunks |
| RETRIEVAL_RELEVANCE_THRESHOLD | 0.30 | Min raw FAISS cosine similarity to call LLM |
| RETRIEVAL_BM25_RELEVANCE_THRESHOLD | 1.0 | Min raw BM25 score to call LLM (OR with FAISS) |

---

## Documentation

| Document | Description |
|---|---|
| [HLD.md](HLD.md) | High Level Design — system context, component diagram, data flows |
| [backend/ARCHITECTURE.md](backend/ARCHITECTURE.md) | Detailed technical architecture — sequence diagrams, decision trees, all design decisions |

---

## Tech Stack

### Backend

| Library | Purpose |
|---|---|
| FastAPI + Uvicorn | Async web framework |
| pdfplumber | PDF text extraction |
| PyMuPDF (fitz) | PDF image extraction |
| sentence-transformers | all-MiniLM-L6-v2 embeddings |
| faiss-cpu | Vector similarity search |
| rank-bm25 | Keyword search |
| scikit-learn | Cosine similarity for deduplication |
| google-genai | Gemini API client |
| Pillow | Image decoding |
| pydantic v2 | Data validation |
| aiofiles | Async file I/O |

### Frontend

| Library | Purpose |
|---|---|
| React 18 | UI framework |
| react-scripts | Build tooling (CRA) |
| Fetch API | HTTP requests (no axios) |
| Plain CSS | Styling (no framework) |

---

## Notes

- **Persistence:** Chunks, FAISS index, and BM25 corpus are saved to `backend/storage/index/` after every upload and restored on server startup — no re-upload needed after restarts. Extracted images and caption sidecars also persist on disk.
- **Duplicate uploads:** Re-uploading the same file evicts the old chunks from the store and indexes before ingesting fresh — no duplicates accumulate.
- **Structured logging:** All log output is JSON-formatted with a `correlation_id` field. Every response includes an `x-correlation-id` header for traceability.
- **API keys:** Keys are passed per-request and never stored or logged. Use HTTPS in production.
- **OpenAI model:** Both chat completions and image captioning use `gpt-4o-mini` when OpenAI is selected as the provider. Configured via `OPENAI_MODEL` and `OPENAI_VISION_MODEL` in `config.py` (both default to `gpt-4o-mini`). A standard `sk-...` key works — no model-specific key needed.
- **Image captioning:** Requires a valid Gemini or OpenAI key. Without a key, images are still extracted and stored but get a fallback caption. Captions are cached as `.caption.txt` sidecars so re-uploads skip API calls.
