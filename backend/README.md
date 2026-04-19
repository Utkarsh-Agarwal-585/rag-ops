# RAG Ingestion Backend

Production-grade document and log ingestion pipeline built with **FastAPI**.  
This service handles the **ingestion layer only** — no embeddings or vector DB yet.

---

## Tech Stack

| Component | Library |
|-----------|---------|
| Web framework | FastAPI + Uvicorn |
| PDF parsing | pdfplumber |
| Async file I/O | aiofiles |
| Data validation | Pydantic v2 |
| Timestamp parsing | python-dateutil |

---

## Architecture

```
backend/
└── app/
    ├── main.py                  # FastAPI app factory
    ├── config.py                # Global constants
    ├── routes/
    │   ├── upload.py            # POST /api/v1/upload
    │   └── chunks.py            # GET  /api/v1/chunks[/stats]
    ├── services/
    │   ├── ingestion/
    │   │   └── ingestor.py      # Pipeline orchestrator
    │   ├── parsing/
    │   │   ├── text_parser.py   # .txt / .log → plain text
    │   │   ├── pdf_parser.py    # .pdf → plain text (pdfplumber)
    │   │   └── log_parser.py    # plain text → list[LogEntry]
    │   └── chunking/
    │       ├── doc_chunker.py   # Sliding-window word chunker
    │       └── log_chunker.py   # Time-window log summariser
    ├── models/
    │   ├── chunk.py             # Chunk Pydantic model
    │   └── log_entry.py         # LogEntry Pydantic model
    ├── storage/
    │   └── memory_store.py      # In-memory list + dict store
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

---

## Running

```bash
# Development (auto-reload on file changes)
uvicorn app.main:app --reload

# Production
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

Interactive docs available at **http://localhost:8000/docs**

---

## API Reference

### `POST /api/v1/upload`

Upload a file for ingestion.

**Accepted formats:** `.pdf`, `.txt`, `.log`  
**Max file size:** 50 MB

```bash
curl -X POST http://localhost:8000/api/v1/upload \
     -F "file=@./sample.pdf"
```

**Response:**
```json
{
  "message": "File processed successfully",
  "chunks_created": 14,
  "source": "sample.pdf",
  "type": "doc"
}
```

---

### `GET /api/v1/chunks`

List stored chunks (paginated).

| Query param | Default | Description |
|-------------|---------|-------------|
| `source` | — | Filter by source filename |
| `limit` | 100 | Max chunks to return (1–1000) |
| `offset` | 0 | Skip N chunks |

---

### `GET /api/v1/chunks/stats`

Returns aggregate counts:

```json
{
  "total_chunks": 42,
  "doc_chunks": 28,
  "log_chunks": 14,
  "sources": ["report.pdf", "app.log"]
}
```

---

### `GET /health`

Liveness probe — returns `{"status": "healthy"}`.

---

## Ingestion Pipeline

```
Upload
  │
  ▼
Validate (ext + size)
  │
  ▼
Parse raw text
  │  .pdf ──► pdfplumber
  │  .txt/.log ──► read + normalize
  │
  ▼
Detect type
  │  log heuristic (30 % lines match pattern) ──► Log path
  │  otherwise ──► Doc path
  │
  ├── Doc path:  sliding-window chunker (400 words, 50-word overlap)
  │
  └── Log path:  parse → group by (level + message signature)
                 → split by 10-min time windows → summarise each bucket
  │
  ▼
Store chunks in memory
  │
  ▼
Return { chunks_created: N }
```

---

## Log Chunk Example

Input (`app.log`):
```
2024-01-15 10:01:23 ERROR retry_failed for VISA user_id=100 attempt=3
2024-01-15 10:03:44 ERROR retry_failed for VISA user_id=201 attempt=2
2024-01-15 10:07:11 ERROR retry_failed for MASTERCARD user_id=305 attempt=1
... (120 similar lines)
```

Output chunk content:
```
[ERROR] 120 log entries between 2024-01-15 10:00:00 and 2024-01-15 10:10:00
Most common: "retry_failed for VISA" (87 occurrences)
Other patterns: retry_failed for MASTERCARD; retry_failed for AMEX
Metadata fields: attempt, user_id
```
