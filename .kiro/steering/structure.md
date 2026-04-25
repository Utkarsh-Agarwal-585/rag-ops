# Project Structure

All source code lives under `backend/`. There is no frontend in this repo.

```
backend/
├── requirements.txt          # Python dependencies
├── README.md                 # Quick-start guide
├── storage/images/           # Extracted PDF images served statically
└── app/
    ├── main.py               # FastAPI app factory, middleware, static mounts, routers
    ├── config.py             # All tunable constants (chunk sizes, limits, API keys)
    ├── models/               # Pydantic data models
    │   ├── chunk.py          # Chunk — the core unit of ingested content
    │   └── log_entry.py      # LogEntry — parsed log line
    ├── routes/               # API endpoints
    │   ├── upload.py         # POST /api/v1/upload
    │   └── chunks.py         # GET /api/v1/chunks, GET /api/v1/chunks/stats
    ├── services/             # Business logic, organised by pipeline stage
    │   ├── ingestion/
    │   │   └── ingestor.py   # Pipeline orchestrator (single entry point)
    │   ├── parsing/
    │   │   ├── pdf_parser.py
    │   │   ├── text_parser.py
    │   │   ├── log_parser.py
    │   │   └── image_extractor.py
    │   ├── chunking/
    │   │   ├── doc_chunker.py    # Sliding-window word chunker
    │   │   └── log_chunker.py    # Time-window log summariser
    │   └── captioning/
    │       └── gemini_captioner.py  # Gemini 2.5 Flash multimodal captions
    ├── storage/
    │   └── memory_store.py   # In-memory store (thin façade, swap for vector DB later)
    └── utils/
        ├── text_utils.py     # Text normalisation helpers
        └── file_utils.py     # Async temp-file helpers
```

## Architecture Conventions

- `services/` is organised by pipeline stage: ingestion → parsing → chunking → captioning.
- `ingestor.py` is the single orchestrator — routes call it, it coordinates everything else.
- `memory_store.py` is a deliberate thin façade. Replacing it with a vector DB should only require changing that one file.
- `config.py` is the single source of truth for all constants and limits.
- Each service subdirectory has its own `__init__.py` (packages, not flat modules).
- Static images are served via FastAPI's `StaticFiles` mount at `/storage/images`.
