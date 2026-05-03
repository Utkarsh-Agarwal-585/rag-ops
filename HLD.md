# RAG Assistant — High Level Design

> A Retrieval-Augmented Generation system that lets users upload documents and ask questions, receiving answers grounded in the uploaded content with source citations and diagram rendering.

---

## 1. System Context

```mermaid
C4Context
    title System Context — RAG Assistant

    Person(user, "User", "Uploads documents, asks questions via browser")

    System(rag, "RAG Assistant", "Ingests documents, retrieves relevant context, answers questions using an LLM")

    System_Ext(gemini, "Google Gemini API", "LLM answering + image captioning")
    System_Ext(openai, "OpenAI API", "Alternative LLM + image captioning")

    Rel(user, rag, "Upload files, ask questions", "HTTPS")
    Rel(rag, gemini, "Caption images, answer questions", "REST / BYOK")
    Rel(rag, openai, "Caption images, answer questions", "REST / BYOK")
```

---

## 2. High Level Architecture

```mermaid
graph TB
    subgraph Browser["Browser (React)"]
        UI["RAG Assistant UI\n─────────────────\n• API key + provider input\n• File upload\n• Chat interface\n• Source + image rendering"]
    end

    subgraph Backend["Backend (FastAPI · Python 3.10+)"]
        direction TB

        subgraph Ingestion["Ingestion Layer"]
            UP["POST /api/v1/upload"]
            ING["Ingestor\n(orchestrator)"]
            PARSE["Parsers\npdf · txt · log"]
            CHUNK["Chunkers\ndoc · log"]
            CAP["Captioner\nGemini / OpenAI vision"]
        end

        subgraph Retrieval["Retrieval Layer"]
            QR["POST /api/v1/query"]
            CACHE["Query Cache\n15-min TTL"]
            HYBRID["Hybrid Retriever\nFAISS + BM25"]
            LLM["LLM Service\nGemini / OpenAI"]
        end

        subgraph Storage["Storage Layer"]
            MEM["In-Memory Store\nChunks + Embeddings"]
            DISK["Disk\nstorage/images/<doc>/\n*.caption.txt sidecars"]
            IDX["Disk\nstorage/index/\nFAISS + BM25 + chunks.pkl"]
        end
    end

    subgraph External["External APIs"]
        GAPI["Google Gemini API"]
        OAPI["OpenAI API"]
    end

    Browser -->|"multipart upload\n+ api_key + provider"| UP
    Browser -->|"query + history\n+ api_key + provider"| QR

    UP --> ING
    ING --> PARSE
    ING --> CHUNK
    ING --> CAP
    CAP -->|"BYOK"| GAPI
    CAP -->|"BYOK"| OAPI
    ING --> MEM
    ING --> DISK
    ING --> IDX

    QR --> CACHE
    CACHE -->|miss| HYBRID
    HYBRID --> MEM
    HYBRID --> LLM
    LLM -->|"BYOK"| GAPI
    LLM -->|"BYOK"| OAPI
    LLM --> CACHE
    CACHE -->|hit / response| QR
    QR -->|answer + sources| Browser

    DISK -->|"GET /storage/images/*"| Browser
```

---

## 3. Ingestion Pipeline

```mermaid
flowchart LR
    A(["File\n.pdf / .txt / .log"]) --> B["Validate\next + size"]
    B --> C["Parse\nraw text"]
    C --> D{"Log\ncontent?"}
    D -->|Yes| E["Log Chunker\ntime-window groups"]
    D -->|No| F["Doc Chunker\n400w sliding window\n50w overlap"]
    E --> G["Text Chunks"]
    F --> G

    A -->|PDF only| H["Image Extractor\nPyMuPDF\nxref-stable names"]
    H --> I{"Caption\ncached?"}
    I -->|Sidecar exists| J["Reuse caption\nno API call"]
    I -->|New image| K["Gemini / OpenAI\nvision caption"]
    K --> L["Save .caption.txt\nsidecar"]
    J --> M["Image Chunks"]
    L --> M

    G --> N["Memory Store"]
    M --> N
    N --> O["Embed\nall-MiniLM-L6-v2"]
    O --> P["Deduplicate\ncosine sim >= 0.85"]
    P --> Q["FAISS Index"]
    P --> R["BM25 Index"]
```

---

## 4. Query Pipeline

```mermaid
flowchart LR
    A(["User query\n+ history\n+ api_key"]) --> B["Enrich query\nheuristic rewrite\nusing history"]
    B --> C{"Cache hit?\nquery + history\n+ provider"}
    C -->|Hit| Z(["Cached response"])
    C -->|Miss| D["Embed query\nall-MiniLM-L6-v2"]
    D --> E["FAISS search\ntop 20"]
    D --> F["BM25 search\ntop 20"]
    E --> G["Normalize + combine\n0.6 x vector\n0.4 x BM25"]
    F --> G
    G --> H["Top 5 chunks\n+ image injection"]
    H --> I["Build prompt\ncontext + history\n+ image paths"]
    I --> J["LLM call\nGemini / OpenAI"]
    J --> K["Cache response\n15-min TTL"]
    K --> L(["Answer\n+ sources"])
```

---

## 5. Data Flow — End to End

```mermaid
sequenceDiagram
    actor U as User
    participant UI as React UI
    participant API as FastAPI Backend
    participant EXT as Gemini / OpenAI

    Note over U,EXT: Phase 1 — Upload
    U->>UI: Select file + enter API key + provider
    UI->>API: POST /upload (file + api_key + provider)
    API->>API: Parse → Chunk → Embed → Index
    API->>EXT: Caption images (BYOK)
    EXT-->>API: Captions
    API-->>UI: {chunks_created, breakdown, images[]}
    UI-->>U: Upload success — chat unlocked

    Note over U,EXT: Phase 2 — Conversation
    U->>UI: Ask question
    UI->>API: POST /query {query, api_key, provider, history[]}
    API->>API: Enrich query using history
    API->>API: Hybrid retrieval (FAISS + BM25)
    API->>EXT: LLM call with context + history (BYOK)
    EXT-->>API: Answer text
    API-->>UI: {answer, sources[]}
    UI-->>U: Render answer + source cards + images
```

---

## 6. Component Responsibilities

| Component | Technology | Responsibility |
|---|---|---|
| **React UI** | React 18, plain CSS | Upload, chat, document list with delete, source rendering, image display |
| **FastAPI** | Python 3.10+, Uvicorn | HTTP routing, request validation, async orchestration |
| **Ingestor** | Python | Single pipeline entry point — coordinates all stages |
| **Delete Handler** | Python | Surgical per-document delete — removes chunks, FAISS vectors, BM25 index, image subdirectory, and persists updated state; failures collected as warnings |
| **PDF Parser** | pdfplumber | Text extraction from PDFs |
| **Image Extractor** | PyMuPDF (fitz) | Image extraction, xref-stable naming, per-doc subdirs |
| **Captioner** | google-genai / requests | Multimodal image captioning; sidecar persistence |
| **Doc Chunker** | Python | 400-word sliding window with 50-word overlap |
| **Log Chunker** | Python | Time-window grouping + summarisation |
| **Embedding Service** | sentence-transformers | `all-MiniLM-L6-v2` — 384-dim vectors |
| **Deduplication** | scikit-learn | Cosine similarity clustering at 0.85 threshold |
| **FAISS Index** | faiss-cpu | Dense vector search (inner product = cosine sim) |
| **BM25 Index** | rank-bm25 | Sparse keyword search |
| **Retrieval Service** | Python | Hybrid scoring + image injection |
| **LLM Service** | requests | Query enrichment, prompt building, BYOK LLM calls |
| **Cache Service** | Python stdlib | In-memory TTL cache keyed by query + history + provider |
| **Memory Store** | Python stdlib | In-process chunk store (list + dict) |
| **Disk Storage** | OS filesystem | Images + caption sidecars — survive server restarts |
| **Persistence Service** | faiss + pickle | Save/load FAISS index + BM25 corpus + chunks to disk on every upload |
| **DocList component** | React 18 | Collapsible sidebar list of all uploaded documents with per-doc delete buttons |
| **Correlation Middleware** | Starlette | Generate `x-correlation-id` per request; inject into all log records |
| **JSON Logger** | Python stdlib | Structured JSON log output with timestamp, level, logger, correlation ID |

---

## 7. Key Design Principles

| Principle | How it's applied |
|---|---|
| **Graceful degradation** | Image pipeline and retrieval indexing failures never break text ingestion |
| **BYOK (Bring Your Own Key)** | API keys passed per-request, never stored or logged |
| **Thin façade storage** | `memory_store.py` is the only file to change when swapping to a real vector DB |
| **Async-first** | All blocking I/O and CPU work runs in thread pool via `asyncio.to_thread()` |
| **Idempotent image storage** | xref-stable filenames + caption sidecars make re-uploads safe and cheap |
| **Conversation-aware retrieval** | Query enrichment rewrites follow-ups into standalone queries before retrieval |
| **History-aware caching** | Cache key includes a digest of recent history to prevent stale hits |
| **No server-side session state** | History is sent by the client on every request |
| **Restart-safe persistence** | FAISS + BM25 + chunks saved to disk after every upload; loaded on startup |
| **Idempotent re-upload** | Re-uploading the same file evicts old chunks before ingesting fresh — no duplicates |
| **Structured observability** | JSON logs + `x-correlation-id` on every request for end-to-end traceability |

---

## 8. Limitations & Future Work

| Area | Current state | Future improvement |
|---|---|---|
| **Persistence** | FAISS + BM25 + chunks saved to disk; loaded on startup | ✅ Implemented — `persistence_service.py` |
| **Vector DB** | In-memory FAISS | Swap `memory_store.py` + `vector_service.py` for Chroma / Pinecone |
| **Auth** | None | Add API key auth or OAuth |
| **Multi-tenancy** | Single shared store | Namespace chunks by user/session |
| **OCR** | Not supported | Add Tesseract for scanned PDFs |
| **Streaming** | Full response only | Stream LLM tokens via SSE |
| **Image cap** | 20 per PDF | Make configurable per-request |
| **Cache eviction** | TTL only, unbounded size | Add LRU eviction |
| **Frontend** | localhost only | Deploy behind HTTPS for key security |
