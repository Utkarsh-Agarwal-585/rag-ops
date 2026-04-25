# Product Overview

RAG Ingestion Backend — a FastAPI service that forms the ingestion layer of a Retrieval-Augmented Generation (RAG) system.

## What It Does

Accepts uploaded documents (.pdf, .txt, .log), parses and normalises content, splits it into semantically meaningful chunks, optionally captions embedded PDF diagrams via Google Gemini, and stores all chunks in memory for downstream embedding and retrieval.

## Key Capabilities

- PDF text extraction (pdfplumber) and image extraction (PyMuPDF)
- Plain text and structured log file ingestion
- Sliding-window document chunking (400 words, 50-word overlap)
- Time-window log grouping and summarisation
- Multimodal image captioning via Gemini 2.5 Flash
- In-memory chunk storage (designed as a swappable façade for future vector DB)

## Current Version

v1.1.0 — ingestion-only. No embeddings, vector DB, query/retrieval, or auth yet.

## Out of Scope (for now)

Embedding generation, vector DB storage, query/retrieval API, OCR for scanned PDFs, user authentication, multi-tenancy, streaming responses.
