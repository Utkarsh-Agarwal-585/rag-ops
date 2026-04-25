# Tech Stack & Build

## Language

Python 3.10+

## Core Dependencies

| Library | Purpose |
|---|---|
| FastAPI (≥0.110) | Web framework, async-first |
| Uvicorn (≥0.27) | ASGI server |
| Pydantic v2 (≥2.0) | Data validation and models |
| pdfplumber (≥0.10) | PDF text extraction |
| PyMuPDF / fitz (≥1.23) | PDF image extraction |
| google-genai (≥1.0) | Gemini API client for image captioning |
| Pillow (≥10.0) | Image decoding |
| aiofiles (≥23.2) | Async file I/O |
| python-dateutil (≥2.8) | Timestamp parsing for logs |
| python-multipart (≥0.0.9) | Multipart form parsing (file uploads) |

## Environment Variables

| Variable | Purpose |
|---|---|
| `GEMINI_API_KEY` | Google AI API key (optional — fallback caption used if unset) |
| `GEMINI_MODEL` | Gemini model identifier (default: `gemini-2.5-flash`) |

## Common Commands

```bash
# Setup
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Run (development with auto-reload)
uvicorn app.main:app --reload

# Run (production)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## Key Patterns

- All blocking I/O and CPU work is offloaded via `asyncio.to_thread()` to keep the event loop free.
- `from __future__ import annotations` is used in every module for PEP 604 style type hints.
- Configuration constants live in `app/config.py` — no scattered magic numbers.
- Pydantic v2 `BaseModel` for all data models.
- Route functions are `async def` and return `JSONResponse` or dicts.
- Logging via stdlib `logging.getLogger(__name__)`.
