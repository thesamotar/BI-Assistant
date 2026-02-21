# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Self-Updating AI Assistant for Business Intelligence — a RAG + RLHF system that ingests live business news (competitors, filings, GitHub repos), indexes them as vector embeddings in Elasticsearch, and answers queries via an LLM with citation-based responses. Feedback on answers drives re-ranking of future results.

## Running the Project

**FastAPI backend (main app):**
```bash
uvicorn app:app --reload
```

**n8n workflow orchestrator (Docker):**
```bash
docker-compose up
# Access at http://localhost:5678  (credentials: admin / admin123)
```

**Streamlit frontend:**
```bash
streamlit run frontend/streamlit_app.py
```

**Data pipeline scripts (run in order):**
```bash
python example2.py           # Fetch articles from EventRegistry
python translation.py        # Translate non-English articles via Gemini
python posting_embeddings_to_elastic.py  # Chunk, embed, and index to Elasticsearch
```

**Tests:**
```bash
python test_app.py
python test_rlhf.py
python vector_search_example.py  # Manual vector search smoke test
```

## Architecture

The system has two runtime components and a data ingestion pipeline:

### Runtime (app.py — the real implementation)
The actual working code lives in **`app.py`** in the root, not in the `backend/` directory (which has empty placeholder files). `app.py` is a single-file FastAPI app with:

- `POST /ask` — encodes the query with SentenceTransformer, does a `script_score` cosine similarity search in Elasticsearch, applies feedback-based score adjustments (±0.1 per positive/negative vote), re-ranks, builds a context block, and calls Gemini to generate a cited answer.
- `POST /feedback` — stores `positive`/`negative` feedback in a separate Elasticsearch index (`bi-assistant-feedback`), which is read back by `/ask` to adjust scores.

### Data Ingestion Pipeline (root-level scripts)
```
example.py / example2.py  →  translation.py  →  posting_embeddings_to_elastic.py
    (fetch articles)           (translate)          (chunk → embed → index)
```

- Chunking: 800-word windows with 100-word overlap, deterministic IDs to avoid duplicates.
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions).
- Elasticsearch index: `bi-assistant-1729` with a `dense_vector` field.

### Planned (not yet implemented)
The `backend/`, `rl/`, `workflows/`, and `frontend/` directories contain empty placeholder files for a planned refactor into a modular architecture with LangGraph pipelines, multi-armed bandit re-ranking, and a Streamlit UI.

## Key Configuration

All credentials are currently hardcoded in source files (no `.env`). The relevant constants are at the top of `app.py`, `posting_embeddings_to_elastic.py`, `translation.py`, and the example scripts:

| Variable | Location | Purpose |
|---|---|---|
| `ES_URL` | app.py | GCP-hosted Elasticsearch cluster |
| `ES_INDEX` | app.py | `"bi-assistant-1729"` — article chunks |
| `FEEDBACK_INDEX` | app.py | `"bi-assistant-feedback"` — RLHF votes |
| Elasticsearch API key | app.py | Auth for all ES operations |
| Gemini API key | app.py, translation.py | LLM generation + translation |
| NewsAPI / EventRegistry key | example.py, example2.py | Article ingestion |

## Requirements

Install all dependencies with:
```bash
pip install fastapi uvicorn pydantic sentence-transformers elasticsearch google-generativeai langdetect newsapi-python eventregistry requests numpy streamlit
```

| Package | pip name | Used in |
|---|---|---|
| `fastapi` | `fastapi` | app.py — API framework |
| `uvicorn` | `uvicorn` | serving FastAPI |
| `pydantic` | `pydantic` | app.py — request/response models |
| `sentence-transformers` | `sentence-transformers` | app.py, posting_embeddings_to_elastic.py, vector_search_example.py |
| `elasticsearch` | `elasticsearch` | app.py, posting_embeddings_to_elastic.py, vector_search_example.py |
| `google-generativeai` | `google-generativeai` | app.py, translation.py — Gemini LLM + translation |
| `langdetect` | `langdetect` | translation.py — language detection |
| `newsapi-python` | `newsapi-python` | example.py — NewsAPI ingestion |
| `eventregistry` | `eventregistry` | example2.py — EventRegistry ingestion |
| `requests` | `requests` | test_app.py, test_rlhf.py |
| `numpy` | `numpy` | vector_search_example.py |
| `streamlit` | `streamlit` | frontend/streamlit_app.py (placeholder) |

> **Note:** `requirements.txt` is currently empty. The above list was compiled from actual imports across all `.py` files.

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI + Pydantic |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | Elasticsearch (GCP, dense_vector + script_score) |
| LLM | Google Gemini (`gemini-2.5-pro`) |
| Translation | Google Gemini + `langdetect` |
| Orchestration | n8n (Docker), LangGraph (planned) |
| Frontend | Streamlit (placeholder) |
| RL/feedback | Score adjustment in `/ask`; bandit/PPO planned in `rl/` |
