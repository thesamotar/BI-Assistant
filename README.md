# BI Assistant — Self-Updating AI for Business Intelligence

A production-ready **RAG + RLHF** system that fetches live business news about GenAI competitors, indexes them as vector embeddings in Supabase (pgvector), and answers queries via Gemini with citation-based responses. User feedback on answers drives UCB1 bandit re-ranking of future results.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    FRONTEND (Streamlit :8501)                │
│  Query input · Answer display · Source scores · 👍👎 feedback│
└─────────────────────────┬───────────────────────────────────┘
                          │ HTTP
┌─────────────────────────▼───────────────────────────────────┐
│               BACKEND (FastAPI :8000)                        │
│                                                              │
│  POST /ask ──► FeedbackAwareRetriever                        │
│                  1. Embed query (all-MiniLM-L6-v2)          │
│                  2. match_documents RPC (top 2×k)            │
│                  3. UCB1 re-rank (vector + bandit score)     │
│                  4. Build context → Gemini → answer          │
│                                                              │
│  POST /feedback ──► store in Supabase → update bandit        │
│  GET  /health   ──► Supabase connectivity check              │
└──────────────┬──────────────────────────┬───────────────────┘
               │                          │
    ┌──────────▼──────────┐   ┌───────────▼──────────┐
    │  Supabase (pgvector) │   │  Gemini 2.5-flash LLM │
    │  documents table     │   │  (LangChain LCEL)     │
    │  feedback table      │   └───────────────────────┘
    └─────────▲───────────┘
              │ upsert (idempotent)
┌─────────────┴───────────────────────────────────────────────┐
│              DATA PIPELINE (LangGraph)                       │
│                                                              │
│  fetch_articles ──► load_articles ──► translate_non_english  │
│       │                                      │               │
│  EventRegistry                          Gemini (non-EN)      │
│  (8 companies, 30 days, 50 art/co)           │               │
│                                    chunk_documents           │
│                                   (3200 chars, 400 overlap)  │
│                                         │                    │
│                                  generate_embeddings         │
│                                   (384-dim, batch=64)        │
│                                         │                    │
│                                  index_to_supabase           │
│                                   (upsert on doc_id)         │
└─────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
BI-Assistant/
├── backend/                        # FastAPI application
│   ├── main.py                     # App entry point, lifespan, CORS
│   ├── config.py                   # Pydantic Settings (reads .env)
│   ├── models/
│   │   ├── request_models.py       # QueryRequest, QueryResponse
│   │   └── feedback_models.py      # FeedbackRequest, FeedbackType
│   ├── routers/
│   │   ├── ask.py                  # POST /ask
│   │   ├── feedback.py             # POST /feedback
│   │   └── health.py               # GET /health
│   └── services/
│       ├── embeddings.py           # HuggingFace embeddings (cached)
│       ├── feedback_rl.py          # Supabase client, UCB1 bandit
│       └── rag_pipeline.py         # FeedbackAwareRetriever + LCEL chain
│
├── workflows/
│   └── langgraph_pipeline.py       # 6-node LangGraph ingestion pipeline
│
├── rl/
│   ├── bandit.py                   # UCB1Bandit (update, get_score, load)
│   └── ppo_experiment.py           # PPO re-ranker (research/educational)
│
├── frontend/
│   └── streamlit_app.py            # Streamlit dashboard (fully implemented)
│
├── _old/                           # Legacy monolithic code (gitignored)
│   ├── app.py                      # Original single-file FastAPI + Elasticsearch
│   ├── example.py                  # NewsAPI fetch script
│   ├── posting_embeddings_to_elastic.py
│   ├── translation.py
│   ├── test_app.py / test_rlhf.py
│   └── vector_search_example.py
│
├── genai_competitors_articles.json # Articles output (pipeline writes here)
├── .env.example                    # Environment variables template
├── docker-compose.yml              # n8n orchestrator
├── requirements.txt
└── setup.sh
```

---

## Quick Start

### 1. Clone and install

```bash
git clone <repo-url>
cd BI-Assistant
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Fill in your Supabase URL/key, Gemini API key, and EventRegistry API key
```

### 3. Run the data pipeline

Fetches live articles, translates, chunks, embeds, and indexes to Supabase in one command:

```bash
python -m workflows.langgraph_pipeline
```

### 4. Start the backend

```bash
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

### 5. Start the frontend

```bash
python -m streamlit run frontend/streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501).

### 6. (Optional) n8n workflow orchestrator

```bash
docker-compose up
# Access at http://localhost:5678  (admin / admin123)
```

---

## Data Pipeline Detail

The LangGraph pipeline (`workflows/langgraph_pipeline.py`) runs 6 nodes in sequence:

| Node | What it does |
|------|-------------|
| `fetch_articles` | Queries EventRegistry for 8 GenAI companies over the last 30 days (50 articles each), saves to `genai_competitors_articles.json` |
| `load_articles` | Reads the JSON file into pipeline state |
| `translate_non_english` | Detects language; translates non-English articles to English via Gemini |
| `chunk_documents` | Splits content into 3200-char chunks with 400-char overlap; assigns deterministic SHA-256 `doc_id` per `url + chunk_index` |
| `generate_embeddings` | Encodes all chunks in batches of 64 using `all-MiniLM-L6-v2` (384 dims) |
| `index_to_supabase` | Upserts chunks + embeddings into the `documents` table using `doc_id` as the conflict key — fully idempotent |

Companies tracked: OpenAI, Anthropic, Google DeepMind, Meta AI, Microsoft AI, Mistral AI, Cohere, Hugging Face.

---

## API Reference

### `POST /ask`

```json
// Request
{ "query": "What are OpenAI's latest product announcements?", "top_k": 5 }

// Response
{
  "answer": "OpenAI announced... [https://example.com]",
  "sources": ["https://example.com", "..."],
  "scores": [0.8421, 0.7903, "..."],
  "model": "gemini-2.5-flash"
}
```

### `POST /feedback`

```json
// Request
{
  "query": "What are OpenAI's latest product announcements?",
  "answer": "...",
  "sources": ["https://example.com"],
  "feedback": "positive"   // or "negative"
}
```

### `GET /health`

```json
{ "status": "ok", "supabase": "ok" }
```

---

## Retrieval & Re-ranking

Retrieval uses a two-stage approach:

1. **Vector similarity** — `match_documents` Supabase RPC fetches 2× `top_k` candidates using cosine similarity on the 384-dim embedding.
2. **UCB1 re-ranking** — Each candidate URL gets a bandit score:
   ```
   score = mean_reward + sqrt(2 · ln(N) / n)
   ```
   where reward = 1.0 (👍) or 0.0 (👎), N = total feedback events, n = feedback for this URL. The final ranking uses `vector_score + ucb1_score`.

The bandit state is rebuilt from Supabase on every startup, so it persists across restarts with no extra infrastructure.

---

## Configuration

All settings are in `backend/config.py` (Pydantic `BaseSettings`). Values can be overridden with a `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `SUPABASE_URL` | — | Supabase project URL |
| `SUPABASE_KEY` | — | Supabase service role key |
| `SUPABASE_DB_PASSWORD` | — | Database password |
| `DOCUMENTS_TABLE` | `documents` | pgvector table name |
| `FEEDBACK_TABLE` | `feedback` | Feedback table name |
| `MATCH_FUNCTION` | `match_documents` | Supabase RPC function |
| `GEMINI_API_KEY` | — | Google Gemini API key |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Generation model |
| `GEMINI_TRANSLATION_MODEL` | `gemini-2.5-flash` | Translation model |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace embedding model |
| `EVENT_REGISTRY_API_KEY` | — | EventRegistry (newsapi.ai) API key |
| `NEWS_LOOKBACK_DAYS` | `30` | Days of news to fetch |
| `NEWS_MAX_ITEMS_PER_COMPANY` | `50` | Max articles per company |
| `ARTICLES_JSON_PATH` | `genai_competitors_articles.json` | Pipeline JSON output path |
| `FRONTEND_ORIGIN` | `http://localhost:8501` | CORS allowed origin |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI + Pydantic |
| LLM | Google Gemini 2.5-flash (via LangChain) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (384d) |
| Vector store | Supabase pgvector |
| RAG framework | LangChain LCEL |
| Data pipeline | LangGraph (6-node graph) |
| News source | EventRegistry (newsapi.ai) |
| Translation | Gemini + `langdetect` |
| Re-ranking | UCB1 multi-armed bandit |
| RL experiment | PPO (`rl/ppo_experiment.py`) |
| Frontend | Streamlit |
| Orchestration | n8n (Docker) |
| CI | GitHub Actions (lint + security audit) |

---

## CI

Two checks run automatically on every push and PR to `main` via GitHub Actions (`.github/workflows/ci-cd.yml`):

| Job | Tool | What it checks |
|-----|------|---------------|
| `lint` | `ruff` | Syntax errors, undefined names, bad imports |
| `security` | `pip-audit` | Known CVEs in `requirements.txt` dependencies |

Lint rules are configured in `ruff.toml` — `_old/` and `logs/` are excluded.

---

## Requirements

```bash
pip install -r requirements.txt
```

Key packages: `fastapi`, `uvicorn`, `langchain`, `langchain-google-genai`, `langchain-huggingface`, `langgraph`, `sentence-transformers`, `supabase`, `google-generativeai`, `eventregistry`, `langdetect`, `streamlit`.

---

## Running Locally

**Prerequisites:** Python 3.10+, `.env` file filled in (copy from `.env.example`).

### Backend

```bash
pip install -r requirements.txt
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

On startup you'll see:
```
[startup] Loading embedding model...
[startup] Embedding model loaded.
[startup] Connecting to Supabase...
[startup] Supabase connected.
[startup] Ready.
```

API is live at `http://localhost:8000`. Docs at `http://localhost:8000/docs`.

### Frontend

In a second terminal:

```bash
python -m streamlit run frontend/streamlit_app.py
```

Open `http://localhost:8501`. The sidebar shows a live green/red indicator for backend and Supabase connectivity.

### Run the data pipeline first (if Supabase is empty)

```bash
python -m workflows.langgraph_pipeline
```

This fetches ~400 fresh articles, translates, chunks, embeds, and indexes them. Run it once before querying, and re-run periodically to keep data fresh.
