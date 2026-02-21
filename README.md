# BI Assistant

> A self-updating AI for Business Intelligence — RAG + RLHF powered by LangChain, Gemini, and Supabase.

[![CI](https://github.com/thesamotar/BI-Assistant/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/thesamotar/BI-Assistant/actions/workflows/ci-cd.yml)
![Python](https://img.shields.io/badge/python-3.10+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green)
![LangChain](https://img.shields.io/badge/LangChain-LCEL-orange)
![Supabase](https://img.shields.io/badge/Supabase-pgvector-3ECF8E)

BI Assistant fetches live news about GenAI competitors (OpenAI, Anthropic, Google DeepMind, and more), indexes them as vector embeddings in Supabase, and answers natural-language queries via Gemini with cited sources. Every 👍/👎 you give re-ranks future results through a UCB1 multi-armed bandit — the assistant gets smarter the more you use it.

---

## Features

- **Live data ingestion** — one command fetches, translates, chunks, embeds, and indexes ~400 articles from EventRegistry
- **Citation-based answers** — Gemini answers are grounded in retrieved sources, with URLs cited inline
- **Feedback-driven re-ranking** — UCB1 bandit adjusts document scores based on user votes, persisted in Supabase
- **Streamlit dashboard** — query input, source score table, 👍👎 buttons, query history, live health indicators
- **Fully modular backend** — FastAPI + LangChain LCEL, clean separation of routers/services/models
- **CI** — ruff linting + pip-audit security scan on every push

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
│                  1. Embed query (all-MiniLM-L6-v2)           │
│                  2. match_documents RPC (top 2×k)            │
│                  3. UCB1 re-rank (vector + bandit score)      │
│                  4. Build context → Gemini → answer           │
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
│  fetch_articles → load_articles → translate_non_english      │
│       │                                   │                  │
│  EventRegistry                       Gemini (non-EN)         │
│  8 companies · 30 days · 50 art/co        │                  │
│                                   chunk_documents            │
│                                  (3200 chars, 400 overlap)   │
│                                          │                   │
│                                  generate_embeddings         │
│                                   (384-dim, batch=64)        │
│                                          │                   │
│                                  index_to_supabase           │
│                                   (upsert on doc_id)         │
└─────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI + Pydantic Settings |
| LLM | Google Gemini 2.5-flash (LangChain LCEL) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (384d) |
| Vector store | Supabase pgvector |
| Data pipeline | LangGraph (6-node graph) |
| News source | EventRegistry (newsapi.ai) |
| Translation | Gemini + `langdetect` |
| Re-ranking | UCB1 multi-armed bandit |
| Frontend | Streamlit |
| Orchestration | n8n (Docker) |
| CI | GitHub Actions — ruff + pip-audit |

---

## Getting Started

### Prerequisites

- Python 3.10+
- A [Supabase](https://supabase.com) project with the `pgvector` extension enabled and a `match_documents` RPC function
- A [Google Gemini](https://aistudio.google.com) API key
- An [EventRegistry](https://newsapi.ai) API key

### Installation

```bash
git clone https://github.com/thesamotar/BI-Assistant.git
cd BI-Assistant
pip install -r requirements.txt
```

### Configuration

```bash
cp .env.example .env
# Fill in SUPABASE_URL, SUPABASE_KEY, GEMINI_API_KEY, EVENT_REGISTRY_API_KEY
```

All available settings are documented in `.env.example`.

### 1. Run the data pipeline

Fetches live articles, translates, chunks, embeds, and indexes to Supabase in one command. Run this first, and re-run periodically to keep data fresh.

```bash
python -m workflows.langgraph_pipeline
```

### 2. Start the backend

```bash
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

API docs available at `http://localhost:8000/docs`.

### 3. Start the frontend

In a second terminal:

```bash
python -m streamlit run frontend/streamlit_app.py
```

Open `http://localhost:8501`.

### 4. (Optional) n8n orchestrator

```bash
docker-compose up
# http://localhost:5678  —  admin / admin123
```

---

## Project Structure

```
BI-Assistant/
├── backend/
│   ├── main.py                  # FastAPI entry point, lifespan, CORS
│   ├── config.py                # Pydantic Settings (reads .env)
│   ├── models/
│   │   ├── request_models.py    # QueryRequest, QueryResponse
│   │   └── feedback_models.py   # FeedbackRequest, FeedbackType
│   ├── routers/
│   │   ├── ask.py               # POST /ask
│   │   ├── feedback.py          # POST /feedback
│   │   └── health.py            # GET /health
│   └── services/
│       ├── embeddings.py        # HuggingFace embeddings (lru_cache)
│       ├── feedback_rl.py       # Supabase client + UCB1 bandit
│       └── rag_pipeline.py      # FeedbackAwareRetriever + LCEL chain
├── workflows/
│   └── langgraph_pipeline.py    # 6-node LangGraph ingestion pipeline
├── rl/
│   ├── bandit.py                # UCB1Bandit implementation
│   └── ppo_experiment.py        # PPO re-ranker (research)
├── frontend/
│   └── streamlit_app.py         # Streamlit dashboard
├── .github/workflows/ci-cd.yml  # Lint + security CI
├── ruff.toml                    # Ruff lint config
├── .env.example                 # Environment variable template
├── docker-compose.yml           # n8n orchestrator
└── requirements.txt
```

---

## API Reference

<details>
<summary><code>POST /ask</code></summary>

**Request**
```json
{
  "query": "What are OpenAI's latest product announcements?",
  "top_k": 5
}
```

**Response**
```json
{
  "answer": "OpenAI announced... [https://example.com]",
  "sources": ["https://example.com"],
  "scores": [0.8421],
  "model": "gemini-2.5-flash"
}
```
</details>

<details>
<summary><code>POST /feedback</code></summary>

**Request**
```json
{
  "query": "What are OpenAI's latest product announcements?",
  "answer": "...",
  "sources": ["https://example.com"],
  "feedback": "positive"
}
```
</details>

<details>
<summary><code>GET /health</code></summary>

**Response**
```json
{ "status": "ok", "supabase": "ok" }
```
</details>

---

## How Re-ranking Works

Retrieval is a two-stage process:

1. **Vector search** — `match_documents` Supabase RPC retrieves 2× `top_k` candidates by cosine similarity.
2. **UCB1 re-ranking** — each candidate URL gets a bandit score:

$$\text{score} = \bar{x} + \sqrt{\frac{2 \ln N}{n}}$$

where $\bar{x}$ is the mean reward (1.0 = 👍, 0.0 = 👎), $N$ is total feedback events, and $n$ is feedback count for that URL. Final rank = `vector_score + ucb1_score`.

The bandit state is rebuilt from Supabase on every startup — no extra infrastructure needed.

---

## Configuration Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `SUPABASE_URL` | — | Supabase project URL |
| `SUPABASE_KEY` | — | Service role key |
| `SUPABASE_DB_PASSWORD` | — | Database password |
| `GEMINI_API_KEY` | — | Google Gemini API key |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Generation model |
| `GEMINI_TRANSLATION_MODEL` | `gemini-2.5-flash` | Translation model |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model |
| `EVENT_REGISTRY_API_KEY` | — | EventRegistry API key |
| `NEWS_LOOKBACK_DAYS` | `30` | Days of history to fetch |
| `NEWS_MAX_ITEMS_PER_COMPANY` | `50` | Articles per company |
| `DOCUMENTS_TABLE` | `documents` | Supabase table for chunks |
| `FEEDBACK_TABLE` | `feedback` | Supabase table for votes |
| `MATCH_FUNCTION` | `match_documents` | Supabase RPC function |
| `ARTICLES_JSON_PATH` | `genai_competitors_articles.json` | Pipeline output file |
| `FRONTEND_ORIGIN` | `http://localhost:8501` | CORS allowed origin |

---

## CI

Every push and PR to `main` runs two GitHub Actions jobs:

| Job | Tool | Checks |
|-----|------|--------|
| `lint` | `ruff` | Syntax errors, undefined names, bad imports |
| `security` | `pip-audit` | Known CVEs in dependencies |

---

## License

[MIT](LICENSE)
