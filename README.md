# Self-Updating AI Assistant for Business Intelligence

An end-to-end **GenAI + RAG + RL project** that ingests live business data 
(news, filings, GitHub repos) and provides citation-based Q&A with feedback-driven improvements.  

## Tech Stack
- **LLMs:** OpenAI GPT, Claude APIs
- **RAG:** LangChain, LangGraph, Vector DB (Pinecone/FAISS)
- **Workflows:** n8n (data ingestion & scheduling)
- **Backend:** FastAPI
- **Frontend:** Streamlit (dashboard)
- **Reinforcement Learning:** Bandit + PPO-lite for re-ranking
- **CI/CD:** GitHub Actions, Docker

## Repo Structure
self-updating-ai-assistant/
│── README.md                  # Project overview with demo, diagrams & setup
│── requirements.txt            # Python dependencies
│── docker-compose.yml          # Optional: containerize FastAPI + vector DB
│── .github/
│   └── workflows/
│       └── ci-cd.yml           # GitHub Actions for auto-deploy
│
├── backend/                    # FastAPI backend
│   ├── main.py                 # Entry point (FastAPI app)
│   ├── routers/
│   │   ├── ask.py              # RAG query endpoint (/ask)
│   │   ├── feedback.py         # Feedback collection (/feedback)
│   │   └── health.py           # Health check
│   ├── services/
│   │   ├── rag_pipeline.py     # LangChain retrieval + LLM Q&A
│   │   ├── embeddings.py       # Embedding + vector DB insert/retrieve
│   │   └── feedback_rl.py      # RL logic for re-ranking retrieval
│   ├── models/
│   │   ├── request_models.py   # Pydantic request/response schemas
│   │   └── feedback_models.py
│   └── config.py               # Env variables, API keys, DB configs
│
├── workflows/                  # n8n + LangGraph workflows
│   ├── n8n_news_workflow.json  # News ingestion
│   ├── n8n_github_workflow.json# GitHub repos ingestion
│   ├── n8n_filings_workflow.json
│   └── langgraph_pipeline.py   # Cleaning, chunking, embedding pipeline
│
├── rl/                         # Reinforcement Learning loop
│   ├── bandit.py               # Multi-armed bandit re-ranking
│   ├── ppo_experiment.py       # (Optional) PPO fine-tuning for retrieval policy
│   └── feedback_store.json     # Example feedback logs
│
├── frontend/                   # Minimal UI (optional)
│   ├── streamlit_app.py        # Streamlit dashboard
│   └── static/                 # CSS/JS if needed
│
└── docs/                       # Documentation + showcase
    ├── architecture.png        # System architecture diagram
    ├── api_flow.png            # Query → retrieval → LLM → feedback
    ├── workflows.png           # n8n pipeline diagram
    └── demo.gif                # Time-lapse demo

## Architecture
![architecture](docs/architecture.png)

## Setup
```bash
git clone https://github.com/your-username/self-updating-ai-assistant.git
cd self-updating-ai-assistant
pip install -r requirements.txt
uvicorn backend.main:app --reload
