import os
# Prevent broken TensorFlow DLL from crashing sentence-transformers on Windows
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import get_settings
from backend.routers import health, ask, feedback
from backend.services.embeddings import get_embeddings
from backend.services.feedback_rl import get_supabase_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: warm up the embedding model and verify Supabase connectivity
    so the first request is not slow. Services that fail are logged as
    warnings — the server still starts so requests can surface the real error.
    Shutdown: nothing to clean up.
    """
    print("[startup] Loading embedding model...")
    try:
        get_embeddings()
        print("[startup] Embedding model loaded.")
    except Exception as exc:
        print(f"[startup] WARNING: Embedding model failed: {exc}")

    print("[startup] Connecting to Supabase...")
    try:
        get_supabase_client().table("documents").select("id").limit(1).execute()
        print("[startup] Supabase connected.")
    except Exception as exc:
        print(f"[startup] WARNING: Supabase not reachable at startup: {exc}")

    print("[startup] Ready.")
    yield


settings = get_settings()

app = FastAPI(
    title="BI Assistant",
    version="2.0",
    description="RAG + RLHF business intelligence assistant powered by LangChain & Gemini.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(ask.router)
app.include_router(feedback.router)
