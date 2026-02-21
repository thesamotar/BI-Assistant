from functools import lru_cache
from langchain_huggingface import HuggingFaceEmbeddings
from backend.config import get_settings


@lru_cache()
def get_embeddings() -> HuggingFaceEmbeddings:
    """Cached HuggingFace embedding model (all-MiniLM-L6-v2, 384 dims)."""
    settings = get_settings()
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
