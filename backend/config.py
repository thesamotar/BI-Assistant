from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Supabase — set via .env
    supabase_url: str = ""
    supabase_key: str = ""
    supabase_db_password: str = ""

    # Table / function names
    documents_table: str = "documents"
    feedback_table: str = "feedback"
    match_function: str = "match_documents"

    # Google Gemini — set via .env
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    gemini_translation_model: str = "gemini-2.5-flash"

    # Embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Data pipeline — set EVENT_REGISTRY_API_KEY via .env
    articles_json_path: str = "genai_competitors_articles.json"
    event_registry_api_key: str = ""
    news_lookback_days: int = 30
    news_max_items_per_company: int = 50

    # Frontend CORS origin
    frontend_origin: str = "http://localhost:8501"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
