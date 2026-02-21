from functools import lru_cache
from typing import Dict, List
import uuid

from supabase import create_client, Client
from backend.config import get_settings
from rl.bandit import UCB1Bandit


@lru_cache()
def get_supabase_client() -> Client:
    """Cached Supabase client (service-role key, bypasses RLS)."""
    settings = get_settings()
    return create_client(settings.supabase_url, settings.supabase_key)


@lru_cache()
def get_bandit() -> UCB1Bandit:
    """
    Cached UCB1Bandit instance, pre-loaded with all historical
    feedback from Supabase so scores survive server restarts.
    """
    settings = get_settings()
    bandit = UCB1Bandit()
    bandit.load_from_supabase(get_supabase_client(), settings.feedback_table)
    return bandit


def ensure_feedback_table_exists() -> None:
    """
    For Supabase the table is created via the SQL migration run in the
    dashboard — nothing to do at runtime. This function is a no-op kept
    for API compatibility with the old Elasticsearch version.
    """
    pass


def store_feedback(
    query: str,
    answer: str,
    sources: List[str],
    feedback: str,
) -> str:
    """
    Persist a feedback event to Supabase and update the UCB1 bandit.
    Returns the generated feedback_id (UUID).
    """
    settings = get_settings()
    client = get_supabase_client()
    bandit = get_bandit()

    feedback_id = str(uuid.uuid4())
    reward = 1.0 if feedback == "positive" else 0.0

    # Update in-memory bandit immediately
    for url in sources:
        bandit.update(url, reward)

    client.table(settings.feedback_table).insert(
        {
            "query": query,
            "answer": answer,
            "sources": sources,
            "feedback": feedback,
            "feedback_id": feedback_id,
        }
    ).execute()

    return feedback_id


def get_feedback_scores(urls: List[str]) -> Dict[str, float]:
    """Return a UCB1 score for each URL (used for re-ranking)."""
    bandit = get_bandit()
    return {url: bandit.get_score(url) for url in urls}
