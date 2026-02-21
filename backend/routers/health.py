from fastapi import APIRouter
from backend.services.feedback_rl import get_supabase_client

router = APIRouter()


@router.get("/health")
async def health():
    """Liveness + dependency check endpoint."""
    try:
        client = get_supabase_client()
        # A lightweight query — just fetch 1 row to confirm DB connectivity
        client.table("documents").select("id").limit(1).execute()
        db_status = "ok"
    except Exception as exc:
        db_status = f"error: {exc}"

    return {
        "status": "ok" if db_status == "ok" else "degraded",
        "supabase": db_status,
    }
