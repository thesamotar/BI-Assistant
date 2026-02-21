from fastapi import APIRouter
from backend.models.request_models import QueryRequest, QueryResponse
from backend.services.rag_pipeline import run_rag_pipeline

router = APIRouter()


@router.post("/ask", response_model=QueryResponse)
async def ask(req: QueryRequest):
    """Answer a business intelligence query using RAG + feedback re-ranking."""
    result = run_rag_pipeline(req.query, req.top_k)
    return QueryResponse(
        query=req.query,
        answer=result["answer"],
        sources=result["sources"],
        scores=result["scores"],
        model=result["model"],
    )
