from fastapi import APIRouter
from backend.models.feedback_models import FeedbackRequest, FeedbackResponse
from backend.services.feedback_rl import store_feedback

router = APIRouter()


@router.post("/feedback", response_model=FeedbackResponse)
async def feedback(req: FeedbackRequest):
    """
    Store thumbs-up / thumbs-down feedback for an answer.
    Invalid feedback types are rejected with HTTP 422 (Pydantic validation).
    """
    feedback_id = store_feedback(
        query=req.query,
        answer=req.answer,
        sources=req.sources,
        feedback=req.feedback.value,
    )
    return FeedbackResponse(
        status="success",
        message="Feedback stored successfully.",
        feedback_id=feedback_id,
    )
