from pydantic import BaseModel
from enum import Enum
from typing import List


class FeedbackType(str, Enum):
    positive = "positive"
    negative = "negative"


class FeedbackRequest(BaseModel):
    query: str
    answer: str
    sources: List[str]
    feedback: FeedbackType


class FeedbackResponse(BaseModel):
    status: str
    message: str
    feedback_id: str
