from pydantic import BaseModel, Field
from typing import List


class QueryRequest(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=20)


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[str]
    scores: List[float]
    model: str
