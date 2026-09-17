from typing import Any

from pydantic import BaseModel, Field, validator


class QueryRequest(BaseModel):
    session_id: str
    question: str
    top_k: int = Field(default=5, description="Number of sources (1-20)")
    filters: dict[str, Any] | None = None
    stream: bool | None = Field(default=True, description="Enable SSE token streaming")

    # 🛡️ Validator works with both Pydantic v1 and v2
    @validator('top_k')
    def validate_top_k(cls, v):
        if v < 1 or v > 20:
            raise ValueError('top_k must be between 1 and 20')
        return v


class Source(BaseModel):
    college_name: str
    tnea_code: str
    district: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]