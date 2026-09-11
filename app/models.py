from pydantic import BaseModel
from typing import List, Optional, Dict, Any

class QueryRequest(BaseModel):
    session_id: str = "default_user"
    question: str
    top_k: int = 5
    filters: Optional[Dict[str, Any]] = None

class Source(BaseModel):
    college_name: str
    tnea_code: str
    district: str
    score: float

class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]