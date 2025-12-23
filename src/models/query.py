from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
from .document import Document
from .citation import Citation


class QueryRequest(BaseModel):
    id: str
    question: str
    query_mode: str  # "book-wide" or "selected-text"
    selected_text: Optional[str] = None  # Required if query_mode is "selected-text"
    user_context: Optional[Dict[str, Any]] = None
    timestamp: datetime
    user_id: Optional[str] = None


class QueryResponse(BaseModel):
    id: str
    query_id: str
    answer: Optional[str]  # Can be None if status is insufficient_context
    citations: List[Citation]
    retrieved_chunks: List[str]  # IDs of the chunks used
    confidence_score: Optional[float]  # Can be None if status is insufficient_context
    generation_time_ms: float
    status: str  # "success", "insufficient_context", "error"
    timestamp: datetime