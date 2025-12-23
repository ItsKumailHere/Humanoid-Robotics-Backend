from pydantic import BaseModel
from typing import Optional


class Citation(BaseModel):
    id: str
    response_id: str
    document_id: str
    chapter: str
    section: str
    file_path: str
    relevance_score: float
    text_snippet: str