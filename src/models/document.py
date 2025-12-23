from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class Document(BaseModel):
    id: str
    title: str
    content: str
    chapter: str
    section: str
    file_path: str
    source_url: str
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    embedding_vector: Optional[List[float]] = None


class DocumentChunk(BaseModel):
    id: str
    document_id: str
    content: str
    chunk_index: int
    embedding_vector: Optional[List[float]] = None
    created_at: datetime