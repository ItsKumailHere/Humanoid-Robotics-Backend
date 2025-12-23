from typing import List, Optional
from qdrant_client.http import models
from ..config.vector_store import vector_store
from ..config.database import db
from ..models.document import DocumentChunk
from ..config.settings import settings
from datetime import datetime


class RetrievalService:
    def __init__(self):
        self.client = vector_store.get_client()

    async def retrieve_for_query(self, query_embedding: List[float], query_mode: str = "book-wide", selected_text: Optional[str] = None, top_k: int = 5) -> List[DocumentChunk]:
        """
        Retrieve relevant chunks for a query based on the query mode.
        """
        if query_mode == "selected-text" and selected_text:
            results = await self._retrieve_from_selected_text(selected_text, top_k)
        else:
            results = await self._retrieve_book_wide(query_embedding, top_k)
        
        return results

    async def _retrieve_book_wide(self, query_embedding: List[float], top_k: int = 5) -> List[DocumentChunk]:
        """Retrieve chunks from the entire textbook"""
        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        
        def sync_search():
            return self.client.query_points(
                collection_name="document_chunks",
                query=query_embedding,
                limit=top_k,
                with_payload=True
            )
        
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            search_results = await loop.run_in_executor(executor, sync_search)

        chunks = []
        for result in search_results.points:
            print(f"Processing result ID: {result.id}")
            print(f"Payload keys: {list(result.payload.keys())}")
            
            # Try multiple possible keys for content in the payload
            content = (
                result.payload.get("content") or 
                result.payload.get("text") or 
                result.payload.get("chunk_text") or
                result.payload.get("page_content") or
                ""
            )
            
            if not content:
                print(f"WARNING: No content found in payload for chunk {result.id}")
                print(f"Available payload: {result.payload}")
                continue  # Skip chunks without content
            
            print(f"Content length: {len(content)}")
            
            # Handle created_at
            created_at = result.payload.get("created_at")
            if created_at is None:
                created_at = datetime.now()
            elif isinstance(created_at, str):
                try:
                    from dateutil import parser
                    created_at = parser.parse(created_at)
                except:
                    created_at = datetime.now()

            chunk = DocumentChunk(
                id=str(result.id),
                document_id=result.payload.get("document_id", ""),
                content=content,
                chunk_index=result.payload.get("chunk_index", 0),
                embedding_vector=None,
                created_at=created_at
            )
            chunks.append(chunk)

        print(f"Successfully created {len(chunks)} chunks with content")
        return chunks

    async def _retrieve_from_selected_text(self, selected_text: str, top_k: int = 5) -> List[DocumentChunk]:
        """
        Retrieve relevant chunks specifically from the selected text.
        """
        chunks = [DocumentChunk(
            id="temp_selected_text",
            document_id="selected_text_session",
            content=selected_text,
            chunk_index=0,
            embedding_vector=None,
            created_at=datetime.now()
        )]
        return chunks

    async def retrieve_by_document_id(self, document_id: str, top_k: int = 10) -> List[DocumentChunk]:
        """Retrieve chunks for a specific document by its ID"""
        import asyncio
        from concurrent.futures import ThreadPoolExecutor
        
        def sync_scroll():
            return self.client.scroll(
                collection_name="document_chunks",
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="document_id",
                            match=models.MatchValue(value=document_id)
                        )
                    ]
                ),
                limit=top_k,
                with_payload=True
            )
        
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            scroll_result = await loop.run_in_executor(executor, sync_scroll)
        
        chunks = []
        for point in scroll_result[0]:
            content = (
                point.payload.get("content") or 
                point.payload.get("text") or 
                point.payload.get("chunk_text") or
                ""
            )
            
            if not content:
                continue
            
            created_at = point.payload.get("created_at")
            if created_at is None:
                created_at = datetime.now()
            
            chunk = DocumentChunk(
                id=str(point.id),
                document_id=point.payload.get("document_id", ""),
                content=content,
                chunk_index=point.payload.get("chunk_index", 0),
                embedding_vector=None,
                created_at=created_at
            )
            chunks.append(chunk)
        
        return chunks