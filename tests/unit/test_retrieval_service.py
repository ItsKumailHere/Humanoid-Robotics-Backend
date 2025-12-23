import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.services.retrieval_service import RetrievalService
from src.models.document import DocumentChunk
from datetime import datetime


class TestRetrievalService:
    @pytest.mark.asyncio
    async def test_retrieve_for_query_book_wide_mode(self):
        """Test that retrieve_for_query works in book-wide mode"""
        service = RetrievalService()
        service.client = MagicMock()

        # Mock the query_points method
        mock_result = MagicMock()
        mock_result.points = []
        service.client.query_points.return_value = mock_result

        query_embedding = [0.1, 0.2, 0.3]
        result = await service.retrieve_for_query(query_embedding, query_mode="book-wide")

        # Verify that the book-wide method was called
        assert service.client.query_points.called
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_retrieve_for_query_selected_text_mode(self):
        """Test that retrieve_for_query works in selected-text mode"""
        service = RetrievalService()
        
        selected_text = "This is the selected text for testing."
        query_embedding = [0.1, 0.2, 0.3]
        
        result = await service.retrieve_for_query(
            query_embedding, 
            query_mode="selected-text", 
            selected_text=selected_text
        )

        # Verify that the result contains the selected text as a chunk
        assert len(result) == 1
        assert isinstance(result[0], DocumentChunk)
        assert result[0].content == selected_text
        assert result[0].document_id == "selected_text_session"

    @pytest.mark.asyncio
    async def test_retrieve_from_selected_text(self):
        """Test the _retrieve_from_selected_text method directly"""
        service = RetrievalService()
        
        selected_text = "Sample text content for testing."
        result = await service._retrieve_from_selected_text(selected_text)
        
        assert len(result) == 1
        assert isinstance(result[0], DocumentChunk)
        assert result[0].content == selected_text
        assert result[0].id == "temp_selected_text"
        assert result[0].document_id == "selected_text_session"

    @pytest.mark.asyncio
    async def test_retrieve_book_wide(self):
        """Test the _retrieve_book_wide method"""
        service = RetrievalService()
        service.client = MagicMock()

        # Mock the query_points response
        mock_point = MagicMock()
        mock_point.id = "test_id"
        mock_point.payload = {
            "content": "Test chunk content",
            "document_id": "doc_123",
            "chunk_index": 1,
            "created_at": datetime.now().isoformat()
        }
        
        mock_result = MagicMock()
        mock_result.points = [mock_point]
        service.client.query_points.return_value = mock_result

        query_embedding = [0.1, 0.2, 0.3]
        result = await service._retrieve_book_wide(query_embedding)

        assert len(result) == 1
        assert isinstance(result[0], DocumentChunk)
        assert result[0].content == "Test chunk content"
        assert result[0].document_id == "doc_123"