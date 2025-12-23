import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.services.generation_service import GenerationService
from src.models.query import QueryRequest
from src.models.document import DocumentChunk


class TestGenerationService:
    
    @pytest.fixture
    def mock_settings(self):
        with patch('src.config.settings.settings') as mock:
            mock.gemini_api_key = "test-gemini-key"
            yield mock

    @pytest.fixture
    def generation_service(self, mock_settings):
        # Mock the Gemini services to avoid actual API calls
        with patch('src.services.gemini_service.GeminiLLMService'), \
             patch('src.services.gemini_service.SelectedTextGeminiLLMService'):
            service = GenerationService()
            return service

    @pytest.mark.asyncio
    async def test_generation_service_initialization(self, mock_settings):
        """Test that GenerationService initializes with Gemini services"""
        with patch('src.services.gemini_service.GeminiLLMService') as mock_gemini, \
             patch('src.services.gemini_service.SelectedTextGeminiLLMService') as mock_selected:
            
            mock_gemini_instance = AsyncMock()
            mock_selected_instance = AsyncMock()
            mock_gemini.return_value = mock_gemini_instance
            mock_selected.return_value = mock_selected_instance
            
            service = GenerationService()
            
            # Verify both services were initialized
            mock_gemini.assert_called_once_with(api_key="test-gemini-key")
            mock_selected.assert_called_once_with(api_key="test-gemini-key")
            assert service.gemini_service == mock_gemini_instance
            assert service.selected_text_gemini_service == mock_selected_instance

    @pytest.mark.asyncio
    async def test_generate_answer_book_wide_mode(self, generation_service):
        """Test generating answer in book-wide mode"""
        # Mock the gemini service response
        mock_response = {
            "answer": "This is the generated answer",
            "status": "success",
            "confidence_score": 0.85
        }
        generation_service.gemini_service.generate_rag_response = AsyncMock(return_value=mock_response)
        
        # Create query and chunks
        query_request = QueryRequest(
            id="query123",
            question="What is humanoid robotics?",
            query_mode="book-wide",
            timestamp="2023-01-01T00:00:00"
        )
        
        chunk = DocumentChunk(
            id="chunk123",
            document_id="doc123",
            content="Humanoid robotics is a field of engineering...",
            chunk_index=0,
            created_at="2023-01-01T00:00:00"
        )
        
        result = await generation_service.generate_answer(query_request, [chunk])
        
        # Verify the correct service was called with correct parameters
        generation_service.gemini_service.generate_rag_response.assert_called_once_with(
            query_request.question,
            "Source: doc123 | Content: Humanoid robotics is a field of engineering...[:500]..."
        )
        
        # Verify the response is as expected
        assert result == mock_response

    @pytest.mark.asyncio
    async def test_generate_answer_selected_text_mode(self, generation_service):
        """Test generating answer in selected-text mode"""
        # Mock the selected text service response
        mock_response = {
            "answer": "Based on the selected text...",
            "status": "success", 
            "confidence_score": 0.75
        }
        generation_service.selected_text_gemini_service.generate_rag_response = AsyncMock(return_value=mock_response)
        
        # Create query with selected text
        query_request = QueryRequest(
            id="query456",
            question="What does the selected text say?",
            query_mode="selected-text",
            selected_text="This is the selected text that should be the context.",
            timestamp="2023-01-01T00:00:00"
        )
        
        chunk = DocumentChunk(
            id="chunk456",
            document_id="doc456",
            content="Some irrelevant content",
            chunk_index=0,
            created_at="2023-01-01T00:00:00"
        )
        
        result = await generation_service.generate_answer(query_request, [chunk])
        
        # Verify the selected text service was called with the selected text as context
        generation_service.selected_text_gemini_service.generate_rag_response.assert_called_once_with(
            query_request.question,
            query_request.selected_text  # Should use selected text, not retrieved chunks
        )
        
        # Verify the response is as expected
        assert result == mock_response

    @pytest.mark.asyncio
    async def test_generate_answer_no_chunks_refusal(self, generation_service):
        """Test that answer generation returns refusal when no chunks are provided"""
        query_request = QueryRequest(
            id="query789",
            question="What is AI?",
            query_mode="book-wide",
            timestamp="2023-01-01T00:00:00"
        )
        
        result = await generation_service.generate_answer(query_request, [])
        
        # Verify refusal response is returned when no chunks are provided
        assert result["answer"] is None
        assert result["status"] == "insufficient_context"
        assert result["reason_code"] == "NO_RELEVANT_CONTEXT"

    @pytest.mark.asyncio
    async def test_validate_answer_grounding_with_content(self, generation_service):
        """Test that answer grounding validation works with content"""
        answer = "The text discusses artificial intelligence and machine learning."
        context = "This document discusses artificial intelligence and machine learning..."
        
        result = generation_service._validate_answer_grounding(answer, context)
        
        # Should return True because there's overlap between answer and context
        assert result is True

    @pytest.mark.asyncio
    async def test_validate_answer_grounding_refusal_response(self, generation_service):
        """Test that refusal responses are considered properly grounded"""
        answer = "Cannot answer based on the provided context."
        context = "This is the context."
        
        result = generation_service._validate_answer_grounding(answer, context)
        
        # Should return True because refusal responses are still grounded
        assert result is True

    @pytest.mark.asyncio
    async def test_validate_answer_grounding_no_content(self, generation_service):
        """Test that grounding validation fails with no content"""
        answer = ""
        context = ""
        
        result = generation_service._validate_answer_grounding(answer, context)
        
        # Should return False because there's no content to validate
        assert result is False

    @pytest.mark.asyncio
    async def test_validate_generation_ability(self, generation_service):
        """Test validation of generation ability"""
        # Valid query
        valid_query = QueryRequest(
            id="query101",
            question="What is robot kinematics?",
            query_mode="book-wide",
            timestamp="2023-01-01T00:00:00"
        )
        
        result = await generation_service.validate_generation_ability(valid_query)
        assert result is True

        # Invalid query (too short)
        invalid_query = QueryRequest(
            id="query102",
            question="Hi",
            query_mode="book-wide",
            timestamp="2023-01-01T00:00:00"
        )
        
        result = await generation_service.validate_generation_ability(invalid_query)
        assert result is False