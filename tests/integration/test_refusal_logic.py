import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from unittest.mock import patch, AsyncMock
import asyncio


class TestRefusalLogic:
    
    @pytest.fixture
    def client(self):
        """Create a test client for the FastAPI app"""
        return TestClient(app)

    @pytest.mark.asyncio
    async def test_refusal_no_relevant_context(self, client):
        """Test refusal logic when no relevant context is found"""
        with patch('src.services.embedding_service.EmbeddingService.create_single_embedding', 
                   new_callable=AsyncMock) as mock_embedding, \
             patch('src.services.retrieval_service.RetrievalService.retrieve_for_query', 
                   new_callable=AsyncMock) as mock_retrieval, \
             patch('src.services.generation_service.GenerationService.generate_answer', 
                   new_callable=AsyncMock) as mock_generation:

            # Mock empty results - no relevant context found
            mock_embedding.return_value = [0.1, 0.2, 0.3]
            mock_retrieval.return_value = []  # No chunks retrieved
            
            # The generation service should return a refusal when no chunks are available
            mock_generation.return_value = {
                "answer": None,
                "status": "insufficient_context",
                "reason_code": "NO_RELEVANT_CONTEXT",
                "explanation": "No relevant context found in the provided information."
            }

            # Make request with question that likely won't have context
            response = client.post("/api/v1/query", json={
                "id": "refusal_query_001",
                "question": "What is the current weather in Mars?",
                "query_mode": "book-wide",
                "timestamp": "2023-01-01T00:00:00"
            })

            # Verify refusal response
            assert response.status_code == 200  # API returns 200 even for refusals
            data = response.json()
            
            assert data["answer"] is None
            assert data["status"] == "insufficient_context"
            assert data["reason_code"] == "NO_RELEVANT_CONTEXT"
            assert "explanation" in data
            print(f"✓ Refusal response correctly returned: {data['reason_code']}")

    @pytest.mark.asyncio
    async def test_refusal_generation_returns_refusal(self, client):
        """Test refusal logic when the generation service explicitly returns refusal"""
        with patch('src.services.embedding_service.EmbeddingService.create_single_embedding', 
                   new_callable=AsyncMock) as mock_embedding, \
             patch('src.services.retrieval_service.RetrievalService.retrieve_for_query', 
                   new_callable=AsyncMock) as mock_retrieval, \
             patch('src.services.generation_service.GenerationService.generate_answer', 
                   new_callable=AsyncMock) as mock_generation:

            # Setup normal flow with chunks but generation returns refusal
            mock_embedding.return_value = [0.4, 0.5, 0.6]
            
            from src.models.document import DocumentChunk
            mock_retrieval.return_value = [
                DocumentChunk(
                    id="chunk_refusal",
                    document_id="doc_refusal",
                    content="This document talks about robotics...",
                    chunk_index=0,
                    created_at="2023-01-01T00:00:00"
                )
            ]
            
            # Generation service returns refusal even with context
            mock_generation.return_value = {
                "answer": None,
                "status": "insufficient_context",
                "reason_code": "NO_RELEVANT_CONTEXT",
                "explanation": "Could not generate an answer based on the provided context."
            }

            response = client.post("/api/v1/query", json={
                "id": "refusal_query_002",
                "question": "What is the meaning of life?",
                "query_mode": "book-wide",
                "timestamp": "2023-01-01T00:00:00"
            })

            # Verify refusal response
            assert response.status_code == 200
            data = response.json()
            
            assert data["answer"] is None
            assert data["status"] == "insufficient_context"
            assert data["reason_code"] == "NO_RELEVANT_CONTEXT"
            print(f"✓ Refusal response correctly handled from generation service: {data['reason_code']}")

    @pytest.mark.asyncio
    async def test_refusal_citation_failure(self, client):
        """Test refusal logic when citation generation fails (as per constitution requirement)"""
        with patch('src.services.embedding_service.EmbeddingService.create_single_embedding', 
                   new_callable=AsyncMock) as mock_embedding, \
             patch('src.services.retrieval_service.RetrievalService.retrieve_for_query', 
                   new_callable=AsyncMock) as mock_retrieval, \
             patch('src.services.generation_service.GenerationService.generate_answer', 
                   new_callable=AsyncMock) as mock_generation, \
             patch('src.services.citation_service.CitationService.generate_citation_for_response', 
                   side_effect=ValueError("Citation could not be generated")):

            # Setup normal flow
            mock_embedding.return_value = [0.7, 0.8, 0.9]
            
            from src.models.document import DocumentChunk
            mock_retrieval.return_value = [
                DocumentChunk(
                    id="chunk_citation",
                    document_id="doc_citation",
                    content="Content that should generate a citation",
                    chunk_index=0,
                    created_at="2023-01-01T00:00:00"
                )
            ]
            
            mock_generation.return_value = {
                "answer": "This is the answer to your question",
                "status": "success",
                "confidence_score": 0.8
            }

            response = client.post("/api/v1/query", json={
                "id": "refusal_query_003",
                "question": "What is this content about?",
                "query_mode": "book-wide",
                "timestamp": "2023-01-01T00:00:00"
            })

            # Since citation generation fails, the entire response should be refused
            assert response.status_code == 200
            data = response.json()
            
            assert data["answer"] is None
            assert data["status"] == "insufficient_context"
            assert data["reason_code"] == "CITATION_FAILURE"
            assert "explanation" in data
            print(f"✓ Refusal correctly triggered by citation failure: {data['reason_code']}")

    @pytest.mark.asyncio
    async def test_refusal_invalid_query_mode(self, client):
        """Test that invalid query mode is properly validated"""
        response = client.post("/api/v1/query", json={
            "id": "invalid_query",
            "question": "What is AI?",
            "query_mode": "invalid_mode",  # Invalid mode
            "timestamp": "2023-01-01T00:00:00"
        })

        # Invalid query should return 400 (validation error), not a refusal response
        assert response.status_code == 400
        print("✓ Invalid query mode returns proper validation error")

    @pytest.mark.asyncio
    async def test_refusal_selected_text_mode_insufficient_context(self, client):
        """Test refusal in selected-text mode when the text doesn't contain needed info"""
        with patch('src.services.embedding_service.EmbeddingService.create_single_embedding', 
                   new_callable=AsyncMock) as mock_embedding, \
             patch('src.services.retrieval_service.RetrievalService.retrieve_for_query', 
                   new_callable=AsyncMock) as mock_retrieval, \
             patch('src.services.generation_service.GenerationService.generate_answer', 
                   new_callable=AsyncMock) as mock_generation:

            mock_embedding.return_value = [0.1, 0.2, 0.3]
            
            # In selected-text mode, retrieval returns chunks based on the provided text
            from src.models.document import DocumentChunk
            mock_retrieval.return_value = [
                DocumentChunk(
                    id="temp_selected_text",
                    document_id="selected_text_session",
                    content="This text talks about flowers and gardening.",
                    chunk_index=0,
                    created_at="2023-01-01T00:00:00"
                )
            ]
            
            # Generation service returns refusal because the selected text doesn't answer the question
            mock_generation.return_value = {
                "answer": None,
                "status": "insufficient_context",
                "reason_code": "NO_RELEVANT_CONTEXT",
                "explanation": "No relevant context found in the provided selected text."
            }

            response = client.post("/api/v1/query", json={
                "id": "refusal_query_004",
                "question": "What are the principles of quantum physics?",
                "query_mode": "selected-text",
                "selected_text": "This text talks about flowers and gardening.",
                "timestamp": "2023-01-01T00:00:00"
            })

            # Verify refusal response
            assert response.status_code == 200
            data = response.json()
            
            assert data["answer"] is None
            assert data["status"] == "insufficient_context"
            assert data["reason_code"] == "NO_RELEVANT_CONTEXT"
            assert "explanation" in data
            print(f"✓ Selected-text refusal correctly handled: {data['reason_code']}")

    @pytest.mark.asyncio
    async def test_no_refusal_when_context_is_sufficient(self, client):
        """Test that no refusal happens when context is sufficient"""
        with patch('src.services.embedding_service.EmbeddingService.create_single_embedding', 
                   new_callable=AsyncMock) as mock_embedding, \
             patch('src.services.retrieval_service.RetrievalService.retrieve_for_query', 
                   new_callable=AsyncMock) as mock_retrieval, \
             patch('src.services.generation_service.GenerationService.generate_answer', 
                   new_callable=AsyncMock) as mock_generation, \
             patch('src.services.citation_service.CitationService.generate_citation_for_response', 
                   new_callable=AsyncMock) as mock_citation:

            mock_embedding.return_value = [0.5, 0.6, 0.7]
            
            from src.models.document import DocumentChunk
            mock_retrieval.return_value = [
                DocumentChunk(
                    id="chunk_valid",
                    document_id="doc_valid",
                    content="Robotics is a field that combines engineering and computer science.",
                    chunk_index=0,
                    created_at="2023-01-01T00:00:00"
                )
            ]
            
            mock_generation.return_value = {
                "answer": "Robotics is a field that combines engineering and computer science.",
                "status": "success",
                "confidence_score": 0.9
            }
            
            from src.models.citation import Citation
            mock_citation.return_value = [
                Citation(
                    id="cit_valid",
                    response_id="resp_valid",
                    document_id="doc_valid",
                    chapter="Chapter 3",
                    section="Section 3.1",
                    file_path="/content/chapter3.md",
                    relevance_score=0.95,
                    text_snippet="Robotics is a field that combines engineering and computer science."
                )
            ]

            response = client.post("/api/v1/query", json={
                "id": "valid_query_001",
                "question": "What is robotics?",
                "query_mode": "book-wide",
                "timestamp": "2023-01-01T00:00:00"
            })

            # This should succeed, not refuse
            assert response.status_code == 200
            data = response.json()
            
            assert data["answer"] is not None
            assert data["status"] == "success"
            assert data["reason_code"] is None  # No reason_code for successful responses
            print("✓ Valid query correctly returned answer instead of refusal")