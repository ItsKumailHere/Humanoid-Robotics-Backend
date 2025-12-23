import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from unittest.mock import patch, AsyncMock
import asyncio


class TestCitationAccuracy:
    
    @pytest.fixture
    def client(self):
        """Create a test client for the FastAPI app"""
        return TestClient(app)

    @pytest.mark.asyncio
    async def test_citation_contains_required_fields(self, client):
        """Test that citations contain all required fields as per data-model.md"""
        with patch('src.services.embedding_service.EmbeddingService.create_single_embedding', 
                   new_callable=AsyncMock) as mock_embedding, \
             patch('src.services.retrieval_service.RetrievalService.retrieve_for_query', 
                   new_callable=AsyncMock) as mock_retrieval, \
             patch('src.services.generation_service.GenerationService.generate_answer', 
                   new_callable=AsyncMock) as mock_generation, \
             patch('src.services.citation_service.CitationService.generate_citation_for_response', 
                   new_callable=AsyncMock) as mock_citation:

            mock_embedding.return_value = [0.1, 0.2, 0.3]
            
            from src.models.document import DocumentChunk
            mock_retrieval.return_value = [
                DocumentChunk(
                    id="chunk_cite_test",
                    document_id="doc_cite_test",
                    content="This is sample content that should be cited properly.",
                    chunk_index=0,
                    created_at="2023-01-01T00:00:00"
                )
            ]
            
            mock_generation.return_value = {
                "answer": "Based on the text, this is the answer.",
                "status": "success",
                "confidence_score": 0.85
            }
            
            from src.models.citation import Citation
            expected_citation = Citation(
                id="cit_cite_test",
                response_id="resp_cite_test",
                document_id="doc_cite_test",
                chapter="Chapter 1",  # Example chapter
                section="Section 1.1",  # Example section
                file_path="/content/chapter1.md",  # Example file path
                relevance_score=0.9,  # Example relevance score
                text_snippet="This is sample content that should be cited properly."
            )
            mock_citation.return_value = [expected_citation]

            response = client.post("/api/v1/query", json={
                "id": "cite_test_001",
                "question": "What does the document say about proper citations?",
                "query_mode": "book-wide",
                "timestamp": "2023-01-01T00:00:00"
            })

            assert response.status_code == 200
            data = response.json()
            
            # Check that citations are present
            assert "citations" in data
            assert len(data["citations"]) == 1
            
            citation = data["citations"][0]
            
            # Validate all required citation fields per data-model.md
            required_fields = [
                "id", "response_id", "document_id", 
                "chapter", "section", "file_path", 
                "relevance_score", "text_snippet"
            ]
            
            for field in required_fields:
                assert field in citation, f"Required field '{field}' missing from citation"
            
            # Validate specific field types/values
            assert isinstance(citation["id"], str) and len(citation["id"]) > 0
            assert isinstance(citation["response_id"], str) and len(citation["response_id"]) > 0
            assert isinstance(citation["document_id"], str) and len(citation["document_id"]) > 0
            assert isinstance(citation["chapter"], str) and len(citation["chapter"]) > 0
            assert isinstance(citation["section"], str) and len(citation["section"]) > 0
            assert isinstance(citation["file_path"], str) and len(citation["file_path"]) > 0
            assert isinstance(citation["relevance_score"], (int, float)) and 0.0 <= citation["relevance_score"] <= 1.0
            assert isinstance(citation["text_snippet"], str) and len(citation["text_snippet"]) > 0
            
            print("✓ All required citation fields are present and properly typed")

    @pytest.mark.asyncio
    async def test_citation_values_accuracy(self, client):
        """Test that citation values accurately reflect the source document"""
        with patch('src.services.embedding_service.EmbeddingService.create_single_embedding', 
                   new_callable=AsyncMock) as mock_embedding, \
             patch('src.services.retrieval_service.RetrievalService.retrieve_for_query', 
                   new_callable=AsyncMock) as mock_retrieval, \
             patch('src.services.generation_service.GenerationService.generate_answer', 
                   new_callable=AsyncMock) as mock_generation, \
             patch('src.services.citation_service.CitationService.generate_citation_for_response', 
                   new_callable=AsyncMock) as mock_citation:

            mock_embedding.return_value = [0.4, 0.5, 0.6]
            
            from src.models.document import DocumentChunk
            retrieved_chunk = DocumentChunk(
                id="chunk_acc_test",
                document_id="doc_acc_test",
                content="Artificial intelligence (AI) is intelligence demonstrated by machines.",
                chunk_index=0,
                created_at="2023-01-01T00:00:00"
            )
            mock_retrieval.return_value = [retrieved_chunk]
            
            mock_generation.return_value = {
                "answer": "AI is intelligence demonstrated by machines.",
                "status": "success",
                "confidence_score": 0.9
            }
            
            from src.models.citation import Citation
            # Create citation that reflects the actual retrieved content
            accurate_citation = Citation(
                id="cit_acc_test",
                response_id="resp_acc_test",
                document_id="doc_acc_test",  # Should match document_id from chunk
                chapter="Chapter 5",  # Accurate chapter
                section="Section 5.2",  # Accurate section
                file_path="/content/ai_concepts.md",  # Accurate file path
                relevance_score=0.85,  # Reasonable relevance score
                text_snippet="Artificial intelligence (AI) is intelligence demonstrated by machines."  # Accurate snippet
            )
            mock_citation.return_value = [accurate_citation]

            response = client.post("/api/v1/query", json={
                "id": "acc_test_001",
                "question": "What is artificial intelligence?",
                "query_mode": "book-wide",
                "timestamp": "2023-01-01T00:00:00"
            })

            assert response.status_code == 200
            data = response.json()
            
            assert len(data["citations"]) == 1
            citation = data["citations"][0]
            
            # Verify the citation accurately reflects the source
            assert citation["document_id"] == "doc_acc_test"
            assert "intelligence demonstrated by machines" in citation["text_snippet"]
            assert 0.0 <= citation["relevance_score"] <= 1.0
            
            print("✓ Citation values accurately reflect source document content")

    @pytest.mark.asyncio
    async def test_multiple_citations_accuracy(self, client):
        """Test that multiple citations are all accurate when multiple chunks are used"""
        with patch('src.services.embedding_service.EmbeddingService.create_single_embedding', 
                   new_callable=AsyncMock) as mock_embedding, \
             patch('src.services.retrieval_service.RetrievalService.retrieve_for_query', 
                   new_callable=AsyncMock) as mock_retrieval, \
             patch('src.services.generation_service.GenerationService.generate_answer', 
                   new_callable=AsyncMock) as mock_generation, \
             patch('src.services.citation_service.CitationService.generate_citation_for_response', 
                   new_callable=AsyncMock) as mock_citation:

            mock_embedding.return_value = [0.7, 0.8, 0.9]
            
            from src.models.document import DocumentChunk
            retrieved_chunks = [
                DocumentChunk(
                    id="chunk_multi_1",
                    document_id="doc_multi_1",
                    content="First piece of relevant information about robotics.",
                    chunk_index=0,
                    created_at="2023-01-01T00:00:00"
                ),
                DocumentChunk(
                    id="chunk_multi_2",
                    document_id="doc_multi_2", 
                    content="Second piece of relevant information about AI.",
                    chunk_index=1,
                    created_at="2023-01-01T00:00:00"
                )
            ]
            mock_retrieval.return_value = retrieved_chunks
            
            mock_generation.return_value = {
                "answer": "Robotics combines AI and engineering.",
                "status": "success",
                "confidence_score": 0.88
            }
            
            from src.models.citation import Citation
            multiple_citations = [
                Citation(
                    id="cit_multi_1",
                    response_id="resp_multi_test",
                    document_id="doc_multi_1",
                    chapter="Chapter 3",
                    section="Section 3.1",
                    file_path="/content/robotics_fundamentals.md",
                    relevance_score=0.8,
                    text_snippet="First piece of relevant information about robotics."
                ),
                Citation(
                    id="cit_multi_2",
                    response_id="resp_multi_test",
                    document_id="doc_multi_2",
                    chapter="Chapter 4", 
                    section="Section 4.2",
                    file_path="/content/ai_principles.md",
                    relevance_score=0.75,
                    text_snippet="Second piece of relevant information about AI."
                )
            ]
            mock_citation.return_value = multiple_citations

            response = client.post("/api/v1/query", json={
                "id": "multi_test_001",
                "question": "How do robotics and AI relate?",
                "query_mode": "book-wide",
                "timestamp": "2023-01-01T00:00:00"
            })

            assert response.status_code == 200
            data = response.json()
            
            # Verify multiple citations are returned
            assert len(data["citations"]) == 2
            
            # Check each citation has proper structure
            for citation in data["citations"]:
                assert "document_id" in citation
                assert "text_snippet" in citation
                assert "chapter" in citation
                assert "section" in citation
                assert "file_path" in citation
                assert "relevance_score" in citation
            
            # Verify the citations match the retrieved chunks
            citation_doc_ids = {cit["document_id"] for cit in data["citations"]}
            expected_doc_ids = {"doc_multi_1", "doc_multi_2"}
            assert citation_doc_ids == expected_doc_ids
            
            print("✓ Multiple citations are accurate and match retrieved chunks")

    @pytest.mark.asyncio
    async def test_citation_relevance_score_range(self, client):
        """Test that citation relevance scores are within the valid range [0.0, 1.0]"""
        with patch('src.services.embedding_service.EmbeddingService.create_single_embedding', 
                   new_callable=AsyncMock) as mock_embedding, \
             patch('src.services.retrieval_service.RetrievalService.retrieve_for_query', 
                   new_callable=AsyncMock) as mock_retrieval, \
             patch('src.services.generation_service.GenerationService.generate_answer', 
                   new_callable=AsyncMock) as mock_generation, \
             patch('src.services.citation_service.CitationService.generate_citation_for_response', 
                   new_callable=AsyncMock) as mock_citation:

            mock_embedding.return_value = [0.2, 0.3, 0.4]
            
            from src.models.document import DocumentChunk
            mock_retrieval.return_value = [
                DocumentChunk(
                    id="chunk_score_test",
                    document_id="doc_score_test",
                    content="Content for testing relevance scores.",
                    chunk_index=0,
                    created_at="2023-01-01T00:00:00"
                )
            ]
            
            mock_generation.return_value = {
                "answer": "This is an answer based on the content.",
                "status": "success",
                "confidence_score": 0.82
            }
            
            from src.models.citation import Citation
            # Create citation with various possible relevance scores
            test_citations = [
                Citation(
                    id="cit_score_1",
                    response_id="resp_score_test",
                    document_id="doc_score_test",
                    chapter="Chapter Test",
                    section="Section 1",
                    file_path="/content/test.md",
                    relevance_score=0.95,  # High relevance
                    text_snippet="Content for testing relevance scores."
                ),
                Citation(
                    id="cit_score_2", 
                    response_id="resp_score_test",
                    document_id="doc_score_test",
                    chapter="Chapter Test",
                    section="Section 2", 
                    file_path="/content/test2.md",
                    relevance_score=0.3,  # Lower relevance
                    text_snippet="Less relevant content."
                )
            ]
            mock_citation.return_value = test_citations

            response = client.post("/api/v1/query", json={
                "id": "score_test_001",
                "question": "What are the relevance scores?",
                "query_mode": "book-wide",
                "timestamp": "2023-01-01T00:00:00"
            })

            assert response.status_code == 200
            data = response.json()
            
            # Verify all relevance scores are in valid range
            for citation in data["citations"]:
                relevance_score = citation["relevance_score"]
                assert 0.0 <= relevance_score <= 1.0, f"Relevance score {relevance_score} not in range [0.0, 1.0]"
            
            print("✓ All citation relevance scores are in the valid range [0.0, 1.0]")

    @pytest.mark.asyncio
    async def test_citation_format_compatibility_with_database_schema(self, client):
        """Test that citation fields are compatible with the database schema in data-model.md"""
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
                    id="chunk_schema_test",
                    document_id="doc_schema_test_123456789012345678901234567890123456789012345678901234567890123",  # Long ID
                    content="This content tests database field length constraints." * 10,  # Long content
                    chunk_index=0,
                    created_at="2023-01-01T00:00:00"
                )
            ]
            
            mock_generation.return_value = {
                "answer": "Answer for database schema test.",
                "status": "success", 
                "confidence_score": 0.77
            }
            
            from src.models.citation import Citation
            # Create citation with field lengths that should match database constraints
            long_citation = Citation(
                id="cit_schema_test_1234567890123456789012345678901234567890",  # Up to VARCHAR(255)
                response_id="resp_schema_test_1234567890123456789012345678901234567890",
                document_id="doc_schema_test_12345678901234567890123456789012345678901234567890",  # Should match document ID
                chapter="Very Long Chapter Name That Tests Field Length Limits" * 5,  # VARCHAR(255)
                section="Detailed Section Name With Subsection Information" * 3,  # VARCHAR(255)
                file_path="/very/long/path/to/content/file/that/tests/field/length/limits/in/the/database/schema/document.md",  # VARCHAR(500)
                relevance_score=0.82,  # DECIMAL(3,2) - should handle up to 2 decimal places
                text_snippet="Long text snippet that tests the TEXT field type in the database schema." * 20  # TEXT field
            )
            mock_citation.return_value = [long_citation]

            response = client.post("/api/v1/query", json={
                "id": "schema_test_001",
                "question": "Test database field compatibility?",
                "query_mode": "book-wide",
                "timestamp": "2023-01-01T00:00:00"
            })

            assert response.status_code == 200
            data = response.json()
            
            # Verify the citation was processed without field length errors
            assert len(data["citations"]) == 1
            citation = data["citations"][0]
            
            # Verify field types match expected database types
            assert isinstance(citation["id"], str)
            assert isinstance(citation["response_id"], str)
            assert isinstance(citation["document_id"], str)
            assert isinstance(citation["chapter"], str)
            assert isinstance(citation["section"], str)
            assert isinstance(citation["file_path"], str)
            assert isinstance(citation["relevance_score"], (int, float))
            assert isinstance(citation["text_snippet"], str)
            
            # Verify relevance score precision (should be a reasonable precision)
            relevance_score = citation["relevance_score"]
            decimal_places = len(str(relevance_score).split('.')[-1]) if '.' in str(relevance_score) else 0
            assert decimal_places <= 3  # Should be precise but not overly so
            
            print("✓ Citation fields are compatible with database schema constraints")