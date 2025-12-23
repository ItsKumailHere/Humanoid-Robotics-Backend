import pytest
import time
from fastapi.testclient import TestClient
from src.api.main import app
from unittest.mock import patch, AsyncMock
import asyncio


class TestBenchmarkResponseTime:
    
    @pytest.fixture
    def client(self):
        """Create a test client for the FastAPI app"""
        return TestClient(app)
    
    @pytest.mark.asyncio
    async def test_response_time_book_wide_query(self, client):
        """Benchmark response time for book-wide queries (should be ≤5 seconds)"""
        # Mock all the services to focus on measuring response time
        with patch('src.services.embedding_service.EmbeddingService.create_single_embedding', 
                   new_callable=AsyncMock) as mock_embedding, \
             patch('src.services.retrieval_service.RetrievalService.retrieve_for_query', 
                   new_callable=AsyncMock) as mock_retrieval, \
             patch('src.services.generation_service.GenerationService.generate_answer', 
                   new_callable=AsyncMock) as mock_generation, \
             patch('src.services.citation_service.CitationService.generate_citation_for_response', 
                   new_callable=AsyncMock) as mock_citation:
            
            # Simulate realistic processing times by adding small delays
            async def slow_embedding(*args, **kwargs):
                await asyncio.sleep(0.01)  # 10ms delay
                return [0.1, 0.2, 0.3, 0.4, 0.5]
            
            async def slow_retrieval(*args, **kwargs):
                await asyncio.sleep(0.02)  # 20ms delay
                from src.models.document import DocumentChunk
                return [
                    DocumentChunk(
                        id="chunk123",
                        document_id="doc123",
                        content="Artificial intelligence content for testing response time...",
                        chunk_index=0,
                        created_at="2023-01-01T00:00:00"
                    )
                ]
            
            async def slow_generation(*args, **kwargs):
                await asyncio.sleep(0.015)  # 15ms delay
                return {
                    "answer": "This is the answer based on the context.",
                    "status": "success",
                    "confidence_score": 0.85
                }
            
            async def slow_citation(*args, **kwargs):
                await asyncio.sleep(0.005)  # 5ms delay
                from src.models.citation import Citation
                return [
                    Citation(
                        id="cit123",
                        response_id="resp123",
                        document_id="doc123",
                        chapter="Chapter 1",
                        section="Section 1.1",
                        file_path="/content/chapter1.md",
                        relevance_score=0.9,
                        text_snippet="Artificial intelligence content for testing..."
                    )
                ]
            
            # Set up the mocks with slow functions to simulate processing time
            mock_embedding.side_effect = slow_embedding
            mock_retrieval.side_effect = slow_retrieval
            mock_generation.side_effect = slow_generation
            mock_citation.side_effect = slow_citation
            
            # Measure the response time for the API call
            start_time = time.time()
            
            response = client.post("/api/v1/query", json={
                "id": "benchmark_query",
                "question": "What is the performance of this system?",
                "query_mode": "book-wide",
                "timestamp": "2023-01-01T00:00:00"
            })
            
            end_time = time.time()
            response_time = end_time - start_time
            
            # Verify the response was successful
            assert response.status_code == 200
            
            # Check that the response time is within the required limit (≤5 seconds)
            assert response_time <= 5.0, f"Response time {response_time:.2f}s exceeded 5 second limit"
            
            # Log the actual response time for monitoring
            print(f"✓ Book-wide query response time: {response_time:.3f}s")
            
            # As an additional check, for a well-optimized system, we'd expect much faster response
            # This is just informational, not a failure condition
            if response_time > 1.0:
                print(f"! Note: Response time ({response_time:.3f}s) is higher than optimal (<1s)")
            else:
                print(f"✓ Response time ({response_time:.3f}s) is within optimal range")

    @pytest.mark.asyncio
    async def test_response_time_selected_text_query(self, client):
        """Benchmark response time for selected-text queries (should be ≤5 seconds)"""
        # Mock all the services to focus on measuring response time
        with patch('src.services.embedding_service.EmbeddingService.create_single_embedding', 
                   new_callable=AsyncMock) as mock_embedding, \
             patch('src.services.retrieval_service.RetrievalService.retrieve_for_query', 
                   new_callable=AsyncMock) as mock_retrieval, \
             patch('src.services.generation_service.GenerationService.generate_answer', 
                   new_callable=AsyncMock) as mock_generation, \
             patch('src.services.citation_service.CitationService.generate_citation_for_response', 
                   new_callable=AsyncMock) as mock_citation:
            
            # Simulate realistic processing times
            async def slow_embedding(*args, **kwargs):
                await asyncio.sleep(0.005)  # 5ms delay
                return [0.1, 0.2, 0.3, 0.4, 0.5]
            
            async def slow_retrieval(*args, **kwargs):
                await asyncio.sleep(0.01)  # 10ms delay
                from src.models.document import DocumentChunk
                return [
                    DocumentChunk(
                        id="chunk456",
                        document_id="selected_text_session",
                        content="Selected text content for performance testing...",
                        chunk_index=0,
                        created_at="2023-01-01T00:00:00"
                    )
                ]
            
            async def slow_generation(*args, **kwargs):
                await asyncio.sleep(0.01)  # 10ms delay
                return {
                    "answer": "Based on the selected text...",
                    "status": "success",
                    "confidence_score": 0.75
                }
            
            async def slow_citation(*args, **kwargs):
                await asyncio.sleep(0.003)  # 3ms delay
                from src.models.citation import Citation
                return [
                    Citation(
                        id="cit456",
                        response_id="resp456",
                        document_id="selected_text_session",
                        chapter="Selected Text",
                        section="General",
                        file_path="",
                        relevance_score=0.8,
                        text_snippet="Selected text content for performance testing..."
                    )
                ]
            
            # Set up the mocks with slow functions
            mock_embedding.side_effect = slow_embedding
            mock_retrieval.side_effect = slow_retrieval
            mock_generation.side_effect = slow_generation
            mock_citation.side_effect = slow_citation
            
            # Measure the response time for the API call
            start_time = time.time()
            
            response = client.post("/api/v1/query", json={
                "id": "benchmark_selected_query",
                "question": "What does the selected text say?",
                "query_mode": "selected-text",
                "selected_text": "Selected text content for performance testing...",
                "timestamp": "2023-01-01T00:00:00"
            })
            
            end_time = time.time()
            response_time = end_time - start_time
            
            # Verify the response was successful
            assert response.status_code == 200
            
            # Check that the response time is within the required limit (≤5 seconds)
            assert response_time <= 5.0, f"Response time {response_time:.2f}s exceeded 5 second limit"
            
            # Log the actual response time for monitoring
            print(f"✓ Selected-text query response time: {response_time:.3f}s")
            
            # As an additional check, for a well-optimized system, we'd expect much faster response
            if response_time > 1.0:
                print(f"! Note: Response time ({response_time:.3f}s) is higher than optimal (<1s)")
            else:
                print(f"✓ Response time ({response_time:.3f}s) is within optimal range")

    def test_response_time_api_direct_call(self, client):
        """Benchmark direct API response time without mocked services"""
        # For this test, we'll check the response time of the API endpoint itself
        # without any processing to establish a baseline
        start_time = time.time()
        
        # Use the root endpoint which requires minimal processing
        response = client.get("/")
        
        end_time = time.time()
        response_time = end_time - start_time
        
        # Verify the response was successful
        assert response.status_code == 200
        
        # The basic API response should be very fast (much less than 5 seconds)
        assert response_time <= 5.0, f"Basic API response time {response_time:.3f}s exceeded 5 second limit"
        
        print(f"✓ Basic API response time: {response_time:.6f}s (baseline)")

    @pytest.mark.asyncio
    async def test_multiple_queries_response_time(self, client):
        """Test response time when handling multiple concurrent queries"""
        import concurrent.futures
        import threading
        
        # Mock all the services to control processing time
        with patch('src.services.embedding_service.EmbeddingService.create_single_embedding', 
                   new_callable=AsyncMock) as mock_embedding, \
             patch('src.services.retrieval_service.RetrievalService.retrieve_for_query', 
                   new_callable=AsyncMock) as mock_retrieval, \
             patch('src.services.generation_service.GenerationService.generate_answer', 
                   new_callable=AsyncMock) as mock_generation, \
             patch('src.services.citation_service.CitationService.generate_citation_for_response', 
                   new_callable=AsyncMock) as mock_citation:
            
            async def slow_embedding(*args, **kwargs):
                await asyncio.sleep(0.01)  # 10ms delay
                return [0.1, 0.2, 0.3]
            
            async def slow_retrieval(*args, **kwargs):
                await asyncio.sleep(0.02)  # 20ms delay
                from src.models.document import DocumentChunk
                return [
                    DocumentChunk(
                        id="chunk_multi",
                        document_id="doc_multi",
                        content="Content for multiple queries test...",
                        chunk_index=0,
                        created_at="2023-01-01T00:00:00"
                    )
                ]
            
            async def slow_generation(*args, **kwargs):
                await asyncio.sleep(0.015)  # 15ms delay
                return {
                    "answer": "Answer to multiple queries test.",
                    "status": "success",
                    "confidence_score": 0.8
                }
            
            async def slow_citation(*args, **kwargs):
                await asyncio.sleep(0.005)  # 5ms delay
                from src.models.citation import Citation
                return [
                    Citation(
                        id="cit_multi",
                        response_id="resp_multi",
                        document_id="doc_multi",
                        chapter="Chapter Multi",
                        section="Section 1",
                        file_path="/content/multi.md",
                        relevance_score=0.85,
                        text_snippet="Content for multiple queries test..."
                    )
                ]
            
            mock_embedding.side_effect = slow_embedding
            mock_retrieval.side_effect = slow_retrieval
            mock_generation.side_effect = slow_generation
            mock_citation.side_effect = slow_citation
            
            # Function to make a single API call
            def make_request(query_id):
                start_time = time.time()
                response = client.post("/api/v1/query", json={
                    "id": f"multi_query_{query_id}",
                    "question": f"What is the performance for query {query_id}?",
                    "query_mode": "book-wide",
                    "timestamp": "2023-01-01T00:00:00"
                })
                end_time = time.time()
                return end_time - start_time, response.status_code
            
            # Make multiple requests concurrently
            num_requests = 5
            start_time_all = time.time()
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_requests) as executor:
                futures = [executor.submit(make_request, i) for i in range(num_requests)]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
            
            end_time_all = time.time()
            total_time = end_time_all - start_time_all
            
            # Verify all requests were successful
            for response_time, status_code in results:
                assert status_code == 200
                assert response_time <= 5.0, f"Individual request time {response_time:.2f}s exceeded 5 second limit"
            
            # Check that total time is reasonable for concurrent requests
            # (Should be less than the sum of individual times if properly concurrent)
            avg_response_time = sum(rt for rt, _ in results) / len(results)
            print(f"✓ Average response time across {num_requests} concurrent queries: {avg_response_time:.3f}s")
            print(f"✓ Total time for {num_requests} concurrent queries: {total_time:.3f}s")
            
            # Even under concurrent load, individual request should meet requirements
            assert avg_response_time <= 5.0, f"Average response time {avg_response_time:.2f}s exceeded 5 second limit"