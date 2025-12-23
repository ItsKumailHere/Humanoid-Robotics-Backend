from fastapi import APIRouter, HTTPException, Depends, status
from typing import Optional
import time
from datetime import datetime
import asyncio

from ...models.query import QueryRequest, QueryResponse
from ...models.document import DocumentChunk
from ...config.database import db
from ...config.settings import settings
from ...config.vector_store import vector_store
from ...services.embedding_service import EmbeddingService
from ...services.retrieval_service import RetrievalService 
from ...services.generation_service import GenerationService
from ...services.citation_service import CitationService
from ...services.validation_service import ValidationService
from ...utils.response_formatter import format_response_with_citations, format_refusal_response

router = APIRouter()

# Initialize services
embedding_service = EmbeddingService()
retrieval_service = RetrievalService()
generation_service = GenerationService()
citation_service = CitationService()
validation_service = ValidationService()



@router.post("/api/v1/query", response_model=QueryResponse)
async def query_endpoint(query_request: QueryRequest):
    """
    POST /api/v1/query endpoint for book-wide questions
    """
    start_time = time.time()
    
    try:
        print(f"\n{'='*60}")
        print(f"Processing query: {query_request.question}")
        print(f"Query mode: {query_request.query_mode}")
        print(f"{'='*60}\n")
        
        # 1. Validate the query request
        validation_errors = validation_service.validate_query_request(query_request)
        if validation_errors:
            print(f"Validation errors: {validation_errors}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail={"error": "Invalid query parameters", "details": validation_errors}
            )
        
        # 2. Check if we can generate an answer
        can_generate = await generation_service.validate_generation_ability(query_request)
        print(f"Can generate: {can_generate}")
        if not can_generate:
            return format_refusal_response(
                query_request.id,
                "NO_RELEVANT_CONTEXT",
                "The query does not contain enough information to generate an answer."
            )
        
        # 3. Create embedding for the query
        print("Creating embedding...")
        query_embedding = await embedding_service.create_single_embedding(query_request.question)
        print(f"Embedding created. Dimension: {len(query_embedding)}")
        print(f"First 5 values: {query_embedding[:5]}")
        
        # 4. Retrieve relevant context
        print(f"\nRetrieving chunks (mode: {query_request.query_mode})...")
        retrieved_chunks = await retrieval_service.retrieve_for_query(
            query_embedding, 
            query_mode=query_request.query_mode,
            selected_text=query_request.selected_text
        )
        
        print(f"Retrieved {len(retrieved_chunks)} chunks")
        for i, chunk in enumerate(retrieved_chunks):
            print(f"  Chunk {i+1}: ID={chunk.id}, DocID={chunk.document_id}, Content length={len(chunk.content) if chunk.content else 0}")
            if chunk.content:
                print(f"    Preview: {chunk.content[:100]}...")
        
        # 5. Validate that we have sufficient context
        print("\nChecking refusal conditions...")
        refusal_response = validation_service.check_refusal_conditions(
            query_request, 
            retrieved_chunks
        )
        
        if refusal_response:
            print(f"REFUSAL: {refusal_response}")
            response_time = (time.time() - start_time) * 1000
            return format_refusal_response(
                query_request.id,
                refusal_response["reason_code"],
                refusal_response["explanation"]
            )
        
        print("Validation passed, generating answer...")

                # In your query_endpoint, update the generation step:

        print("Validation passed, generating answer...")
        print(f"Chunks being sent to generation: {len(retrieved_chunks)}")
        for i, chunk in enumerate(retrieved_chunks):
          print(f"  Chunk {i+1} content preview: {chunk.content[:200]}...")
        
        # 6. Generate the answer
        generation_result = await generation_service.generate_answer(query_request, retrieved_chunks)
        print(f"Generation result status: {generation_result['status']}")
        print(f"Generation result status: {generation_result['status']}")
        print(f"Generation result keys: {generation_result.keys()}")
        print(f"Generation result: {generation_result}")
        
        # 7. If generation resulted in a refusal, return it
        if generation_result["status"] == "insufficient_context":
            return format_refusal_response(
                query_request.id,
                generation_result.get("reason_code", "NO_RELEVANT_CONTEXT"),
                generation_result.get("explanation", "Could not generate an answer based on the provided context.")
            )
        
        # 8. Generate citations for the response
        try:
            citations = await citation_service.generate_citation_for_response(
                retrieved_chunks, 
                query_request.id
            )
        except ValueError as e:
            # If citation generation fails, return refusal as per constitution requirement
            return format_refusal_response(
                query_request.id,
                "CITATION_FAILURE",
                f"Citations could not be generated: {str(e)}"
            )
        
        # 9. Format and return the response
        response_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        # Create the response object
        response = QueryResponse(
            id=f"resp_{query_request.id}",
            query_id=query_request.id,
            answer=generation_result["answer"],
            citations=citations,
            retrieved_chunks=[chunk.id for chunk in retrieved_chunks],  # List of chunk IDs used
            confidence_score=generation_result.get("confidence_score", 0.0),
            generation_time_ms=response_time,
            status=generation_result["status"],
            timestamp=datetime.now()
        )
        
        # Format the response according to the API contract
        formatted_response = format_response_with_citations(response)
        formatted_response["timestamp"] = datetime.now().isoformat()
        
        # Check if response time exceeds the 5-second limit
        if response_time > (settings.rag_response_timeout_seconds * 1000):
            # In a real system, we might want to handle timeout differently
            # For now, we'll log it as a warning
            print(f"WARNING: Response time ({response_time}ms) exceeded timeout threshold")
        
        return formatted_response
        
    except Exception as e:
        # Log the error for debugging
        print(f"Error in query endpoint: {str(e)}")
        
        # Return a generic error response
        response_time = (time.time() - start_time) * 1000
        return format_refusal_response(
            query_request.id,
            "ERROR",
            f"An error occurred while processing your request: {str(e)}"
        )


@router.get("/api/v1/debug/status")
async def debug_status():
    """Debug endpoint to check system status"""
    try:
        # Check Qdrant through the service
        client = retrieval_service.client
        collection_info = client.get_collection("document_chunks")
        
        # Get a sample point to check structure
        sample_points = client.scroll(
            collection_name="document_chunks",
            limit=1,
            with_payload=True,
            with_vectors=False
        )
        
        sample_payload = None
        if sample_points[0]:
            sample_payload = sample_points[0][0].payload
        
        # Check database
        pool = await db.get_pool()
        async with pool.acquire() as connection:
            chunk_count = await connection.fetchval(
                "SELECT COUNT(*) FROM document_chunks"
            )
            sample_chunk = await connection.fetchrow(
                "SELECT id, document_id, chunk_index, LENGTH(content) as content_length FROM document_chunks LIMIT 1"
            )
        
        return {
            "qdrant": {
                "connected": True,
                "collection": "document_chunks",
                "points_count": collection_info.points_count,
                "vectors_count": collection_info.vectors_count,
                "vector_size": collection_info.config.params.vectors.size,
                "sample_payload": sample_payload
            },
            "database": {
                "connected": True,
                "chunks_count": chunk_count,
                "sample_chunk": dict(sample_chunk) if sample_chunk else None
            },
            "embedding_service": {
                "configured": True
            }
        }
    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }


@router.get("/api/v1/debug/compare-ids")
async def compare_ids():
    """Compare IDs between Qdrant and PostgreSQL"""
    try:
        # Get sample IDs from Qdrant
        client = retrieval_service.client
        qdrant_points = client.scroll(
            collection_name="document_chunks",
            limit=5,
            with_payload=True,
            with_vectors=False
        )
        
        qdrant_ids = [str(point.id) for point in qdrant_points[0]]
        
        # Get sample IDs from PostgreSQL
        pool = await db.get_pool()
        async with pool.acquire() as connection:
            pg_rows = await connection.fetch(
                "SELECT id, document_id, LENGTH(content) as content_len FROM document_chunks LIMIT 5"
            )
            pg_ids = [row['id'] for row in pg_rows]
            
        return {
            "qdrant_sample_ids": qdrant_ids,
            "postgres_sample_ids": pg_ids,
            "match": any(qid in pg_ids for qid in qdrant_ids)
        }
    except Exception as e:
        import traceback
        return {"error": str(e), "traceback": traceback.format_exc()}        



@router.post("/api/v1/debug/test-generation")
async def test_generation(query_request: QueryRequest):
    """Test generation with mock chunks"""
    from datetime import datetime
    
    # Create a mock chunk with content
    mock_chunks = [
        DocumentChunk(
            id="test_1",
            document_id="test_doc",
            content="Humanoid robotics is the field focused on designing and building robots with human-like appearance and behavior.",
            chunk_index=0,
            embedding_vector=None,
            created_at=datetime.now()
        )
    ]
    
    try:
        result = await generation_service.generate_answer(query_request, mock_chunks)
        return {
            "success": True,
            "result": result
        }
    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc()
        }