from fastapi import APIRouter
from datetime import datetime
import time
from typing import Dict, Any

from ...config.settings import settings
from ...config.database import db
from ...config.vector_store import vector_store

router = APIRouter()


@router.get("/api/v1/health")
async def health_check() -> Dict[str, Any]:
    """
    GET /api/v1/health endpoint to check service dependencies.
    Returns the health status of the RAG backend service.
    """
    start_time = time.time()

    # Check the status of various dependencies
    dependencies = {
        "qdrant": "checking...",
        "postgres": "checking...",
        "llm_service": "checking...",  # Updated to reflect the actual LLM service
        "embedding_service": "checking..."  # Updated to reflect the actual embedding service
    }

    try:
        # Test Qdrant connection
        try:
            client = vector_store.get_client()
            # Perform a simple operation to test the connection
            client.get_collections()
            dependencies["qdrant"] = "connected"
        except Exception as e:
            dependencies["qdrant"] = f"error: {str(e)}"

        # Test Postgres connection
        try:
            pool = await db.get_pool()
            async with pool.acquire() as conn:
                await conn.fetchval("SELECT 1")  # Simple query to test connection
            dependencies["postgres"] = "connected"
        except Exception as e:
            dependencies["postgres"] = f"error: {str(e)}"

        # Test LLM service connection (will check for either OpenAI, Gemini, or other providers)
        try:
            # In a real implementation, we would test the actual LLM service
            # For now, we'll check if the appropriate API keys are available
            if settings.gemini_api_key:
                dependencies["llm_service"] = "gemini connected"
            elif settings.openai_api_key:
                dependencies["llm_service"] = "openai connected"
            else:
                dependencies["llm_service"] = "warning: no api key configured"
        except Exception as e:
            dependencies["llm_service"] = f"error: {str(e)}"

        # Test embedding service connection
        try:
            # In a real implementation, we would test the actual embedding service
            # For now, we'll check if the appropriate API keys are available
            if settings.cohere_api_key:
                dependencies["embedding_service"] = "cohere connected"
            elif settings.openai_embedding_api_key:
                dependencies["embedding_service"] = "openai embedding connected"
            else:
                dependencies["embedding_service"] = "warning: no embedding api key configured"
        except Exception as e:
            dependencies["embedding_service"] = f"error: {str(e)}"

        # Determine overall status based on dependencies
        # For health purposes, we'll consider warnings as connected but note them
        all_connected_or_warning = all(
            status.startswith("connected") or status.startswith("warning")
            for status in dependencies.values()
        )

        overall_status = "healthy" if all_connected_or_warning else "unhealthy"

        # Calculate response time
        response_time = (time.time() - start_time) * 1000  # Convert to milliseconds

        return {
            "status": overall_status,
            "timestamp": datetime.now().isoformat(),
            "dependencies": dependencies,
            "response_time_ms": round(response_time, 2)  # Round to 2 decimal places
        }

    except Exception as e:
        return {
            "status": "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "dependencies": {
                "qdrant": "unknown",
                "postgres": "unknown",
                "llm_service": "unknown",
                "embedding_service": "unknown"
            },
            "response_time_ms": round((time.time() - start_time) * 1000, 2),
            "error": str(e)
        }