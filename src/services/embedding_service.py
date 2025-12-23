import asyncio
from typing import List
from ..config.settings import settings
from .cohere_embedding_service import EmbeddingService as CohereEmbeddingService


class EmbeddingService:
    def __init__(self, provider: str = "cohere", api_key: str = None):
        # Initialize the Cohere embedding service as the primary embedding provider
        if provider == "cohere":
            self._embedding_service = CohereEmbeddingService(provider="cohere", api_key=api_key or settings.cohere_api_key)
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")

    async def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Create embeddings for a list of texts using Cohere.
        """
        if not texts:
            return []
        return await self._embedding_service.create_embeddings(texts)

    async def create_single_embedding(self, text: str) -> List[float]:
        """Create a single embedding for a text string"""
        if not text:
            return []
        return await self._embedding_service.create_single_embedding(text)

    async def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        return await self._embedding_service.cosine_similarity(vec1, vec2)