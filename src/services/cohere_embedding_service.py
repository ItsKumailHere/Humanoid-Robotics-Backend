from abc import ABC, abstractmethod
from typing import List
import cohere
import asyncio
from concurrent.futures import ThreadPoolExecutor
from ..config.settings import settings


class EmbeddingServiceInterface(ABC):
    """
    Abstract interface for embedding services.
    This allows swapping between different embedding providers (OpenAI, Cohere, etc.)
    """

    @abstractmethod
    async def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings for a list of texts"""
        pass

    @abstractmethod
    async def create_single_embedding(self, text: str) -> List[float]:
        """Create a single embedding for a text string"""
        pass


class CohereEmbeddingService(EmbeddingServiceInterface):
    """
    Cohere implementation of EmbeddingService
    """

    def __init__(self, api_key: str = None, model_name: str = "embed-english-v3.0"):
        api_key = api_key or settings.cohere_api_key
        if not api_key:
            raise ValueError("COHERE_API_KEY must be provided in settings")
        
        self.client = cohere.Client(api_key)
        self.model_name = model_name

    async def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Create embeddings using Cohere API
        """
        def sync_create_embeddings():
            # Cohere embedding API call
            response = self.client.embed(
                texts=texts,
                model=self.model_name,
                input_type="search_document"  # Using search_document for RAG context
            )
            return response.embeddings

        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            embeddings = await loop.run_in_executor(executor, sync_create_embeddings)

        return embeddings

    async def create_single_embedding(self, text: str) -> List[float]:
        """Create a single embedding for a text string"""
        embeddings = await self.create_embeddings([text])
        return embeddings[0]


class EmbeddingService:
    """
    Main embedding service that can use different providers
    """
    def __init__(self, provider: str = "cohere", api_key: str = None):
        if provider.lower() == "cohere":
            self.provider = CohereEmbeddingService(api_key=api_key)
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")

    async def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings using the configured provider"""
        return await self.provider.create_embeddings(texts)

    async def create_single_embedding(self, text: str) -> List[float]:
        """Create a single embedding using the configured provider"""
        return await self.provider.create_single_embedding(text)

    async def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        # Calculate dot product
        dot_product = sum(a * b for a, b in zip(vec1, vec2))

        # Calculate magnitudes
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(a * a for a in vec2) ** 0.5

        # Calculate cosine similarity
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)