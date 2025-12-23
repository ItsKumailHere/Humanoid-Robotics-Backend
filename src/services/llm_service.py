from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import asyncio


class LLMService(ABC):
    """
    Abstract interface for Language Model services.
    This allows swapping between different LLM providers (OpenAI, Gemini, etc.)
    """

    @abstractmethod
    async def generate_text(self, prompt: str, context: Optional[str] = None) -> str:
        """Generate text based on a prompt and optional context"""
        pass

    @abstractmethod
    async def generate_rag_response(self, question: str, context: str) -> Dict[str, Any]:
        """Generate a RAG-based response with the question and retrieved context"""
        pass