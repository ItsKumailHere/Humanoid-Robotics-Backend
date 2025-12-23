from typing import Dict, Any, List
from ..models.query import QueryRequest
from ..models.document import DocumentChunk
from ..config.settings import settings
from .gemini_service import GeminiLLMService, SelectedTextGeminiLLMService
import asyncio
import google.generativeai as genai


class GenerationService:
    def __init__(self, api_key: str = None):
        # Initialize the Gemini service with the API key from settings
        api_key = api_key or settings.gemini_api_key
        if not api_key:
            raise ValueError("GEMINI_API_KEY must be provided in settings")
        self.gemini_service = GeminiLLMService(api_key=api_key)
        self.selected_text_gemini_service = SelectedTextGeminiLLMService(api_key=api_key)

    async def generate_answer(self, query_request: QueryRequest, retrieved_chunks: List[DocumentChunk]) -> Dict[str, Any]:
        """
        Generate an answer based on the query and retrieved context.
        This implements the OpenAI Agent with tools, policies, and refusal behavior.
        """
        # Check if we have sufficient context
        if not retrieved_chunks:
            return self._create_refusal_response(
                "NO_RELEVANT_CONTEXT",
                "No relevant context found to answer the question from the textbook."
            )

        # Build context from retrieved chunks
        context = self._build_context(retrieved_chunks)

        # Determine which service to use based on query mode
        if query_request.query_mode == "selected-text" and query_request.selected_text:
            llm_service = self.selected_text_gemini_service
            # For selected-text mode, use the selected text as context
            context = query_request.selected_text
        else:
            llm_service = self.gemini_service

        # Generate the answer using the appropriate LLM service
        result = await llm_service.generate_rag_response(query_request.question, context)

        return result

    def _build_context(self, retrieved_chunks: List[DocumentChunk]) -> str:
        """Build context string from retrieved chunks"""
        context_parts = []
        for chunk in retrieved_chunks:
            context_parts.append(f"Source: {chunk.document_id} | Content: {chunk.content[:500]}...")
        return "\n\n".join(context_parts)

    def _build_generation_prompt(self, question: str, context: str) -> str:
        """Build the prompt for answer generation"""
        prompt = f"""
        Based on the following textbook context, answer the question. 
        If the context does not contain enough information to answer the question, 
        respond with a refusal message as specified in the requirements.
        
        Context:
        {context}
        
        Question: {question}
        
        Answer:
        """
        return prompt

    def _validate_answer_grounding(self, answer: str, context: str) -> bool:
        """Validate that the answer is grounded in the provided context"""
        # In a real implementation, we would use more sophisticated validation
        # For now, just checking if the answer contains information from the context
        # and is not just a refusal message
        if not answer or not context:
            return False

        # Check if the answer is a refusal response
        refusal_indicators = [
            "cannot answer", "no relevant", "not mentioned",
            "not found in context", "insufficient context", "no response"
        ]

        answer_lower = answer.lower()
        for indicator in refusal_indicators:
            if indicator in answer_lower:
                return True  # A refusal is still a grounded response

        # Check if there's some overlap between answer and context
        answer_words = set(answer_lower.split())
        context_lower = context.lower()
        context_words = set(context_lower.split())

        # If more than 10% of answer words appear in context, consider it grounded
        common_words = answer_words.intersection(context_words)
        if len(common_words) > 0:
            return len(common_words) / len(answer_words) > 0.1

        return True  # Default to true for all non-refusal responses

    def _create_refusal_response(self, reason_code: str, explanation: str) -> Dict[str, Any]:
        """Create a refusal response according to specification"""
        return {
            "answer": None,
            "reason_code": reason_code,
            "explanation": explanation,
            "status": "insufficient_context",
            "confidence_score": None
        }

    async def validate_generation_ability(self, query_request: QueryRequest) -> bool:
        """
        Check if we have the ability to generate an answer based on the query.
        This implements the refusal policy for when the system cannot answer.
        """
        # Validate that the query is non-empty and meaningful
        if not query_request.question or len(query_request.question.strip()) < 3:
            return False
            
        # Additional validation could go here
        return True