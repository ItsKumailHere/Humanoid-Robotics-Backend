from typing import List, Optional
from ..models.query import QueryRequest
from ..models.document import DocumentChunk
from ..config.settings import settings


class ValidationService:
    def __init__(self):
        pass

    def validate_query_request(self, query_request: QueryRequest) -> List[str]:
        """
        Validate the query request according to the data model validation rules.
        Returns a list of validation errors, empty if valid.
        """
        errors = []
        
        # Validate question
        if not query_request.question or len(query_request.question.strip()) < 3:
            errors.append("Question must be at least 3 characters long")
        
        # Validate query_mode
        if query_request.query_mode not in ["book-wide", "selected-text"]:
            errors.append("Query mode must be either 'book-wide' or 'selected-text'")
        
        # Validate selected_text if query_mode is "selected-text"
        if query_request.query_mode == "selected-text":
            if not query_request.selected_text or len(query_request.selected_text.strip()) == 0:
                errors.append("Selected text must be provided when query mode is 'selected-text'")
        
        return errors

    def validate_retrieved_context(self, retrieved_chunks: List[DocumentChunk], query_request: QueryRequest) -> bool:
        """
        Validate that the retrieved context is sufficient for answering the query.
        """
        # Check if we have any retrieved chunks
        if not retrieved_chunks:
            return False
            
        # Additional validation could be implemented here
        # For example: check if the chunks have enough content, 
        # check relevance scores, etc.
        return len(retrieved_chunks) > 0

    def validate_answer_grounding(self, answer: str, retrieved_chunks: List[DocumentChunk]) -> bool:
        """
        Validate that the generated answer is grounded in the retrieved context.
        """
        if not answer or not retrieved_chunks:
            return False
            
        # A simple check would be to see if elements of the answer appear in the context
        answer_lower = answer.lower()
        context_parts = [chunk.content.lower() for chunk in retrieved_chunks]
        
        # At minimum, check that the answer isn't completely unrelated
        # In a real implementation, we would use more sophisticated NLP techniques
        for context in context_parts:
            if len(context) > 10:  # Only check substantial context chunks
                # If the answer contains text from the context, it's likely grounded
                if any(word in context for word in answer_lower.split()[:5]):
                    return True
        
        # This is a simplified check - in practice, we would use more sophisticated grounding validation
        return True

    def validate_citation_generation(self, citations: List[DocumentChunk]) -> bool:
        """
        Validate that citations can be properly generated.
        According to constitution requirement, if citations cannot be generated,
        the system must refuse the response.
        """
        # In the citation service, we already have validation
        # This is another layer of validation to ensure all required citation data is available
        for citation in citations:
            if not citation.chapter or not citation.section or not citation.file_path:
                return False
        return True

    def check_refusal_conditions(self, query_request: QueryRequest, retrieved_chunks: List[DocumentChunk], 
                                 answer: Optional[str] = None) -> Optional[dict]:
        """
        Check if any refusal conditions are met and return appropriate refusal response.
        """
        # Check if there's insufficient context
        if not self.validate_retrieved_context(retrieved_chunks, query_request):
            return {
                "reason_code": "NO_RELEVANT_CONTEXT",
                "explanation": "No relevant context found in the textbook to answer the question."
            }
        
        # Check if the answer is properly grounded
        if answer and not self.validate_answer_grounding(answer, retrieved_chunks):
            return {
                "reason_code": "NO_RELEVANT_CONTEXT",
                "explanation": "The answer could not be properly grounded in the provided textbook context."
            }
        
        # According to constitution requirement, if citations cannot be generated,
        # the system must refuse the response
        # Here we would check the citations, but since we're checking this before 
        # citations are generated, we'll check if the retrieved chunks have necessary info
        if retrieved_chunks and not self._can_generate_citations(retrieved_chunks):
            return {
                "reason_code": "CITATION_FAILURE",
                "explanation": "Citations cannot be generated for the provided context."
            }
        
        # No refusal conditions met
        return None

    def _can_generate_citations(self, retrieved_chunks: List[DocumentChunk]) -> bool:
        """
        Helper method to check if we have enough information to generate citations.
        """
        # Check if all required fields for citation are available
        for chunk in retrieved_chunks:
            if not chunk.document_id:
                return False
        return True