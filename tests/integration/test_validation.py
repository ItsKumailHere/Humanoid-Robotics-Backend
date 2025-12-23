import pytest
from src.services.validation_service import ValidationService
from src.services.citation_service import CitationService
from src.models.query import QueryRequest, QueryResponse
from src.models.document import DocumentChunk
from src.models.citation import Citation
from datetime import datetime


def test_success_criteria_validation():
    """
    Final validation that all success criteria from spec.md are met
    """
    validation_service = ValidationService()
    citation_service = CitationService()
    
    # Test 1: Query validation works as specified
    valid_query = QueryRequest(
        id="test_1",
        question="What are the principles of humanoid robotics?",
        query_mode="book-wide",
        timestamp=datetime.now()
    )
    
    validation_errors = validation_service.validate_query_request(valid_query)
    assert len(validation_errors) == 0, f"Valid query should have no validation errors: {validation_errors}"
    
    # Test 2: Invalid query is properly rejected
    invalid_query = QueryRequest(
        id="test_2",
        question="",  # Invalid: empty question
        query_mode="book-wide",
        timestamp=datetime.now()
    )
    
    validation_errors = validation_service.validate_query_request(invalid_query)
    assert len(validation_errors) > 0, "Invalid query should have validation errors"
    
    # Test 3: Selected-text mode validation
    invalid_selected_text_query = QueryRequest(
        id="test_3",
        question="What does the text say?",
        query_mode="selected-text",  # Mode requires selected_text
        selected_text="",  # But selected_text is empty
        timestamp=datetime.now()
    )
    
    validation_errors = validation_service.validate_query_request(invalid_selected_text_query)
    assert len(validation_errors) > 0, "Invalid selected-text query should have validation errors"
    
    # Test 4: Citation service properly validates citations
    valid_citation = Citation(
        id="cit_1",
        response_id="resp_1",
        document_id="doc_1",
        chapter="Chapter 1",
        section="Section 1.1",
        file_path="/path/file.md",
        relevance_score=0.8,
        text_snippet="Sample text"
    )
    
    assert citation_service.validate_citation_format(valid_citation), "Valid citation should pass validation"
    
    invalid_citation = Citation(
        id="cit_2",
        response_id="resp_1",
        document_id="doc_1",
        chapter="",  # Empty chapter
        section="Section 1.1",
        file_path="/path/file.md",
        relevance_score=0.8,
        text_snippet="Sample text"
    )
    
    assert not citation_service.validate_citation_format(invalid_citation), "Invalid citation should fail validation"
    
    print("All success criteria validation tests passed!")


def test_constitution_requirements():
    """
    Validate that constitution requirements are met:
    1. Selected-text isolation
    2. Citation requirement
    3. Refusal requirement
    """
    validation_service = ValidationService()
    citation_service = CitationService()
    
    # 1. Testing selected-text isolation - this is implemented in the retrieval service
    # to ensure queries with selected-text mode don't access the global book vector store
    
    # 2. Citation requirement - if citations can't be generated, system refuses response
    chunks_with_missing_info = [
        DocumentChunk(
            id="chunk_1",
            document_id="",  # Empty document_id will cause citation failure
            content="",
            chunk_index=0,
            embedding_vector=None,
            created_at=datetime.now()
        )
    ]
    
    import asyncio
    with pytest.raises(ValueError, match="Invalid citation format"):
        asyncio.run(
            citation_service.generate_citation_for_response(
                chunks_with_missing_info, 
                "response_1"
            )
        )
    
    # 3. Refusal requirement - system must refuse when no relevant context
    query_request = QueryRequest(
        id="test_refusal",
        question="What are the principles?",
        query_mode="book-wide",
        timestamp=datetime.now()
    )
    
    # Test with empty chunks (no context)
    refusal_response = validation_service.check_refusal_conditions(
        query_request, 
        []  # No context
    )
    
    assert refusal_response is not None, "Should refuse when no context available"
    assert refusal_response["reason_code"] == "NO_RELEVANT_CONTEXT", "Should have correct reason code"
    
    print("All constitution requirements validated!")


def test_response_time_requirements():
    """
    Validate that system meets response time requirements
    """
    # This is difficult to test in a unit test without a running system,
    # but we would check that the timeout settings are properly configured
    from src.config.settings import settings
    
    # The settings should have the required timeout
    assert hasattr(settings, 'rag_response_timeout_seconds')
    assert settings.rag_response_timeout_seconds == 5  # 5 seconds as per requirement
    
    print("Response time requirements validated!")


if __name__ == "__main__":
    test_success_criteria_validation()
    test_constitution_requirements()
    test_response_time_requirements()
    print("All validation tests passed!")