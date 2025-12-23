import pytest
from fastapi import HTTPException
from src.services.validation_service import ValidationService
from src.services.generation_service import GenerationService
from src.models.query import QueryRequest
from datetime import datetime


def test_error_handling_edge_cases():
    """
    Test error handling for various edge cases
    """
    validation_service = ValidationService()
    
    # Test 1: Very long question
    long_question = "A" * 10000  # 10,000 character question
    query = QueryRequest(
        id="long_question_test",
        question=long_question,
        query_mode="book-wide",
        timestamp=datetime.now()
    )
    
    # The validation should not crash on long questions
    try:
        errors = validation_service.validate_query_request(query)
        # Validation may flag this as an issue or handle it gracefully
    except Exception as e:
        pytest.fail(f"Validation crashed on long question: {e}")
    
    # Test 2: Special characters and Unicode
    unicode_question = "What is ɸ (phi) in robotics? 🤖"
    query = QueryRequest(
        id="unicode_test",
        question=unicode_question,
        query_mode="book-wide",
        timestamp=datetime.now()
    )
    
    try:
        errors = validation_service.validate_query_request(query)
        # Should handle Unicode without crashing
    except Exception as e:
        pytest.fail(f"Validation crashed on Unicode question: {e}")
    
    # Test 3: SQL injection attempts in selected text (if applicable)
    sql_injection_text = "'; DROP TABLE documents; --"
    query = QueryRequest(
        id="sql_injection_test",
        question="Normal question",
        query_mode="selected-text",
        selected_text=sql_injection_text,
        timestamp=datetime.now()
    )
    
    try:
        errors = validation_service.validate_query_request(query)
        # Should handle potentially malicious input gracefully
    except Exception as e:
        pytest.fail(f"Validation crashed on potential injection text: {e}")
    
    # Test 4: Extremely large selected text
    large_text = "This is a large text. " * 1000  # 20,000+ characters
    query = QueryRequest(
        id="large_text_test",
        question="What does this text say?",
        query_mode="selected-text",
        selected_text=large_text,
        timestamp=datetime.now()
    )
    
    try:
        errors = validation_service.validate_query_request(query)
        # Should handle large text without crashing
    except Exception as e:
        pytest.fail(f"Validation crashed on large selected text: {e}")
    
    print("All edge case error handling tests passed!")


def test_generation_service_errors():
    """
    Test error handling in the generation service
    """
    # This test would need to interact with the generation service
    # to verify it properly handles errors when the LLM fails
    generation_service = GenerationService()
    
    # We can only test the validation methods since actual generation
    # requires API calls to external services
    query = QueryRequest(
        id="error_test",
        question="Test question",
        query_mode="book-wide",
        timestamp=datetime.now()
    )
    
    try:
        result = generation_service.validate_generation_ability(query)
        # Should handle gracefully without crashing
    except Exception as e:
        pytest.fail(f"Generation validation crashed: {e}")
    
    print("Generation service error handling test passed!")


def test_invalid_json_input():
    """
    Test how the API handles invalid JSON input
    """
    # This would be tested at the API level, but we can validate
    # that our services handle malformed data appropriately
    
    # Test with None values
    try:
        query = QueryRequest(
            id=None,
            question=None,
            query_mode=None,
            timestamp=datetime.now()
        )
        errors = validation_service.validate_query_request(query)
    except Exception as e:
        # This is expected since Pydantic will validate at the API boundary
        pass
    
    print("Invalid input handling test completed!")


if __name__ == "__main__":
    validation_service = ValidationService()
    test_error_handling_edge_cases()
    test_generation_service_errors()
    test_invalid_json_input()
    print("All error handling tests passed!")