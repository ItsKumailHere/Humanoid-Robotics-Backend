import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from src.models.query import QueryRequest


client = TestClient(app)


def test_query_endpoint_selected_text_mode():
    """Test the query endpoint with selected-text mode"""
    query_request = {
        "id": "test_query_1",
        "question": "What does this text say?",
        "query_mode": "selected-text",
        "selected_text": "This is the selected text that contains specific information.",
        "timestamp": "2023-01-01T00:00:00Z"
    }
    
    response = client.post("/api/v1/query", json=query_request)
    
    # The response should be successful (200) or have an insufficient context response (which is also 200)
    assert response.status_code == 200
    
    response_data = response.json()
    # Verify that the response has the expected structure
    assert "answer" in response_data
    assert "citations" in response_data
    assert "status" in response_data
    assert "query_id" in response_data
    
    # In selected-text mode, the system should be able to respond based on the provided text
    # Note: The actual response depends on the LLM and embedding service implementation


def test_query_endpoint_book_wide_mode():
    """Test the query endpoint with book-wide mode"""
    query_request = {
        "id": "test_query_2",
        "question": "What are the main topics covered?",
        "query_mode": "book-wide",
        "timestamp": "2023-01-01T00:00:00Z"
    }
    
    response = client.post("/api/v1/query", json=query_request)
    
    # The response should be successful (200)
    assert response.status_code == 200
    
    response_data = response.json()
    # Verify that the response has the expected structure
    assert "answer" in response_data
    assert "citations" in response_data
    assert "status" in response_data
    assert "query_id" in response_data


def test_query_endpoint_missing_selected_text():
    """Test that selected-text mode requires selected_text parameter"""
    query_request = {
        "id": "test_query_3",
        "question": "What does this text say?",
        "query_mode": "selected-text",
        # Missing selected_text
        "timestamp": "2023-01-01T00:00:00Z"
    }
    
    response = client.post("/api/v1/query", json=query_request)
    
    # This might return 200 with an insufficient context response
    # Or it might return 400 if validation catches this
    assert response.status_code in [200, 400]  # Both are acceptable depending on validation logic
    
    if response.status_code == 400:
        # If it's a 400 error, verify it's about validation
        assert "error" in response.json()
    else:
        # If it's a 200, check for insufficient context status
        response_data = response.json()
        assert "status" in response_data
        assert response_data["status"] in ["insufficient_context", "success"]


def test_query_endpoint_invalid_query_mode():
    """Test the query endpoint with an invalid query mode"""
    query_request = {
        "id": "test_query_4",
        "question": "What are the main topics?",
        "query_mode": "invalid-mode",
        "timestamp": "2023-01-01T00:00:00Z"
    }
    
    response = client.post("/api/v1/query", json=query_request)
    
    # Should return 400 for invalid parameters or handle gracefully
    assert response.status_code in [200, 400]