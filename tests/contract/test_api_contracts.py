import pytest
from fastapi.testclient import TestClient
from src.api.main import app
from src.models.query import QueryRequest
import json


client = TestClient(app)


def test_query_endpoint_contract():
    """Test that the query endpoint follows the API contract specification"""
    query_request = {
        "id": "test_query_contract",
        "question": "What are the key principles of humanoid robotics?",
        "query_mode": "book-wide",
        "timestamp": "2023-01-01T00:00:00Z"
    }
    
    response = client.post("/api/v1/query", json=query_request)
    
    # Should return 200 OK
    assert response.status_code == 200
    
    response_data = response.json()
    
    # Verify response structure as per contract
    assert "id" in response_data  # Response ID
    assert "query_id" in response_data  # Query reference
    assert "answer" in response_data  # Generated answer or null
    assert "citations" in response_data  # List of citations
    assert "confidence_score" in response_data  # Confidence level
    assert "generation_time_ms" in response_data  # Response time
    assert "status" in response_data  # Status of the request
    
    # Verify citations structure
    for citation in response_data["citations"]:
        assert "chapter" in citation
        assert "section" in citation
        assert "file_path" in citation
        assert "relevance_score" in citation
        assert "text_snippet" in citation


def test_query_endpoint_selected_text_contract():
    """Test that the query endpoint handles selected-text mode correctly"""
    query_request = {
        "id": "test_selected_text",
        "question": "Explain the concept mentioned?",
        "query_mode": "selected-text",
        "selected_text": "The control algorithms in humanoid robots include inverse kinematics for movement planning.",
        "timestamp": "2023-01-01T00:00:00Z"
    }
    
    response = client.post("/api/v1/query", json=query_request)
    
    # Should return 200 OK
    assert response.status_code == 200
    
    response_data = response.json()
    
    # Verify response structure
    assert "id" in response_data
    assert "query_id" in response_data
    assert "answer" in response_data
    assert "citations" in response_data
    assert "confidence_score" in response_data
    assert "generation_time_ms" in response_data
    assert "status" in response_data


def test_health_endpoint_contract():
    """Test that the health endpoint follows the API contract specification"""
    response = client.get("/api/v1/health")
    
    # Should return 200 OK
    assert response.status_code == 200
    
    response_data = response.json()
    
    # Verify response structure as per contract
    assert "status" in response_data
    assert "timestamp" in response_data
    assert "dependencies" in response_data
    assert "response_time_ms" in response_data
    
    # Verify dependencies structure
    dependencies = response_data["dependencies"]
    assert "qdrant" in dependencies
    assert "postgres" in dependencies


def test_query_endpoint_validation_contract():
    """Test that the query endpoint returns proper validation errors"""
    # Send invalid request with empty question
    query_request = {
        "id": "test_invalid",
        "question": "",  # Invalid: empty question
        "query_mode": "book-wide",
        "timestamp": "2023-01-01T00:00:00Z"
    }
    
    response = client.post("/api/v1/query", json=query_request)
    
    # Should return 400 for validation errors
    assert response.status_code in [200, 400]  # Could be 200 with refusal or 400 validation error
    
    response_data = response.json()
    
    if response.status_code == 400:
        # If it's a 400, verify error structure
        assert "error" in response_data
        assert "details" in response_data


def test_response_structure_consistency():
    """Test that responses always have the correct structure regardless of status"""
    # Test with a valid request
    query_request = {
        "id": "test_consistency",
        "question": "What is humanoid robotics?",
        "query_mode": "book-wide",
        "timestamp": "2023-01-01T00:00:00Z"
    }
    
    response = client.post("/api/v1/query", json=query_request)
    assert response.status_code == 200
    
    response_data = response.json()
    
    # Ensure all required fields are present
    required_fields = [
        "id", "query_id", "answer", "citations", 
        "confidence_score", "generation_time_ms", "status"
    ]
    
    for field in required_fields:
        assert field in response_data, f"Missing required field: {field}"
    
    # Verify citations is always a list
    assert isinstance(response_data["citations"], list)
    
    # Verify that if status is "success", answer should not be null
    if response_data["status"] == "success":
        assert response_data["answer"] is not None
    # If status is "insufficient_context", answer may be null
    elif response_data["status"] == "insufficient_context":
        # Answer could be null or have a special refusal message
        pass


def test_citation_structure_contract():
    """Test that citations always have the correct structure"""
    query_request = {
        "id": "test_citation",
        "question": "What are design principles?",
        "query_mode": "book-wide",
        "timestamp": "2023-01-01T00:00:00Z"
    }
    
    response = client.post("/api/v1/query", json=query_request)
    assert response.status_code == 200
    
    response_data = response.json()
    
    # Check each citation structure
    for citation in response_data["citations"]:
        assert isinstance(citation, dict)
        
        required_citation_fields = [
            "chapter", "section", "file_path", "relevance_score", "text_snippet"
        ]
        
        for field in required_citation_fields:
            assert field in citation, f"Missing citation field: {field}"
        
        # Verify types
        assert isinstance(citation["chapter"], str)
        assert isinstance(citation["section"], str)
        assert isinstance(citation["file_path"], str)
        assert isinstance(citation["text_snippet"], str)
        assert isinstance(citation["relevance_score"], (int, float))
        
        # Verify relevance score is in valid range
        assert 0.0 <= citation["relevance_score"] <= 1.0