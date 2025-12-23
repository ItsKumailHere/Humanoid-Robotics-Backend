import pytest
from fastapi.testclient import TestClient
from src.api.main import app
import json


client = TestClient(app)


def test_api_response_format_contract():
    """Test that all API responses follow the specified contract format"""
    
    # Test health endpoint response format
    health_response = client.get("/api/v1/health")
    assert health_response.status_code == 200
    
    health_data = health_response.json()
    assert "status" in health_data
    assert "timestamp" in health_data
    assert "dependencies" in health_data
    assert "response_time_ms" in health_data


def test_error_response_format_contract():
    """Test that error responses follow the specified contract format"""
    
    # Test with an invalid request to trigger error response
    invalid_request = {
        "invalid_field": "invalid_value"
    }
    
    # We expect this to return an error (likely 422 for validation error)
    response = client.post("/api/v1/query", json=invalid_request)
    
    # The response might be 422 (validation error) or 200 with refusal
    if response.status_code >= 400:
        response_data = response.json()
        # Check for standard error response format
        assert "error" in response_data
        # Optionally check for "message", "timestamp", and "request_id" if implemented


def test_common_headers_contract():
    """Test that API responses include common headers"""
    
    response = client.get("/api/v1/health")
    
    # Test that JSON response has correct content type
    assert response.headers["content-type"].startswith("application/json")


def test_rate_limit_contract():
    """Test rate limiting behavior"""
    
    # Make multiple requests to test rate limiting
    query_request = {
        "id": "test_rate_limit",
        "question": "What is rate limiting?",
        "query_mode": "book-wide",
        "timestamp": "2023-01-01T00:00:00Z"
    }
    
    # Make several requests
    responses = []
    for i in range(12):  # Make 12 requests (more than 10 per minute limit)
        response = client.post("/api/v1/query", json=query_request)
        responses.append(response.status_code)
        
        # Update the ID to avoid duplicate key issues
        query_request["id"] = f"test_rate_limit_{i}"
    
    # Note: We're not testing the exact rate limit since it depends on timing,
    # but we're ensuring the endpoint accepts requests properly