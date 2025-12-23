import asyncio
import time
import pytest
from concurrent.futures import ThreadPoolExecutor
import requests
import threading

# This is a simplified performance test to validate that the system can handle
# multiple concurrent queries within the performance requirements

def test_concurrent_queries_performance():
    """
    Performance validation: Test 10+ concurrent queries to verify performance requirements
    """
    # This test should be run against a running server
    # For now, we'll create a mock test that would work with a real server
    
    base_url = "http://localhost:8000"
    
    # Sample query data
    query_data = {
        "id": "perf_test_query",
        "question": "What are the principles of humanoid robotics?",
        "query_mode": "book-wide",
        "timestamp": "2023-01-01T00:00:00Z"
    }
    
    def make_query(query_id):
        """Function to make a single query"""
        # Update the ID for each query to avoid conflicts
        local_query = query_data.copy()
        local_query["id"] = f"perf_test_query_{query_id}"
        
        start_time = time.time()
        try:
            response = requests.post(f"{base_url}/api/v1/query", json=local_query, timeout=10)
            end_time = time.time()
            
            response_time = (end_time - start_time) * 1000  # Convert to milliseconds
            
            return {
                "status_code": response.status_code,
                "response_time": response_time,
                "success": response.status_code == 200,
                "query_id": query_id
            }
        except requests.exceptions.Timeout:
            return {
                "status_code": -1,
                "response_time": 10000,  # 10 seconds timeout
                "success": False,
                "query_id": query_id
            }
        except Exception as e:
            return {
                "status_code": -1,
                "response_time": -1,
                "success": False,
                "query_id": query_id,
                "error": str(e)
            }
    
    # Run 10 concurrent queries
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(make_query, i) for i in range(10)]
        results = [future.result() for future in futures]
    
    # Collect statistics
    successful_queries = [r for r in results if r["success"]]
    failed_queries = [r for r in results if not r["success"]]
    response_times = [r["response_time"] for r in successful_queries if r["response_time"] > 0]
    
    print(f"Total queries: {len(results)}")
    print(f"Successful queries: {len(successful_queries)}")
    print(f"Failed queries: {len(failed_queries)}")
    
    if response_times:
        avg_response_time = sum(response_times) / len(response_times)
        max_response_time = max(response_times)
        print(f"Average response time: {avg_response_time:.2f} ms")
        print(f"Max response time: {max_response_time:.2f} ms")
        
        # Performance requirements:
        # - Response time should be ≤ 5 seconds (5000ms) for most queries
        slow_queries = [rt for rt in response_times if rt > 5000]
        print(f"Queries exceeding 5-second limit: {len(slow_queries)}")
        
        # This is a validation point - in a real test, we would assert these values
        assert len(slow_queries) <= 2, f"Too many queries exceeded the 5-second limit: {len(slow_queries)}"
    
    # Check that most queries were successful
    assert len(successful_queries) >= 7, f"Too many queries failed: {len(failed_queries)}/{len(results)}"
    
    return results


if __name__ == "__main__":
    # This test requires a running server to function properly
    # For now, we'll skip execution but keep the test function
    print("Performance test function defined. Requires running server to execute.")