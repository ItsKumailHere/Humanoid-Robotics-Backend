from typing import List
from ..models.query import QueryResponse, Citation


def format_response_with_citations(response: QueryResponse) -> dict:
    """Format response with properly structured citations"""
    formatted_response = {
        "id": response.id,
        "query_id": response.query_id,
        "answer": response.answer,
        "citations": format_citations_list(response.citations),
        "retrieved_chunks": response.retrieved_chunks,
        "confidence_score": response.confidence_score,
        "generation_time_ms": response.generation_time_ms,
        "status": response.status,
        "timestamp": response.timestamp.isoformat() if hasattr(response.timestamp, 'isoformat') else str(response.timestamp)
    }
    
    return formatted_response


def format_citations_list(citations: List[Citation]) -> List[dict]:
    """Format a list of Citation objects to dictionaries"""
    formatted_citations = []
    
    for citation in citations:
        formatted_citation = {
            "id": getattr(citation, 'id', None),
            "response_id": getattr(citation, 'response_id', None),
            "document_id": getattr(citation, 'document_id', None),
            "chapter": citation.chapter,
            "section": citation.section,
            "file_path": citation.file_path,
            "relevance_score": citation.relevance_score,
            "text_snippet": citation.text_snippet
        }
        formatted_citations.append(formatted_citation)
    
    return formatted_citations


def format_refusal_response(query_id: str, reason_code: str, explanation: str) -> dict:
    """Format a refusal response according to the specification"""
    from datetime import datetime

    response = {
        "id": f"ref_{query_id}",
        "query_id": query_id,
        "answer": None,
        # Include reason_code and explanation for clients/tests that expect them
        "reason_code": reason_code,  # NO_RELEVANT_CONTEXT | OUT_OF_SCOPE | CITATION_FAILURE
        "explanation": explanation,
        "citations": [],
        "retrieved_chunks": [],
        "confidence_score": None,
        "generation_time_ms": 0,  # Placeholder
        "status": "insufficient_context",
        "timestamp": datetime.now().isoformat()
    }
    
    return response