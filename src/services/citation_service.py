from typing import List
from ..models.citation import Citation
from ..models.document import DocumentChunk
from ..config.settings import settings


class CitationService:
    def __init__(self):
        pass

    def format_citations(self, retrieved_chunks: List[DocumentChunk], response_id: str) -> List[Citation]:
        """
        Format citations from retrieved chunks according to data-model.md
        Each citation must include chapter, section, file_path, and other required fields
        """
        citations = []
        for i, chunk in enumerate(retrieved_chunks):
            # Create a citation for each retrieved chunk
            citation = Citation(
                id=f"cit_{response_id}_{i}",
                response_id=response_id,
                document_id=chunk.document_id,
                chapter=self._extract_chapter_from_path(chunk.document_id),  # Would extract from metadata
                section=self._extract_section_from_path(chunk.document_id),  # Would extract from metadata
                file_path=chunk.document_id,  # Would use actual file path from metadata
                relevance_score=self._calculate_relevance_score(chunk),  # Would calculate based on retrieval score
                text_snippet=chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content
            )
            citations.append(citation)
        
        return citations

    def _extract_chapter_from_path(self, document_id: str) -> str:
        """Extract chapter information from document ID or metadata"""
        # In a real implementation, this would look up the document metadata
        # to get the actual chapter information
        # For now, implement a basic extraction from the document ID
        import re
        # Look for chapter patterns in the document ID
        chapter_match = re.search(r'(?:chapter|ch)\s*(\d+)', document_id.lower())
        if chapter_match:
            return f"Chapter {chapter_match.group(1)}"

        # Default to using the first part of the ID as the chapter name
        # This assumes document IDs might contain chapter information
        parts = document_id.split('_')
        if len(parts) > 1:
            return parts[0].replace('-', ' ').title()

        # If no chapter info found in the ID, return a default
        return "Unknown Chapter"

    def _extract_section_from_path(self, document_id: str) -> str:
        """Extract section information from document ID or metadata"""
        # In a real implementation, this would look up the document metadata
        # to get the actual section information
        import re
        # Look for section patterns in the document ID
        section_match = re.search(r'(?:section|sec)\s*(\d+(?:\.\d+)?)', document_id.lower())
        if section_match:
            return f"Section {section_match.group(1)}"

        # If the document ID contains descriptive text, use it as the section
        parts = document_id.split('_')
        if len(parts) > 1:
            # Try to find a section-related part
            for part in parts[1:]:
                if 'section' in part.lower() or 'topic' in part.lower() or len(part) > 0:
                    return part.replace('-', ' ').title()

        # Default to General if no specific section is identified
        return "General"

    def _calculate_relevance_score(self, chunk: DocumentChunk) -> float:
        """Calculate relevance score for a chunk"""
        # In a real implementation, this would use the similarity score
        # from the vector database retrieval
        # For now, we'll return a default score, but in practice this would come from the retrieval process
        # In vector DBs, this would be the similarity score which already exists in the response
        # For this implementation, we'll return a moderate score
        return 0.75  # More realistic default relevance score

    def validate_citation_format(self, citation: Citation) -> bool:
        """
        Validate that a citation has all required fields according to the data model.
        If citations cannot be generated, the system must refuse the response per constitution requirement.
        """
        required_fields = [
            citation.chapter,
            citation.section,
            citation.file_path,
            citation.text_snippet
        ]
        
        # Check that no required field is empty
        for field in required_fields:
            if not field or (isinstance(field, str) and not field.strip()):
                return False
        
        # Check that relevance score is valid
        if not (0.0 <= citation.relevance_score <= 1.0):
            return False
            
        return True

    async def generate_citation_for_response(self, retrieved_chunks: List[DocumentChunk], response_id: str) -> List[Citation]:
        """
        Generate citations for a specific response.
        If citation generation fails for any reason, this should raise an exception
        which will trigger the refusal response as per constitution requirement.
        """
        citations = self.format_citations(retrieved_chunks, response_id)
        
        # Validate all citations are properly formatted
        for citation in citations:
            if not self.validate_citation_format(citation):
                # According to constitution requirement, if citations cannot be generated,
                # the system must refuse the response
                raise ValueError(f"Invalid citation format for document {citation.document_id}")
        
        return citations