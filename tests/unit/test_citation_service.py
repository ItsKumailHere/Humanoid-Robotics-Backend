import pytest
from unittest.mock import MagicMock
from src.services.citation_service import CitationService
from src.models.document import DocumentChunk
from src.models.citation import Citation
from datetime import datetime


class TestCitationService:
    def setup_method(self):
        self.citation_service = CitationService()

    def test_format_citations(self):
        """Test that citations are formatted correctly from document chunks"""
        chunks = [
            DocumentChunk(
                id="chunk_1",
                document_id="doc_123",
                content="This is the content of the first chunk.",
                chunk_index=0,
                embedding_vector=None,
                created_at=datetime.now()
            )
        ]
        
        citations = self.citation_service.format_citations(chunks, "response_1")
        
        assert len(citations) == 1
        citation = citations[0]
        assert citation.document_id == "doc_123"
        assert citation.response_id == "response_1"
        assert citation.chapter == "Doc"  # Extracted from document_id
        assert citation.text_snippet == "This is the content of the first chunk."
        assert 0.0 <= citation.relevance_score <= 1.0

    def test_extract_chapter_from_path(self):
        """Test chapter extraction from document ID"""
        # Test with chapter pattern
        result = self.citation_service._extract_chapter_from_path("chapter_5_introduction")
        assert result == "Chapter 5"
        
        # Test with different pattern
        result = self.citation_service._extract_chapter_from_path("ch3_methods")
        assert result == "Chapter 3"
        
        # Test with default case
        result = self.citation_service._extract_chapter_from_path("intro_to_ai")
        assert result == "Intro To Ai"

    def test_extract_section_from_path(self):
        """Test section extraction from document ID"""
        # Test with section pattern
        result = self.citation_service._extract_section_from_path("section_2_3_detailed_analysis")
        assert result == "Section 2.3"
        
        # Test with different pattern
        result = self.citation_service._extract_section_from_path("ch4_sec1_implementation")
        assert result == "Section 1"
        
        # Test with default case
        result = self.citation_service._extract_section_from_path("random_doc_part_2")
        assert result == "Part 2"

    def test_calculate_relevance_score(self):
        """Test relevance score calculation"""
        # This method returns a fixed value, so just verify it returns a valid score
        chunk = DocumentChunk(
            id="chunk_1",
            document_id="doc_123",
            content="Sample content",
            chunk_index=0,
            embedding_vector=None,
            created_at=datetime.now()
        )
        
        score = self.citation_service._calculate_relevance_score(chunk)
        assert 0.0 <= score <= 1.0

    def test_validate_citation_format(self):
        """Test citation validation"""
        valid_citation = Citation(
            id="cit_1",
            response_id="resp_1",
            document_id="doc_1",
            chapter="Chapter 1",
            section="Section 1",
            file_path="/path/to/file",
            relevance_score=0.8,
            text_snippet="Sample snippet"
        )
        
        assert self.citation_service.validate_citation_format(valid_citation) is True
        
        # Test invalid citation with empty chapter
        invalid_citation = Citation(
            id="cit_2",
            response_id="resp_1",
            document_id="doc_1",
            chapter="",  # Empty chapter
            section="Section 1",
            file_path="/path/to/file",
            relevance_score=0.8,
            text_snippet="Sample snippet"
        )
        
        assert self.citation_service.validate_citation_format(invalid_citation) is False
        
        # Test invalid citation with invalid relevance score
        invalid_citation2 = Citation(
            id="cit_3",
            response_id="resp_1",
            document_id="doc_1",
            chapter="Chapter 1",
            section="Section 1",
            file_path="/path/to/file",
            relevance_score=1.5,  # Invalid score
            text_snippet="Sample snippet"
        )
        
        assert self.citation_service.validate_citation_format(invalid_citation2) is False

    @pytest.mark.asyncio
    async def test_generate_citation_for_response(self):
        """Test citation generation for a response"""
        chunks = [
            DocumentChunk(
                id="chunk_1",
                document_id="doc_123",
                content="This is the content of the first chunk.",
                chunk_index=0,
                embedding_vector=None,
                created_at=datetime.now()
            )
        ]
        
        citations = await self.citation_service.generate_citation_for_response(chunks, "response_1")
        
        assert len(citations) == 1
        assert citations[0].response_id == "response_1"
        
    @pytest.mark.asyncio
    async def test_generate_citation_for_response_failure(self):
        """Test citation generation fails when formatting is invalid"""
        chunks = [
            DocumentChunk(
                id="chunk_1",
                document_id="",
                content="",
                chunk_index=0,
                embedding_vector=None,
                created_at=datetime.now()
            )
        ]
        
        # This should raise a ValueError due to invalid citation format
        with pytest.raises(ValueError):
            await self.citation_service.generate_citation_for_response(chunks, "response_1")