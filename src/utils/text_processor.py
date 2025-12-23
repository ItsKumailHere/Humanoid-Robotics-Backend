import re
from typing import List


def clean_text(text: str) -> str:
    """Clean text by removing extra whitespace and normalizing characters"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters if needed
    # text = re.sub(r'[^\w\s]', '', text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks"""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        
        start = end - overlap  # Move with overlap
        
        # Handle the case where the remaining text is less than chunk_size
        if start >= len(text):
            break
    
    return chunks


def extract_sentences(text: str) -> List[str]:
    """Extract sentences from text"""
    # Simple sentence splitting using punctuation
    sentences = re.split(r'[.!?]+', text)
    # Clean and filter empty sentences
    sentences = [clean_text(s) for s in sentences if clean_text(s)]
    return sentences


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text"""
    return ' '.join(text.split())


def sanitize_for_search(text: str) -> str:
    """Sanitize text for search operations"""
    # Remove special characters that might interfere with search
    sanitized = re.sub(r'[^\w\s\-\._]', ' ', text)
    return normalize_whitespace(sanitized)