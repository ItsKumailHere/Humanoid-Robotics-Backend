import os
import asyncio
from pathlib import Path
from datetime import datetime
from typing import List
import asyncpg

# Add the src directory to the path so we can import our modules
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.config.settings import settings
from src.config.database import db
from src.config.vector_store import vector_store
from src.models.document import Document, DocumentChunk
from src.services.embedding_service import EmbeddingService
import hashlib


async def load_textbook_content():
    """Load textbook content from data/textbook-content/ directory"""
    BASE_DIR = Path(__file__).resolve().parent  # scripts/
    REPO_ROOT = BASE_DIR.parent.parent          # herewegoagain/
    content_dir = REPO_ROOT / "textbook-physical-ai" / "docs"

    if not content_dir.exists():
       raise RuntimeError(f"Docs directory not found at: {content_dir.resolve()}")
    documents = []

    for file_path in content_dir.rglob("*.md"):  # Assuming markdown files
        print(f"Scanning docs directory: {content_dir.resolve()}")
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    
            # Create document object
            doc = Document(
                id=hashlib.sha256(
                  str(file_path).encode()
                 ).hexdigest()[:16],  # Simple ID generation
                title=file_path.stem,
                content=content,
                chapter=file_path.parent.name,
                section=file_path.name,
                file_path=str(file_path),
                source_url="",
                created_at=datetime.now(),
                updated_at=datetime.now(),
                embedding_vector=None  # Will be filled during processing
            )
            documents.append(doc)

    return documents


def chunk_document(document: Document, chunk_size: int = 1000) -> List[DocumentChunk]:
    """Chunk a document into smaller pieces"""
    content = document.content
    chunks = []
    
    # Simple chunking by character count
    for i in range(0, len(content), chunk_size):
        chunk_text = content[i:i+chunk_size]
        chunk = DocumentChunk(
            id=f"{document.id}_chunk_{i//chunk_size}",
            document_id=document.id,
            content=chunk_text,
            chunk_index=i//chunk_size,
            embedding_vector=None,  # Will be filled during processing
            created_at=datetime.now()
        )
        chunks.append(chunk)
    
    return chunks


async def store_document_metadata(document: Document):
    """Store document metadata in Neon Postgres"""
    pool = await db.get_pool()

    async with pool.acquire() as connection:
        await connection.execute(
            """
            INSERT INTO documents (id, title, content, chapter, section, file_path, source_url, created_at, updated_at, embedding_vector)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            ON CONFLICT (id) DO UPDATE SET
                title = EXCLUDED.title,
                content = EXCLUDED.content,
                chapter = EXCLUDED.chapter,
                section = EXCLUDED.section,
                file_path = EXCLUDED.file_path,
                source_url = EXCLUDED.source_url,
                updated_at = EXCLUDED.updated_at,
                embedding_vector = EXCLUDED.embedding_vector
            """,
            document.id,
            document.title,
            document.content,
            document.chapter,
            document.section,
            document.file_path,
            document.source_url,
            document.created_at,
            document.updated_at,
            document.embedding_vector  # Store the embedding vector (might be None initially)
        )


async def store_chunk_metadata(chunk: DocumentChunk):
    """Store chunk metadata in Neon Postgres"""
    pool = await db.get_pool()
    
    async with pool.acquire() as connection:
        await connection.execute(
            """
            INSERT INTO document_chunks (id, document_id, content, chunk_index, created_at)
            VALUES ($1, $2, $3, $4, $5)
            ON CONFLICT (id) DO UPDATE SET
                content = EXCLUDED.content,
                chunk_index = EXCLUDED.chunk_index,
                created_at = EXCLUDED.created_at
            """,
            chunk.id,
            chunk.document_id,
            chunk.content,
            chunk.chunk_index,
            chunk.created_at
        )


async def store_chunk_embeddings(chunk: DocumentChunk, embedding_vector: List[float]):
    """Store chunk embeddings in Qdrant and update metadata in Neon Postgres"""
    from qdrant_client.models import PointStruct

    client = vector_store.get_client()

    # Create collection if it doesn't exist
    vector_store.create_collection_if_not_exists("document_chunks", len(embedding_vector))

    # Convert chunk ID to integer for Qdrant (it only accepts UUID or unsigned int)
    chunk_id_int = int(hashlib.sha256(chunk.id.encode()).hexdigest()[:16], 16)

    # Prepare the payload with metadata
    # payload = {
    #     "document_id": chunk.document_id,
    #     "chunk_id": chunk.id,
    #     "chunk_index": chunk.chunk_index,
    #     "file_path": chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content  # Store content snippet
    # }

    payload = {
        "document_id": chunk.document_id,
        "chunk_id": chunk.id,
        "chunk_index": chunk.chunk_index,
        "content": chunk.content[:200] + "..." if len(chunk.content) > 200 else chunk.content  # Store content snippet
    }

    # Store in Qdrant
    client.upsert(
        collection_name="document_chunks",
        points=[
            PointStruct(
                id=chunk_id_int,
                vector=embedding_vector,
                payload=payload
            )
        ]
    )


# Initialize the embedding service globally
embedding_service = EmbeddingService(provider="cohere")


async def generate_embeddings(text: str) -> List[float]:
    """
    Generate embeddings for text using Cohere API.
    """
    if not text.strip():
        return []

    try:
        # Use the Cohere embedding service
        embedding = await embedding_service.create_single_embedding(text)
        return embedding
    except Exception as e:
        print(f"Error generating embedding: {str(e)}")
        # Return a zero vector in case of error, though this shouldn't happen in production
        return [0.0] * 1024  # Cohere embeddings are typically 1024 dimensions


async def main():
    """Main ingestion function"""
    print("Starting content ingestion process...")

    # Check if the required API key is available
    if not settings.cohere_api_key:
        raise ValueError("COHERE_API_KEY must be provided in settings")

    # Connect to database and vector store
    await db.connect()
    vector_store.connect()

    # Load documents
    documents = await load_textbook_content()
    print(f"Loaded {len(documents)} documents")

    # Process each document
    for doc in documents:
        print(f"Processing document: {doc.title}")

        # Store document metadata
        await store_document_metadata(doc)

        # Chunk the document
        chunks = chunk_document(doc)
        print(f"Created {len(chunks)} chunks")

        # Process each chunk
        for chunk in chunks:
            # Store chunk metadata
            await store_chunk_metadata(chunk)

            # Generate embedding
            embedding = await generate_embeddings(chunk.content)

            # Store embedding
            await store_chunk_embeddings(chunk, embedding)

    print("Content ingestion completed!")


if __name__ == "__main__":
    asyncio.run(main())