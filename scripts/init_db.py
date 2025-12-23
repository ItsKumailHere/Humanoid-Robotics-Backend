import asyncpg
import sys
import os
from datetime import datetime

# Add the src directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.config.settings import settings


async def create_database_schema():
    """Create database schema according to data-model.md"""
    
    # Connect to database
    conn = await asyncpg.connect(settings.neon_database_url)
    
    try:
        # Create documents table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id VARCHAR(255) PRIMARY KEY,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                chapter VARCHAR(255),
                section VARCHAR(255),
                file_path VARCHAR(500),
                source_url VARCHAR(500),
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # Create document_chunks table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id VARCHAR(255) PRIMARY KEY,
                document_id VARCHAR(255) REFERENCES documents(id),
                content TEXT NOT NULL,
                chunk_index INTEGER,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # Create queries table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS queries (
                id VARCHAR(255) PRIMARY KEY,
                question TEXT NOT NULL,
                query_mode VARCHAR(50) CHECK (query_mode IN ('book-wide', 'selected-text')),
                selected_text TEXT,
                user_context JSONB,
                timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                user_id VARCHAR(255)
            );
        """)
        
        # Create responses table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS responses (
                id VARCHAR(255) PRIMARY KEY,
                query_id VARCHAR(255) REFERENCES queries(id),
                answer TEXT,
                confidence_score DECIMAL(3,2),
                generation_time_ms DECIMAL(10,2),
                status VARCHAR(50) CHECK (status IN ('success', 'insufficient_context', 'error')),
                timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
            );
        """)
        
        # Create citations table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS citations (
                id VARCHAR(255) PRIMARY KEY,
                response_id VARCHAR(255) REFERENCES responses(id),
                document_id VARCHAR(255) REFERENCES documents(id),
                chapter VARCHAR(255),
                section VARCHAR(255),
                file_path VARCHAR(500),
                relevance_score DECIMAL(3,2),
                text_snippet TEXT
            );
        """)
        
        print("Database schema created successfully!")
        
    except Exception as e:
        print(f"Error creating database schema: {e}")
        
    finally:
        await conn.close()


if __name__ == "__main__":
    import asyncio
    asyncio.run(create_database_schema())