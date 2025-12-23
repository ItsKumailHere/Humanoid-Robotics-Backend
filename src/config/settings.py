from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # OpenAI Configuration (for backward compatibility)
    openai_api_key: Optional[str] = None

    # Gemini Configuration (Primary LLM service)
    gemini_api_key: Optional[str] = None

    # Cohere Configuration (Embedding service)
    cohere_api_key: Optional[str] = None

    # OpenAI Embedding Configuration (for backward compatibility)
    openai_embedding_api_key: Optional[str] = None

    # Qdrant Configuration
    qdrant_url: str
    qdrant_api_key: str

    # Neon Postgres Configuration
    neon_database_url: str

    # Application Settings
    app_env: str = "development"
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30

    # RAG-specific Settings
    rag_relevance_threshold: float = 0.7
    rag_max_context_tokens: int = 2048
    rag_response_timeout_seconds: int = 5

    class Config:
        env_file = ".env"


settings = Settings()