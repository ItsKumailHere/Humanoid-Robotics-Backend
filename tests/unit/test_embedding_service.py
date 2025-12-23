import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from src.services.embedding_service import EmbeddingService
from src.services.cohere_embedding_service import EmbeddingService as CohereEmbeddingService


class TestEmbeddingService:
    
    @pytest.fixture
    def mock_cohere_service(self):
        with patch('src.services.cohere_embedding_service.CohereEmbeddingService') as mock:
            # Create a mock instance
            mock_instance = AsyncMock()
            mock_instance.create_embeddings = AsyncMock(return_value=[[0.1, 0.2, 0.3]] * 2)
            mock_instance.create_single_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3])
            mock_instance.cosine_similarity = AsyncMock(return_value=0.8)
            mock.return_value = mock_instance
            yield mock_instance

    @pytest.mark.asyncio
    async def test_embedding_service_creation(self, mock_cohere_service):
        """Test that EmbeddingService is created with Cohere as the default provider"""
        service = EmbeddingService()
        assert service._embedding_service is not None

    @pytest.mark.asyncio
    async def test_create_embeddings(self, mock_cohere_service):
        """Test creating embeddings for multiple texts"""
        service = EmbeddingService()
        texts = ["text1", "text2"]
        
        result = await service.create_embeddings(texts)
        
        mock_cohere_service.create_embeddings.assert_called_once_with(texts)
        assert result == [[0.1, 0.2, 0.3]] * 2

    @pytest.mark.asyncio
    async def test_create_single_embedding(self, mock_cohere_service):
        """Test creating embedding for a single text"""
        service = EmbeddingService()
        text = "test text"
        
        result = await service.create_single_embedding(text)
        
        mock_cohere_service.create_single_embedding.assert_called_once_with(text)
        assert result == [0.1, 0.2, 0.3]

    @pytest.mark.asyncio
    async def test_cosine_similarity(self, mock_cohere_service):
        """Test cosine similarity calculation"""
        service = EmbeddingService()
        vec1 = [1.0, 0.0]
        vec2 = [0.0, 1.0]
        
        result = await service.cosine_similarity(vec1, vec2)
        
        mock_cohere_service.cosine_similarity.assert_called_once_with(vec1, vec2)
        assert result == 0.8

    @pytest.mark.asyncio
    async def test_create_embeddings_empty_input(self):
        """Test that creating embeddings with empty input returns empty list"""
        service = EmbeddingService()
        result = await service.create_embeddings([])
        assert result == []

    @pytest.mark.asyncio
    async def test_create_single_embedding_empty_input(self):
        """Test that creating embedding with empty input returns empty list"""
        service = EmbeddingService()
        result = await service.create_single_embedding("")
        assert result == []


class TestCohereEmbeddingService:
    
    @pytest.fixture
    def mock_cohere_client(self):
        with patch('src.services.cohere_embedding_service.cohere.Client') as mock:
            mock_client = MagicMock()
            mock_client.embed.return_value = MagicMock(embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
            mock.return_value = mock_client
            yield mock_client

    @pytest.mark.asyncio
    async def test_cohere_embedding_service_creation(self, mock_cohere_client):
        """Test CohereEmbeddingService creation"""
        with patch('src.config.settings.settings') as mock_settings:
            mock_settings.cohere_api_key = "test-key"
            service = CohereEmbeddingService()
            assert service.client == mock_cohere_client

    @pytest.mark.asyncio
    async def test_cohere_create_embeddings(self, mock_cohere_client):
        """Test creating embeddings with Cohere service"""
        with patch('src.config.settings.settings') as mock_settings:
            mock_settings.cohere_api_key = "test-key"
            service = CohereEmbeddingService()
            
            texts = ["test1", "test2"]
            result = await service.create_embeddings(texts)
            
            mock_cohere_client.embed.assert_called_once_with(
                texts=texts,
                model="embed-english-v3.0",
                input_type="search_document"
            )
            assert result == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]

    @pytest.mark.asyncio
    async def test_cohere_create_single_embedding(self, mock_cohere_client):
        """Test creating single embedding with Cohere service"""
        with patch('src.config.settings.settings') as mock_settings:
            mock_settings.cohere_api_key = "test-key"
            service = CohereEmbeddingService()
            
            text = "test text"
            result = await service.create_single_embedding(text)
            
            # The single embedding calls create_embeddings with a single item
            mock_cohere_client.embed.assert_called_once_with(
                texts=[text],
                model="embed-english-v3.0",
                input_type="search_document"
            )
            assert result == [0.1, 0.2, 0.3]