from qdrant_client import QdrantClient
from qdrant_client.http import models
from ..config.settings import settings


class VectorStoreConnection:
    def __init__(self):
        self.client = None
        # Default vector size for Cohere embeddings
        self.default_vector_size = 1024

    def connect(self):
        """Initialize the Qdrant client"""
        self.client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            prefer_grpc=True
        )
        return self.client

    def get_client(self):
        """Get the Qdrant client"""
        if self.client is None:
            self.connect()
        return self.client

    def create_collection_if_not_exists(self, collection_name: str, vector_size: int = None):
        """Create a collection if it doesn't exist"""
        if vector_size is None:
            vector_size = self.default_vector_size

        try:
            self.client.get_collection(collection_name)
        except:
            # Collection doesn't exist, create it
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE
                )
            )

    async def async_search(self, collection_name: str, query_vector: list, limit: int = 5, with_payload: bool = True, query_filter=None):
        """Async wrapper for searching Qdrant using the synchronous client.

        This tries common method names on the client for compatibility across
        different qdrant-client versions and falls back to the HTTP helper.
        """
        import asyncio
        from concurrent.futures import ThreadPoolExecutor

        def _sync_search(collection_name, query_vector, limit, with_payload, query_filter):
            client = self.get_client()

            # Preferred method names to try
            method_names = [
                "search",
                "search_points",
                "search_collection",
                "search_vectors",
            ]

            for name in method_names:
                if hasattr(client, name):
                    method = getattr(client, name)
                    # adapt argument names for different clients
                    try:
                        return method(
                            collection_name=collection_name,
                            query_vector=query_vector,
                            limit=limit,
                            with_payload=with_payload,
                            query_filter=query_filter
                        )
                    except TypeError:
                        # try positional style or different param names
                        return method(collection_name, query_vector, limit)

            # Fallback: try the http client interface
            try:
                return client.http.search_points(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    limit=limit,
                    with_payload=with_payload,
                    query_filter=query_filter
                )
            except Exception as e:
                raise RuntimeError("Qdrant search not available on client: %s" % e)

        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor() as executor:
            results = await loop.run_in_executor(
                executor, _sync_search, collection_name, query_vector, limit, with_payload, query_filter
            )
        return results


# Global vector store instance
vector_store = VectorStoreConnection()