import asyncpg
from ..config.settings import settings


class DatabaseConnection:
    def __init__(self):
        self.pool = None

    async def connect(self):
        """Initialize the database connection pool"""
        self.pool = await asyncpg.create_pool(
            dsn=settings.neon_database_url,
            min_size=5,
            max_size=20,
            command_timeout=60
        )
        return self.pool

    async def get_pool(self):
        """Get the connection pool"""
        if self.pool is None:
            await self.connect()
        return self.pool

    async def close(self):
        """Close the connection pool"""
        if self.pool:
            await self.pool.close()


# Global database instance
db = DatabaseConnection()