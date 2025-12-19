"""Application configuration using Pydantic settings."""

from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Server Configuration
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    environment: str = Field(default="development", description="Environment")
    frontend_url: str = Field(default="http://localhost:5173", description="Frontend URL")

    # MongoDB
    mongodb_host: str = Field(default="localhost", description="MongoDB host")
    mongodb_port: int = Field(default=27017, description="MongoDB port")
    mongodb_database: str = Field(default="skullking", description="MongoDB database name")
    mongodb_username: Optional[str] = Field(default=None, description="MongoDB username")
    mongodb_password: Optional[str] = Field(default=None, description="MongoDB password")

    # Redis
    broker_redis_host: str = Field(default="localhost", description="Redis host")
    broker_redis_port: int = Field(default=6379, description="Redis port")
    broker_redis_password: Optional[str] = Field(default=None, description="Redis password")
    redis_db: int = Field(default=0, description="Redis database number")

    # JWT Authentication
    jwt_secret: str = Field(default="your-secret-key", description="JWT secret key")
    jwt_algorithm: str = Field(default="HS256", description="JWT algorithm")

    # Game Configuration
    max_players: int = Field(default=7, description="Maximum players per game")
    rounds_count: int = Field(default=10, description="Number of rounds")
    wait_time_seconds: int = Field(default=15, description="Wait time for players")

    # Bot Configuration
    enable_bots: bool = Field(default=True, description="Enable bot players")
    bot_think_time_min: float = Field(default=0.5, description="Min bot think time")
    bot_think_time_max: float = Field(default=2.0, description="Max bot think time")
    default_bot_strategy: str = Field(default="rule_based", description="Default bot strategy")

    @property
    def mongodb_uri(self) -> str:
        """Build MongoDB connection URI."""
        if self.mongodb_username and self.mongodb_password:
            return f"mongodb://{self.mongodb_username}:{self.mongodb_password}@{self.mongodb_host}:{self.mongodb_port}"
        return f"mongodb://{self.mongodb_host}:{self.mongodb_port}"

    @property
    def redis_url(self) -> str:
        """Build Redis connection URL."""
        if self.broker_redis_password:
            return f"redis://:{self.broker_redis_password}@{self.broker_redis_host}:{self.broker_redis_port}/{self.redis_db}"
        return f"redis://{self.broker_redis_host}:{self.broker_redis_port}/{self.redis_db}"


# Global settings instance
settings = Settings()
