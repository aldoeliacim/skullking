"""Redis publisher service for game events."""

import json
import logging
from typing import Any

import redis.asyncio as redis
from redis.exceptions import RedisError

from app.config import settings

logger = logging.getLogger(__name__)


class PublisherService:
    """Service for publishing game events to Redis pub/sub.

    Used for inter-service communication and event broadcasting.
    """

    def __init__(self) -> None:
        """Initialize publisher service."""
        self.redis_client: redis.Redis | None = None

    async def connect(self) -> None:
        """Connect to Redis and verify connection."""
        try:
            self.redis_client = redis.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
            )
            # Actually verify the connection works
            await self.redis_client.ping()
            logger.info("Connected to Redis")
        except (RedisError, TimeoutError, OSError):
            logger.warning("Redis not available, running without pub/sub")
            self.redis_client = None

    async def publish(self, channel: str, message: dict[str, Any]) -> bool:
        """Publish message to Redis channel.

        Args:
            channel: Channel name
            message: Message payload

        Returns:
            True if successful

        """
        if not self.redis_client:
            logger.warning("Redis not connected, skipping publish")
            return False

        try:
            message_json = json.dumps(message)
            await self.redis_client.publish(channel, message_json)
            logger.debug("Published message to channel %s", channel)
        except (RedisError, json.JSONEncodeError):
            logger.exception("Error publishing message")
            return False
        else:
            return True

    async def publish_game_event(self, event_type: str, game_id: str, data: dict[str, Any]) -> bool:
        """Publish game event.

        Args:
            event_type: Type of event (e.g., "game_started", "round_ended")
            game_id: Game identifier
            data: Event data

        Returns:
            True if successful

        """
        message = {
            "event": event_type,
            "game_id": game_id,
            "data": data,
            "timestamp": __import__("time").time(),
        }

        return await self.publish(f"game_events:{game_id}", message)

    async def close(self) -> None:
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()
            logger.info("Redis connection closed")
