"""Redis publisher service for game events.

Provides pub/sub functionality for multi-instance scaling.
When running multiple backend instances, game events are broadcast
through Redis so all instances stay synchronized.
"""

import asyncio
import contextlib
import json
import logging
import time
from collections.abc import Callable, Coroutine
from typing import Any

import redis.asyncio as redis
from redis.exceptions import RedisError

from app.config import settings

logger = logging.getLogger(__name__)

# Type for event handlers
EventHandler = Callable[[str, str, dict[str, Any]], Coroutine[Any, Any, None]]


class PublisherService:
    """Service for publishing and subscribing to game events via Redis pub/sub.

    Used for inter-service communication and event broadcasting across
    multiple backend instances.
    """

    def __init__(self) -> None:
        """Initialize publisher service."""
        self.redis_client: redis.Redis[str] | None = None
        self.pubsub: redis.client.PubSub | None = None
        self._handlers: dict[str, list[EventHandler]] = {}
        self._subscriber_task: asyncio.Task[None] | None = None
        self._running = False
        self._instance_id = f"instance_{int(time.time() * 1000)}"

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
            logger.info("Connected to Redis (instance: %s)", self._instance_id)
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
            return False

        try:
            # Add instance ID to prevent echo
            message["_instance_id"] = self._instance_id
            message_json = json.dumps(message)
            await self.redis_client.publish(channel, message_json)
            logger.debug("Published message to channel %s", channel)
        except (RedisError, TypeError):
            logger.exception("Error publishing message")
            return False
        else:
            return True

    async def publish_game_event(
        self,
        event_type: str,
        game_id: str,
        data: dict[str, Any],
    ) -> bool:
        """Publish game event to Redis.

        Args:
            event_type: Type of event (e.g., "game_started", "card_played")
            game_id: Game identifier
            data: Event data

        Returns:
            True if successful
        """
        message = {
            "event": event_type,
            "game_id": game_id,
            "data": data,
            "timestamp": time.time(),
        }

        return await self.publish(f"game_events:{game_id}", message)

    async def subscribe(self, pattern: str, handler: EventHandler) -> None:
        """Subscribe to a channel pattern.

        Args:
            pattern: Channel pattern (e.g., "game_events:*")
            handler: Async function to call when messages arrive
                     Signature: async def handler(event_type, game_id, data)
        """
        if pattern not in self._handlers:
            self._handlers[pattern] = []
        self._handlers[pattern].append(handler)

        logger.info("Registered handler for pattern: %s", pattern)

    async def start_subscriber(self) -> None:
        """Start the background subscriber task."""
        if not self.redis_client or self._running:
            return

        self._running = True
        self.pubsub = self.redis_client.pubsub()

        # Subscribe to all registered patterns
        for pattern in self._handlers:
            await self.pubsub.psubscribe(pattern)
            logger.info("Subscribed to pattern: %s", pattern)

        self._subscriber_task = asyncio.create_task(self._subscriber_loop())
        logger.info("Redis subscriber started")

    async def _subscriber_loop(self) -> None:
        """Background loop processing incoming messages."""
        if not self.pubsub:
            return

        while self._running:
            try:
                message = await self.pubsub.get_message(
                    ignore_subscribe_messages=True,
                    timeout=1.0,
                )

                if message and message["type"] == "pmessage":
                    await self._handle_message(message)

            except asyncio.CancelledError:
                break
            except (RedisError, ConnectionError):
                logger.warning("Redis connection lost, attempting reconnect...")
                await asyncio.sleep(5)
                try:
                    await self.connect()
                    if self.redis_client:
                        self.pubsub = self.redis_client.pubsub()
                        for pattern in self._handlers:
                            await self.pubsub.psubscribe(pattern)
                except Exception:
                    logger.exception("Failed to reconnect to Redis")
            except Exception:
                logger.exception("Error in subscriber loop")
                await asyncio.sleep(1)

    async def _handle_message(self, message: dict[str, Any]) -> None:
        """Process an incoming pub/sub message.

        Args:
            message: Redis pub/sub message
        """
        try:
            raw_pattern = message.get("pattern", "")
            pattern = raw_pattern.decode() if isinstance(raw_pattern, bytes) else raw_pattern
            data_str = message.get("data", "{}")

            data = json.loads(data_str)

            # Skip messages from our own instance
            if data.get("_instance_id") == self._instance_id:
                return

            event_type = data.get("event", "unknown")
            game_id = data.get("game_id", "")
            event_data = data.get("data", {})

            logger.debug(
                "Received event %s for game %s from instance %s",
                event_type,
                game_id,
                data.get("_instance_id", "unknown"),
            )

            # Call all handlers for this pattern
            handlers = self._handlers.get(pattern, [])
            for handler in handlers:
                try:
                    await handler(event_type, game_id, event_data)
                except Exception:
                    logger.exception("Error in event handler")

        except json.JSONDecodeError:
            logger.warning("Invalid JSON in pub/sub message")
        except Exception:
            logger.exception("Error handling pub/sub message")

    async def stop_subscriber(self) -> None:
        """Stop the background subscriber task."""
        self._running = False

        if self._subscriber_task:
            self._subscriber_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._subscriber_task

        if self.pubsub:
            await self.pubsub.close()
            self.pubsub = None

        logger.info("Redis subscriber stopped")

    async def close(self) -> None:
        """Close Redis connection."""
        await self.stop_subscriber()

        if self.redis_client:
            await self.redis_client.close()
            logger.info("Redis connection closed")

    @property
    def is_connected(self) -> bool:
        """Check if Redis is connected."""
        return self.redis_client is not None
