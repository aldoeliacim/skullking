"""Game repository for MongoDB persistence."""

import logging
from typing import Any

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo import ASCENDING, DESCENDING, ReplaceOne
from pymongo.errors import PyMongoError

from app.config import settings
from app.models.game import Game
from app.services.game_serializer import deserialize_game, serialize_game

logger = logging.getLogger(__name__)


class GameRepository:
    """Repository for game persistence using MongoDB.

    Handles game CRUD operations with async Motor driver.
    Provides full game state serialization for recovery after restarts.
    """

    def __init__(self) -> None:
        """Initialize repository."""
        self.client: AsyncIOMotorClient[dict[str, Any]] | None = None
        self.db: AsyncIOMotorDatabase[dict[str, Any]] | None = None

    async def connect(self) -> None:
        """Connect to MongoDB and create indexes."""
        try:
            self.client = AsyncIOMotorClient(
                settings.mongodb_uri,
                serverSelectionTimeoutMS=2000,  # 2 second timeout
            )
            self.db = self.client[settings.mongodb_database]

            # Verify connection
            await self.client.admin.command("ping")
            logger.info("Connected to MongoDB: %s", settings.mongodb_database)

            # Create indexes for efficient queries
            await self._create_indexes()

        except PyMongoError:
            logger.warning("MongoDB not available")
            raise

    async def _create_indexes(self) -> None:
        """Create indexes for efficient queries."""
        if self.db is None:
            return

        try:
            # Index on slug for lookups by game code
            await self.db.games.create_index("slug", unique=True, sparse=True)

            # Index on state for finding active games
            await self.db.games.create_index("state")

            # Index on created_at for sorting/filtering
            await self.db.games.create_index([("created_at", DESCENDING)])

            # Compound index for finding active games by state and update time
            await self.db.games.create_index([("state", ASCENDING), ("updated_at", DESCENDING)])

            logger.info("MongoDB indexes created successfully")
        except PyMongoError:
            logger.exception("Error creating indexes")

    async def disconnect(self) -> None:
        """Disconnect from MongoDB."""
        if self.client:
            self.client.close()
            logger.info("Disconnected from MongoDB")

    async def save(self, game: Game) -> bool:
        """Save or update a game in the database (upsert).

        Uses the full game serializer to persist complete game state.

        Args:
            game: Game instance to save

        Returns:
            True if successful
        """
        if self.db is None:
            return False

        try:
            game_dict = serialize_game(game)

            # Upsert: insert if not exists, update if exists
            result = await self.db.games.replace_one(
                {"_id": game.id},
                game_dict,
                upsert=True,
            )

            success = result.acknowledged
        except PyMongoError:
            logger.exception("Error saving game %s", game.id)
            return False
        else:
            if success:
                logger.debug("Game %s saved to database", game.id)
            return success

    async def create(self, game: Game) -> bool:
        """Save a new game to database.

        Args:
            game: Game instance to save

        Returns:
            True if successful

        """
        return await self.save(game)

    async def find_by_id(self, game_id: str) -> Game | None:
        """Find and restore a game by ID.

        Args:
            game_id: Game identifier

        Returns:
            Restored Game instance or None
        """
        if self.db is None:
            return None

        try:
            result = await self.db.games.find_one({"_id": game_id})
        except PyMongoError:
            logger.exception("Error finding game %s", game_id)
            return None
        else:
            if result:
                return deserialize_game(result)
            return None

    async def find_by_slug(self, slug: str) -> Game | None:
        """Find and restore a game by slug (game code).

        Args:
            slug: Game slug/code

        Returns:
            Restored Game instance or None
        """
        if self.db is None:
            return None

        try:
            result = await self.db.games.find_one({"slug": slug})
        except PyMongoError:
            logger.exception("Error finding game by slug %s", slug)
            return None
        else:
            if result:
                return deserialize_game(result)
            return None

    async def find_active_games(self, limit: int = 100) -> list[Game]:
        """Find all active (non-ended) games.

        Useful for restoring games after server restart.

        Args:
            limit: Maximum number of games to return

        Returns:
            List of active Game instances
        """
        if self.db is None:
            return []

        try:
            cursor = self.db.games.find(
                {"state": {"$ne": "ENDED"}},
            ).sort("updated_at", DESCENDING).limit(limit)

            games = []
            async for doc in cursor:
                try:
                    game = deserialize_game(doc)
                    games.append(game)
                except (KeyError, ValueError) as e:
                    logger.warning("Error deserializing game %s: %s", doc.get("_id"), e)

        except PyMongoError:
            logger.exception("Error finding active games")
            return []
        else:
            logger.info("Found %d active games in database", len(games))
            return games

    async def update(self, game: Game) -> bool:
        """Update existing game.

        Args:
            game: Game instance to update

        Returns:
            True if successful

        """
        return await self.save(game)

    async def delete(self, game_id: str) -> bool:
        """Delete a game from the database.

        Args:
            game_id: Game identifier

        Returns:
            True if successful
        """
        if self.db is None:
            return False

        try:
            result = await self.db.games.delete_one({"_id": game_id})
        except PyMongoError:
            logger.exception("Error deleting game %s", game_id)
            return False
        else:
            return result.deleted_count > 0

    async def save_many(self, games: list[Game]) -> int:
        """Save multiple games in bulk.

        Args:
            games: List of games to save

        Returns:
            Number of games successfully saved
        """
        if self.db is None or not games:
            return 0

        try:
            operations = [
                ReplaceOne(
                    {"_id": game.id},
                    serialize_game(game),
                    upsert=True,
                )
                for game in games
            ]

            result = await self.db.games.bulk_write(operations)
            saved_count = result.upserted_count + result.modified_count
            logger.debug("Bulk saved %d games", saved_count)
        except PyMongoError:
            logger.exception("Error bulk saving games")
            return 0
        else:
            return saved_count
