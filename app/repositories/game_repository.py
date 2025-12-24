"""Game repository for MongoDB persistence."""

import logging

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo.errors import PyMongoError

from app.config import settings
from app.models.game import Game

logger = logging.getLogger(__name__)


class GameRepository:
    """Repository for game persistence using MongoDB.

    Handles game CRUD operations with async Motor driver.
    """

    def __init__(self) -> None:
        """Initialize repository."""
        self.client: AsyncIOMotorClient | None = None
        self.db: AsyncIOMotorDatabase | None = None

    async def connect(self) -> None:
        """Connect to MongoDB."""
        try:
            self.client = AsyncIOMotorClient(
                settings.mongodb_uri,
                serverSelectionTimeoutMS=2000,  # 2 second timeout
            )
            self.db = self.client[settings.mongodb_database]

            # Verify connection
            await self.client.admin.command("ping")
            logger.info("Connected to MongoDB: %s", settings.mongodb_database)

        except PyMongoError:
            logger.warning("MongoDB not available")
            raise

    async def disconnect(self) -> None:
        """Disconnect from MongoDB."""
        if self.client:
            self.client.close()
            logger.info("Disconnected from MongoDB")

    async def create(self, game: Game) -> bool:
        """Save a new game to database.

        Args:
            game: Game instance to save

        Returns:
            True if successful

        """
        if not self.db:
            logger.error("Database not connected")
            return False

        try:
            game_dict = {
                "_id": game.id,
                "slug": game.slug,
                "state": game.state.value,
                "current_round": game.current_round_number,
                "players": [
                    {
                        "id": p.id,
                        "username": p.username,
                        "score": p.score,
                        "index": p.index,
                        "is_bot": p.is_bot,
                    }
                    for p in game.players
                ],
                "created_at": game.created_at,
            }

            await self.db.games.insert_one(game_dict)
            logger.info("Game %s saved to database", game.id)
        except PyMongoError:
            logger.exception("Error saving game %s", game.id)
            return False
        else:
            return True

    async def find_by_id(self, game_id: str) -> dict | None:
        """Find game by ID.

        Args:
            game_id: Game identifier

        Returns:
            Game document or None

        """
        if not self.db:
            return None

        try:
            return await self.db.games.find_one({"_id": game_id})
        except PyMongoError:
            logger.exception("Error finding game %s", game_id)
            return None

    async def update(self, game: Game) -> bool:
        """Update existing game.

        Args:
            game: Game instance to update

        Returns:
            True if successful

        """
        if not self.db:
            return False

        try:
            update_dict = {
                "state": game.state.value,
                "current_round": game.current_round_number,
                "players": [
                    {
                        "id": p.id,
                        "username": p.username,
                        "score": p.score,
                        "index": p.index,
                        "is_bot": p.is_bot,
                    }
                    for p in game.players
                ],
            }

            result = await self.db.games.update_one({"_id": game.id}, {"$set": update_dict})
        except PyMongoError:
            logger.exception("Error updating game %s", game.id)
            return False
        else:
            return result.modified_count > 0
