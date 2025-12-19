"""Game repository for MongoDB persistence."""

import logging
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from pymongo.errors import PyMongoError

from app.config import settings
from app.models.game import Game

logger = logging.getLogger(__name__)


class GameRepository:
    """
    Repository for game persistence using MongoDB.

    Handles game CRUD operations with async Motor driver.
    """

    def __init__(self) -> None:
        """Initialize repository."""
        self.client: Optional[AsyncIOMotorClient] = None
        self.db: Optional[AsyncIOMotorDatabase] = None

    async def connect(self) -> None:
        """Connect to MongoDB."""
        try:
            self.client = AsyncIOMotorClient(settings.mongodb_uri)
            self.db = self.client[settings.mongodb_database]

            # Verify connection
            await self.client.admin.command("ping")
            logger.info(f"Connected to MongoDB: {settings.mongodb_database}")

        except PyMongoError as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise

    async def disconnect(self) -> None:
        """Disconnect from MongoDB."""
        if self.client:
            self.client.close()
            logger.info("Disconnected from MongoDB")

    async def create(self, game: Game) -> bool:
        """
        Save a new game to database.

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
            logger.info(f"Game {game.id} saved to database")
            return True

        except PyMongoError as e:
            logger.error(f"Error saving game {game.id}: {e}")
            return False

    async def find_by_id(self, game_id: str) -> Optional[dict]:
        """
        Find game by ID.

        Args:
            game_id: Game identifier

        Returns:
            Game document or None
        """
        if not self.db:
            return None

        try:
            return await self.db.games.find_one({"_id": game_id})
        except PyMongoError as e:
            logger.error(f"Error finding game {game_id}: {e}")
            return None

    async def update(self, game: Game) -> bool:
        """
        Update existing game.

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

            result = await self.db.games.update_one(
                {"_id": game.id}, {"$set": update_dict}
            )

            return result.modified_count > 0

        except PyMongoError as e:
            logger.error(f"Error updating game {game.id}: {e}")
            return False
