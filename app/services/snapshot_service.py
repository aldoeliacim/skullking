"""Game snapshot service for periodic persistence.

Periodically saves active games to MongoDB for crash recovery.
"""

import asyncio
import contextlib
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.api.websocket import ConnectionManager
    from app.repositories.game_repository import GameRepository

logger = logging.getLogger(__name__)

# Snapshot interval in seconds
SNAPSHOT_INTERVAL = 30


class SnapshotService:
    """Service for periodic game state snapshots to MongoDB.

    Runs as a background task and saves all active games every SNAPSHOT_INTERVAL seconds.
    This ensures game state can be recovered after server restarts.
    """

    def __init__(
        self,
        connection_manager: "ConnectionManager",
        game_repository: "GameRepository | None",
    ) -> None:
        """Initialize snapshot service.

        Args:
            connection_manager: WebSocket connection manager with active games
            game_repository: MongoDB repository for persistence
        """
        self.connection_manager = connection_manager
        self.game_repository = game_repository
        self._running = False
        self._task: asyncio.Task[None] | None = None

    async def start(self) -> None:
        """Start the background snapshot task."""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("Snapshot service started (interval: %ds)", SNAPSHOT_INTERVAL)

    async def stop(self) -> None:
        """Stop the background snapshot task."""
        self._running = False
        if self._task:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
        logger.info("Snapshot service stopped")

    async def _run_loop(self) -> None:
        """Background loop that periodically saves game states."""
        while self._running:
            try:
                await asyncio.sleep(SNAPSHOT_INTERVAL)
                await self.snapshot_all_games()
            except asyncio.CancelledError:
                break
            except Exception:
                logger.exception("Error in snapshot loop")
                await asyncio.sleep(5)  # Brief pause before retry

    async def snapshot_all_games(self) -> int:
        """Save all active games to MongoDB.

        Returns:
            Number of games saved
        """
        if not self.game_repository:
            return 0

        games = list(self.connection_manager.games.values())
        if not games:
            return 0

        try:
            saved = await self.game_repository.save_many(games)
        except Exception:
            logger.exception("Error saving game snapshots")
            return 0
        else:
            if saved > 0:
                logger.info("Snapshot: saved %d active games to MongoDB", saved)
            return saved

    async def snapshot_game(self, game_id: str) -> bool:
        """Save a specific game to MongoDB.

        Args:
            game_id: ID of game to save

        Returns:
            True if saved successfully
        """
        if not self.game_repository:
            return False

        game = self.connection_manager.games.get(game_id)
        if not game:
            return False

        try:
            return await self.game_repository.save(game)
        except Exception:
            logger.exception("Error saving game %s", game_id)
            return False

    async def restore_games(self) -> int:
        """Restore active games from MongoDB on startup.

        Returns:
            Number of games restored
        """
        if not self.game_repository:
            return 0

        try:
            games = await self.game_repository.find_active_games()

            for game in games:
                # Only restore games that aren't already in memory
                if game.id not in self.connection_manager.games:
                    self.connection_manager.games[game.id] = game
                    logger.info("Restored game %s (%s) from database", game.slug, game.state.value)

            if games:
                logger.info("Restored %d games from MongoDB", len(games))
            return len(games)

        except Exception:
            logger.exception("Error restoring games from database")
            return 0
