"""WebSocket connection manager and hub."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING

from fastapi import WebSocket, WebSocketDisconnect

from app.api.game_handler import GameHandler

if TYPE_CHECKING:
    from app.api.responses import ServerMessage
    from app.models.game import Game
    from app.repositories.game_repository import GameRepository
    from app.services.publisher_service import PublisherService

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections for multiplayer games.

    Handles:
    - Player connections per game
    - Spectator connections per game
    - Message broadcasting
    - Connection lifecycle
    """

    def __init__(self) -> None:
        """Initialize the connection manager."""
        # game_id -> player_id -> WebSocket
        self.active_connections: dict[str, dict[str, WebSocket]] = {}
        # game_id -> spectator_id -> WebSocket
        self.spectator_connections: dict[str, dict[str, WebSocket]] = {}
        # Message queue for broadcasting
        self.message_queue: asyncio.Queue[ServerMessage] = asyncio.Queue()
        self.games: dict[str, Game] = {}
        self.game_handler: GameHandler
        # External services (set via set_services)
        self._game_repository: GameRepository | None = None
        self._publisher_service: PublisherService | None = None

    def set_services(
        self,
        game_repository: GameRepository | None,
        publisher_service: PublisherService | None,
    ) -> None:
        """Set external services for persistence and pub/sub.

        Args:
            game_repository: MongoDB repository for game persistence
            publisher_service: Redis pub/sub service
        """
        self._game_repository = game_repository
        self._publisher_service = publisher_service

    def set_game_handler(self, game_handler: GameHandler) -> None:
        """Set the game handler after initialization to avoid circular imports.

        Args:
            game_handler: The game handler instance

        """
        self.game_handler = game_handler

    async def connect(self, websocket: WebSocket, game_id: str, player_id: str) -> None:
        """Accept a new WebSocket connection for a player.

        If the game is not in memory, attempts to restore it from MongoDB.

        Args:
            websocket: WebSocket connection
            game_id: Game identifier
            player_id: Player identifier

        """
        await websocket.accept()

        if game_id not in self.active_connections:
            self.active_connections[game_id] = {}

        self.active_connections[game_id][player_id] = websocket
        logger.info("Player %s connected to game %s", player_id, game_id)

        # Try to restore game from MongoDB if not in memory
        if game_id not in self.games:
            await self._try_restore_game(game_id)

        # Send current game state if game exists
        if game_id in self.games:
            await self.game_handler.send_game_state(self.games[game_id], player_id)

    async def _try_restore_game(self, game_id: str) -> bool:
        """Try to restore a game from MongoDB.

        Args:
            game_id: Game identifier

        Returns:
            True if game was restored
        """
        if not self._game_repository:
            return False

        try:
            game = await self._game_repository.find_by_id(game_id)
            if game:
                self.games[game_id] = game
                logger.info("Restored game %s from MongoDB on reconnection", game_id)
                return True
        except Exception:
            logger.exception("Error restoring game %s from MongoDB", game_id)

        return False

    async def connect_spectator(
        self, websocket: WebSocket, game_id: str, spectator_id: str
    ) -> None:
        """Accept a new WebSocket connection for a spectator.

        Args:
            websocket: WebSocket connection
            game_id: Game identifier
            spectator_id: Spectator identifier

        """
        await websocket.accept()

        if game_id not in self.spectator_connections:
            self.spectator_connections[game_id] = {}

        self.spectator_connections[game_id][spectator_id] = websocket
        logger.info("Spectator %s connected to game %s", spectator_id, game_id)

    def disconnect(self, game_id: str, player_id: str) -> None:
        """Remove a player WebSocket connection.

        Args:
            game_id: Game identifier
            player_id: Player identifier

        """
        if game_id in self.active_connections and player_id in self.active_connections[game_id]:
            del self.active_connections[game_id][player_id]
            logger.info("Player %s disconnected from game %s", player_id, game_id)

            # Clean up empty game (only if no players AND no spectators)
            if not self.active_connections[game_id]:
                del self.active_connections[game_id]
                # Only delete game if no spectators either
                no_spectators = (
                    game_id not in self.spectator_connections
                    or not self.spectator_connections[game_id]
                )
                if no_spectators and game_id in self.games:
                    del self.games[game_id]

    def disconnect_spectator(self, game_id: str, spectator_id: str) -> None:
        """Remove a spectator WebSocket connection.

        Args:
            game_id: Game identifier
            spectator_id: Spectator identifier

        """
        if (
            game_id in self.spectator_connections
            and spectator_id in self.spectator_connections[game_id]
        ):
            del self.spectator_connections[game_id][spectator_id]
            logger.info("Spectator %s disconnected from game %s", spectator_id, game_id)

            # Clean up empty spectator dict
            if not self.spectator_connections[game_id]:
                del self.spectator_connections[game_id]

    def get_spectator_count(self, game_id: str) -> int:
        """Get the number of spectators for a game."""
        if game_id not in self.spectator_connections:
            return 0
        return len(self.spectator_connections[game_id])

    async def send_personal_message(
        self, message: ServerMessage, game_id: str, player_id: str
    ) -> None:
        """Send message to specific player.

        Args:
            message: Message to send
            game_id: Game identifier
            player_id: Player identifier

        """
        if game_id in self.active_connections and player_id in self.active_connections[game_id]:
            websocket = self.active_connections[game_id][player_id]
            try:
                await websocket.send_json(message.to_dict())
            except (WebSocketDisconnect, RuntimeError, ConnectionError, OSError):
                logger.warning("Connection lost to %s", player_id)
                self.disconnect(game_id, player_id)

    async def broadcast_to_game(
        self,
        message: ServerMessage,
        game_id: str,
        excluded_player_id: str | None = None,
    ) -> None:
        """Broadcast message to all players and spectators in a game.

        Args:
            message: Message to broadcast
            game_id: Game identifier
            excluded_player_id: Player to exclude from broadcast

        """
        disconnected_players = []
        disconnected_spectators = []

        # Send to players
        if game_id in self.active_connections:
            for player_id, websocket in self.active_connections[game_id].items():
                if excluded_player_id and player_id == excluded_player_id:
                    continue

                try:
                    await websocket.send_json(message.to_dict())
                except (WebSocketDisconnect, RuntimeError, ConnectionError, OSError):
                    logger.warning("Connection lost to player %s", player_id)
                    disconnected_players.append(player_id)

        # Send to spectators (spectators receive all public broadcasts)
        if game_id in self.spectator_connections:
            for spectator_id, websocket in self.spectator_connections[game_id].items():
                try:
                    await websocket.send_json(message.to_dict())
                except (WebSocketDisconnect, RuntimeError, ConnectionError, OSError):
                    logger.warning("Connection lost to spectator %s", spectator_id)
                    disconnected_spectators.append(spectator_id)

        # Clean up disconnected
        for player_id in disconnected_players:
            self.disconnect(game_id, player_id)
        for spectator_id in disconnected_spectators:
            self.disconnect_spectator(game_id, spectator_id)

    async def dispatch_message(self, message: ServerMessage) -> None:
        """Queue a message for dispatch.

        Args:
            message: Message to dispatch

        """
        await self.message_queue.put(message)

    async def run(self) -> None:
        """Background task to process message queue.

        This runs continuously, processing messages from the queue
        and dispatching them to the appropriate recipients.
        """
        logger.info("WebSocket manager started")

        while True:
            try:
                message = await self.message_queue.get()

                # Log message
                logger.info("Dispatching %s to game %s", message.command.value, message.game_id)

                # Send to specific recipient or broadcast
                if message.receiver_id:
                    await self.send_personal_message(message, message.game_id, message.receiver_id)
                else:
                    await self.broadcast_to_game(message, message.game_id, message.excluded_id)

            except asyncio.CancelledError:
                logger.info("WebSocket manager shutting down")
                break
            except (WebSocketDisconnect, RuntimeError, ConnectionError, OSError) as e:
                logger.warning("WebSocket manager connection error: %s", e)
                await asyncio.sleep(0.1)

    async def handle_player_message(
        self, websocket: WebSocket, game_id: str, player_id: str
    ) -> None:
        """Handle incoming messages from a player.

        Args:
            websocket: WebSocket connection
            game_id: Game identifier
            player_id: Player identifier

        """
        try:
            while True:
                # Receive message from player
                data = await websocket.receive_text()
                message = json.loads(data)

                command = message.get("command", "")
                content = message.get("content", {})

                logger.info("Received %s from player %s in game %s", command, player_id, game_id)

                # Get game
                if game_id not in self.games:
                    logger.warning("Game %s not found", game_id)
                    continue

                game = self.games[game_id]

                # Handle command via game handler
                await self.game_handler.handle_command(game, player_id, command, content)

        except WebSocketDisconnect:
            logger.info("Player %s disconnected from game %s", player_id, game_id)
            self.disconnect(game_id, player_id)

        except (RuntimeError, ConnectionError, OSError, json.JSONDecodeError) as e:
            logger.warning("Error handling message from %s: %s", player_id, e)
            self.disconnect(game_id, player_id)

    async def handle_spectator_message(
        self, websocket: WebSocket, game_id: str, spectator_id: str
    ) -> None:
        """Handle incoming messages from a spectator.

        Spectators can only observe, not send game commands.

        Args:
            websocket: WebSocket connection
            game_id: Game identifier
            spectator_id: Spectator identifier

        """
        try:
            while True:
                # Receive message (spectators can't send game commands, but need to handle pings)
                data = await websocket.receive_text()
                message = json.loads(data)

                command = message.get("command", "")
                logger.debug("Spectator %s sent %s (ignored)", spectator_id, command)

                # Spectators can only send PING or similar non-game commands
                if command == "PING":
                    await websocket.send_json({"command": "PONG"})

        except WebSocketDisconnect:
            logger.info("Spectator %s disconnected from game %s", spectator_id, game_id)
            self.disconnect_spectator(game_id, spectator_id)

        except (RuntimeError, ConnectionError, OSError, json.JSONDecodeError) as e:
            logger.warning("Error handling spectator message from %s: %s", spectator_id, e)
            self.disconnect_spectator(game_id, spectator_id)

    def get_game(self, game_id: str) -> Game | None:
        """Get game by ID or slug.

        First tries direct lookup by game_id (UUID).
        If not found, searches by slug (4-char hex code).
        """
        # Direct lookup by UUID
        if game_id in self.games:
            return self.games[game_id]

        # Search by slug (case-insensitive)
        game_id_upper = game_id.upper()
        for game in self.games.values():
            if game.slug.upper() == game_id_upper:
                return game

        return None

    def add_game(self, game: Game) -> None:
        """Add game to manager."""
        self.games[game.id] = game


# Global WebSocket manager instance
websocket_manager = ConnectionManager()

# Initialize game handler to resolve circular dependency
websocket_manager.set_game_handler(GameHandler(websocket_manager))
