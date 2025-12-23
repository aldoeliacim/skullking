"""WebSocket connection manager and hub."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING

from fastapi import WebSocket, WebSocketDisconnect

from app.api.responses import ServerMessage
from app.models.game import Game

if TYPE_CHECKING:
    from app.api.game_handler import GameHandler

logger = logging.getLogger(__name__)


class ConnectionManager:
    """
    Manages WebSocket connections for multiplayer games.

    Handles:
    - Player connections per game
    - Message broadcasting
    - Connection lifecycle
    """

    def __init__(self) -> None:
        """Initialize the connection manager."""
        # game_id -> player_id -> WebSocket
        self.active_connections: dict[str, dict[str, WebSocket]] = {}
        # Message queue for broadcasting
        self.message_queue: asyncio.Queue[ServerMessage] = asyncio.Queue()
        self.games: dict[str, Game] = {}
        self._game_handler: GameHandler | None = None

    @property
    def game_handler(self) -> GameHandler:
        """Lazy-initialize game handler to avoid circular imports."""
        if self._game_handler is None:
            from app.api.game_handler import GameHandler

            self._game_handler = GameHandler(self)
        return self._game_handler

    async def connect(self, websocket: WebSocket, game_id: str, player_id: str) -> None:
        """
        Accept a new WebSocket connection.

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

        # Send current game state if game exists
        if game_id in self.games:
            await self.game_handler.send_game_state(self.games[game_id], player_id)

    def disconnect(self, game_id: str, player_id: str) -> None:
        """
        Remove a WebSocket connection.

        Args:
            game_id: Game identifier
            player_id: Player identifier
        """
        if game_id in self.active_connections and player_id in self.active_connections[game_id]:
            del self.active_connections[game_id][player_id]
            logger.info("Player %s disconnected from game %s", player_id, game_id)

            # Clean up empty game
            if not self.active_connections[game_id]:
                del self.active_connections[game_id]
                if game_id in self.games:
                    del self.games[game_id]

    async def send_personal_message(
        self, message: ServerMessage, game_id: str, player_id: str
    ) -> None:
        """
        Send message to specific player.

        Args:
            message: Message to send
            game_id: Game identifier
            player_id: Player identifier
        """
        if game_id in self.active_connections and player_id in self.active_connections[game_id]:
            websocket = self.active_connections[game_id][player_id]
            try:
                await websocket.send_json(message.to_dict())
            except Exception:
                logger.exception("Error sending message to %s", player_id)
                self.disconnect(game_id, player_id)

    async def broadcast_to_game(
        self,
        message: ServerMessage,
        game_id: str,
        excluded_player_id: str | None = None,
    ) -> None:
        """
        Broadcast message to all players in a game.

        Args:
            message: Message to broadcast
            game_id: Game identifier
            excluded_player_id: Player to exclude from broadcast
        """
        if game_id not in self.active_connections:
            return

        disconnected = []

        for player_id, websocket in self.active_connections[game_id].items():
            if excluded_player_id and player_id == excluded_player_id:
                continue

            try:
                await websocket.send_json(message.to_dict())
            except Exception:
                logger.exception("Error broadcasting to %s", player_id)
                disconnected.append(player_id)

        # Clean up disconnected players
        for player_id in disconnected:
            self.disconnect(game_id, player_id)

    async def dispatch_message(self, message: ServerMessage) -> None:
        """
        Queue a message for dispatch.

        Args:
            message: Message to dispatch
        """
        await self.message_queue.put(message)

    async def run(self) -> None:
        """
        Background task to process message queue.

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
            except Exception:
                logger.exception("Error in WebSocket manager")
                await asyncio.sleep(0.1)

    async def handle_player_message(
        self, websocket: WebSocket, game_id: str, player_id: str
    ) -> None:
        """
        Handle incoming messages from a player.

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

        except Exception:
            logger.exception("Error handling message from %s", player_id)
            self.disconnect(game_id, player_id)

    def get_game(self, game_id: str) -> Game | None:
        """Get game by ID."""
        return self.games.get(game_id)

    def add_game(self, game: Game) -> None:
        """Add game to manager."""
        self.games[game.id] = game


# Global WebSocket manager instance
websocket_manager = ConnectionManager()
