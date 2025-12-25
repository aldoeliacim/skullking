"""Response models and DTOs."""

from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from app.models.enums import Command

__all__ = [
    "BidInfo",
    "CardListResponse",
    "Command",
    "CreateGameRequest",
    "CreateGameResponse",
    "ErrorResponse",
    "GameInfo",
    "PlayerInfo",
    "ScoreUpdate",
    "ServerMessage",
    "TableCard",
    "TrickWinnerInfo",
]


class TableCard(BaseModel):
    """Card on the table with player info."""

    player_id: str
    card_id: int  # CardId as int for JSON serialization


class PlayerInfo(BaseModel):
    """Player information for responses."""

    id: str
    username: str
    avatar_id: int
    score: int
    index: int
    is_bot: bool
    is_connected: bool


class GameInfo(BaseModel):
    """Game information response."""

    id: str
    slug: str
    state: str
    players: list[PlayerInfo]
    current_round: int


class BidInfo(BaseModel):
    """Bid information."""

    player_id: str
    bid: int


class TrickWinnerInfo(BaseModel):
    """Trick winner information."""

    player_id: str
    card_id: int
    bonus_points: int


class ScoreUpdate(BaseModel):
    """Score update for a player."""

    player_id: str
    score_delta: int
    total_score: int
    tricks_won: int
    bid: int


@dataclass
class ServerMessage:
    """Message sent from server to clients via WebSocket.

    Attributes:
        command: Command type
        game_id: Game identifier
        content: Message payload (varies by command)
        receiver_id: Specific player to receive (empty = broadcast)
        excluded_id: Player to exclude from broadcast

    """

    command: Command
    game_id: str
    content: Any
    receiver_id: str = ""
    excluded_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "command": self.command.value,
            "content": self.content,
        }


class CardListResponse(BaseModel):
    """Response for card list endpoint."""

    cards: list[dict[str, Any]]


class CreateGameRequest(BaseModel):
    """Request to create a new game."""

    lobby_id: str


class CreateGameResponse(BaseModel):
    """Response for game creation."""

    game_id: str
    slug: str
    message: str = "Game created successfully"


class ErrorResponse(BaseModel):
    """Error response."""

    error: str
    detail: str | None = None
