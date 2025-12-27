"""Response models and DTOs."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from pydantic import BaseModel

from app.models.enums import Command

__all__ = [
    "BidInfo",
    "CardListResponse",
    "Command",
    "CreateGameRequest",
    "CreateGameResponse",
    "ErrorCode",
    "ErrorResponse",
    "GameInfo",
    "PlayerInfo",
    "ScoreUpdate",
    "ServerMessage",
    "TableCard",
    "TrickWinnerInfo",
]


class ErrorCode(StrEnum):
    """Error codes for i18n translation on the frontend."""

    # Game state errors
    GAME_ALREADY_STARTED = "error.gameAlreadyStarted"
    NOT_ENOUGH_PLAYERS = "error.notEnoughPlayers"
    NOT_IN_BIDDING_PHASE = "error.notInBiddingPhase"
    NOT_IN_PICKING_PHASE = "error.notInPickingPhase"
    NO_ACTIVE_TRICK = "error.noActiveTrick"
    NO_ACTIVE_ROUND = "error.noActiveRound"

    # Player errors
    PLAYER_NOT_FOUND = "error.playerNotFound"
    NOT_YOUR_TURN = "error.notYourTurn"
    ALREADY_PLACED_BID = "error.alreadyPlacedBid"
    ALREADY_PLAYED = "error.alreadyPlayed"

    # Bid errors
    MISSING_BID_VALUE = "error.missingBidValue"
    BID_MUST_BE_NUMBER = "error.bidMustBeNumber"
    INVALID_BID = "error.invalidBid"

    # Card errors
    INVALID_CARD = "error.invalidCard"
    CARD_NOT_IN_HAND = "error.cardNotInHand"
    TIGRESS_REQUIRES_CHOICE = "error.tigressRequiresChoice"
    MUST_FOLLOW_SUIT = "error.mustFollowSuit"

    # Ability errors
    NO_PENDING_ABILITY = "error.noPendingAbility"
    UNKNOWN_ABILITY = "error.unknownAbility"

    # Bot errors
    CANNOT_ADD_BOT_AFTER_START = "error.cannotAddBotAfterStart"
    GAME_IS_FULL = "error.gameIsFull"
    CANNOT_REMOVE_BOT_AFTER_START = "error.cannotRemoveBotAfterStart"
    MISSING_BOT_ID = "error.missingBotId"
    BOT_NOT_FOUND = "error.botNotFound"

    # Ability-specific errors
    INVALID_PLAYER_CHOSEN = "error.invalidPlayerChosen"
    CANNOT_RESOLVE_ABILITY = "error.cannotResolveAbility"
    INVALID_CARD_IDS = "error.invalidCardIds"
    INVALID_BET_AMOUNT = "error.invalidBetAmount"
    INVALID_MODIFIER = "error.invalidModifier"


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
