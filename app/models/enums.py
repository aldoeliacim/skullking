"""Enums and constants for the game."""

from enum import Enum


class GameState(str, Enum):
    """Game states during the lifecycle."""

    PENDING = "PENDING"
    DEALING = "DEALING"
    BIDDING = "BIDDING"
    PICKING = "PICKING"
    ENDED = "ENDED"


class Command(str, Enum):
    """WebSocket commands."""

    # Commands sent to players
    INIT = "INIT"
    LEFT = "LEFT"
    JOINED = "JOINED"
    START_BIDDING = "START_BIDDING"
    END_BIDDING = "END_BIDDING"
    BADE = "BADE"  # Note: Original has typo "BADE" instead of "BID"
    START_PICKING = "START_PICKING"
    PICKED = "PICKED"
    DEAL = "DEAL"
    STARTED = "STARTED"
    ANNOUNCE_TRICK_WINNER = "ANNOUNCE_TRICK_WINNER"
    ANNOUNCE_SCORES = "ANNOUNCE_SCORES"
    NEXT_TRICK = "NEXT_TRICK"
    END_GAME = "END_GAME"
    REPORT_ERROR = "REPORT_ERROR"
    STATISTICS_FETCHED = "STATISTICS_FETCHED"
    GAME_STATE = "GAME_STATE"  # Full state sync

    # Commands from client
    PICK = "PICK"
    BID = "BID"
    FETCH_STATISTICS = "FETCH_STATISTICS"
    SYNC_STATE = "SYNC_STATE"  # Request state sync


class CardType(str, Enum):
    """Card types in Skull King."""

    KING = "king"
    WHALE = "whale"
    KRAKEN = "kraken"
    MERMAID = "mermaid"
    PARROT = "parrot"
    MAP = "map"
    CHEST = "chest"
    ROGER = "roger"
    PIRATE = "pirate"
    ESCAPE = "escape"


# Game configuration constants
MAX_PLAYERS = 7
MAX_ROUNDS = 10
WAIT_TIME_SECONDS = 15
