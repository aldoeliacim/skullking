"""Game event model for replay system.

Captures all significant game events for later replay.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any


def _utc_now() -> datetime:
    """Return current UTC time (timezone-aware)."""
    return datetime.now(UTC)


class GameEventType(str, Enum):
    """Types of game events that can be recorded."""

    # Game lifecycle
    GAME_STARTED = "GAME_STARTED"
    GAME_ENDED = "GAME_ENDED"

    # Round events
    ROUND_STARTED = "ROUND_STARTED"
    ROUND_ENDED = "ROUND_ENDED"

    # Bidding
    BID_PLACED = "BID_PLACED"
    BIDDING_COMPLETE = "BIDDING_COMPLETE"

    # Card play
    CARD_PLAYED = "CARD_PLAYED"
    TRICK_WON = "TRICK_WON"

    # Special events
    ABILITY_TRIGGERED = "ABILITY_TRIGGERED"
    ABILITY_RESOLVED = "ABILITY_RESOLVED"

    # Scoring
    SCORES_ANNOUNCED = "SCORES_ANNOUNCED"


@dataclass
class GameEvent:
    """Represents a single game event for replay."""

    game_id: str
    event_type: GameEventType
    timestamp: datetime = field(default_factory=_utc_now)
    round_number: int = 0
    trick_number: int | None = None
    player_id: str | None = None
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage/transmission."""
        return {
            "game_id": self.game_id,
            "event_type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "round_number": self.round_number,
            "trick_number": self.trick_number,
            "player_id": self.player_id,
            "data": self.data,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GameEvent":
        """Create from dictionary."""
        return cls(
            game_id=data["game_id"],
            event_type=GameEventType(data["event_type"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            round_number=data.get("round_number", 0),
            trick_number=data.get("trick_number"),
            player_id=data.get("player_id"),
            data=data.get("data", {}),
        )


@dataclass
class GameHistory:
    """Complete game history for replay."""

    game_id: str
    slug: str
    created_at: datetime
    ended_at: datetime
    duration_seconds: int
    players: list[dict[str, Any]]  # Player info with final scores
    winner_id: str
    winner_username: str
    total_rounds: int
    events: list[GameEvent] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "game_id": self.game_id,
            "slug": self.slug,
            "created_at": self.created_at.isoformat(),
            "ended_at": self.ended_at.isoformat(),
            "duration_seconds": self.duration_seconds,
            "players": self.players,
            "winner_id": self.winner_id,
            "winner_username": self.winner_username,
            "total_rounds": self.total_rounds,
            "events": [e.to_dict() for e in self.events],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GameHistory":
        """Create from dictionary."""
        return cls(
            game_id=data["game_id"],
            slug=data["slug"],
            created_at=datetime.fromisoformat(data["created_at"]),
            ended_at=datetime.fromisoformat(data["ended_at"]),
            duration_seconds=data["duration_seconds"],
            players=data["players"],
            winner_id=data["winner_id"],
            winner_username=data["winner_username"],
            total_rounds=data["total_rounds"],
            events=[GameEvent.from_dict(e) for e in data.get("events", [])],
        )

    def get_summary(self) -> dict[str, Any]:
        """Get summary without full events (for listing)."""
        return {
            "game_id": self.game_id,
            "slug": self.slug,
            "created_at": self.created_at.isoformat(),
            "duration_seconds": self.duration_seconds,
            "players": self.players,
            "winner_id": self.winner_id,
            "winner_username": self.winner_username,
            "total_rounds": self.total_rounds,
            "event_count": len(self.events),
        }
