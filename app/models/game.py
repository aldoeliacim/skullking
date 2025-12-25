"""Game model for managing game state."""

import random
from dataclasses import dataclass, field
from typing import Any

from app.models.deck import Deck
from app.models.enums import MAX_PLAYERS, MAX_ROUNDS, MIN_PLAYERS, GameState
from app.models.player import Player
from app.models.round import Round


@dataclass
class Game:
    """Represents a complete Skull King game.

    A game consists of up to 10 rounds, with 2-8 players.
    In round N, each player receives N cards.

    Attributes:
        id: Unique game identifier
        slug: Human-readable game slug
        state: Current game state
        players: List of players in turn order
        rounds: List of completed and current rounds
        current_round_number: Current round (1-10)
        deck: The deck of cards
        created_at: Timestamp when game was created

    """

    id: str
    slug: str
    state: GameState = GameState.PENDING
    players: list[Player] = field(default_factory=list)
    rounds: list[Round] = field(default_factory=list)
    current_round_number: int = 0
    deck: Deck = field(default_factory=Deck)
    created_at: str | None = None

    def add_player(self, player: Player) -> bool:
        """Add a player to the game."""
        if len(self.players) >= MAX_PLAYERS:
            return False
        if any(p.id == player.id for p in self.players):
            return False

        player.index = len(self.players)
        player.game_id = self.id
        self.players.append(player)
        return True

    def remove_player(self, player_id: str) -> bool:
        """Remove a player from the game."""
        for i, player in enumerate(self.players):
            if player.id == player_id:
                self.players.pop(i)
                # Update indices
                for j in range(i, len(self.players)):
                    self.players[j].index = j
                return True
        return False

    def get_player(self, player_id: str) -> Player | None:
        """Get a player by ID."""
        for player in self.players:
            if player.id == player_id:
                return player
        return None

    def get_player_by_index(self, index: int) -> Player | None:
        """Get a player by their turn index."""
        if 0 <= index < len(self.players):
            return self.players[index]
        return None

    def is_full(self) -> bool:
        """Check if game is at max capacity."""
        return len(self.players) >= MAX_PLAYERS

    def can_start(self) -> bool:
        """Check if game has enough players to start."""
        return len(self.players) >= MIN_PLAYERS

    def get_current_round(self) -> Round | None:
        """Get the current round."""
        if self.rounds:
            return self.rounds[-1]
        return None

    def start_new_round(self) -> Round:
        """Start a new round."""
        self.current_round_number += 1

        # First round: random starter. Subsequent rounds: rotate from round 1 starter
        if self.current_round_number == 1:
            starter_index = random.randint(0, len(self.players) - 1)  # noqa: S311
        else:
            # Get round 1 starter and rotate from there
            round1_starter = self.rounds[0].starter_player_index if self.rounds else 0
            starter_index = (round1_starter + self.current_round_number - 1) % len(self.players)

        round_obj = Round(
            number=self.current_round_number,
            starter_player_index=starter_index,
        )
        self.rounds.append(round_obj)

        # Reset player round state
        for player in self.players:
            player.reset_round()

        return round_obj

    def deal_cards(self) -> None:
        """Deal cards for the current round."""
        current_round = self.get_current_round()
        if not current_round:
            return

        self.deck.shuffle()
        hands = self.deck.deal(len(self.players), self.current_round_number)

        for player, hand in zip(self.players, hands, strict=False):
            player.hand = hand
            current_round.dealt_cards[player.id] = hand.copy()

    def is_game_complete(self) -> bool:
        """Check if all rounds are complete."""
        return self.current_round_number >= MAX_ROUNDS

    def get_leaderboard(self) -> list[dict[str, Any]]:
        """Get sorted leaderboard."""
        sorted_players = sorted(self.players, key=lambda p: p.score, reverse=True)
        return [
            {
                "player_id": p.id,
                "username": p.username,
                "score": p.score,
                "is_bot": p.is_bot,
            }
            for p in sorted_players
        ]

    def get_winner(self) -> Player | None:
        """Get the winning player."""
        if not self.is_game_complete():
            return None
        return max(self.players, key=lambda p: p.score)

    def __str__(self) -> str:
        """Return string representation."""
        return (
            f"Game {self.slug}: {len(self.players)} players, "
            f"Round {self.current_round_number}, State: {self.state.value}"
        )
