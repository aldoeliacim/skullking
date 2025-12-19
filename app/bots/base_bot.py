"""Base class for all bot strategies."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Optional

from app.models.card import CardId
from app.models.game import Game


class BotDifficulty(str, Enum):
    """Bot difficulty levels."""

    RANDOM = "random"
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class BaseBot(ABC):
    """
    Abstract base class for bot AI strategies.

    All bot implementations must inherit from this class and implement
    the make_bid() and pick_card() methods.
    """

    def __init__(self, player_id: str, difficulty: BotDifficulty = BotDifficulty.MEDIUM):
        """
        Initialize the bot.

        Args:
            player_id: ID of the player this bot controls
            difficulty: Bot difficulty level
        """
        self.player_id = player_id
        self.difficulty = difficulty

    @abstractmethod
    def make_bid(self, game: Game, round_number: int, hand: List[CardId]) -> int:
        """
        Make a bid for the current round.

        Args:
            game: Current game state
            round_number: Current round number
            hand: Bot's cards for this round

        Returns:
            Bid amount (0 to round_number)
        """
        pass

    @abstractmethod
    def pick_card(
        self,
        game: Game,
        hand: List[CardId],
        cards_in_trick: List[CardId],
        valid_cards: Optional[List[CardId]] = None,
    ) -> CardId:
        """
        Pick a card to play in the current trick.

        Args:
            game: Current game state
            hand: Bot's remaining cards
            cards_in_trick: Cards already played in this trick
            valid_cards: List of valid cards to play (if None, all cards in hand are valid)

        Returns:
            CardId to play
        """
        pass

    def _get_valid_cards(self, hand: List[CardId], valid_cards: Optional[List[CardId]] = None) -> List[CardId]:
        """
        Get list of valid cards to play.

        Args:
            hand: Bot's current hand
            valid_cards: Optionally pre-computed valid cards

        Returns:
            List of valid card IDs
        """
        if valid_cards is not None:
            return [c for c in valid_cards if c in hand]
        return hand.copy()

    def __str__(self) -> str:
        """String representation."""
        return f"{self.__class__.__name__} ({self.difficulty.value})"
