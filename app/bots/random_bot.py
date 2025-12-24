"""Random bot that makes random valid moves."""

import random

from app.bots.base_bot import BaseBot, BotDifficulty
from app.models.card import CardId
from app.models.game import Game


class RandomBot(BaseBot):
    """Bot that makes completely random decisions.

    This serves as a baseline for evaluating other bot strategies
    and provides a simple opponent for testing.
    """

    def __init__(self, player_id: str, _difficulty: BotDifficulty = BotDifficulty.RANDOM) -> None:
        """Initialize random bot."""
        # Random bot ignores difficulty but accepts it for API compatibility
        super().__init__(player_id, BotDifficulty.RANDOM)

    def make_bid(self, _game: Game, round_number: int, _hand: list[CardId]) -> int:
        """Make a random bid between 0 and round_number.

        Args:
            _game: Current game state
            round_number: Current round number
            _hand: Bot's cards for this round

        Returns:
            Random bid

        """
        return random.randint(0, round_number)  # noqa: S311

    def pick_card(
        self,
        _game: Game,
        hand: list[CardId],
        _cards_in_trick: list[CardId],
        valid_cards: list[CardId] | None = None,
    ) -> CardId:
        """Pick a random valid card.

        Args:
            _game: Current game state
            hand: Bot's remaining cards
            _cards_in_trick: Cards already played in this trick
            valid_cards: List of valid cards (or None for all cards)

        Returns:
            Random card from valid choices

        """
        playable = self._get_valid_cards(hand, valid_cards)

        if not playable:
            # Fallback: play any card from hand
            playable = hand

        return random.choice(playable)  # noqa: S311
