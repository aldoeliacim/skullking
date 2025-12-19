"""Deck model for shuffling and dealing cards."""

import random
from typing import List

from app.models.card import CardId, get_all_cards


class Deck:
    """
    Represents a deck of Skull King cards.

    The deck contains 63 cards total:
    - 1 Skull King
    - 1 White Whale, 1 Kraken
    - 2 Mermaids
    - 5 Pirates
    - 14 Jolly Rogers (trump suit)
    - 14 Parrots, 14 Maps, 14 Chests (standard suits)
    - 5 Escapes
    """

    def __init__(self) -> None:
        """Initialize an empty deck."""
        self.cards: List[CardId] = []

    def fill(self) -> None:
        """Fill the deck with all 63 cards."""
        all_cards = get_all_cards()
        self.cards = list(all_cards.keys())

    def shuffle(self) -> None:
        """Fill and shuffle the deck."""
        self.fill()
        random.shuffle(self.cards)

    def deal(self, num_players: int, cards_per_player: int) -> List[List[CardId]]:
        """
        Deal cards to players.

        Args:
            num_players: Number of players to deal to
            cards_per_player: Number of cards per player

        Returns:
            List of hands, where each hand is a list of CardIds
        """
        if not self.cards:
            self.shuffle()

        hands: List[List[CardId]] = []
        index = 0

        for _ in range(num_players):
            hand = self.cards[index : index + cards_per_player]
            hands.append(hand)
            index += cards_per_player

        return hands

    def reset(self) -> None:
        """Reset the deck."""
        self.cards = []
