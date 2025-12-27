"""Trick model for a single trick within a round."""

from dataclasses import dataclass, field
from enum import Enum

from app.models.card import CardId, CardType, determine_winner, get_card


class TigressChoice(Enum):
    """Choice for how Tigress (Scary Mary) is played."""

    PIRATE = "pirate"
    ESCAPE = "escape"


@dataclass
class PickedCard:
    """Represents a card picked by a player in a trick."""

    player_id: str
    card_id: CardId
    tigress_choice: TigressChoice | None = None  # Only set when card is Tigress


@dataclass
class Trick:
    """Represents a single trick within a round.

    A trick consists of each player playing one card in turn order.
    The winner is determined by card hierarchy rules.

    Attributes:
        number: Trick number within the round (1-indexed)
        picking_player_id: ID of player whose turn it is to pick
        picked_cards: Cards played so far, in order
        winner_player_id: ID of player who won this trick
        winner_card_id: Card that won this trick
        starter_player_index: Index of player who started this trick

    """

    number: int
    starter_player_index: int
    picking_player_id: str = ""
    picked_cards: list[PickedCard] = field(default_factory=list)
    winner_player_id: str | None = None
    winner_card_id: CardId | None = None

    def has_player_picked(self, player_id: str) -> bool:
        """Check if a player has already picked a card in this trick."""
        return any(pc.player_id == player_id for pc in self.picked_cards)

    def get_all_card_ids(self) -> list[CardId]:
        """Get all card IDs picked in this trick."""
        return [pc.card_id for pc in self.picked_cards]

    def add_card(
        self, player_id: str, card_id: CardId, tigress_choice: TigressChoice | None = None
    ) -> bool:
        """Add a picked card to this trick.

        Args:
            player_id: ID of the player playing the card
            card_id: The card being played
            tigress_choice: If playing Tigress, whether it's pirate or escape

        Returns:
            True if card was added, False if player already picked.

        """
        # Guard: don't allow duplicate picks from same player
        if self.has_player_picked(player_id):
            return False
        self.picked_cards.append(PickedCard(player_id, card_id, tigress_choice))
        return True

    def determine_winner(self) -> tuple[CardId | None, str | None]:
        """Determine the winner of this trick.

        Returns:
            Tuple of (winner_card_id, winner_player_id)
            If Kraken wins, returns (None, None)

        """
        if not self.picked_cards:
            return None, None

        card_ids = self.get_all_card_ids()

        # Build tigress choices dict for cards that have a choice set
        tigress_choices: dict[CardId, str] = {}
        for picked_card in self.picked_cards:
            if picked_card.tigress_choice is not None:
                tigress_choices[picked_card.card_id] = picked_card.tigress_choice.value

        winner_card_id = determine_winner(card_ids, tigress_choices)

        if winner_card_id is None:
            # Kraken wins - no one gets the trick
            return None, None

        # Find the player who played the winning card
        winner_player_id = None
        for picked_card in self.picked_cards:
            if picked_card.card_id == winner_card_id:
                winner_player_id = picked_card.player_id
                break

        self.winner_card_id = winner_card_id
        self.winner_player_id = winner_player_id

        return winner_card_id, winner_player_id

    def calculate_bonus_points(self) -> int:
        """Calculate bonus points for the trick winner.

        Bonus points are awarded for:
        - Capturing 14s of standard suits: +10 each
        - Capturing 14 of Jolly Roger: +20
        - Pirate capturing Mermaid: +20
        - Skull King capturing Pirate: +30
        - Mermaid capturing Skull King: +40

        Returns:
            Total bonus points for this trick

        """
        if not self.winner_card_id or not self.winner_player_id:
            return 0

        bonus = 0
        winner_card = get_card(self.winner_card_id)

        # Special card IDs for 14s
        for picked_card in self.picked_cards:
            card = get_card(picked_card.card_id)

            # Bonus for capturing 14s
            if picked_card.card_id in [CardId.PARROT14, CardId.CHEST14, CardId.MAP14]:
                bonus += 10
            elif picked_card.card_id == CardId.ROGER14:
                bonus += 20

            # Character bonuses
            if winner_card.is_pirate() and card.is_mermaid():
                bonus += 20
            elif winner_card.is_king() and card.is_pirate():
                bonus += 30
            elif winner_card.is_mermaid() and card.is_king():
                bonus += 40

        return bonus

    def is_complete(self, num_players: int) -> bool:
        """Check if all players have picked a card."""
        return len(self.picked_cards) == num_players

    def _get_lead_suit(self, cards_in_trick: list[CardId]) -> CardType | None:
        """Get the lead suit from cards in trick (first suit card played)."""
        for card_id in cards_in_trick:
            card = get_card(card_id)
            if card.is_suit():
                return card.card_type
        return None

    def _categorize_hand(
        self, hand: list[CardId], lead_suit: CardType
    ) -> tuple[list[CardId], list[CardId], list[CardId]]:
        """Categorize hand into suit cards, special cards, and trump cards."""
        suit_cards = []
        special_cards = []
        trump_cards = []
        for card_id in hand:
            card = get_card(card_id)
            if card.is_special():
                special_cards.append(card_id)
            elif card.is_roger():
                trump_cards.append(card_id)
            elif card.card_type == lead_suit:
                suit_cards.append(card_id)
        return suit_cards, special_cards, trump_cards

    def get_valid_cards(self, hand: list[CardId], cards_in_trick: list[CardId]) -> list[CardId]:
        """Get valid cards that can be played from the hand.

        In Skull King, the rules are:
        - Special cards (Pirates, Mermaids, Escapes, etc.) can always be played
        - If leading, any card can be played
        - If following, must follow the lead suit if possible (unless playing special)
        """
        if not hand:
            return []

        # If leading (no cards in trick), can play anything
        if not cards_in_trick:
            return list(hand)

        # Get the lead suit - if no suit was led, can play anything
        lead_suit = self._get_lead_suit(cards_in_trick)
        if lead_suit is None:
            return list(hand)

        # Categorize hand and apply suit-following rules
        suit_cards, special_cards, trump_cards = self._categorize_hand(hand, lead_suit)

        # Must follow suit if possible, but can always play special cards or trump
        if suit_cards:
            return suit_cards + special_cards + trump_cards
        # Can't follow suit, can play anything
        return list(hand)

    def get_loot_alliances(self) -> list[tuple[str, str]]:
        """Get loot alliances formed in this trick.

        When a player plays a Loot card, they form an alliance with the trick winner.
        If both make their bids at round end, each gets +20 bonus.

        Returns:
            List of (loot_player_id, ally_player_id) tuples.
            Empty if no winner (Kraken) or no loot cards played.

        """
        if not self.winner_player_id:
            return []

        alliances = []
        for picked_card in self.picked_cards:
            card = get_card(picked_card.card_id)
            if card.is_loot() and picked_card.player_id != self.winner_player_id:
                # Loot player allies with trick winner (unless they are the winner)
                alliances.append((picked_card.player_id, self.winner_player_id))

        return alliances

    def __str__(self) -> str:
        """Return string representation of the trick."""
        if self.winner_player_id:
            return f"Trick {self.number}: Winner {self.winner_player_id}"
        return f"Trick {self.number}: {len(self.picked_cards)} cards played"
