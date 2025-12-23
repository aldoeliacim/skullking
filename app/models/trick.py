"""Trick model for a single trick within a round."""

from dataclasses import dataclass, field

from app.models.card import CardId, determine_winner, get_card


@dataclass
class PickedCard:
    """Represents a card picked by a player in a trick."""

    player_id: str
    card_id: CardId


@dataclass
class Trick:
    """
    Represents a single trick within a round.

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

    def add_card(self, player_id: str, card_id: CardId) -> None:
        """Add a picked card to this trick."""
        self.picked_cards.append(PickedCard(player_id, card_id))

    def determine_winner(self) -> tuple[CardId | None, str | None]:
        """
        Determine the winner of this trick.

        Returns:
            Tuple of (winner_card_id, winner_player_id)
            If Kraken wins, returns (None, None)
        """
        if not self.picked_cards:
            return None, None

        card_ids = self.get_all_card_ids()
        winner_card_id = determine_winner(card_ids)

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
        """
        Calculate bonus points for the trick winner.

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
        from app.models.card import CardId

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

    def __str__(self) -> str:
        """String representation."""
        if self.winner_player_id:
            return f"Trick {self.number}: Winner {self.winner_player_id}"
        return f"Trick {self.number}: {len(self.picked_cards)} cards played"
