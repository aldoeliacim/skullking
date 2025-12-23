"""Round model representing one round of the game."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from app.models.card import CardId
from app.models.trick import Trick


@dataclass
class Round:
    """
    Represents a single round of Skull King.

    In round N, each player is dealt N cards and must bid how many tricks they'll win.
    The round consists of N tricks.

    Attributes:
        number: Round number (1-10)
        dealt_cards: Cards dealt to each player, keyed by player_id
        bids: Each player's bid for this round
        tricks: List of tricks in this round
        starter_player_index: Index of player who starts bidding/picking
        scores: Score changes for each player this round
    """

    number: int
    starter_player_index: int
    dealt_cards: Dict[str, List[CardId]] = field(default_factory=dict)
    bids: Dict[str, int] = field(default_factory=dict)
    tricks: List[Trick] = field(default_factory=list)
    scores: Dict[str, int] = field(default_factory=dict)

    def get_player_hand(self, player_id: str) -> List[CardId]:
        """Get the cards dealt to a player."""
        return self.dealt_cards.get(player_id, [])

    def get_picked_cards(self, player_id: str) -> List[CardId]:
        """Get all cards a player has picked in this round."""
        picked = []
        for trick in self.tricks:
            for picked_card in trick.picked_cards:
                if picked_card.player_id == player_id:
                    picked.append(picked_card.card_id)
        return picked

    def get_remaining_cards(self, player_id: str) -> List[CardId]:
        """Get cards remaining in player's hand."""
        dealt = self.get_player_hand(player_id)
        picked = self.get_picked_cards(player_id)
        return [card_id for card_id in dealt if card_id not in picked]

    def get_tricks_won(self, player_id: str) -> int:
        """Count how many tricks a player has won."""
        return sum(
            1 for trick in self.tricks if trick.winner_player_id == player_id
        )

    def has_player_bid(self, player_id: str) -> bool:
        """Check if a player has made their bid."""
        return player_id in self.bids

    def add_bid(self, player_id: str, bid: int) -> None:
        """Add a player's bid."""
        self.bids[player_id] = bid

    def get_bonus_points(self, player_id: str) -> int:
        """Calculate total bonus points for a player in this round."""
        bonus = 0
        for trick in self.tricks:
            if trick.winner_player_id == player_id:
                bonus += trick.calculate_bonus_points()
        return bonus

    def calculate_scores(self) -> None:
        """
        Calculate scores for all players in this round.

        Scoring rules:
        - Bid correct (non-zero): 20 * bid + bonus points
        - Bid correct (zero): 10 * round_number
        - Bid wrong (non-zero): -10 * difference
        - Bid wrong (zero): -10 * round_number
        """
        won_tricks: Dict[str, int] = {}

        # Count tricks won by each player
        for player_id in self.bids:
            won_tricks[player_id] = self.get_tricks_won(player_id)

        # Calculate scores
        for player_id, bid in self.bids.items():
            tricks_won = won_tricks[player_id]

            if tricks_won == bid:
                # Bid correct
                if bid == 0:
                    self.scores[player_id] = 10 * self.number
                else:
                    bonus = self.get_bonus_points(player_id)
                    self.scores[player_id] = 20 * bid + bonus
            else:
                # Bid wrong
                if bid == 0:
                    self.scores[player_id] = -10 * self.number
                else:
                    diff = abs(tricks_won - bid)
                    self.scores[player_id] = -10 * diff

    def is_complete(self) -> bool:
        """Check if the round is complete (all tricks played)."""
        # A round is complete when we have the right number of tricks,
        # regardless of whether they have winners (Kraken can result in no winner)
        return len(self.tricks) == self.number

    def get_current_trick(self) -> Optional[Trick]:
        """Get the current (incomplete) trick, if any."""
        if not self.tricks:
            return None
        last_trick = self.tricks[-1]
        if last_trick.winner_player_id is None:
            return last_trick
        return None

    def __str__(self) -> str:
        """String representation."""
        return f"Round {self.number}: {len(self.bids)} bids, {len(self.tricks)} tricks"
