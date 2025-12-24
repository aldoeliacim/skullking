"""Player model."""

from dataclasses import dataclass, field

from app.models.card import CardId


@dataclass
class Player:
    """Represents a player in the game.

    Attributes:
        id: Unique player identifier
        username: Player's display name
        game_id: ID of the game they're in
        avatar_id: Avatar identifier (0-255)
        score: Current total score
        index: Player position/turn order (0-6)
        is_connected: Whether player is currently connected
        is_bot: Whether this is an AI player
        hand: Current cards in hand
        bid: Current round bid (None if not yet bid)
        tricks_won: Number of tricks won this round

    """

    id: str
    username: str
    game_id: str
    avatar_id: int = 0
    score: int = 0
    index: int = 0
    is_connected: bool = True
    is_bot: bool = False
    hand: list[CardId] = field(default_factory=list)
    bid: int | None = None
    tricks_won: int = 0

    def reset_round(self) -> None:
        """Reset player state for a new round."""
        self.hand = []
        self.bid = None
        self.tricks_won = 0

    def has_card(self, card_id: CardId) -> bool:
        """Check if player has a card in their hand."""
        return card_id in self.hand

    def remove_card(self, card_id: CardId) -> None:
        """Remove a card from player's hand."""
        if card_id in self.hand:
            self.hand.remove(card_id)

    def add_card(self, card_id: CardId) -> None:
        """Add a card to player's hand."""
        self.hand.append(card_id)

    def made_bid(self) -> bool:
        """Check if player has made their bid."""
        return self.bid is not None

    def update_score(self, points: int) -> None:
        """Update player's score."""
        self.score += points

    def bid_correct(self) -> bool:
        """Check if player's bid matches tricks won."""
        return self.bid == self.tricks_won

    def __str__(self) -> str:
        """Return string representation."""
        bot_str = " (Bot)" if self.is_bot else ""
        return f"{self.username}{bot_str} - Score: {self.score}"
