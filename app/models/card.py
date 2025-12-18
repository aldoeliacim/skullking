"""Card model and game logic."""

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

from app.models.enums import CardType


class CardId(IntEnum):
    """Card identifiers using IntEnum for type safety."""

    # Special character cards
    SKULL_KING = 1
    WHALE = 2
    KRAKEN = 3
    MERMAID1 = 4
    MERMAID2 = 5

    # Pirates (5)
    PIRATE1 = 6
    PIRATE2 = 7
    PIRATE3 = 8
    PIRATE4 = 9
    PIRATE5 = 10

    # Jolly Roger / Trump suit (14)
    ROGER1 = 11
    ROGER2 = 12
    ROGER3 = 13
    ROGER4 = 14
    ROGER5 = 15
    ROGER6 = 16
    ROGER7 = 17
    ROGER8 = 18
    ROGER9 = 19
    ROGER10 = 20
    ROGER11 = 21
    ROGER12 = 22
    ROGER13 = 23
    ROGER14 = 24

    # Parrot suit (14)
    PARROT1 = 25
    PARROT2 = 26
    PARROT3 = 27
    PARROT4 = 28
    PARROT5 = 29
    PARROT6 = 30
    PARROT7 = 31
    PARROT8 = 32
    PARROT9 = 33
    PARROT10 = 34
    PARROT11 = 35
    PARROT12 = 36
    PARROT13 = 37
    PARROT14 = 38

    # Map suit (14)
    MAP1 = 39
    MAP2 = 40
    MAP3 = 41
    MAP4 = 42
    MAP5 = 43
    MAP6 = 44
    MAP7 = 45
    MAP8 = 46
    MAP9 = 47
    MAP10 = 48
    MAP11 = 49
    MAP12 = 50
    MAP13 = 51
    MAP14 = 52

    # Chest suit (14)
    CHEST1 = 53
    CHEST2 = 54
    CHEST3 = 55
    CHEST4 = 56
    CHEST5 = 57
    CHEST6 = 58
    CHEST7 = 59
    CHEST8 = 60
    CHEST9 = 61
    CHEST10 = 62
    CHEST11 = 63
    CHEST12 = 64
    CHEST13 = 65
    CHEST14 = 66

    # Escape cards (5)
    ESCAPE1 = 67
    ESCAPE2 = 68
    ESCAPE3 = 69
    ESCAPE4 = 70
    ESCAPE5 = 71


@dataclass(frozen=True)
class Card:
    """
    Represents a card in Skull King.

    Attributes:
        id: Unique card identifier
        number: Card number (0 for special cards, 1-14 for suit cards)
        card_type: Type of card (suit, character, escape, etc.)
    """

    id: CardId
    number: int
    card_type: CardType

    # Card type checking methods
    def is_suit(self) -> bool:
        """Check if card is a suit card (numbered 1-14)."""
        return self.is_standard_suit() or self.is_roger()

    def is_standard_suit(self) -> bool:
        """Check if card is a standard suit (Parrot, Map, Chest)."""
        return self.is_parrot() or self.is_map() or self.is_chest()

    def is_king(self) -> bool:
        """Check if card is Skull King."""
        return self.card_type == CardType.KING

    def is_whale(self) -> bool:
        """Check if card is White Whale."""
        return self.card_type == CardType.WHALE

    def is_kraken(self) -> bool:
        """Check if card is Kraken."""
        return self.card_type == CardType.KRAKEN

    def is_mermaid(self) -> bool:
        """Check if card is Mermaid."""
        return self.card_type == CardType.MERMAID

    def is_character(self) -> bool:
        """Check if card is a character (King, Mermaid, or Pirate)."""
        return self.is_king() or self.is_mermaid() or self.is_pirate()

    def is_beast(self) -> bool:
        """Check if card is a beast (Kraken or Whale)."""
        return self.is_kraken() or self.is_whale()

    def is_special(self) -> bool:
        """Check if card is special (character, beast, or escape)."""
        return self.is_character() or self.is_beast() or self.is_escape()

    def is_parrot(self) -> bool:
        """Check if card is Parrot suit."""
        return self.card_type == CardType.PARROT

    def is_map(self) -> bool:
        """Check if card is Map suit."""
        return self.card_type == CardType.MAP

    def is_chest(self) -> bool:
        """Check if card is Chest suit."""
        return self.card_type == CardType.CHEST

    def is_roger(self) -> bool:
        """Check if card is Jolly Roger (trump) suit."""
        return self.card_type == CardType.ROGER

    def is_pirate(self) -> bool:
        """Check if card is Pirate."""
        return self.card_type == CardType.PIRATE

    def is_escape(self) -> bool:
        """Check if card is Escape."""
        return self.card_type == CardType.ESCAPE

    def __str__(self) -> str:
        """String representation of card."""
        if self.number > 0:
            return f"{self.card_type.value.title()}{self.number}"
        return self.card_type.value.title().replace("_", " ")


# All cards in the deck (63 total)
_CARDS: dict[CardId, Card] = {
    # Special characters
    CardId.SKULL_KING: Card(CardId.SKULL_KING, 0, CardType.KING),
    CardId.WHALE: Card(CardId.WHALE, 0, CardType.WHALE),
    CardId.KRAKEN: Card(CardId.KRAKEN, 0, CardType.KRAKEN),
    CardId.MERMAID1: Card(CardId.MERMAID1, 0, CardType.MERMAID),
    CardId.MERMAID2: Card(CardId.MERMAID2, 0, CardType.MERMAID),

    # Pirates
    CardId.PIRATE1: Card(CardId.PIRATE1, 0, CardType.PIRATE),
    CardId.PIRATE2: Card(CardId.PIRATE2, 0, CardType.PIRATE),
    CardId.PIRATE3: Card(CardId.PIRATE3, 0, CardType.PIRATE),
    CardId.PIRATE4: Card(CardId.PIRATE4, 0, CardType.PIRATE),
    CardId.PIRATE5: Card(CardId.PIRATE5, 0, CardType.PIRATE),
}

# Generate suit cards dynamically
for num in range(1, 15):
    # Jolly Roger (trump)
    roger_id = CardId[f"ROGER{num}"]
    _CARDS[roger_id] = Card(roger_id, num, CardType.ROGER)

    # Parrot
    parrot_id = CardId[f"PARROT{num}"]
    _CARDS[parrot_id] = Card(parrot_id, num, CardType.PARROT)

    # Map
    map_id = CardId[f"MAP{num}"]
    _CARDS[map_id] = Card(map_id, num, CardType.MAP)

    # Chest
    chest_id = CardId[f"CHEST{num}"]
    _CARDS[chest_id] = Card(chest_id, num, CardType.CHEST)

# Escape cards
for num in range(1, 6):
    escape_id = CardId[f"ESCAPE{num}"]
    _CARDS[escape_id] = Card(escape_id, 0, CardType.ESCAPE)


def get_card(card_id: CardId) -> Card:
    """Get card by ID."""
    return _CARDS[card_id]


def get_all_cards() -> dict[CardId, Card]:
    """Get all cards in the deck."""
    return _CARDS.copy()


def determine_winner(card_ids: list[CardId]) -> Optional[CardId]:
    """
    Determine the winner of a trick given the cards played.

    Args:
        card_ids: List of card IDs played in order

    Returns:
        CardId of winning card, or None if Kraken wins (no one gets the trick)

    Rules:
        1. Highest card of same type wins among same types
        2. Beasts (Kraken/Whale) beat suits
        3. Jolly Roger beats standard suits and escapes
        4. Characters beat suits and escapes
        5. Pirates beat Mermaids
        6. Skull King beats Pirates
        7. Mermaid beats Skull King (when Pirate + King + Mermaid present)
        8. White Whale: Highest suit card wins
        9. Kraken: No one wins the trick
    """
    if not card_ids:
        return None

    lead: Optional[Card] = None
    suit_lead: Optional[Card] = None
    mermaid_lead: Optional[Card] = None
    has_pirate = False
    has_king = False

    for card_id in card_ids:
        card = get_card(card_id)

        # Track special card presence
        if card.is_pirate():
            has_pirate = True
        if card.is_king():
            has_king = True

        # Track suit leader (highest suit card)
        if card.is_suit():
            if suit_lead is None or suit_lead.number < card.number:
                suit_lead = card

        # Track first mermaid
        if card.is_mermaid() and mermaid_lead is None:
            mermaid_lead = card

        # Determine overall leader
        if lead is None:
            lead = card
            continue

        # Same type: higher number wins
        if lead.card_type == card.card_type:
            if lead.number < card.number:
                lead = card
        else:
            # Different types: apply special rules

            # Beasts beat everything except themselves
            if card.is_whale() or card.is_kraken():
                lead = card

            # Standard suit beats escape
            elif card.is_standard_suit() and lead.is_escape():
                lead = card

            # Jolly Roger beats standard suits and escapes
            elif card.is_roger() and (lead.is_standard_suit() or lead.is_escape()):
                lead = card

            # Characters beat suits and escapes
            elif card.is_character() and (lead.is_suit() or lead.is_escape()):
                lead = card

            # Pirate beats Mermaid
            elif card.is_pirate() and lead.is_mermaid():
                lead = card

            # Skull King beats Pirate
            elif card.is_king() and lead.is_pirate():
                lead = card

            # Mermaid beats Skull King
            elif card.is_mermaid() and lead.is_king():
                lead = card

    # Special end-game rules

    # White Whale: Highest suit card wins
    if lead and lead.is_whale():
        return suit_lead.id if suit_lead else None

    # Kraken: No one wins
    if lead and lead.is_kraken():
        return None

    # Mermaid + Pirate + King: Mermaid wins
    if mermaid_lead and has_pirate and has_king:
        return mermaid_lead.id

    return lead.id if lead else None
