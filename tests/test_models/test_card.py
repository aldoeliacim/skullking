"""Tests for Card model and winner determination."""

import pytest

from app.models.card import Card, CardId, CardType, determine_winner, get_card


class TestCard:
    """Test Card model."""

    def test_card_creation(self):
        """Test creating cards."""
        card = get_card(CardId.SKULL_KING)
        assert card.id == CardId.SKULL_KING
        assert card.card_type == CardType.KING
        assert card.number == 0

    def test_suit_cards(self):
        """Test suit card properties."""
        parrot = get_card(CardId.PARROT10)
        assert parrot.is_parrot()
        assert parrot.is_suit()
        assert parrot.is_standard_suit()
        assert parrot.number == 10

        roger = get_card(CardId.ROGER14)
        assert roger.is_roger()
        assert roger.is_suit()
        assert not roger.is_standard_suit()
        assert roger.number == 14

    def test_special_cards(self):
        """Test special card properties."""
        king = get_card(CardId.SKULL_KING)
        assert king.is_king()
        assert king.is_character()
        assert king.is_special()

        mermaid = get_card(CardId.MERMAID1)
        assert mermaid.is_mermaid()
        assert mermaid.is_character()

        pirate = get_card(CardId.PIRATE1)
        assert pirate.is_pirate()
        assert pirate.is_character()

        escape = get_card(CardId.ESCAPE1)
        assert escape.is_escape()
        assert escape.is_special()


class TestWinnerDetermination:
    """Test winner determination logic."""

    def test_simple_suit_winner(self):
        """Test simple suit card winner."""
        # Parrot 5, Parrot 10, Parrot 3
        cards = [CardId.PARROT5, CardId.PARROT10, CardId.PARROT3]
        winner = determine_winner(cards)
        assert winner == CardId.PARROT10

    def test_trump_beats_standard_suit(self):
        """Test that Jolly Roger beats standard suits."""
        # Parrot 14, Roger 1, Chest 14
        cards = [CardId.PARROT14, CardId.ROGER1, CardId.CHEST14]
        winner = determine_winner(cards)
        assert winner == CardId.ROGER1

    def test_character_beats_suits(self):
        """Test that characters beat suits."""
        # Roger 14, Pirate 1, Parrot 14
        cards = [CardId.ROGER14, CardId.PIRATE1, CardId.PARROT14]
        winner = determine_winner(cards)
        assert winner == CardId.PIRATE1

    def test_skull_king_beats_pirate(self):
        """Test that Skull King beats Pirates."""
        # Pirate 1, Skull King, Pirate 2
        cards = [CardId.PIRATE1, CardId.SKULL_KING, CardId.PIRATE2]
        winner = determine_winner(cards)
        assert winner == CardId.SKULL_KING

    def test_mermaid_beats_skull_king(self):
        """Test that Mermaid beats Skull King."""
        # Skull King, Mermaid
        cards = [CardId.SKULL_KING, CardId.MERMAID1]
        winner = determine_winner(cards)
        assert winner == CardId.MERMAID1

    def test_pirate_beats_mermaid(self):
        """Test that Pirate beats Mermaid."""
        # Mermaid, Pirate
        cards = [CardId.MERMAID1, CardId.PIRATE1]
        winner = determine_winner(cards)
        assert winner == CardId.PIRATE1

    def test_mermaid_with_king_and_pirate(self):
        """Test special rule: Mermaid wins with King + Pirate present."""
        # Skull King, Pirate, Mermaid
        cards = [CardId.SKULL_KING, CardId.PIRATE1, CardId.MERMAID1]
        winner = determine_winner(cards)
        assert winner == CardId.MERMAID1

    def test_kraken_no_winner(self):
        """Test that Kraken results in no winner."""
        # Kraken, Roger 14
        cards = [CardId.KRAKEN, CardId.ROGER14]
        winner = determine_winner(cards)
        assert winner is None

    def test_whale_highest_suit_wins(self):
        """Test that White Whale means highest suit card wins."""
        # Whale, Parrot 5, Roger 10
        cards = [CardId.WHALE, CardId.PARROT5, CardId.ROGER10]
        winner = determine_winner(cards)
        assert winner == CardId.ROGER10

    def test_escape_loses_to_everything(self):
        """Test that Escape cards lose to everything."""
        # Escape, Parrot 1
        cards = [CardId.ESCAPE1, CardId.PARROT1]
        winner = determine_winner(cards)
        assert winner == CardId.PARROT1
