"""Comprehensive tests for all Skull King game rules.

Based on the official rulebook (reglas.md), this test file validates:
- Card hierarchy and winner determination
- Suit following rules
- Beast card interactions (Kraken, Whale)
- Scoring system
- Bonus points calculation
- Rule-based bot behavior
"""

from app.models.card import CardId, determine_winner
from app.models.game import Game
from app.models.player import Player
from app.models.round import Round
from app.models.trick import Trick

# =============================================================================
# SECTION 6.1: SUIT CARDS - BASIC HIERARCHY
# =============================================================================


class TestSuitCardHierarchy:
    """Test basic suit card hierarchy rules (Section 6.1)."""

    def test_same_suit_higher_number_wins(self):
        """Rule: Among same suit, highest number wins."""
        cards = [CardId.PARROT5, CardId.PARROT10, CardId.PARROT3]
        assert determine_winner(cards) == CardId.PARROT10

    def test_same_suit_all_cards_1_to_14(self):
        """Rule: Cards are numbered 1-14, 14 is highest."""
        cards = [CardId.MAP1, CardId.MAP14]
        assert determine_winner(cards) == CardId.MAP14

    def test_trump_beats_all_standard_suits(self):
        """Rule: Trump (Roger/Black flag) beats all standard suits."""
        # Roger 1 should beat even Parrot 14
        cards = [CardId.PARROT14, CardId.ROGER1]
        assert determine_winner(cards) == CardId.ROGER1

    def test_trump_beats_multiple_standard_suits(self):
        """Rule: Any trump beats any combination of standard suits."""
        cards = [CardId.PARROT14, CardId.MAP14, CardId.CHEST14, CardId.ROGER1]
        assert determine_winner(cards) == CardId.ROGER1

    def test_higher_trump_beats_lower_trump(self):
        """Rule: Among trumps, higher number wins."""
        cards = [CardId.ROGER5, CardId.ROGER14, CardId.ROGER1]
        assert determine_winner(cards) == CardId.ROGER14

    def test_different_standard_suits_lead_wins(self):
        """Rule: If no trump, first suit card determines suit to follow."""
        # Lead suit (Parrot) wins, others are off-suit and can't win
        cards = [CardId.PARROT10, CardId.MAP14, CardId.CHEST14]
        assert determine_winner(cards) == CardId.PARROT10

    def test_lead_suit_highest_wins(self):
        """Rule: Among lead suit cards, highest wins."""
        cards = [CardId.CHEST5, CardId.CHEST14, CardId.CHEST1]
        assert determine_winner(cards) == CardId.CHEST14


# =============================================================================
# SECTION 6.2: ESCAPE CARDS
# =============================================================================


class TestEscapeCards:
    """Test Escape card rules (Section 6.2)."""

    def test_escape_loses_to_any_suit(self):
        """Rule: Escape always loses to suit cards."""
        cards = [CardId.ESCAPE1, CardId.PARROT1]
        assert determine_winner(cards) == CardId.PARROT1

    def test_escape_loses_to_trump(self):
        """Rule: Escape loses to trump."""
        cards = [CardId.ESCAPE1, CardId.ROGER1]
        assert determine_winner(cards) == CardId.ROGER1

    def test_escape_loses_to_characters(self):
        """Rule: Escape loses to all characters."""
        # Escape vs Pirate
        assert determine_winner([CardId.ESCAPE1, CardId.PIRATE1]) == CardId.PIRATE1
        # Escape vs Mermaid
        assert determine_winner([CardId.ESCAPE1, CardId.MERMAID1]) == CardId.MERMAID1
        # Escape vs Skull King
        assert determine_winner([CardId.ESCAPE1, CardId.SKULL_KING]) == CardId.SKULL_KING

    def test_all_escapes_first_wins(self):
        """Rule: If all cards are escapes, first escape wins."""
        cards = [CardId.ESCAPE1, CardId.ESCAPE2, CardId.ESCAPE3]
        assert determine_winner(cards) == CardId.ESCAPE1

    def test_escape_purpose_avoid_winning(self):
        """Rule: Escape is used to avoid winning a trick."""
        # Playing escape when you could play suit
        cards = [CardId.PARROT14, CardId.ESCAPE1]
        assert determine_winner(cards) == CardId.PARROT14  # Escape intentionally loses


# =============================================================================
# SECTION 6.3: CHARACTER CARDS - PIRATES
# =============================================================================


class TestPirateCards:
    """Test Pirate card rules (Section 6.3)."""

    def test_pirate_beats_all_suits(self):
        """Rule: Pirates beat all suit cards."""
        cards = [CardId.ROGER14, CardId.PIRATE1]
        assert determine_winner(cards) == CardId.PIRATE1

    def test_pirate_beats_all_suit_types(self):
        """Rule: Pirates beat trumps and standard suits."""
        cards = [CardId.ROGER14, CardId.PARROT14, CardId.MAP14, CardId.PIRATE1]
        assert determine_winner(cards) == CardId.PIRATE1

    def test_multiple_pirates_first_wins(self):
        """Rule: If multiple pirates, first one played wins."""
        cards = [CardId.PIRATE1, CardId.PIRATE2, CardId.PIRATE3]
        assert determine_winner(cards) == CardId.PIRATE1

    def test_pirate_beats_mermaid(self):
        """Rule: Pirates beat Mermaids."""
        cards = [CardId.MERMAID1, CardId.PIRATE1]
        assert determine_winner(cards) == CardId.PIRATE1

    def test_all_five_pirates_same_rank(self):
        """Rule: All 5 pirates have equal rank among themselves."""
        # PIRATE5 played first should beat PIRATE1 played second
        cards = [CardId.PIRATE5, CardId.PIRATE1]
        assert determine_winner(cards) == CardId.PIRATE5


# =============================================================================
# SECTION 6.3: CHARACTER CARDS - TIGRESS (SCARY MARY)
# =============================================================================


class TestTigressCard:
    """Test Tigress (Scary Mary) card rules (Section 6.3)."""

    def test_tigress_as_pirate_beats_suits(self):
        """Rule: Tigress as pirate beats all suits."""
        cards = [CardId.ROGER14, CardId.TIGRESS]
        tigress_choices = {CardId.TIGRESS: "pirate"}
        assert determine_winner(cards, tigress_choices) == CardId.TIGRESS

    def test_tigress_as_pirate_beats_mermaid(self):
        """Rule: Tigress as pirate beats mermaids like a regular pirate."""
        cards = [CardId.MERMAID1, CardId.TIGRESS]
        tigress_choices = {CardId.TIGRESS: "pirate"}
        assert determine_winner(cards, tigress_choices) == CardId.TIGRESS

    def test_tigress_as_escape_loses_to_suits(self):
        """Rule: Tigress as escape loses like regular escape."""
        cards = [CardId.TIGRESS, CardId.PARROT1]
        tigress_choices = {CardId.TIGRESS: "escape"}
        assert determine_winner(cards, tigress_choices) == CardId.PARROT1

    def test_tigress_as_escape_loses_to_characters(self):
        """Rule: Tigress as escape loses to all characters."""
        cards = [CardId.TIGRESS, CardId.PIRATE1]
        tigress_choices = {CardId.TIGRESS: "escape"}
        assert determine_winner(cards, tigress_choices) == CardId.PIRATE1

    def test_skull_king_beats_tigress_as_pirate(self):
        """Rule: Skull King beats Tigress when played as pirate."""
        cards = [CardId.TIGRESS, CardId.SKULL_KING]
        tigress_choices = {CardId.TIGRESS: "pirate"}
        assert determine_winner(cards, tigress_choices) == CardId.SKULL_KING

    def test_tigress_as_pirate_first_beats_regular_pirate(self):
        """Rule: First pirate wins, including Tigress as pirate."""
        cards = [CardId.TIGRESS, CardId.PIRATE1]
        tigress_choices = {CardId.TIGRESS: "pirate"}
        assert determine_winner(cards, tigress_choices) == CardId.TIGRESS

    def test_all_tigress_escape_loot_first_wins(self):
        """Rule: If all escape-like cards, first one wins."""
        cards = [CardId.TIGRESS, CardId.ESCAPE1, CardId.LOOT1]
        tigress_choices = {CardId.TIGRESS: "escape"}
        assert determine_winner(cards, tigress_choices) == CardId.TIGRESS


# =============================================================================
# SECTION 6.3: CHARACTER CARDS - SKULL KING
# =============================================================================


class TestSkullKingCard:
    """Test Skull King card rules (Section 6.3)."""

    def test_skull_king_beats_all_suits(self):
        """Rule: Skull King beats all suit cards."""
        cards = [CardId.ROGER14, CardId.SKULL_KING]
        assert determine_winner(cards) == CardId.SKULL_KING

    def test_skull_king_beats_all_pirates(self):
        """Rule: Skull King beats all pirates."""
        cards = [CardId.PIRATE1, CardId.PIRATE2, CardId.SKULL_KING]
        assert determine_winner(cards) == CardId.SKULL_KING

    def test_skull_king_beats_tigress_pirate(self):
        """Rule: Skull King beats Tigress played as pirate."""
        cards = [CardId.TIGRESS, CardId.SKULL_KING]
        tigress_choices = {CardId.TIGRESS: "pirate"}
        assert determine_winner(cards, tigress_choices) == CardId.SKULL_KING


# =============================================================================
# SECTION 6.3: CHARACTER CARDS - MERMAIDS
# =============================================================================


class TestMermaidCards:
    """Test Mermaid card rules (Section 6.3)."""

    def test_mermaid_beats_all_suits(self):
        """Rule: Mermaids beat all suit cards."""
        cards = [CardId.ROGER14, CardId.MERMAID1]
        assert determine_winner(cards) == CardId.MERMAID1

    def test_mermaid_loses_to_pirate(self):
        """Rule: Mermaids lose to pirates."""
        cards = [CardId.MERMAID1, CardId.PIRATE1]
        assert determine_winner(cards) == CardId.PIRATE1

    def test_mermaid_beats_skull_king(self):
        """Rule: Mermaid beats Skull King (captures him)."""
        cards = [CardId.SKULL_KING, CardId.MERMAID1]
        assert determine_winner(cards) == CardId.MERMAID1

    def test_multiple_mermaids_first_wins(self):
        """Rule: If multiple mermaids, first one wins."""
        cards = [CardId.MERMAID1, CardId.MERMAID2]
        assert determine_winner(cards) == CardId.MERMAID1

    def test_mermaid_order_matters(self):
        """Rule: Order of mermaids matters - first wins."""
        cards = [CardId.MERMAID2, CardId.MERMAID1]
        assert determine_winner(cards) == CardId.MERMAID2


# =============================================================================
# SECTION 6.3: SPECIAL THREE-WAY INTERACTION
# =============================================================================


class TestThreeWayInteraction:
    """Test Mermaid + Pirate + Skull King interaction (Section 6.3)."""

    def test_mermaid_pirate_king_mermaid_wins(self):
        """Rule: When Mermaid + Pirate + King present, Mermaid always wins."""
        cards = [CardId.SKULL_KING, CardId.PIRATE1, CardId.MERMAID1]
        assert determine_winner(cards) == CardId.MERMAID1

    def test_three_way_any_order(self):
        """Rule: Three-way rule works regardless of play order."""
        # Order 1: King, Mermaid, Pirate
        cards = [CardId.SKULL_KING, CardId.MERMAID1, CardId.PIRATE1]
        assert determine_winner(cards) == CardId.MERMAID1

        # Order 2: Pirate, King, Mermaid
        cards = [CardId.PIRATE1, CardId.SKULL_KING, CardId.MERMAID1]
        assert determine_winner(cards) == CardId.MERMAID1

        # Order 3: Mermaid, Pirate, King
        cards = [CardId.MERMAID1, CardId.PIRATE1, CardId.SKULL_KING]
        assert determine_winner(cards) == CardId.MERMAID1

    def test_three_way_first_mermaid_wins(self):
        """Rule: If multiple mermaids in three-way, first mermaid wins."""
        cards = [CardId.SKULL_KING, CardId.PIRATE1, CardId.MERMAID2, CardId.MERMAID1]
        assert determine_winner(cards) == CardId.MERMAID2  # First mermaid played

    def test_three_way_with_tigress_pirate(self):
        """Rule: Three-way works with Tigress as pirate."""
        cards = [CardId.SKULL_KING, CardId.TIGRESS, CardId.MERMAID1]
        tigress_choices = {CardId.TIGRESS: "pirate"}
        assert determine_winner(cards, tigress_choices) == CardId.MERMAID1


# =============================================================================
# SECTION 11.1: KRAKEN
# =============================================================================


class TestKrakenCard:
    """Test Kraken card rules (Section 11.1)."""

    def test_kraken_no_winner(self):
        """Rule: When Kraken is played, no one wins the trick."""
        cards = [CardId.KRAKEN, CardId.ROGER14]
        assert determine_winner(cards) is None

    def test_kraken_destroys_all_cards(self):
        """Rule: Kraken destroys the trick regardless of other cards."""
        cards = [CardId.SKULL_KING, CardId.KRAKEN, CardId.MERMAID1]
        assert determine_winner(cards) is None

    def test_kraken_beats_all_characters(self):
        """Rule: Kraken effect applies even with Skull King."""
        cards = [CardId.SKULL_KING, CardId.PIRATE1, CardId.MERMAID1, CardId.KRAKEN]
        assert determine_winner(cards) is None

    def test_kraken_alone(self):
        """Rule: Kraken alone still results in no winner."""
        cards = [CardId.KRAKEN]
        assert determine_winner(cards) is None


# =============================================================================
# SECTION 11.2: WHITE WHALE
# =============================================================================


class TestWhaleCard:
    """Test White Whale card rules (Section 11.2)."""

    def test_whale_highest_suit_wins(self):
        """Rule: With Whale, highest numbered suit card wins."""
        cards = [CardId.WHALE, CardId.PARROT5, CardId.ROGER10]
        assert determine_winner(cards) == CardId.ROGER10

    def test_whale_ignores_suit_colors(self):
        """Rule: Whale makes suit colors irrelevant, only number matters."""
        # Parrot 14 beats Roger 10 when Whale is present (14 > 10)
        cards = [CardId.WHALE, CardId.PARROT14, CardId.ROGER10]
        assert determine_winner(cards) == CardId.PARROT14

    def test_whale_destroys_special_cards(self):
        """Rule: Whale makes all special cards lose (treated as destroyed)."""
        cards = [CardId.WHALE, CardId.SKULL_KING, CardId.PARROT1]
        # Skull King is destroyed, Parrot 1 wins
        assert determine_winner(cards) == CardId.PARROT1

    def test_whale_destroys_pirates(self):
        """Rule: Pirates are destroyed when Whale is present."""
        cards = [CardId.WHALE, CardId.PIRATE1, CardId.PIRATE2, CardId.CHEST5]
        assert determine_winner(cards) == CardId.CHEST5

    def test_whale_destroys_mermaids(self):
        """Rule: Mermaids are destroyed when Whale is present."""
        cards = [CardId.WHALE, CardId.MERMAID1, CardId.MAP3]
        assert determine_winner(cards) == CardId.MAP3

    def test_whale_only_specials_no_winner(self):
        """Rule: If Whale + only special cards, trick is destroyed like Kraken."""
        cards = [CardId.WHALE, CardId.SKULL_KING, CardId.PIRATE1]
        assert determine_winner(cards) is None

    def test_whale_tie_first_wins(self):
        """Rule: If same highest number, first played wins."""
        cards = [CardId.WHALE, CardId.PARROT14, CardId.ROGER14]
        # Both are 14, first one (Parrot14) wins
        assert determine_winner(cards) == CardId.PARROT14


# =============================================================================
# SECTION 11.3: KRAKEN + WHALE INTERACTION
# =============================================================================


class TestKrakenWhaleInteraction:
    """Test Kraken + Whale interaction (Section 11.3)."""

    def test_kraken_then_whale_whale_wins(self):
        """Rule: Last beast takes effect. Kraken then Whale = Whale effect."""
        cards = [CardId.KRAKEN, CardId.WHALE, CardId.PARROT10]
        # Whale is last, so highest suit wins
        assert determine_winner(cards) == CardId.PARROT10

    def test_whale_then_kraken_no_winner(self):
        """Rule: Last beast takes effect. Whale then Kraken = no winner."""
        cards = [CardId.WHALE, CardId.KRAKEN, CardId.PARROT14]
        # Kraken is last, no one wins
        assert determine_winner(cards) is None

    def test_first_beast_acts_like_escape(self):
        """Rule: First beast is 'defeated' and acts like escape."""
        # When Kraken is first and Whale is second, Kraken is defeated
        # Whale effect applies: highest suit wins
        cards = [CardId.KRAKEN, CardId.WHALE, CardId.ROGER5, CardId.PARROT10]
        assert determine_winner(cards) == CardId.PARROT10  # 10 > 5


# =============================================================================
# SECTION 11.4: LOOT CARDS
# =============================================================================


class TestLootCards:
    """Test Loot (Botín) card rules (Section 11.4)."""

    def test_loot_acts_like_escape(self):
        """Rule: Loot functions like escape in trick hierarchy."""
        cards = [CardId.LOOT1, CardId.PARROT1]
        assert determine_winner(cards) == CardId.PARROT1

    def test_loot_loses_to_everything(self):
        """Rule: Loot loses to all other card types."""
        assert determine_winner([CardId.LOOT1, CardId.PARROT1]) == CardId.PARROT1
        assert determine_winner([CardId.LOOT1, CardId.ROGER1]) == CardId.ROGER1
        assert determine_winner([CardId.LOOT1, CardId.PIRATE1]) == CardId.PIRATE1
        assert determine_winner([CardId.LOOT1, CardId.MERMAID1]) == CardId.MERMAID1
        assert determine_winner([CardId.LOOT1, CardId.SKULL_KING]) == CardId.SKULL_KING

    def test_all_loot_first_wins(self):
        """Rule: If all loot/escape cards, first one wins."""
        cards = [CardId.LOOT1, CardId.LOOT2]
        assert determine_winner(cards) == CardId.LOOT1

    def test_loot_escape_mixed_first_wins(self):
        """Rule: Mixed loot and escape, first one wins."""
        cards = [CardId.ESCAPE1, CardId.LOOT1, CardId.LOOT2]
        assert determine_winner(cards) == CardId.ESCAPE1

        cards = [CardId.LOOT1, CardId.ESCAPE1, CardId.LOOT2]
        assert determine_winner(cards) == CardId.LOOT1


class TestLootAllianceBonus:
    """Test Loot alliance bonus rules (Section 11.4)."""

    def test_loot_alliance_formed_with_trick_winner(self):
        """Rule: Playing Loot forms alliance with trick winner."""
        trick = Trick(number=1, starter_player_index=0)
        trick.add_card("player1", CardId.LOOT1)
        trick.add_card("player2", CardId.PARROT10)
        trick.determine_winner()

        alliances = trick.get_loot_alliances()
        assert len(alliances) == 1
        assert alliances[0] == ("player1", "player2")

    def test_loot_alliance_no_bonus_if_player_is_winner(self):
        """Rule: No alliance if loot player wins the trick themselves."""
        trick = Trick(number=1, starter_player_index=0)
        trick.add_card("player1", CardId.LOOT1)
        trick.add_card("player2", CardId.ESCAPE1)
        trick.determine_winner()  # player1 wins (first loot)

        alliances = trick.get_loot_alliances()
        assert len(alliances) == 0  # No alliance since loot player won

    def test_loot_alliance_bonus_both_correct_bids(self):
        """Rule: +20 bonus each if both alliance members make their bids."""
        game_round = Round(number=3, starter_player_index=0)
        game_round.bids = {"player1": 1, "player2": 1}

        # Trick 1: player1 plays Loot, player2 wins with Parrot
        trick1 = Trick(number=1, starter_player_index=0)
        trick1.add_card("player1", CardId.LOOT1)
        trick1.add_card("player2", CardId.PARROT10)
        trick1.determine_winner()
        game_round.tricks.append(trick1)

        # Trick 2: player2 plays Escape, player1 wins
        trick2 = Trick(number=2, starter_player_index=1)
        trick2.add_card("player2", CardId.ESCAPE1)
        trick2.add_card("player1", CardId.PARROT5)
        trick2.determine_winner()
        game_round.tricks.append(trick2)

        # Trick 3: both play something neutral
        trick3 = Trick(number=3, starter_player_index=0)
        trick3.add_card("player1", CardId.MAP1)
        trick3.add_card("player2", CardId.MAP2)
        trick3.determine_winner()
        game_round.tricks.append(trick3)

        # Both made their bids: player1 won 1, player2 won 2
        # Wait - player2 won 2 tricks but bid 1, so only player1 made bid
        # Let me adjust: player1 bid 1 (won 1 ✓), player2 bid 2 (won 2 ✓)
        game_round.bids = {"player1": 1, "player2": 2}
        game_round.calculate_scores()

        # Base scores: player1 = 20*1=20, player2 = 20*2=40
        # Alliance bonus: +20 each since both made bids
        assert game_round.scores["player1"] == 20 + 20  # 40
        assert game_round.scores["player2"] == 40 + 20  # 60

    def test_loot_alliance_no_bonus_if_one_fails_bid(self):
        """Rule: No bonus if either alliance member fails their bid."""
        game_round = Round(number=2, starter_player_index=0)
        game_round.bids = {"player1": 0, "player2": 1}

        # Trick 1: player1 plays Loot, player2 wins
        trick1 = Trick(number=1, starter_player_index=0)
        trick1.add_card("player1", CardId.LOOT1)
        trick1.add_card("player2", CardId.PARROT10)
        trick1.determine_winner()
        game_round.tricks.append(trick1)

        # Trick 2: player2 wins again
        trick2 = Trick(number=2, starter_player_index=1)
        trick2.add_card("player2", CardId.PARROT5)
        trick2.add_card("player1", CardId.ESCAPE1)
        trick2.determine_winner()
        game_round.tricks.append(trick2)

        # player1 bid 0 but won 0 ✓, player2 bid 1 but won 2 ✗
        game_round.calculate_scores()

        # player1: 10 * round = 20 (zero bid correct, no alliance bonus)
        # player2: -10 * diff = -10 (bid wrong)
        assert game_round.scores["player1"] == 20  # No +20 bonus
        assert game_round.scores["player2"] == -10


# =============================================================================
# SECTION 2.4 & 6.4: SUIT FOLLOWING RULES
# =============================================================================


class TestSuitFollowing:
    """Test suit following rules (Section 2.4 and 6.4)."""

    def test_must_follow_suit_if_possible(self):
        """Rule: Must follow lead suit if you have cards of that suit."""
        trick = Trick(number=1, starter_player_index=0)
        hand = [CardId.PARROT5, CardId.MAP10, CardId.CHEST14]
        cards_in_trick = [CardId.PARROT10]  # Parrot led

        valid = trick.get_valid_cards(hand, cards_in_trick)

        # Must play Parrot (or special cards if any)
        assert CardId.PARROT5 in valid
        assert CardId.MAP10 not in valid  # Can't play off-suit
        assert CardId.CHEST14 not in valid  # Can't play off-suit

    def test_special_cards_always_valid(self):
        """Rule: Special cards can always be played, regardless of suit."""
        trick = Trick(number=1, starter_player_index=0)
        hand = [CardId.PARROT5, CardId.ESCAPE1, CardId.PIRATE1]
        cards_in_trick = [CardId.MAP10]  # Map led

        valid = trick.get_valid_cards(hand, cards_in_trick)

        # Special cards always valid
        assert CardId.ESCAPE1 in valid
        assert CardId.PIRATE1 in valid
        # Can't follow suit (no Map cards), so Parrot is valid
        assert CardId.PARROT5 in valid

    def test_cant_follow_suit_play_anything(self):
        """Rule: If can't follow suit, can play any card."""
        trick = Trick(number=1, starter_player_index=0)
        hand = [CardId.MAP5, CardId.CHEST10, CardId.ROGER1]
        cards_in_trick = [CardId.PARROT10]  # Parrot led, no Parrots in hand

        valid = trick.get_valid_cards(hand, cards_in_trick)

        # Can play anything since no Parrots
        assert CardId.MAP5 in valid
        assert CardId.CHEST10 in valid
        assert CardId.ROGER1 in valid

    def test_leading_can_play_anything(self):
        """Rule: When leading (first to play), can play any card."""
        trick = Trick(number=1, starter_player_index=0)
        hand = [CardId.PARROT5, CardId.MAP10, CardId.ESCAPE1, CardId.SKULL_KING]
        cards_in_trick = []  # Leading

        valid = trick.get_valid_cards(hand, cards_in_trick)

        # Can play any card when leading
        assert set(valid) == set(hand)

    def test_escape_lead_no_suit_set(self):
        """Rule: Opening with escape doesn't set a suit."""
        trick = Trick(number=1, starter_player_index=0)
        hand = [CardId.MAP5, CardId.CHEST10]
        cards_in_trick = [CardId.ESCAPE1]  # Escape led

        valid = trick.get_valid_cards(hand, cards_in_trick)

        # No suit set, can play anything
        assert set(valid) == set(hand)

    def test_character_lead_no_suit_set(self):
        """Rule: Opening with character doesn't set a suit."""
        trick = Trick(number=1, starter_player_index=0)
        hand = [CardId.MAP5, CardId.CHEST10]
        cards_in_trick = [CardId.PIRATE1]  # Pirate led

        valid = trick.get_valid_cards(hand, cards_in_trick)

        # No suit set, can play anything
        assert set(valid) == set(hand)

    def test_kraken_lead_no_suit_set(self):
        """Rule: Opening with Kraken doesn't set a suit."""
        trick = Trick(number=1, starter_player_index=0)
        hand = [CardId.MAP5, CardId.CHEST10]
        cards_in_trick = [CardId.KRAKEN]

        valid = trick.get_valid_cards(hand, cards_in_trick)

        # No suit set
        assert set(valid) == set(hand)


# =============================================================================
# SECTION 8: SCORING SYSTEM
# =============================================================================


class TestScoringSystem:
    """Test scoring rules (Section 8)."""

    def test_correct_bid_nonzero(self):
        """Rule: Correct non-zero bid = 20 * bid + bonus."""
        round_obj = Round(number=5, starter_player_index=0)
        round_obj.bids = {"player1": 3}

        # Create tricks where player1 wins exactly 3
        for i in range(5):
            trick = Trick(number=i + 1, starter_player_index=0)
            if i < 3:
                trick.winner_player_id = "player1"
            else:
                trick.winner_player_id = "player2"
            round_obj.tricks.append(trick)

        round_obj.calculate_scores()

        # 20 * 3 = 60 points (no bonus)
        assert round_obj.scores["player1"] == 60

    def test_correct_bid_zero(self):
        """Rule: Correct zero bid = 10 * round_number."""
        round_obj = Round(number=7, starter_player_index=0)
        round_obj.bids = {"player1": 0}

        # Player1 wins 0 tricks
        for i in range(7):
            trick = Trick(number=i + 1, starter_player_index=0)
            trick.winner_player_id = "player2"
            round_obj.tricks.append(trick)

        round_obj.calculate_scores()

        # 10 * 7 = 70 points
        assert round_obj.scores["player1"] == 70

    def test_wrong_bid_nonzero(self):
        """Rule: Wrong non-zero bid = -10 * |difference|."""
        round_obj = Round(number=5, starter_player_index=0)
        round_obj.bids = {"player1": 2}

        # Player1 wins 4 tricks (2 off)
        for i in range(5):
            trick = Trick(number=i + 1, starter_player_index=0)
            if i < 4:
                trick.winner_player_id = "player1"
            else:
                trick.winner_player_id = "player2"
            round_obj.tricks.append(trick)

        round_obj.calculate_scores()

        # -10 * |4 - 2| = -20 points
        assert round_obj.scores["player1"] == -20

    def test_wrong_bid_zero(self):
        """Rule: Wrong zero bid = -10 * round_number."""
        round_obj = Round(number=9, starter_player_index=0)
        round_obj.bids = {"player1": 0}

        # Player1 wins 1 trick (should have won 0)
        for i in range(9):
            trick = Trick(number=i + 1, starter_player_index=0)
            if i < 1:
                trick.winner_player_id = "player1"
            else:
                trick.winner_player_id = "player2"
            round_obj.tricks.append(trick)

        round_obj.calculate_scores()

        # -10 * 9 = -90 points
        assert round_obj.scores["player1"] == -90

    def test_under_bid_penalty(self):
        """Rule: Under-bidding also incurs penalty."""
        round_obj = Round(number=5, starter_player_index=0)
        round_obj.bids = {"player1": 4}

        # Player1 wins only 1 trick (3 under)
        for i in range(5):
            trick = Trick(number=i + 1, starter_player_index=0)
            if i < 1:
                trick.winner_player_id = "player1"
            else:
                trick.winner_player_id = "player2"
            round_obj.tricks.append(trick)

        round_obj.calculate_scores()

        # -10 * |1 - 4| = -30 points
        assert round_obj.scores["player1"] == -30


# =============================================================================
# SECTION 8.3: BONUS POINTS
# =============================================================================


class TestBonusPoints:
    """Test bonus point rules (Section 8.3)."""

    def test_bonus_standard_14_plus_10(self):
        """Rule: Capturing standard suit 14 = +10 bonus."""
        trick = Trick(number=1, starter_player_index=0)
        trick.add_card("player1", CardId.PIRATE1)  # Winner
        trick.add_card("player2", CardId.PARROT14)  # 14 captured
        trick.determine_winner()

        bonus = trick.calculate_bonus_points()
        assert bonus == 10

    def test_bonus_all_standard_14s(self):
        """Rule: Each standard 14 gives +10."""
        trick = Trick(number=1, starter_player_index=0)
        trick.add_card("player1", CardId.SKULL_KING)  # Winner
        trick.add_card("player2", CardId.PARROT14)
        trick.add_card("player3", CardId.MAP14)
        trick.add_card("player4", CardId.CHEST14)
        trick.determine_winner()

        bonus = trick.calculate_bonus_points()
        assert bonus == 30  # 10 + 10 + 10

    def test_bonus_roger_14_plus_20(self):
        """Rule: Capturing Roger/Trump 14 = +20 bonus."""
        trick = Trick(number=1, starter_player_index=0)
        trick.add_card("player1", CardId.SKULL_KING)  # Winner
        trick.add_card("player2", CardId.ROGER14)
        trick.determine_winner()

        bonus = trick.calculate_bonus_points()
        assert bonus == 20

    def test_bonus_pirate_captures_mermaid_plus_20(self):
        """Rule: Pirate winning with Mermaid present = +20."""
        trick = Trick(number=1, starter_player_index=0)
        trick.add_card("player1", CardId.PIRATE1)  # Winner
        trick.add_card("player2", CardId.MERMAID1)
        trick.determine_winner()

        bonus = trick.calculate_bonus_points()
        assert bonus == 20

    def test_bonus_king_captures_pirate_plus_30(self):
        """Rule: Skull King winning with Pirate present = +30."""
        trick = Trick(number=1, starter_player_index=0)
        trick.add_card("player1", CardId.SKULL_KING)  # Winner
        trick.add_card("player2", CardId.PIRATE1)
        trick.determine_winner()

        bonus = trick.calculate_bonus_points()
        assert bonus == 30

    def test_bonus_king_captures_multiple_pirates(self):
        """Rule: +30 for each pirate captured by King."""
        trick = Trick(number=1, starter_player_index=0)
        trick.add_card("player1", CardId.SKULL_KING)  # Winner
        trick.add_card("player2", CardId.PIRATE1)
        trick.add_card("player3", CardId.PIRATE2)
        trick.determine_winner()

        bonus = trick.calculate_bonus_points()
        assert bonus == 60  # 30 + 30

    def test_bonus_mermaid_captures_king_plus_40(self):
        """Rule: Mermaid winning with Skull King present = +40."""
        trick = Trick(number=1, starter_player_index=0)
        trick.add_card("player1", CardId.MERMAID1)  # Winner
        trick.add_card("player2", CardId.SKULL_KING)
        trick.determine_winner()

        bonus = trick.calculate_bonus_points()
        assert bonus == 40

    def test_bonus_combined(self):
        """Rule: Multiple bonuses can combine."""
        trick = Trick(number=1, starter_player_index=0)
        trick.add_card("player1", CardId.SKULL_KING)  # Winner
        trick.add_card("player2", CardId.PIRATE1)  # +30
        trick.add_card("player3", CardId.ROGER14)  # +20
        trick.add_card("player4", CardId.PARROT14)  # +10
        trick.determine_winner()

        bonus = trick.calculate_bonus_points()
        assert bonus == 60  # 30 + 20 + 10

    def test_no_bonus_if_bid_wrong(self):
        """Rule: Bonuses only count if bid is correct."""
        round_obj = Round(number=3, starter_player_index=0)
        round_obj.bids = {"player1": 1}

        # Create a trick where player1 wins with bonus
        trick = Trick(number=1, starter_player_index=0)
        trick.add_card("player1", CardId.SKULL_KING)
        trick.add_card("player2", CardId.PIRATE1)
        trick.picked_cards[0].player_id = "player1"
        trick.picked_cards[1].player_id = "player2"
        trick.determine_winner()
        round_obj.tricks.append(trick)

        # Player1 wins 2 more tricks (total 3, but bid was 1)
        for i in range(2):
            t = Trick(number=i + 2, starter_player_index=0)
            t.winner_player_id = "player1"
            round_obj.tricks.append(t)

        round_obj.calculate_scores()

        # Bid wrong: -10 * |3 - 1| = -20 (no bonus added)
        assert round_obj.scores["player1"] == -20


# =============================================================================
# GAME STRUCTURE RULES
# =============================================================================


class TestGameStructure:
    """Test game structure rules (Section 4)."""

    def test_ten_rounds_total(self):
        """Rule: Standard game has 10 rounds."""
        from app.models.enums import MAX_ROUNDS

        assert MAX_ROUNDS == 10

    def test_round_n_has_n_cards(self):
        """Rule: Round N deals N cards per player."""
        game = Game(id="test", slug="test")
        game.add_player(Player(id="p1", username="P1", avatar_id=1, index=0, game_id="test"))
        game.add_player(Player(id="p2", username="P2", avatar_id=2, index=1, game_id="test"))

        # Start round 1
        game.start_new_round()
        game.deal_cards()
        assert len(game.players[0].hand) == 1

        # Start round 2
        game.start_new_round()
        game.deal_cards()
        assert len(game.players[0].hand) == 2

    def test_max_players(self):
        """Rule: Maximum 8 players (2-8 supported)."""
        from app.models.enums import MAX_PLAYERS

        assert MAX_PLAYERS == 8

    def test_min_players_to_start(self):
        """Rule: Minimum 2 players to start."""
        game = Game(id="test", slug="test")
        game.add_player(Player(id="p1", username="P1", avatar_id=1, index=0, game_id="test"))
        assert not game.can_start()

        game.add_player(Player(id="p2", username="P2", avatar_id=2, index=1, game_id="test"))
        assert game.can_start()


# =============================================================================
# DECK COMPOSITION
# =============================================================================


class TestDeckComposition:
    """Test deck composition (Section 0)."""

    def test_deck_has_74_cards(self):
        """Rule: Full deck has 74 cards."""
        from app.models.card import get_all_cards

        cards = get_all_cards()
        assert len(cards) == 74

    def test_deck_has_56_suit_cards(self):
        """Rule: 56 suit cards (14 each of 4 suits)."""
        from app.models.card import get_all_cards

        cards = get_all_cards()
        suit_cards = [c for c in cards.values() if c.is_suit()]
        assert len(suit_cards) == 56

    def test_deck_has_5_pirates(self):
        """Rule: 5 pirate cards."""
        from app.models.card import get_all_cards

        cards = get_all_cards()
        pirates = [c for c in cards.values() if c.is_pirate()]
        assert len(pirates) == 5

    def test_deck_has_5_escapes(self):
        """Rule: 5 escape cards."""
        from app.models.card import get_all_cards

        cards = get_all_cards()
        escapes = [c for c in cards.values() if c.is_escape()]
        assert len(escapes) == 5

    def test_deck_has_2_mermaids(self):
        """Rule: 2 mermaid cards."""
        from app.models.card import get_all_cards

        cards = get_all_cards()
        mermaids = [c for c in cards.values() if c.is_mermaid()]
        assert len(mermaids) == 2

    def test_deck_has_1_skull_king(self):
        """Rule: 1 Skull King card."""
        from app.models.card import get_all_cards

        cards = get_all_cards()
        kings = [c for c in cards.values() if c.is_king()]
        assert len(kings) == 1

    def test_deck_has_1_tigress(self):
        """Rule: 1 Tigress (Scary Mary) card."""
        from app.models.card import get_all_cards

        cards = get_all_cards()
        tigress = [c for c in cards.values() if c.is_tigress()]
        assert len(tigress) == 1

    def test_deck_has_1_kraken(self):
        """Rule: 1 Kraken card."""
        from app.models.card import get_all_cards

        cards = get_all_cards()
        krakens = [c for c in cards.values() if c.is_kraken()]
        assert len(krakens) == 1

    def test_deck_has_1_whale(self):
        """Rule: 1 White Whale card."""
        from app.models.card import get_all_cards

        cards = get_all_cards()
        whales = [c for c in cards.values() if c.is_whale()]
        assert len(whales) == 1

    def test_deck_has_2_loot(self):
        """Rule: 2 Loot (Botín) cards."""
        from app.models.card import get_all_cards

        cards = get_all_cards()
        loots = [c for c in cards.values() if c.is_loot()]
        assert len(loots) == 2


# =============================================================================
# RULE-BASED BOT VALIDATION
# =============================================================================


class TestRuleBasedBot:
    """Test that rule-based bot follows game rules."""

    def test_bot_respects_valid_cards(self):
        """Rule: Bot must play from valid cards provided."""
        # Test without importing bot to avoid stable_baselines3 dependency
        trick = Trick(number=1, starter_player_index=0)

        # Hand with Parrot cards
        hand = [CardId.PARROT5, CardId.PARROT10, CardId.MAP14]

        # Parrot led - must follow
        cards_in_trick = [CardId.PARROT1]
        valid_cards = trick.get_valid_cards(hand, cards_in_trick)

        # Valid cards should only include Parrots (following suit)
        assert CardId.PARROT5 in valid_cards
        assert CardId.PARROT10 in valid_cards
        assert CardId.MAP14 not in valid_cards

    def test_valid_cards_respects_suit_following(self):
        """Rule: Valid cards must follow suit when possible."""
        trick = Trick(number=1, starter_player_index=0)

        # Hand with mixed suits
        hand = [CardId.PARROT5, CardId.PARROT10, CardId.MAP14, CardId.ESCAPE1]
        cards_in_trick = [CardId.PARROT1]  # Parrot led

        valid_cards = trick.get_valid_cards(hand, cards_in_trick)

        # Must follow Parrot suit, plus special cards (Escape) always valid
        assert CardId.PARROT5 in valid_cards
        assert CardId.PARROT10 in valid_cards
        assert CardId.ESCAPE1 in valid_cards  # Special cards always valid
        assert CardId.MAP14 not in valid_cards  # Can't play off-suit

    def test_card_strength_hierarchy(self):
        """Test that game hierarchy is correct: King > Pirate > Mermaid > Suits."""
        # Skull King beats Pirate
        assert determine_winner([CardId.PIRATE1, CardId.SKULL_KING]) == CardId.SKULL_KING

        # Pirate beats Mermaid
        assert determine_winner([CardId.MERMAID1, CardId.PIRATE1]) == CardId.PIRATE1

        # Mermaid beats suits (including trump)
        assert determine_winner([CardId.ROGER14, CardId.MERMAID1]) == CardId.MERMAID1

        # Trump beats standard suits
        assert determine_winner([CardId.PARROT14, CardId.ROGER1]) == CardId.ROGER1

        # Any suit beats Escape
        assert determine_winner([CardId.ESCAPE1, CardId.PARROT1]) == CardId.PARROT1

    def test_escape_always_loses(self):
        """Rule: Escape cards have strength 0 - always lose."""
        # Escape loses to lowest suit card
        assert determine_winner([CardId.ESCAPE1, CardId.PARROT1]) == CardId.PARROT1

        # Multiple escapes - first wins
        assert determine_winner([CardId.ESCAPE1, CardId.ESCAPE2]) == CardId.ESCAPE1
