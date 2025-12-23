"""Tests for game flow, turn order, and bot behavior.

These tests verify:
- Correct turn order during tricks
- Bots respecting suit following constraints
- Game state transitions
- Trick winner leading next trick
- Round progression
"""

from app.models.card import CardId
from app.models.enums import GameState
from app.models.game import Game
from app.models.player import Player
from app.models.round import Round
from app.models.trick import TigressChoice, Trick

# =============================================================================
# TURN ORDER TESTS
# =============================================================================


class TestTurnOrder:
    """Test turn order mechanics."""

    def test_starter_opens_first_trick(self):
        """Rule: Player at starter_player_index opens first trick."""
        round_obj = Round(number=3, starter_player_index=2)
        trick = Trick(number=1, starter_player_index=2)
        round_obj.tricks.append(trick)

        assert trick.starter_player_index == 2

    def test_trick_winner_opens_next_trick(self):
        """Rule: Winner of trick opens the next trick."""
        # Simulate first trick with player 1 winning
        trick1 = Trick(number=1, starter_player_index=0)
        trick1.add_card("player-0", CardId.PARROT5)
        trick1.add_card("player-1", CardId.PARROT14)  # Wins
        trick1.add_card("player-2", CardId.PARROT3)
        trick1.determine_winner()

        assert trick1.winner_player_id == "player-1"

        # Next trick should start with player-1
        # (In actual game, game_handler sets this based on winner)

    def test_players_play_in_order(self):
        """Rule: Players pick in turn order starting from starter."""
        trick = Trick(number=1, starter_player_index=1)

        # Simulate 3 players playing in order starting from index 1
        trick.add_card("player-1", CardId.PARROT5)
        trick.add_card("player-2", CardId.PARROT10)
        trick.add_card("player-0", CardId.PARROT3)

        # Verify order is preserved
        assert trick.picked_cards[0].player_id == "player-1"
        assert trick.picked_cards[1].player_id == "player-2"
        assert trick.picked_cards[2].player_id == "player-0"

    def test_player_cannot_pick_twice(self):
        """Rule: Player can only play one card per trick."""
        trick = Trick(number=1, starter_player_index=0)

        # First pick succeeds
        assert trick.add_card("player-0", CardId.PARROT5) is True

        # Second pick fails
        assert trick.add_card("player-0", CardId.PARROT10) is False

        # Verify only one card from player-0
        assert len([pc for pc in trick.picked_cards if pc.player_id == "player-0"]) == 1


# =============================================================================
# SUIT FOLLOWING CONSTRAINT TESTS
# =============================================================================


class TestSuitFollowingConstraints:
    """Test that suit following rules are enforced correctly."""

    def test_must_follow_lead_suit(self):
        """Rule: Must play lead suit if you have it."""
        trick = Trick(number=1, starter_player_index=0)
        hand = [CardId.PARROT5, CardId.PARROT10, CardId.MAP14, CardId.CHEST7]
        cards_in_trick = [CardId.PARROT1]  # Parrot led

        valid = trick.get_valid_cards(hand, cards_in_trick)

        # Only Parrots should be valid (no special cards in hand)
        assert CardId.PARROT5 in valid
        assert CardId.PARROT10 in valid
        assert CardId.MAP14 not in valid
        assert CardId.CHEST7 not in valid

    def test_special_cards_bypass_suit_following(self):
        """Rule: Special cards can always be played regardless of suit."""
        trick = Trick(number=1, starter_player_index=0)
        hand = [CardId.PARROT5, CardId.ESCAPE1, CardId.PIRATE1, CardId.MERMAID1]
        cards_in_trick = [CardId.MAP10]  # Map led, no Maps in hand

        valid = trick.get_valid_cards(hand, cards_in_trick)

        # All cards valid since can't follow suit
        assert set(valid) == set(hand)

    def test_can_play_special_even_with_suit(self):
        """Rule: Can play special cards even if you have the lead suit."""
        trick = Trick(number=1, starter_player_index=0)
        hand = [CardId.PARROT5, CardId.PARROT10, CardId.ESCAPE1, CardId.SKULL_KING]
        cards_in_trick = [CardId.PARROT1]  # Parrot led

        valid = trick.get_valid_cards(hand, cards_in_trick)

        # Parrots and special cards valid
        assert CardId.PARROT5 in valid
        assert CardId.PARROT10 in valid
        assert CardId.ESCAPE1 in valid
        assert CardId.SKULL_KING in valid

    def test_trump_is_separate_suit(self):
        """Rule: Trump (Roger) is its own suit, not a 'special' card."""
        trick = Trick(number=1, starter_player_index=0)
        hand = [CardId.ROGER5, CardId.ROGER14, CardId.PARROT10]
        cards_in_trick = [CardId.ROGER1]  # Roger led

        valid = trick.get_valid_cards(hand, cards_in_trick)

        # Must follow Roger suit
        assert CardId.ROGER5 in valid
        assert CardId.ROGER14 in valid
        assert CardId.PARROT10 not in valid

    def test_no_suit_when_special_leads(self):
        """Rule: No suit to follow when special card leads."""
        trick = Trick(number=1, starter_player_index=0)
        hand = [CardId.PARROT5, CardId.MAP10, CardId.CHEST14]
        cards_in_trick = [CardId.PIRATE1]  # Pirate led

        valid = trick.get_valid_cards(hand, cards_in_trick)

        # Any card valid
        assert set(valid) == set(hand)

    def test_escape_lead_no_suit(self):
        """Rule: Escape doesn't set a suit."""
        trick = Trick(number=1, starter_player_index=0)
        hand = [CardId.PARROT5, CardId.MAP10]
        cards_in_trick = [CardId.ESCAPE1]

        valid = trick.get_valid_cards(hand, cards_in_trick)

        assert set(valid) == set(hand)

    def test_loot_lead_no_suit(self):
        """Rule: Loot doesn't set a suit."""
        trick = Trick(number=1, starter_player_index=0)
        hand = [CardId.PARROT5, CardId.MAP10]
        cards_in_trick = [CardId.LOOT1]

        valid = trick.get_valid_cards(hand, cards_in_trick)

        assert set(valid) == set(hand)

    def test_multiple_escapes_then_suit(self):
        """Rule: First suit card sets the suit even after escapes."""
        trick = Trick(number=1, starter_player_index=0)
        hand = [CardId.PARROT5, CardId.MAP10]
        cards_in_trick = [CardId.ESCAPE1, CardId.ESCAPE2, CardId.PARROT1]

        valid = trick.get_valid_cards(hand, cards_in_trick)

        # Parrot set the suit
        assert CardId.PARROT5 in valid
        assert CardId.MAP10 not in valid

    def test_tigress_as_escape_no_suit(self):
        """Rule: Tigress as escape doesn't set suit."""
        trick = Trick(number=1, starter_player_index=0)
        hand = [CardId.PARROT5, CardId.MAP10]
        # Tigress played as escape (card type is TIGRESS, not suit)
        cards_in_trick = [CardId.TIGRESS]

        valid = trick.get_valid_cards(hand, cards_in_trick)

        # Tigress is special card, no suit set
        assert set(valid) == set(hand)


# =============================================================================
# GAME STATE TRANSITION TESTS
# =============================================================================


class TestGameStateTransitions:
    """Test game state transitions."""

    def test_game_starts_pending(self):
        """Rule: Game starts in PENDING state."""
        game = Game(id="test", slug="test")
        assert game.state == GameState.PENDING

    def test_round_transitions_to_bidding(self):
        """Test that starting a round sets up bidding."""
        game = Game(id="test", slug="test")
        game.add_player(Player(id="p1", username="P1", avatar_id=1, index=0, game_id="test"))
        game.add_player(Player(id="p2", username="P2", avatar_id=2, index=1, game_id="test"))

        game.start_new_round()
        # State should be ready for bidding (game_handler sets BIDDING)
        assert game.current_round_number == 1

    def test_trick_complete_after_all_pick(self):
        """Rule: Trick is complete when all players have picked."""
        trick = Trick(number=1, starter_player_index=0)

        assert not trick.is_complete(3)

        trick.add_card("p0", CardId.PARROT5)
        assert not trick.is_complete(3)

        trick.add_card("p1", CardId.PARROT10)
        assert not trick.is_complete(3)

        trick.add_card("p2", CardId.PARROT3)
        assert trick.is_complete(3)

    def test_round_complete_after_all_tricks(self):
        """Rule: Round is complete when N tricks played in round N."""
        round_obj = Round(number=3, starter_player_index=0)

        # Need 3 tricks for round 3
        assert not round_obj.is_complete()

        for i in range(3):
            trick = Trick(number=i + 1, starter_player_index=0)
            trick.winner_player_id = "p0"
            round_obj.tricks.append(trick)

        assert round_obj.is_complete()


# =============================================================================
# TRICK WINNER DETERMINATION FLOW
# =============================================================================


class TestTrickWinnerFlow:
    """Test trick winner determination and its effects."""

    def test_winner_recorded_in_trick(self):
        """Test that winner is correctly recorded."""
        trick = Trick(number=1, starter_player_index=0)
        trick.add_card("p0", CardId.PARROT5)
        trick.add_card("p1", CardId.PARROT14)
        trick.add_card("p2", CardId.PARROT3)

        winner_card, winner_player = trick.determine_winner()

        assert winner_card == CardId.PARROT14
        assert winner_player == "p1"
        assert trick.winner_card_id == CardId.PARROT14
        assert trick.winner_player_id == "p1"

    def test_kraken_no_winner_recorded(self):
        """Test Kraken results in no winner."""
        trick = Trick(number=1, starter_player_index=0)
        trick.add_card("p0", CardId.PARROT14)
        trick.add_card("p1", CardId.KRAKEN)
        trick.add_card("p2", CardId.SKULL_KING)

        winner_card, winner_player = trick.determine_winner()

        assert winner_card is None
        assert winner_player is None

    def test_tricks_won_counted(self):
        """Test that tricks won are counted correctly."""
        round_obj = Round(number=3, starter_player_index=0)
        round_obj.bids = {"p0": 2, "p1": 1}

        # Player 0 wins 2 tricks
        for i in range(2):
            trick = Trick(number=i + 1, starter_player_index=0)
            trick.winner_player_id = "p0"
            round_obj.tricks.append(trick)

        # Player 1 wins 1 trick
        trick = Trick(number=3, starter_player_index=0)
        trick.winner_player_id = "p1"
        round_obj.tricks.append(trick)

        assert round_obj.get_tricks_won("p0") == 2
        assert round_obj.get_tricks_won("p1") == 1


# =============================================================================
# ROUND PROGRESSION TESTS
# =============================================================================


class TestRoundProgression:
    """Test round progression mechanics."""

    def test_round_number_increments(self):
        """Test that round number increments correctly."""
        game = Game(id="test", slug="test")
        game.add_player(Player(id="p1", username="P1", avatar_id=1, index=0, game_id="test"))
        game.add_player(Player(id="p2", username="P2", avatar_id=2, index=1, game_id="test"))

        game.start_new_round()
        assert game.current_round_number == 1

        game.start_new_round()
        assert game.current_round_number == 2

        game.start_new_round()
        assert game.current_round_number == 3

    def test_cards_dealt_match_round(self):
        """Test that N cards are dealt in round N."""
        game = Game(id="test", slug="test")
        game.add_player(Player(id="p1", username="P1", avatar_id=1, index=0, game_id="test"))
        game.add_player(Player(id="p2", username="P2", avatar_id=2, index=1, game_id="test"))

        for round_num in range(1, 6):
            game.start_new_round()
            game.deal_cards()
            assert len(game.players[0].hand) == round_num
            assert len(game.players[1].hand) == round_num

    def test_player_state_reset_between_rounds(self):
        """Test that player state is reset between rounds."""
        game = Game(id="test", slug="test")
        p1 = Player(id="p1", username="P1", avatar_id=1, index=0, game_id="test")
        game.add_player(p1)
        game.add_player(Player(id="p2", username="P2", avatar_id=2, index=1, game_id="test"))

        # Round 1
        game.start_new_round()
        p1.bid = 1
        p1.tricks_won = 1

        # Round 2 should reset
        game.start_new_round()
        assert p1.bid is None
        assert p1.tricks_won == 0

    def test_game_complete_after_10_rounds(self):
        """Test that game is complete after 10 rounds."""
        game = Game(id="test", slug="test")
        game.add_player(Player(id="p1", username="P1", avatar_id=1, index=0, game_id="test"))
        game.add_player(Player(id="p2", username="P2", avatar_id=2, index=1, game_id="test"))

        for _ in range(10):
            game.start_new_round()

        assert game.is_game_complete()

    def test_starter_rotates_from_round1(self):
        """Test that starter position rotates from round 1's random starter."""
        game = Game(id="test", slug="test")
        game.add_player(Player(id="p1", username="P1", avatar_id=1, index=0, game_id="test"))
        game.add_player(Player(id="p2", username="P2", avatar_id=2, index=1, game_id="test"))
        game.add_player(Player(id="p3", username="P3", avatar_id=3, index=2, game_id="test"))

        game.start_new_round()
        round1_starter = game.rounds[0].starter_player_index

        game.start_new_round()
        round2_starter = game.rounds[1].starter_player_index

        game.start_new_round()
        round3_starter = game.rounds[2].starter_player_index

        # Should rotate: if round1 was X, round2 is (X+1)%3, round3 is (X+2)%3
        assert round2_starter == (round1_starter + 1) % 3
        assert round3_starter == (round1_starter + 2) % 3


# =============================================================================
# BIDDING PHASE TESTS
# =============================================================================


class TestBiddingPhase:
    """Test bidding phase mechanics."""

    def test_bid_must_be_within_range(self):
        """Test that bids are validated within 0 to round_number."""
        round_obj = Round(number=5, starter_player_index=0)

        # Valid bids
        round_obj.add_bid("p0", 0)
        round_obj.add_bid("p1", 5)
        round_obj.add_bid("p2", 3)

        assert round_obj.bids["p0"] == 0
        assert round_obj.bids["p1"] == 5
        assert round_obj.bids["p2"] == 3

    def test_all_bids_placed_detection(self):
        """Test detection of all bids placed."""
        round_obj = Round(number=3, starter_player_index=0)

        assert not round_obj.all_bids_placed(3)

        round_obj.add_bid("p0", 1)
        assert not round_obj.all_bids_placed(3)

        round_obj.add_bid("p1", 2)
        assert not round_obj.all_bids_placed(3)

        round_obj.add_bid("p2", 0)
        assert round_obj.all_bids_placed(3)

    def test_player_can_bid_once(self):
        """Test that player's bid is recorded."""
        round_obj = Round(number=3, starter_player_index=0)

        assert not round_obj.has_player_bid("p0")

        round_obj.add_bid("p0", 2)
        assert round_obj.has_player_bid("p0")


# =============================================================================
# CARD PLAYING CONSTRAINTS TESTS
# =============================================================================


class TestCardPlayingConstraints:
    """Test card playing constraints and validation."""

    def test_can_only_play_cards_in_hand(self):
        """Test that players can only play cards from their hand."""
        trick = Trick(number=1, starter_player_index=0)
        hand = [CardId.PARROT5, CardId.MAP10]

        valid = trick.get_valid_cards(hand, [])

        # Can only play cards in hand
        assert CardId.PARROT5 in valid
        assert CardId.MAP10 in valid
        assert CardId.ROGER14 not in valid

    def test_empty_hand_returns_empty(self):
        """Test that empty hand returns no valid cards."""
        trick = Trick(number=1, starter_player_index=0)
        hand = []

        valid = trick.get_valid_cards(hand, [])

        assert valid == []

    def test_tigress_requires_choice(self):
        """Test that Tigress card requires pirate/escape choice."""
        trick = Trick(number=1, starter_player_index=0)

        # Add Tigress with choice
        trick.add_card("p0", CardId.TIGRESS, TigressChoice.PIRATE)

        assert trick.picked_cards[0].tigress_choice == TigressChoice.PIRATE

    def test_tigress_choice_affects_winner(self):
        """Test that Tigress choice affects winner determination."""
        # Tigress as pirate beats suit
        trick1 = Trick(number=1, starter_player_index=0)
        trick1.add_card("p0", CardId.TIGRESS, TigressChoice.PIRATE)
        trick1.add_card("p1", CardId.ROGER14)
        winner_card, _ = trick1.determine_winner()
        assert winner_card == CardId.TIGRESS

        # Tigress as escape loses to suit
        trick2 = Trick(number=1, starter_player_index=0)
        trick2.add_card("p0", CardId.TIGRESS, TigressChoice.ESCAPE)
        trick2.add_card("p1", CardId.PARROT1)
        winner_card, _ = trick2.determine_winner()
        assert winner_card == CardId.PARROT1


# =============================================================================
# MULTI-PLAYER GAME FLOW TESTS
# =============================================================================


class TestMultiPlayerGameFlow:
    """Test game flow with multiple players."""

    def test_three_player_trick(self):
        """Test a complete trick with 3 players."""
        trick = Trick(number=1, starter_player_index=0)

        trick.add_card("p0", CardId.PARROT5)
        trick.add_card("p1", CardId.PARROT14)
        trick.add_card("p2", CardId.PARROT10)

        assert trick.is_complete(3)
        winner_card, winner = trick.determine_winner()
        assert winner == "p1"
        assert winner_card == CardId.PARROT14

    def test_eight_player_trick(self):
        """Test a complete trick with 8 players (max)."""
        trick = Trick(number=1, starter_player_index=0)

        cards = [
            CardId.PARROT1,
            CardId.PARROT5,
            CardId.PARROT3,
            CardId.PARROT14,  # Winner
            CardId.PARROT7,
            CardId.PARROT2,
            CardId.PARROT9,
            CardId.PARROT11,
        ]

        for i, card in enumerate(cards):
            trick.add_card(f"p{i}", card)

        assert trick.is_complete(8)
        winner_card, winner = trick.determine_winner()
        assert winner == "p3"
        assert winner_card == CardId.PARROT14

    def test_full_round_three_players(self):
        """Test a complete round with 3 players, 3 tricks."""
        round_obj = Round(number=3, starter_player_index=0)
        round_obj.bids = {"p0": 1, "p1": 1, "p2": 1}

        # Play 3 tricks
        for trick_num in range(3):
            trick = Trick(number=trick_num + 1, starter_player_index=trick_num % 3)
            for i in range(3):
                card = CardId(11 + trick_num * 3 + i)  # Use different Roger cards
                trick.add_card(f"p{i}", card)
            trick.determine_winner()
            round_obj.tricks.append(trick)

        assert round_obj.is_complete()

    def test_scoring_at_round_end(self):
        """Test that scoring is calculated at round end."""
        round_obj = Round(number=3, starter_player_index=0)
        round_obj.bids = {"p0": 2, "p1": 1, "p2": 0}

        # p0 wins 2, p1 wins 1, p2 wins 0
        tricks_per_player = [2, 1, 0]
        trick_num = 0

        for player_idx, wins in enumerate(tricks_per_player):
            for _ in range(wins):
                trick = Trick(number=trick_num + 1, starter_player_index=0)
                trick.winner_player_id = f"p{player_idx}"
                round_obj.tricks.append(trick)
                trick_num += 1

        round_obj.calculate_scores()

        # p0: bid 2, won 2 = 20 * 2 = 40
        assert round_obj.scores["p0"] == 40
        # p1: bid 1, won 1 = 20 * 1 = 20
        assert round_obj.scores["p1"] == 20
        # p2: bid 0, won 0 = 10 * 3 = 30
        assert round_obj.scores["p2"] == 30


# =============================================================================
# EDGE CASES AND ERROR HANDLING
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_trick_no_winner(self):
        """Test that empty trick has no winner."""
        trick = Trick(number=1, starter_player_index=0)
        winner_card, winner = trick.determine_winner()

        assert winner_card is None
        assert winner is None

    def test_single_card_wins(self):
        """Test that single card in trick wins."""
        trick = Trick(number=1, starter_player_index=0)
        trick.add_card("p0", CardId.PARROT5)
        winner_card, winner = trick.determine_winner()

        assert winner_card == CardId.PARROT5
        assert winner == "p0"

    def test_all_same_card_number_first_wins(self):
        """Test that first card wins when all same number."""
        trick = Trick(number=1, starter_player_index=0)
        # All 14s of different suits
        trick.add_card("p0", CardId.PARROT14)
        trick.add_card("p1", CardId.MAP14)
        trick.add_card("p2", CardId.CHEST14)

        winner_card, winner = trick.determine_winner()

        # First played 14 wins (Parrot14)
        assert winner_card == CardId.PARROT14
        assert winner == "p0"

    def test_kraken_with_all_special_cards(self):
        """Test Kraken with only special cards."""
        trick = Trick(number=1, starter_player_index=0)
        trick.add_card("p0", CardId.SKULL_KING)
        trick.add_card("p1", CardId.PIRATE1)
        trick.add_card("p2", CardId.KRAKEN)
        trick.add_card("p3", CardId.MERMAID1)

        winner_card, _ = trick.determine_winner()

        # Kraken destroys everything
        assert winner_card is None

    def test_whale_with_only_specials(self):
        """Test Whale with only special cards results in no winner."""
        trick = Trick(number=1, starter_player_index=0)
        trick.add_card("p0", CardId.WHALE)
        trick.add_card("p1", CardId.PIRATE1)
        trick.add_card("p2", CardId.ESCAPE1)

        winner_card, _ = trick.determine_winner()

        # No suit cards, whale destroys specials
        assert winner_card is None


# =============================================================================
# LEADERBOARD AND GAME END TESTS
# =============================================================================


class TestLeaderboardAndGameEnd:
    """Test leaderboard and game end mechanics."""

    def test_leaderboard_sorted_by_score(self):
        """Test that leaderboard is sorted by score descending."""
        game = Game(id="test", slug="test")
        p1 = Player(id="p1", username="P1", avatar_id=1, index=0, game_id="test")
        p2 = Player(id="p2", username="P2", avatar_id=2, index=1, game_id="test")
        p3 = Player(id="p3", username="P3", avatar_id=3, index=2, game_id="test")

        p1.score = 50
        p2.score = 100
        p3.score = 75

        game.add_player(p1)
        game.add_player(p2)
        game.add_player(p3)

        leaderboard = game.get_leaderboard()

        assert leaderboard[0]["player_id"] == "p2"
        assert leaderboard[0]["score"] == 100
        assert leaderboard[1]["player_id"] == "p3"
        assert leaderboard[1]["score"] == 75
        assert leaderboard[2]["player_id"] == "p1"
        assert leaderboard[2]["score"] == 50

    def test_winner_highest_score(self):
        """Test that winner has highest score."""
        game = Game(id="test", slug="test")
        p1 = Player(id="p1", username="P1", avatar_id=1, index=0, game_id="test")
        p2 = Player(id="p2", username="P2", avatar_id=2, index=1, game_id="test")

        p1.score = 150
        p2.score = 200

        game.add_player(p1)
        game.add_player(p2)

        # Simulate game complete
        for _ in range(10):
            game.start_new_round()

        winner = game.get_winner()
        assert winner.id == "p2"

    def test_negative_scores_allowed(self):
        """Test that negative scores are handled correctly."""
        game = Game(id="test", slug="test")
        p1 = Player(id="p1", username="P1", avatar_id=1, index=0, game_id="test")
        p2 = Player(id="p2", username="P2", avatar_id=2, index=1, game_id="test")

        p1.score = -50
        p2.score = 30

        game.add_player(p1)
        game.add_player(p2)

        leaderboard = game.get_leaderboard()

        assert leaderboard[0]["player_id"] == "p2"
        assert leaderboard[1]["player_id"] == "p1"
        assert leaderboard[1]["score"] == -50
