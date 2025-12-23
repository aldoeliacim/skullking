"""Tests for pirate abilities system.

Tests cover all 5 pirate abilities from the official rulebook:
1. Rosie de Laney - Choose who opens next trick
2. El bandido Bendt - Draw 2 cards, discard 2
3. Bribón de Roatán - Additional bet of 0/10/20 points
4. Juanita Jade - Look at undealt cards
5. Harry, el Gigante - Modify bid by +1/-1/0 at end of round
"""

from app.models.card import CardId
from app.models.pirate_ability import (
    PIRATE_ABILITY,
    PIRATE_IDENTITY,
    AbilityState,
    AbilityType,
    PirateType,
    get_card_ability,
    get_pirate_type,
)
from app.models.round import Round


class TestPirateIdentity:
    """Test pirate card to identity mapping."""

    def test_pirate1_is_rosie(self):
        """PIRATE1 should be Rosie de Laney."""
        assert get_pirate_type(CardId.PIRATE1) == PirateType.ROSIE

    def test_pirate2_is_bendt(self):
        """PIRATE2 should be El bandido Bendt."""
        assert get_pirate_type(CardId.PIRATE2) == PirateType.BENDT

    def test_pirate3_is_roatan(self):
        """PIRATE3 should be Bribón de Roatán."""
        assert get_pirate_type(CardId.PIRATE3) == PirateType.ROATAN

    def test_pirate4_is_jade(self):
        """PIRATE4 should be Juanita Jade."""
        assert get_pirate_type(CardId.PIRATE4) == PirateType.JADE

    def test_pirate5_is_harry(self):
        """PIRATE5 should be Harry, el Gigante."""
        assert get_pirate_type(CardId.PIRATE5) == PirateType.HARRY

    def test_non_pirate_returns_none(self):
        """Non-pirate cards should return None."""
        assert get_pirate_type(CardId.SKULL_KING) is None
        assert get_pirate_type(CardId.MERMAID1) is None
        assert get_pirate_type(CardId.ROGER14) is None
        assert get_pirate_type(CardId.ESCAPE1) is None

    def test_all_pirates_have_identity(self):
        """All 5 pirate cards should have an identity."""
        assert len(PIRATE_IDENTITY) == 5


class TestPirateAbilities:
    """Test pirate identity to ability mapping."""

    def test_rosie_has_choose_starter(self):
        """Rosie should have the choose_starter ability."""
        assert PIRATE_ABILITY[PirateType.ROSIE] == AbilityType.CHOOSE_STARTER

    def test_bendt_has_draw_discard(self):
        """Bendt should have the draw_discard ability."""
        assert PIRATE_ABILITY[PirateType.BENDT] == AbilityType.DRAW_DISCARD

    def test_roatan_has_extra_bet(self):
        """Roatán should have the extra_bet ability."""
        assert PIRATE_ABILITY[PirateType.ROATAN] == AbilityType.EXTRA_BET

    def test_jade_has_view_deck(self):
        """Jade should have the view_deck ability."""
        assert PIRATE_ABILITY[PirateType.JADE] == AbilityType.VIEW_DECK

    def test_harry_has_modify_bid(self):
        """Harry should have the modify_bid ability."""
        assert PIRATE_ABILITY[PirateType.HARRY] == AbilityType.MODIFY_BID

    def test_get_card_ability_for_pirates(self):
        """Get ability type directly from card ID."""
        assert get_card_ability(CardId.PIRATE1) == AbilityType.CHOOSE_STARTER
        assert get_card_ability(CardId.PIRATE2) == AbilityType.DRAW_DISCARD
        assert get_card_ability(CardId.PIRATE3) == AbilityType.EXTRA_BET
        assert get_card_ability(CardId.PIRATE4) == AbilityType.VIEW_DECK
        assert get_card_ability(CardId.PIRATE5) == AbilityType.MODIFY_BID

    def test_get_card_ability_for_non_pirates(self):
        """Non-pirates should return None for abilities."""
        assert get_card_ability(CardId.SKULL_KING) is None
        assert get_card_ability(CardId.MERMAID1) is None


class TestAbilityStateRosie:
    """Test Rosie's ability - choose who starts next trick."""

    def test_trigger_rosie_creates_pending(self):
        """Winning with Rosie should create pending ability."""
        state = AbilityState()
        pending = state.trigger_ability("player1", CardId.PIRATE1, 1)

        assert pending is not None
        assert pending.ability_type == AbilityType.CHOOSE_STARTER
        assert pending.player_id == "player1"
        assert pending.pirate_type == PirateType.ROSIE
        assert not pending.resolved

    def test_resolve_rosie_sets_next_starter(self):
        """Resolving Rosie should set the next trick starter."""
        state = AbilityState()
        state.trigger_ability("player1", CardId.PIRATE1, 1)

        result = state.resolve_rosie("player1", "player2")

        assert result is True
        assert state.rosie_next_starter == "player2"

    def test_resolve_rosie_without_pending_fails(self):
        """Cannot resolve Rosie without pending ability."""
        state = AbilityState()
        result = state.resolve_rosie("player1", "player2")
        assert result is False

    def test_clear_rosie_override(self):
        """Clearing Rosie override should reset starter."""
        state = AbilityState()
        state.trigger_ability("player1", CardId.PIRATE1, 1)
        state.resolve_rosie("player1", "player2")

        assert state.rosie_next_starter == "player2"
        state.clear_rosie_override()
        assert state.rosie_next_starter is None


class TestAbilityStateBendt:
    """Test Bendt's ability - draw 2, discard 2."""

    def test_trigger_bendt_creates_pending(self):
        """Winning with Bendt should create pending ability."""
        state = AbilityState()
        pending = state.trigger_ability("player1", CardId.PIRATE2, 1)

        assert pending is not None
        assert pending.ability_type == AbilityType.DRAW_DISCARD
        assert pending.pirate_type == PirateType.BENDT

    def test_resolve_bendt_with_correct_discard(self):
        """Resolving Bendt with 2 discards should succeed."""
        state = AbilityState()
        pending = state.trigger_ability("player1", CardId.PIRATE2, 1)
        drawn = [CardId.ROGER1, CardId.ROGER2]
        pending.drawn_cards = drawn

        result = state.resolve_bendt("player1", drawn, [CardId.PARROT1, CardId.PARROT2])

        assert result is True

    def test_resolve_bendt_wrong_discard_count_fails(self):
        """Resolving Bendt with wrong discard count should fail."""
        state = AbilityState()
        pending = state.trigger_ability("player1", CardId.PIRATE2, 1)
        drawn = [CardId.ROGER1, CardId.ROGER2]
        pending.drawn_cards = drawn

        # Only 1 discard when 2 drawn
        result = state.resolve_bendt("player1", drawn, [CardId.PARROT1])

        assert result is False

    def test_resolve_bendt_without_pending_fails(self):
        """Cannot resolve Bendt without pending ability."""
        state = AbilityState()
        result = state.resolve_bendt("player1", [], [])
        assert result is False


class TestAbilityStateRoatan:
    """Test Roatán's ability - extra bet of 0/10/20."""

    def test_trigger_roatan_creates_pending(self):
        """Winning with Roatán should create pending ability."""
        state = AbilityState()
        pending = state.trigger_ability("player1", CardId.PIRATE3, 1)

        assert pending is not None
        assert pending.ability_type == AbilityType.EXTRA_BET
        assert pending.pirate_type == PirateType.ROATAN

    def test_resolve_roatan_with_0(self):
        """Resolving Roatán with 0 extra bet should work."""
        state = AbilityState()
        state.trigger_ability("player1", CardId.PIRATE3, 1)

        result = state.resolve_roatan("player1", 0)

        assert result is True
        assert state.roatan_bets["player1"] == 0

    def test_resolve_roatan_with_10(self):
        """Resolving Roatán with 10 extra bet should work."""
        state = AbilityState()
        state.trigger_ability("player1", CardId.PIRATE3, 1)

        result = state.resolve_roatan("player1", 10)

        assert result is True
        assert state.roatan_bets["player1"] == 10

    def test_resolve_roatan_with_20(self):
        """Resolving Roatán with 20 extra bet should work."""
        state = AbilityState()
        state.trigger_ability("player1", CardId.PIRATE3, 1)

        result = state.resolve_roatan("player1", 20)

        assert result is True
        assert state.roatan_bets["player1"] == 20

    def test_resolve_roatan_invalid_amount_fails(self):
        """Resolving Roatán with invalid amount should fail."""
        state = AbilityState()
        state.trigger_ability("player1", CardId.PIRATE3, 1)

        assert state.resolve_roatan("player1", 5) is False
        assert state.resolve_roatan("player1", 15) is False
        assert state.resolve_roatan("player1", 30) is False

    def test_roatan_bonus_on_correct_bid(self):
        """Roatán bonus should add points when bid is correct."""
        state = AbilityState()
        state.trigger_ability("player1", CardId.PIRATE3, 1)
        state.resolve_roatan("player1", 20)

        bonus = state.get_roatan_bonus("player1", bid_correct=True)
        assert bonus == 20

    def test_roatan_penalty_on_wrong_bid(self):
        """Roatán bonus should subtract points when bid is wrong."""
        state = AbilityState()
        state.trigger_ability("player1", CardId.PIRATE3, 1)
        state.resolve_roatan("player1", 20)

        bonus = state.get_roatan_bonus("player1", bid_correct=False)
        assert bonus == -20

    def test_multiple_roatan_bets_stack(self):
        """Multiple Roatán wins should stack the extra bets."""
        state = AbilityState()

        # First win with Roatán
        state.trigger_ability("player1", CardId.PIRATE3, 1)
        state.resolve_roatan("player1", 10)

        # Second win with Roatán (different trick)
        state.trigger_ability("player1", CardId.PIRATE3, 2)
        state.resolve_roatan("player1", 20)

        # Total should be 30
        assert state.roatan_bets["player1"] == 30
        assert state.get_roatan_bonus("player1", bid_correct=True) == 30


class TestAbilityStateJade:
    """Test Jade's ability - view undealt cards."""

    def test_trigger_jade_creates_pending(self):
        """Winning with Jade should create pending ability."""
        state = AbilityState()
        pending = state.trigger_ability("player1", CardId.PIRATE4, 1)

        assert pending is not None
        assert pending.ability_type == AbilityType.VIEW_DECK
        assert pending.pirate_type == PirateType.JADE

    def test_resolve_jade_marks_resolved(self):
        """Resolving Jade should mark as resolved."""
        state = AbilityState()
        state.trigger_ability("player1", CardId.PIRATE4, 1)

        result = state.resolve_jade("player1")

        assert result is True
        pending = state.get_pending_ability("player1")
        assert pending is None  # No longer pending


class TestAbilityStateHarry:
    """Test Harry's ability - modify bid at end of round."""

    def test_trigger_harry_arms_ability(self):
        """Winning with Harry should arm the ability (not create pending)."""
        state = AbilityState()
        pending = state.trigger_ability("player1", CardId.PIRATE5, 1)

        # Harry doesn't create pending ability immediately
        assert pending is None
        # But is armed
        assert state.has_armed_harry("player1") is True

    def test_resolve_harry_with_plus_one(self):
        """Resolving Harry with +1 should work."""
        state = AbilityState()
        state.trigger_ability("player1", CardId.PIRATE5, 1)

        result = state.resolve_harry("player1", 1)

        assert result is True
        assert state.get_harry_modifier("player1") == 1

    def test_resolve_harry_with_minus_one(self):
        """Resolving Harry with -1 should work."""
        state = AbilityState()
        state.trigger_ability("player1", CardId.PIRATE5, 1)

        result = state.resolve_harry("player1", -1)

        assert result is True
        assert state.get_harry_modifier("player1") == -1

    def test_resolve_harry_with_zero(self):
        """Resolving Harry with 0 (no change) should work."""
        state = AbilityState()
        state.trigger_ability("player1", CardId.PIRATE5, 1)

        result = state.resolve_harry("player1", 0)

        assert result is True
        assert state.get_harry_modifier("player1") == 0

    def test_resolve_harry_invalid_modifier_fails(self):
        """Resolving Harry with invalid modifier should fail."""
        state = AbilityState()
        state.trigger_ability("player1", CardId.PIRATE5, 1)

        assert state.resolve_harry("player1", 2) is False
        assert state.resolve_harry("player1", -2) is False
        assert state.resolve_harry("player1", 5) is False

    def test_resolve_harry_without_armed_fails(self):
        """Cannot resolve Harry without winning with him."""
        state = AbilityState()
        result = state.resolve_harry("player1", 1)
        assert result is False

    def test_harry_disarms_after_resolve(self):
        """Harry ability should be disarmed after resolution."""
        state = AbilityState()
        state.trigger_ability("player1", CardId.PIRATE5, 1)
        state.resolve_harry("player1", 0)

        assert state.has_armed_harry("player1") is False


class TestRoundWithAbilities:
    """Test Round integration with pirate abilities."""

    def test_round_has_ability_state(self):
        """Round should have an ability state."""
        round_obj = Round(number=3, starter_player_index=0)
        assert round_obj.ability_state is not None

    def test_get_next_trick_starter_default(self):
        """Without Rosie override, default starter is used."""
        round_obj = Round(number=3, starter_player_index=0)
        starter = round_obj.get_next_trick_starter("player1")
        assert starter == "player1"

    def test_get_next_trick_starter_with_rosie(self):
        """With Rosie override, chosen player starts."""
        round_obj = Round(number=3, starter_player_index=0)
        round_obj.ability_state.trigger_ability("player1", CardId.PIRATE1, 1)
        round_obj.ability_state.resolve_rosie("player1", "player3")

        starter = round_obj.get_next_trick_starter("player1")

        # Rosie chose player3
        assert starter == "player3"
        # Override is cleared after use
        assert round_obj.ability_state.rosie_next_starter is None


class TestScoringWithAbilities:
    """Test scoring system with pirate abilities."""

    def test_harry_modifier_affects_scoring(self):
        """Harry's bid modifier should affect final scoring."""
        round_obj = Round(number=3, starter_player_index=0)
        round_obj.bids = {"player1": 2}  # Bid 2

        # Player won with Harry and will modify bid
        round_obj.ability_state.trigger_ability("player1", CardId.PIRATE5, 1)
        round_obj.ability_state.resolve_harry("player1", 1)  # +1 to bid

        # Simulate 3 tricks won - matches 2+1=3
        from app.models.trick import Trick

        for i in range(3):
            trick = Trick(number=i + 1, starter_player_index=0)
            trick.winner_player_id = "player1"
            round_obj.tricks.append(trick)

        round_obj.calculate_scores()

        # Effective bid is 3, won 3, so score = 20*3 = 60
        assert round_obj.scores["player1"] == 60

    def test_harry_modifier_negative(self):
        """Harry's -1 modifier should reduce effective bid."""
        round_obj = Round(number=3, starter_player_index=0)
        round_obj.bids = {"player1": 2}  # Bid 2

        round_obj.ability_state.trigger_ability("player1", CardId.PIRATE5, 1)
        round_obj.ability_state.resolve_harry("player1", -1)  # -1 to bid

        # Simulate 1 trick won - matches 2-1=1
        from app.models.trick import Trick

        trick = Trick(number=1, starter_player_index=0)
        trick.winner_player_id = "player1"
        round_obj.tricks.append(trick)

        round_obj.calculate_scores()

        # Effective bid is 1, won 1, so score = 20*1 = 20
        assert round_obj.scores["player1"] == 20

    def test_harry_cannot_make_bid_negative(self):
        """Harry's modifier cannot make bid go below 0."""
        round_obj = Round(number=1, starter_player_index=0)
        round_obj.bids = {"player1": 0}  # Bid 0

        round_obj.ability_state.trigger_ability("player1", CardId.PIRATE5, 1)
        round_obj.ability_state.resolve_harry("player1", -1)  # -1 to bid

        # Won 0 tricks
        from app.models.trick import Trick

        trick = Trick(number=1, starter_player_index=0)
        trick.winner_player_id = "player2"  # Someone else won
        round_obj.tricks.append(trick)

        round_obj.calculate_scores()

        # Effective bid is max(0, 0-1) = 0, won 0, score = 10 * round_number = 10
        assert round_obj.scores["player1"] == 10

    def test_roatan_bonus_on_correct_bid(self):
        """Roatán's extra bet adds to score on correct bid."""
        round_obj = Round(number=2, starter_player_index=0)
        round_obj.bids = {"player1": 2}

        # Win with Roatán and bet 20 extra
        round_obj.ability_state.trigger_ability("player1", CardId.PIRATE3, 1)
        round_obj.ability_state.resolve_roatan("player1", 20)

        # Simulate 2 tricks won
        from app.models.trick import Trick

        for i in range(2):
            trick = Trick(number=i + 1, starter_player_index=0)
            trick.winner_player_id = "player1"
            round_obj.tricks.append(trick)

        round_obj.calculate_scores()

        # Base: 20*2 = 40, Roatán bonus: +20, total: 60
        assert round_obj.scores["player1"] == 60

    def test_roatan_penalty_on_wrong_bid(self):
        """Roatán's extra bet subtracts from score on wrong bid."""
        round_obj = Round(number=2, starter_player_index=0)
        round_obj.bids = {"player1": 2}

        # Win with Roatán and bet 20 extra
        round_obj.ability_state.trigger_ability("player1", CardId.PIRATE3, 1)
        round_obj.ability_state.resolve_roatan("player1", 20)

        # Simulate 1 trick won (wrong bid)
        from app.models.trick import Trick

        trick = Trick(number=1, starter_player_index=0)
        trick.winner_player_id = "player1"
        round_obj.tricks.append(trick)
        trick2 = Trick(number=2, starter_player_index=0)
        trick2.winner_player_id = "player2"
        round_obj.tricks.append(trick2)

        round_obj.calculate_scores()

        # Base: -10*1 = -10, Roatán penalty: -20, total: -30
        assert round_obj.scores["player1"] == -30

    def test_combined_harry_and_roatan(self):
        """Harry modifier and Roatán bonus can combine."""
        round_obj = Round(number=3, starter_player_index=0)
        round_obj.bids = {"player1": 2}

        # Win with Harry and Roatán
        round_obj.ability_state.trigger_ability("player1", CardId.PIRATE5, 1)
        round_obj.ability_state.resolve_harry("player1", 1)  # Effective bid = 3
        round_obj.ability_state.trigger_ability("player1", CardId.PIRATE3, 2)
        round_obj.ability_state.resolve_roatan("player1", 10)

        # Simulate 3 tricks won
        from app.models.trick import Trick

        for i in range(3):
            trick = Trick(number=i + 1, starter_player_index=0)
            trick.winner_player_id = "player1"
            round_obj.tricks.append(trick)

        round_obj.calculate_scores()

        # Base: 20*3 = 60, Roatán bonus: +10, total: 70
        assert round_obj.scores["player1"] == 70


class TestPendingAbilityTracking:
    """Test ability tracking and resolution."""

    def test_get_pending_ability(self):
        """Can retrieve pending ability for a player."""
        state = AbilityState()
        state.trigger_ability("player1", CardId.PIRATE1, 1)

        pending = state.get_pending_ability("player1")

        assert pending is not None
        assert pending.player_id == "player1"

    def test_has_pending_abilities_true(self):
        """Check if there are pending abilities."""
        state = AbilityState()
        state.trigger_ability("player1", CardId.PIRATE1, 1)

        assert state.has_pending_abilities() is True
        assert state.has_pending_abilities("player1") is True

    def test_has_pending_abilities_false(self):
        """No pending abilities when resolved."""
        state = AbilityState()
        state.trigger_ability("player1", CardId.PIRATE1, 1)
        state.resolve_rosie("player1", "player2")

        assert state.has_pending_abilities("player1") is False

    def test_has_pending_abilities_wrong_player(self):
        """Check pending abilities for wrong player."""
        state = AbilityState()
        state.trigger_ability("player1", CardId.PIRATE1, 1)

        assert state.has_pending_abilities("player2") is False

    def test_multiple_pending_abilities(self):
        """Can track multiple pending abilities."""
        state = AbilityState()
        state.trigger_ability("player1", CardId.PIRATE1, 1)  # Rosie
        state.trigger_ability("player2", CardId.PIRATE2, 2)  # Bendt

        assert state.has_pending_abilities("player1") is True
        assert state.has_pending_abilities("player2") is True

        # Resolve player1's ability
        state.resolve_rosie("player1", "player3")

        assert state.has_pending_abilities("player1") is False
        assert state.has_pending_abilities("player2") is True
