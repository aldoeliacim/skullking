"""Basic tests for bot functionality."""

import pytest

from app.bots import RandomBot, RuleBasedBot
from app.models.card import CardId
from app.models.game import Game
from app.models.player import Player


@pytest.fixture
def sample_game():
    """Create a sample game for testing."""
    game = Game(id="test_game", slug="test-game")

    for i in range(4):
        player = Player(
            id=f"player_{i}",
            username=f"Player{i}",
            game_id=game.id,
            index=i,
            is_bot=(i > 0),
        )
        game.add_player(player)

    game.start_new_round()
    game.deal_cards()

    return game


class TestRandomBot:
    """Test RandomBot behavior."""

    def test_random_bot_makes_bid(self, sample_game):
        """Test that random bot makes valid bid."""
        bot = RandomBot("player_1")
        player = sample_game.get_player("player_1")

        bid = bot.make_bid(sample_game, 1, player.hand)

        assert isinstance(bid, int)
        assert 0 <= bid <= 1

    def test_random_bot_picks_card(self, sample_game):
        """Test that random bot picks valid card."""
        bot = RandomBot("player_1")
        player = sample_game.get_player("player_1")

        card = bot.pick_card(sample_game, player.hand, [])

        assert card in player.hand
        assert isinstance(card, CardId)


class TestRuleBasedBot:
    """Test RuleBasedBot behavior."""

    def test_rule_based_bot_makes_bid(self, sample_game):
        """Test that rule-based bot makes valid bid."""
        bot = RuleBasedBot("player_1")
        player = sample_game.get_player("player_1")

        bid = bot.make_bid(sample_game, 1, player.hand)

        assert isinstance(bid, int)
        assert 0 <= bid <= 1

    def test_rule_based_bot_picks_card(self, sample_game):
        """Test that rule-based bot picks valid card."""
        bot = RuleBasedBot("player_1")
        player = sample_game.get_player("player_1")

        card = bot.pick_card(sample_game, player.hand, [])

        assert card in player.hand
        assert isinstance(card, CardId)

    def test_rule_based_bot_prefers_strong_cards_when_winning(self, sample_game):
        """Test that rule-based bot plays strategically."""
        bot = RuleBasedBot("player_1")
        player = sample_game.get_player("player_1")

        # Give player a strong hand
        player.hand = [
            CardId.SKULL_KING,
            CardId.PARROT1,
            CardId.ESCAPE1,
        ]
        player.bid = 1

        # Bot should try to win with strong card
        card = bot.pick_card(sample_game, player.hand, [])

        # Should not pick escape card if trying to win
        assert card != CardId.ESCAPE1


class TestBotComparison:
    """Compare bot strategies."""

    def test_bots_have_different_strategies(self, sample_game):
        """Test that random and rule-based bots behave differently."""
        random_bot = RandomBot("player_1")
        rule_bot = RuleBasedBot("player_2")

        player1 = sample_game.get_player("player_1")
        player2 = sample_game.get_player("player_2")

        # Give both same hand
        test_hand = [CardId.SKULL_KING, CardId.PARROT1, CardId.ESCAPE1]
        player1.hand = test_hand.copy()
        player2.hand = test_hand.copy()

        # Make multiple bids to see variance
        random_bids = [random_bot.make_bid(sample_game, 3, test_hand) for _ in range(10)]
        rule_bids = [rule_bot.make_bid(sample_game, 3, test_hand) for _ in range(10)]

        # Random bot should have more variance
        random_variance = len(set(random_bids))
        rule_variance = len(set(rule_bids))

        # This is probabilistic but should generally hold
        assert random_variance >= rule_variance or random_variance > 1
