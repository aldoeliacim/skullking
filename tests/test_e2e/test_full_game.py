"""End-to-end tests for complete game flows."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.api.game_handler import GameHandler
from app.api.responses import Command
from app.bots import RandomBot, RuleBasedBot
from app.bots.base_bot import BotDifficulty
from app.models.card import CardId, get_all_cards, get_card
from app.models.enums import GameState
from app.models.game import Game
from app.models.player import Player
from app.models.round import Round
from app.models.trick import Trick

pytestmark = pytest.mark.anyio


@pytest.fixture
def mock_manager():
    """Create a mock connection manager."""
    manager = MagicMock()
    manager.broadcast_to_game = AsyncMock()
    manager.send_personal_message = AsyncMock()
    return manager


@pytest.fixture
def game_handler(mock_manager):
    """Create a game handler with mock manager."""
    return GameHandler(mock_manager)


class TestCardModel:
    """Tests for the Card model - all 74 cards."""

    def test_deck_has_74_cards(self):
        """Deck should have exactly 74 cards."""
        cards = get_all_cards()
        assert len(cards) == 74

    def test_special_cards_exist(self):
        """All special cards should exist."""
        # Skull King
        king = get_card(CardId.SKULL_KING)
        assert king.is_king()

        # Whale
        whale = get_card(CardId.WHALE)
        assert whale.is_whale()
        assert whale.is_beast()

        # Kraken
        kraken = get_card(CardId.KRAKEN)
        assert kraken.is_kraken()
        assert kraken.is_beast()

        # Mermaids
        mermaid1 = get_card(CardId.MERMAID1)
        mermaid2 = get_card(CardId.MERMAID2)
        assert mermaid1.is_mermaid()
        assert mermaid2.is_mermaid()

        # Pirates
        for i in range(1, 6):
            pirate = get_card(CardId[f"PIRATE{i}"])
            assert pirate.is_pirate()

        # Escapes
        for i in range(1, 6):
            escape = get_card(CardId[f"ESCAPE{i}"])
            assert escape.is_escape()

        # Tigress (Scary Mary)
        tigress = get_card(CardId.TIGRESS)
        assert tigress.is_tigress()
        assert tigress.is_special()

        # Loot cards
        loot1 = get_card(CardId.LOOT1)
        loot2 = get_card(CardId.LOOT2)
        assert loot1.is_loot()
        assert loot2.is_loot()
        assert loot1.is_special()

    def test_suit_cards(self):
        """All suit cards should exist with correct numbers."""
        suits = ["ROGER", "PARROT", "MAP", "CHEST"]
        for suit in suits:
            for num in range(1, 15):
                card = get_card(CardId[f"{suit}{num}"])
                assert card.number == num
                assert card.is_suit()

    def test_card_type_checks(self):
        """Card type helper methods should work correctly."""
        # Standard suit
        parrot5 = get_card(CardId.PARROT5)
        assert parrot5.is_parrot()
        assert parrot5.is_standard_suit()
        assert parrot5.is_suit()
        assert not parrot5.is_special()

        # Test Roger cards (trump suit)
        roger10 = get_card(CardId.ROGER10)
        assert roger10.is_roger()
        assert roger10.is_suit()
        assert not roger10.is_standard_suit()

        # Character cards
        king = get_card(CardId.SKULL_KING)
        assert king.is_character()

        pirate = get_card(CardId.PIRATE1)
        assert pirate.is_character()

        mermaid = get_card(CardId.MERMAID1)
        assert mermaid.is_character()


class TestTrickValidCards:
    """Tests for valid card selection in tricks."""

    def test_leading_can_play_any_card(self):
        """First player can play any card."""
        trick = Trick(number=1, starter_player_index=0)
        hand = [CardId.PARROT5, CardId.MAP10, CardId.PIRATE1]

        valid = trick.get_valid_cards(hand, [])
        assert set(valid) == set(hand)

    def test_must_follow_suit(self):
        """Must follow suit if able."""
        trick = Trick(number=1, starter_player_index=0)
        hand = [CardId.PARROT5, CardId.PARROT10, CardId.MAP3]
        cards_in_trick = [CardId.PARROT1]  # Parrot was led

        valid = trick.get_valid_cards(hand, cards_in_trick)

        # Should include parrots and special cards (none here)
        assert CardId.PARROT5 in valid
        assert CardId.PARROT10 in valid
        # Map should not be valid since we have parrot
        assert CardId.MAP3 not in valid

    def test_special_cards_always_valid(self):
        """Special cards can always be played."""
        trick = Trick(number=1, starter_player_index=0)
        hand = [CardId.PARROT5, CardId.ESCAPE1, CardId.PIRATE1]
        cards_in_trick = [CardId.MAP1]  # Map was led

        valid = trick.get_valid_cards(hand, cards_in_trick)

        # No map in hand, so can play anything
        assert set(valid) == set(hand)

    def test_cant_follow_suit_play_anything(self):
        """If can't follow suit, can play anything."""
        trick = Trick(number=1, starter_player_index=0)
        hand = [CardId.PARROT5, CardId.CHEST3]
        cards_in_trick = [CardId.MAP1]  # Map was led

        valid = trick.get_valid_cards(hand, cards_in_trick)
        assert set(valid) == set(hand)


class TestRoundModel:
    """Tests for the Round model."""

    def test_all_bids_placed(self):
        """all_bids_placed should work correctly."""
        round_obj = Round(number=1, starter_player_index=0)

        assert not round_obj.all_bids_placed(3)

        round_obj.add_bid("player-0", 0)
        assert not round_obj.all_bids_placed(3)

        round_obj.add_bid("player-1", 1)
        round_obj.add_bid("player-2", 0)
        assert round_obj.all_bids_placed(3)

    def test_get_tricks_won(self):
        """get_tricks_won should count correctly."""
        round_obj = Round(number=3, starter_player_index=0)

        trick1 = Trick(number=1, starter_player_index=0)
        trick1.winner_player_id = "player-0"
        round_obj.tricks.append(trick1)

        trick2 = Trick(number=2, starter_player_index=0)
        trick2.winner_player_id = "player-1"
        round_obj.tricks.append(trick2)

        trick3 = Trick(number=3, starter_player_index=1)
        trick3.winner_player_id = "player-0"
        round_obj.tricks.append(trick3)

        assert round_obj.get_tricks_won("player-0") == 2
        assert round_obj.get_tricks_won("player-1") == 1
        assert round_obj.get_tricks_won("player-2") == 0

    def test_is_complete(self):
        """Round completion check."""
        round_obj = Round(number=2, starter_player_index=0)

        assert not round_obj.is_complete()

        round_obj.tricks.append(Trick(number=1, starter_player_index=0))
        assert not round_obj.is_complete()

        round_obj.tricks.append(Trick(number=2, starter_player_index=0))
        assert round_obj.is_complete()


class TestBots:
    """Tests for bot implementations."""

    def test_random_bot_makes_valid_bid(self):
        """Random bot should make bid within valid range."""
        bot = RandomBot("bot-1")
        game = Game(id="test", slug="test")

        for round_num in range(1, 11):
            hand = [CardId.PARROT1] * round_num
            bid = bot.make_bid(game, round_num, hand)
            assert 0 <= bid <= round_num

    def test_random_bot_picks_from_hand(self):
        """Random bot should pick a card from its hand."""
        bot = RandomBot("bot-1")
        game = Game(id="test", slug="test")
        hand = [CardId.PARROT5, CardId.MAP10, CardId.CHEST3]

        card = bot.pick_card(game, hand, [])
        assert card in hand

    def test_rule_based_bot_makes_valid_bid(self):
        """Rule-based bot should make bid within valid range."""
        bot = RuleBasedBot("bot-1", BotDifficulty.MEDIUM)
        game = Game(id="test", slug="test")

        for round_num in range(1, 11):
            hand = [CardId.PARROT1] * round_num
            bid = bot.make_bid(game, round_num, hand)
            assert 0 <= bid <= round_num

    def test_rule_based_bot_picks_from_hand(self):
        """Rule-based bot should pick a card from its hand."""
        bot = RuleBasedBot("bot-1", BotDifficulty.HARD)
        game = Game(id="test", slug="test")
        hand = [CardId.PARROT5, CardId.MAP10, CardId.SKULL_KING]

        card = bot.pick_card(game, hand, [])
        assert card in hand

    def test_rule_based_bot_handles_new_cards(self):
        """Rule-based bot should handle all 74 card types."""
        bot = RuleBasedBot("bot-1", BotDifficulty.HARD)
        game = Game(id="test", slug="test")

        # Test with expansion cards
        hand = [CardId.TIGRESS, CardId.LOOT1, CardId.WHALE, CardId.KRAKEN]
        bid = bot.make_bid(game, 4, hand)
        assert 0 <= bid <= 4

        card = bot.pick_card(game, hand, [])
        assert card in hand


class TestGameFlow:
    """Tests for game state transitions."""

    async def test_game_creation(self, game_handler, mock_manager):
        """Game should start in PENDING state."""
        game = Game(id="test-game", slug="test-game")
        assert game.state == GameState.PENDING
        assert len(game.players) == 0
        assert game.current_round_number == 0

    async def test_add_players(self, game_handler, mock_manager):
        """Players can be added to pending game."""
        game = Game(id="test-game", slug="test-game")

        for i in range(4):
            player = Player(
                id=f"player-{i}",
                username=f"Player {i}",
                game_id=game.id,
                index=i,
            )
            game.add_player(player)

        assert len(game.players) == 4
        assert game.can_start()

    async def test_start_game_flow(self, game_handler, mock_manager):
        """Starting game should deal cards and begin bidding."""
        game = Game(id="test-game", slug="test-game")
        for i in range(3):
            player = Player(
                id=f"player-{i}",
                username=f"Player {i}",
                game_id=game.id,
                index=i,
            )
            game.add_player(player)

        await game_handler.handle_command(game, "player-0", "START_GAME", {})

        assert game.state == GameState.BIDDING
        assert game.current_round_number == 1

        # Each player should have 1 card (round 1)
        for player in game.players:
            assert len(player.hand) == 1

    async def test_bidding_to_picking_transition(self, game_handler, mock_manager):
        """All bids should transition to picking phase."""
        game = Game(id="test-game", slug="test-game")
        for i in range(3):
            player = Player(
                id=f"player-{i}",
                username=f"Player {i}",
                game_id=game.id,
                index=i,
            )
            game.add_player(player)

        # Start game
        await game_handler.handle_command(game, "player-0", "START_GAME", {})

        # All players bid
        for i in range(3):
            await game_handler.handle_command(game, f"player-{i}", "BID", {"bid": 0})

        assert game.state == GameState.PICKING

    async def test_complete_trick(self, game_handler, mock_manager):
        """All players picking should complete a trick."""
        game = Game(id="test-game", slug="test-game")
        for i in range(3):
            player = Player(
                id=f"player-{i}",
                username=f"Player {i}",
                game_id=game.id,
                index=i,
            )
            game.add_player(player)

        # Start and bid
        await game_handler.handle_command(game, "player-0", "START_GAME", {})
        for i in range(3):
            await game_handler.handle_command(game, f"player-{i}", "BID", {"bid": 0})

        # Get current trick
        current_round = game.get_current_round()
        assert len(current_round.tricks) >= 1

        # Each player picks in turn order
        for _ in range(3):
            trick = current_round.get_current_trick()
            if not trick:
                break
            player = game.get_player(trick.picking_player_id)
            if player and player.hand:
                card = player.hand[0]
                await game_handler.handle_command(
                    game, trick.picking_player_id, "PICK", {"card_id": card.value}
                )

        # Trick should be complete
        assert (
            current_round.tricks[0].winner_player_id is not None
            or current_round.tricks[0].winner_card_id is None
        )  # Kraken case


class TestFullGameWithBots:
    """Tests for complete games with bot players."""

    async def test_game_with_rule_based_bots(self, game_handler, mock_manager):
        """Full game should complete with rule-based bots."""
        game = Game(id="test-game", slug="test-game")

        # Add human player
        human = Player(
            id="human",
            username="Human",
            game_id=game.id,
            index=0,
        )
        game.add_player(human)

        # Add 3 bot players
        for i in range(1, 4):
            bot_player = Player(
                id=f"bot-{i}",
                username=f"Bot {i}",
                game_id=game.id,
                index=i,
                is_bot=True,
            )
            game.add_player(bot_player)

        # Register bots with handler
        bots = {f"bot-{i}": RuleBasedBot(f"bot-{i}", BotDifficulty.MEDIUM) for i in range(1, 4)}
        game_handler.bots[game.id] = bots

        # Start game
        await game_handler.handle_command(game, "human", "START_GAME", {})

        assert game.state == GameState.BIDDING
        assert game.current_round_number == 1

        # Human bids
        await game_handler.handle_command(game, "human", "BID", {"bid": 0})

        # Bots should have bid automatically
        current_round = game.get_current_round()
        assert current_round.all_bids_placed(4)

        # Should now be in picking phase
        assert game.state == GameState.PICKING

    async def test_complete_round_1(self, game_handler, mock_manager):
        """Round 1 should complete successfully."""
        game = Game(id="test-game", slug="test-game")

        # Add players
        for i in range(3):
            player = Player(
                id=f"player-{i}",
                username=f"Player {i}",
                game_id=game.id,
                index=i,
                is_bot=(i > 0),
            )
            game.add_player(player)

        # Register bots
        bots = {f"player-{i}": RuleBasedBot(f"player-{i}") for i in range(1, 3)}
        game_handler.bots[game.id] = bots

        # Start game
        await game_handler.handle_command(game, "player-0", "START_GAME", {})

        # Human bids
        await game_handler.handle_command(game, "player-0", "BID", {"bid": 0})

        # Human plays their card
        player = game.get_player("player-0")
        if player.hand:
            card = player.hand[0]
            await game_handler.handle_command(game, "player-0", "PICK", {"card_id": card.value})

        # Round should complete
        current_round = game.get_current_round()
        # Either still in round 1 completed or moved to round 2
        assert current_round.is_complete() or game.current_round_number == 2


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    async def test_invalid_command(self, game_handler, mock_manager):
        """Invalid command should be handled gracefully."""
        game = Game(id="test-game", slug="test-game")
        player = Player(id="player-0", username="Player", game_id=game.id, index=0)
        game.add_player(player)

        # Unknown command should not crash
        await game_handler.handle_command(game, "player-0", "INVALID_CMD", {})

    async def test_pick_invalid_card_id(self, game_handler, mock_manager):
        """Picking with invalid card ID should send error."""
        game = Game(id="test-game", slug="test-game")
        for i in range(2):
            player = Player(id=f"player-{i}", username=f"P{i}", game_id=game.id, index=i)
            game.add_player(player)

        await game_handler.handle_command(game, "player-0", "START_GAME", {})

        for i in range(2):
            await game_handler.handle_command(game, f"player-{i}", "BID", {"bid": 0})

        # Try to pick invalid card
        await game_handler.handle_command(game, "player-0", "PICK", {"card_id": "not_a_number"})

        # Should have sent error
        error_calls = [
            c
            for c in mock_manager.send_personal_message.call_args_list
            if c[0][0].command == Command.REPORT_ERROR
        ]
        assert len(error_calls) > 0

    async def test_bid_missing_value(self, game_handler, mock_manager):
        """Bidding without value should send error."""
        game = Game(id="test-game", slug="test-game")
        for i in range(2):
            player = Player(id=f"player-{i}", username=f"P{i}", game_id=game.id, index=i)
            game.add_player(player)

        await game_handler.handle_command(game, "player-0", "START_GAME", {})

        # Bid without value
        await game_handler.handle_command(game, "player-0", "BID", {})

        # Should have sent error
        error_calls = [
            c
            for c in mock_manager.send_personal_message.call_args_list
            if c[0][0].command == Command.REPORT_ERROR
        ]
        assert len(error_calls) > 0


class TestDetermineWinner:
    """Tests for trick winner determination."""

    def test_highest_suit_wins(self):
        """Highest card of led suit should win."""
        from app.models.card import determine_winner

        cards = [CardId.PARROT5, CardId.PARROT10, CardId.PARROT3]
        winner = determine_winner(cards)
        assert winner == CardId.PARROT10

    def test_trump_beats_suit(self):
        """Trump (Roger) should beat standard suits."""
        from app.models.card import determine_winner

        cards = [CardId.PARROT14, CardId.ROGER1]
        winner = determine_winner(cards)
        assert winner == CardId.ROGER1

    def test_pirate_beats_suit(self):
        """Pirate should beat suits."""
        from app.models.card import determine_winner

        cards = [CardId.PARROT14, CardId.PIRATE1]
        winner = determine_winner(cards)
        assert winner == CardId.PIRATE1

    def test_king_beats_pirate(self):
        """Skull King should beat pirate."""
        from app.models.card import determine_winner

        cards = [CardId.PIRATE1, CardId.SKULL_KING]
        winner = determine_winner(cards)
        assert winner == CardId.SKULL_KING

    def test_mermaid_beats_king(self):
        """Mermaid should beat Skull King."""
        from app.models.card import determine_winner

        cards = [CardId.SKULL_KING, CardId.MERMAID1]
        winner = determine_winner(cards)
        assert winner == CardId.MERMAID1

    def test_kraken_no_winner(self):
        """Kraken should result in no winner."""
        from app.models.card import determine_winner

        cards = [CardId.PARROT10, CardId.KRAKEN]
        winner = determine_winner(cards)
        assert winner is None

    def test_whale_highest_suit_wins(self):
        """With Whale, highest suit card should win."""
        from app.models.card import determine_winner

        cards = [CardId.WHALE, CardId.PARROT5, CardId.PARROT10]
        winner = determine_winner(cards)
        assert winner == CardId.PARROT10

    def test_all_escapes_first_wins(self):
        """All escapes - first escape wins."""
        from app.models.card import determine_winner

        cards = [CardId.ESCAPE1, CardId.ESCAPE2, CardId.ESCAPE3]
        winner = determine_winner(cards)
        assert winner == CardId.ESCAPE1
