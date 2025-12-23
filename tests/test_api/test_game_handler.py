"""Tests for WebSocket game handler."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.api.game_handler import GameHandler
from app.api.responses import Command
from app.models.enums import GameState
from app.models.game import Game
from app.models.player import Player
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


@pytest.fixture
def game_with_players():
    """Create a game with 3 players."""
    game = Game(id="test-game-123", slug="test-game")
    for i in range(3):
        player = Player(
            id=f"player-{i}",
            username=f"Player {i}",
            avatar_id=i,
            index=i,
            game_id=game.id,
        )
        game.add_player(player)
    return game


class TestStartGame:
    """Tests for START_GAME command."""

    async def test_start_game_success(self, game_handler, game_with_players, mock_manager):
        """Starting a valid game should broadcast STARTED and deal cards."""
        game = game_with_players
        assert game.state == GameState.PENDING

        await game_handler.handle_command(game, "player-0", "START_GAME", {})

        # Game state should transition
        assert game.state == GameState.BIDDING
        assert game.current_round_number == 1

        # STARTED should be broadcast
        calls = mock_manager.broadcast_to_game.call_args_list
        assert any(call[0][0].command == Command.STARTED for call in calls)

        # START_BIDDING should be broadcast
        assert any(call[0][0].command == Command.START_BIDDING for call in calls)

        # DEAL should be sent to each player
        deal_calls = [
            call
            for call in mock_manager.send_personal_message.call_args_list
            if call[0][0].command == Command.DEAL
        ]
        assert len(deal_calls) == 3  # One per player

    async def test_start_game_already_started(self, game_handler, game_with_players, mock_manager):
        """Starting an already-started game should send error."""
        game = game_with_players
        game.state = GameState.BIDDING

        await game_handler.handle_command(game, "player-0", "START_GAME", {})

        # Error should be sent to player
        error_call = mock_manager.send_personal_message.call_args
        assert error_call[0][0].command == Command.REPORT_ERROR
        assert "already started" in error_call[0][0].content["error"]

    async def test_start_game_not_enough_players(self, game_handler, mock_manager):
        """Starting with 1 player should send error."""
        game = Game(id="test-game", slug="test-game")
        game.add_player(
            Player(id="player-0", username="Player 0", avatar_id=0, game_id="test-game")
        )

        await game_handler.handle_command(game, "player-0", "START_GAME", {})

        error_call = mock_manager.send_personal_message.call_args
        assert error_call[0][0].command == Command.REPORT_ERROR
        assert "Not enough players" in error_call[0][0].content["error"]


class TestBidding:
    """Tests for BID command."""

    async def test_bid_success(self, game_handler, game_with_players, mock_manager):
        """Valid bid should be recorded and broadcast."""
        game = game_with_players
        game.state = GameState.BIDDING
        game.start_new_round()

        await game_handler.handle_command(game, "player-0", "BID", {"bid": 1})

        # Bid should be recorded
        player = game.get_player("player-0")
        assert player.bid == 1

        # BADE should be broadcast
        bade_call = mock_manager.broadcast_to_game.call_args
        assert bade_call[0][0].command == Command.BADE
        assert bade_call[0][0].content["player_id"] == "player-0"
        assert bade_call[0][0].content["bid"] == 1

    async def test_bid_invalid_amount(self, game_handler, game_with_players, mock_manager):
        """Bid exceeding round number should send error."""
        game = game_with_players
        game.state = GameState.BIDDING
        game.start_new_round()  # Round 1 = max bid of 1

        await game_handler.handle_command(game, "player-0", "BID", {"bid": 5})

        error_call = mock_manager.send_personal_message.call_args
        assert error_call[0][0].command == Command.REPORT_ERROR
        assert "Invalid bid" in error_call[0][0].content["error"]

    async def test_bid_not_in_bidding_phase(self, game_handler, game_with_players, mock_manager):
        """Bid during wrong phase should send error."""
        game = game_with_players
        game.state = GameState.PICKING

        await game_handler.handle_command(game, "player-0", "BID", {"bid": 1})

        error_call = mock_manager.send_personal_message.call_args
        assert error_call[0][0].command == Command.REPORT_ERROR
        assert "Not in bidding phase" in error_call[0][0].content["error"]

    async def test_bid_already_placed(self, game_handler, game_with_players, mock_manager):
        """Double bid should send error."""
        game = game_with_players
        game.state = GameState.BIDDING
        game.start_new_round()
        player = game.get_player("player-0")
        player.bid = 0

        await game_handler.handle_command(game, "player-0", "BID", {"bid": 1})

        error_call = mock_manager.send_personal_message.call_args
        assert error_call[0][0].command == Command.REPORT_ERROR
        assert "Already placed bid" in error_call[0][0].content["error"]

    async def test_all_bids_triggers_picking(self, game_handler, game_with_players, mock_manager):
        """All players bidding should trigger picking phase."""
        game = game_with_players
        game.state = GameState.BIDDING
        game.start_new_round()
        game.deal_cards()

        # All players bid
        for i in range(3):
            await game_handler.handle_command(game, f"player-{i}", "BID", {"bid": 0})

        # Game should now be in picking phase
        assert game.state == GameState.PICKING

        # START_PICKING should be broadcast
        calls = mock_manager.broadcast_to_game.call_args_list
        start_picking = [c for c in calls if c[0][0].command == Command.START_PICKING]
        assert len(start_picking) == 1


class TestPicking:
    """Tests for PICK command."""

    async def test_pick_success(self, game_handler, game_with_players, mock_manager):
        """Valid card pick should be recorded and broadcast."""
        game = game_with_players
        game.state = GameState.BIDDING
        game.start_new_round()
        game.deal_cards()

        # All players bid
        for i in range(3):
            player = game.get_player(f"player-{i}")
            player.bid = 0

        # Set up picking phase
        game.state = GameState.PICKING
        current_round = game.get_current_round()
        trick = Trick(number=1, starter_player_index=0, picking_player_id="player-0")
        current_round.tricks.append(trick)

        # Player 0 picks their first card
        player = game.get_player("player-0")
        card_to_play = player.hand[0]

        await game_handler.handle_command(game, "player-0", "PICK", {"card_id": card_to_play.value})

        # Card should be removed from hand
        assert card_to_play not in player.hand

        # PICKED should be broadcast (might be followed by NEXT_TRICK)
        calls = mock_manager.broadcast_to_game.call_args_list
        picked_calls = [c for c in calls if c[0][0].command == Command.PICKED]
        assert len(picked_calls) >= 1
        picked_call = picked_calls[0]
        assert picked_call[0][0].content["player_id"] == "player-0"
        assert picked_call[0][0].content["card_id"] == card_to_play.value

    async def test_pick_not_your_turn(self, game_handler, game_with_players, mock_manager):
        """Picking when not your turn should send error."""
        game = game_with_players
        game.state = GameState.PICKING
        game.start_new_round()
        game.deal_cards()

        current_round = game.get_current_round()
        trick = Trick(number=1, starter_player_index=0, picking_player_id="player-0")
        current_round.tricks.append(trick)

        # Player 1 tries to pick (not their turn)
        player = game.get_player("player-1")
        card = player.hand[0]

        await game_handler.handle_command(game, "player-1", "PICK", {"card_id": card.value})

        error_call = mock_manager.send_personal_message.call_args
        assert error_call[0][0].command == Command.REPORT_ERROR
        assert "Not your turn" in error_call[0][0].content["error"]

    async def test_pick_card_not_in_hand(self, game_handler, game_with_players, mock_manager):
        """Picking a card not in hand should send error."""
        game = game_with_players
        game.state = GameState.PICKING
        game.start_new_round()
        game.deal_cards()

        current_round = game.get_current_round()
        trick = Trick(number=1, starter_player_index=0, picking_player_id="player-0")
        current_round.tricks.append(trick)

        # Try to pick a card not in player's hand
        await game_handler.handle_command(game, "player-0", "PICK", {"card_id": 999})

        error_call = mock_manager.send_personal_message.call_args
        assert error_call[0][0].command == Command.REPORT_ERROR
        # Either "Invalid card ID" or "Card not in hand"
        assert "card" in error_call[0][0].content["error"].lower()


class TestSyncState:
    """Tests for SYNC_STATE command."""

    async def test_sync_state_pending_game(self, game_handler, game_with_players, mock_manager):
        """Sync state should send full game state."""
        game = game_with_players

        await game_handler.handle_command(game, "player-0", "SYNC_STATE", {})

        # GAME_STATE should be sent
        state_call = mock_manager.send_personal_message.call_args
        assert state_call[0][0].command == Command.GAME_STATE

        content = state_call[0][0].content
        assert content["game_id"] == "test-game-123"
        assert content["state"] == "PENDING"
        assert len(content["players"]) == 3

    async def test_sync_state_with_round(self, game_handler, game_with_players, mock_manager):
        """Sync state during game should include round info."""
        game = game_with_players
        game.state = GameState.BIDDING
        game.start_new_round()
        game.deal_cards()

        await game_handler.handle_command(game, "player-0", "SYNC_STATE", {})

        content = mock_manager.send_personal_message.call_args[0][0].content
        assert content["current_round"] is not None
        assert content["current_round"]["number"] == 1
        assert len(content["hand"]) == 1  # Round 1 = 1 card

    async def test_send_game_state_on_connect(self, game_handler, game_with_players, mock_manager):
        """send_game_state should send full state to player."""
        game = game_with_players
        game.state = GameState.BIDDING
        game.start_new_round()
        game.deal_cards()

        await game_handler.send_game_state(game, "player-1")

        state_call = mock_manager.send_personal_message.call_args
        assert state_call[0][0].command == Command.GAME_STATE
        assert state_call[0][2] == "player-1"  # Sent to player-1


class TestTrickCompletion:
    """Tests for trick completion and round scoring."""

    async def test_trick_completion_broadcasts_winner(
        self, game_handler, game_with_players, mock_manager
    ):
        """Completing a trick should broadcast the winner."""
        game = game_with_players
        game.state = GameState.BIDDING
        game.start_new_round()
        game.deal_cards()

        # Set all bids to 0
        for player in game.players:
            player.bid = 0

        game.state = GameState.PICKING
        current_round = game.get_current_round()
        trick = Trick(number=1, starter_player_index=0, picking_player_id="player-0")
        current_round.tricks.append(trick)

        # Each player picks their card in order
        for i in range(3):
            player = game.get_player(f"player-{i}")
            card = player.hand[0]
            trick.picking_player_id = f"player-{i}"
            await game_handler.handle_command(game, f"player-{i}", "PICK", {"card_id": card.value})

        # ANNOUNCE_TRICK_WINNER should be broadcast
        calls = mock_manager.broadcast_to_game.call_args_list
        winner_calls = [c for c in calls if c[0][0].command == Command.ANNOUNCE_TRICK_WINNER]
        assert len(winner_calls) >= 1

        # Since round 1 has only 1 trick, ANNOUNCE_SCORES should also be broadcast
        score_calls = [c for c in calls if c[0][0].command == Command.ANNOUNCE_SCORES]
        assert len(score_calls) >= 1
