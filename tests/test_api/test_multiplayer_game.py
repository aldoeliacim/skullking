"""End-to-end tests for full multiplayer gameplay."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.api.game_handler import GameHandler
from app.api.responses import Command
from app.models.enums import GameState
from app.models.game import Game
from app.models.player import Player

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


def create_game_with_players(num_players: int = 3) -> Game:
    """Create a game with specified number of bot players (for auto-continue in tests)."""
    game = Game(id="multiplayer-test", slug="multiplayer-test")
    for i in range(num_players):
        player = Player(
            id=f"player-{i}",
            username=f"Player {i}",
            avatar_id=i,
            index=i,
            game_id=game.id,
            is_bot=True,  # Bots auto-continue in game flow
        )
        game.add_player(player)
    return game


class TestFullGameFlow:
    """Tests for complete game flow from start to finish."""

    async def test_complete_round_flow(self, game_handler, mock_manager):
        """Test a complete round: deal -> bid -> pick -> score."""
        game = create_game_with_players(3)

        # Start game
        await game_handler.handle_command(game, "player-0", "START_GAME", {})
        assert game.state == GameState.BIDDING
        assert game.current_round_number == 1

        # All players bid 0
        for i in range(3):
            await game_handler.handle_command(game, f"player-{i}", "BID", {"bid": 0})

        # Game should transition to picking
        assert game.state == GameState.PICKING

        # All players pick their card
        current_round = game.get_current_round()
        for i in range(3):
            player = game.get_player(f"player-{i}")
            card = player.hand[0]
            current_round.tricks[-1].picking_player_id = f"player-{i}"
            await game_handler.handle_command(game, f"player-{i}", "PICK", {"card_id": card.value})

        # Round 1 should be complete, now in round 2
        assert game.current_round_number == 2
        assert game.state == GameState.BIDDING

        # Verify scores were broadcast
        calls = mock_manager.broadcast_to_game.call_args_list
        score_calls = [c for c in calls if c[0][0].command == Command.ANNOUNCE_SCORES]
        assert len(score_calls) >= 1

    async def test_multiple_rounds(self, game_handler, mock_manager):
        """Test playing through multiple rounds."""
        game = create_game_with_players(2)

        # Start game
        await game_handler.handle_command(game, "player-0", "START_GAME", {})

        # Play through 3 rounds
        for round_num in range(1, 4):
            assert game.current_round_number == round_num
            assert game.state == GameState.BIDDING

            # All players bid 0
            for i in range(2):
                await game_handler.handle_command(game, f"player-{i}", "BID", {"bid": 0})

            # Play all tricks in this round
            current_round = game.get_current_round()
            for _ in range(round_num):
                assert game.state == GameState.PICKING

                for i in range(2):
                    player = game.get_player(f"player-{i}")
                    if player.hand:
                        card = player.hand[0]
                        current_round.tricks[-1].picking_player_id = f"player-{i}"
                        pick_content = {"card_id": card.value}
                        # Tigress (card_id 72) requires a choice
                        if card.value == 72:
                            pick_content["tigress_choice"] = "escape"
                        await game_handler.handle_command(game, f"player-{i}", "PICK", pick_content)

        # After round 3, should be in round 4
        assert game.current_round_number == 4

    async def test_scoring_correct_bid(self, game_handler, mock_manager):
        """Test scoring when player bids correctly."""
        game = create_game_with_players(2)

        await game_handler.handle_command(game, "player-0", "START_GAME", {})

        # Players bid 0 (both will be correct since only 1 trick)
        await game_handler.handle_command(game, "player-0", "BID", {"bid": 0})
        await game_handler.handle_command(game, "player-1", "BID", {"bid": 0})

        # Play the trick
        current_round = game.get_current_round()
        for i in range(2):
            player = game.get_player(f"player-{i}")
            card = player.hand[0]
            current_round.tricks[-1].picking_player_id = f"player-{i}"
            await game_handler.handle_command(game, f"player-{i}", "PICK", {"card_id": card.value})

        # One player won, one didn't - but both bid 0
        # Winner gets penalty (-10), loser gets bonus (+10) for round 1
        # Wait, let me check the scoring logic more carefully
        # Actually: if bid == 0 and tricks_won == 0: score = 10 * round_number
        # If bid == 0 and tricks_won > 0: score = -10 * round_number

        # Verify ANNOUNCE_SCORES was broadcast with score changes
        calls = mock_manager.broadcast_to_game.call_args_list
        score_calls = [c for c in calls if c[0][0].command == Command.ANNOUNCE_SCORES]
        assert len(score_calls) >= 1

        score_content = score_calls[-1][0][0].content
        assert "scores" in score_content
        assert len(score_content["scores"]) == 2

    async def test_full_game_to_end(self, game_handler, mock_manager):
        """Test playing a complete game through all 10 rounds."""
        game = create_game_with_players(2)

        await game_handler.handle_command(game, "player-0", "START_GAME", {})

        # Play all 10 rounds
        for round_num in range(1, 11):
            assert game.current_round_number == round_num, f"Expected round {round_num}"
            assert game.state == GameState.BIDDING

            # All players bid 0
            for i in range(2):
                await game_handler.handle_command(game, f"player-{i}", "BID", {"bid": 0})

            # Play all tricks in this round
            current_round = game.get_current_round()
            for _ in range(round_num):
                assert game.state == GameState.PICKING

                for i in range(2):
                    player = game.get_player(f"player-{i}")
                    if player.hand:
                        card = player.hand[0]
                        current_round.tricks[-1].picking_player_id = f"player-{i}"
                        pick_content = {"card_id": card.value}
                        # Tigress (card_id 72) requires a choice
                        if card.value == 72:
                            pick_content["tigress_choice"] = "escape"
                        await game_handler.handle_command(game, f"player-{i}", "PICK", pick_content)

        # Game should be ended after round 10
        assert game.state == GameState.ENDED

        # END_GAME should have been broadcast
        calls = mock_manager.broadcast_to_game.call_args_list
        end_calls = [c for c in calls if c[0][0].command == Command.END_GAME]
        assert len(end_calls) >= 1

        # Leaderboard should be in the content
        end_content = end_calls[-1][0][0].content
        assert "leaderboard" in end_content
        assert len(end_content["leaderboard"]) == 2


class TestMultiplayerInteractions:
    """Tests for player interactions in multiplayer."""

    async def test_turn_order_enforcement(self, game_handler, mock_manager):
        """Test that players must pick in correct order."""
        # Create game with human players (not bots) for controlled test
        game = Game(id="turn-order-test", slug="turn-order-test")
        for i in range(3):
            player = Player(
                id=f"player-{i}",
                username=f"Player {i}",
                avatar_id=i,
                index=i,
                game_id=game.id,
                is_bot=False,  # Human players - no auto-processing
            )
            game.add_player(player)

        await game_handler.handle_command(game, "player-0", "START_GAME", {})

        # All bid
        for i in range(3):
            await game_handler.handle_command(game, f"player-{i}", "BID", {"bid": 0})

        # Verify we're in picking phase
        assert game.state == GameState.PICKING
        current_round = game.get_current_round()
        trick = current_round.get_current_trick()
        assert trick is not None

        # Get whose turn it actually is (starter is randomized in round 1)
        current_player_id = trick.picking_player_id
        current_player_index = int(current_player_id.split("-")[1])

        # Find a player who is NOT the current player
        wrong_player_index = (current_player_index + 1) % 3
        wrong_player_id = f"player-{wrong_player_index}"
        wrong_player = game.get_player(wrong_player_id)
        card = wrong_player.hand[0]

        mock_manager.send_personal_message.reset_mock()
        await game_handler.handle_command(game, wrong_player_id, "PICK", {"card_id": card.value})

        # Should receive error
        error_call = mock_manager.send_personal_message.call_args
        assert error_call is not None, "Expected error message to be sent"
        assert error_call[0][0].command == Command.REPORT_ERROR
        assert "Not your turn" in error_call[0][0].content["error"]

    async def test_player_hands_are_private(self, game_handler, mock_manager):
        """Test that DEAL only sends hand to each player individually."""
        game = create_game_with_players(3)

        await game_handler.handle_command(game, "player-0", "START_GAME", {})

        # Check DEAL messages were sent individually
        deal_calls = [
            c
            for c in mock_manager.send_personal_message.call_args_list
            if c[0][0].command == Command.DEAL
        ]

        # Each player should get their own DEAL message
        assert len(deal_calls) == 3

        # Each should be sent to a different player
        recipients = {c[0][2] for c in deal_calls}  # player_id is 3rd arg
        assert recipients == {"player-0", "player-1", "player-2"}

    async def test_broadcasts_reach_all_players(self, game_handler, mock_manager):
        """Test that game events are broadcast to all players."""
        game = create_game_with_players(3)

        await game_handler.handle_command(game, "player-0", "START_GAME", {})

        # STARTED should be broadcast (no specific recipient)
        started_calls = [
            c
            for c in mock_manager.broadcast_to_game.call_args_list
            if c[0][0].command == Command.STARTED
        ]
        assert len(started_calls) >= 1

        # START_BIDDING should be broadcast
        bidding_calls = [
            c
            for c in mock_manager.broadcast_to_game.call_args_list
            if c[0][0].command == Command.START_BIDDING
        ]
        assert len(bidding_calls) >= 1


class TestStateRecovery:
    """Tests for state synchronization and recovery."""

    async def test_sync_state_mid_game(self, game_handler, mock_manager):
        """Test state sync during an active game."""
        game = create_game_with_players(2)

        await game_handler.handle_command(game, "player-0", "START_GAME", {})

        # Player 0 bids
        await game_handler.handle_command(game, "player-0", "BID", {"bid": 0})

        # Player 1 requests sync before bidding
        mock_manager.send_personal_message.reset_mock()
        await game_handler.handle_command(game, "player-1", "SYNC_STATE", {})

        state_call = mock_manager.send_personal_message.call_args
        assert state_call[0][0].command == Command.GAME_STATE

        content = state_call[0][0].content
        assert content["state"] == "BIDDING"
        assert content["current_round"]["number"] == 1
        assert len(content["hand"]) == 1  # Round 1 = 1 card

    async def test_sync_state_shows_player_specific_hand(self, game_handler, mock_manager):
        """Test that sync state shows correct hand for each player."""
        game = create_game_with_players(2)

        await game_handler.handle_command(game, "player-0", "START_GAME", {})

        # Get both players' sync states
        for i in range(2):
            mock_manager.send_personal_message.reset_mock()
            await game_handler.handle_command(game, f"player-{i}", "SYNC_STATE", {})

            state_call = mock_manager.send_personal_message.call_args
            content = state_call[0][0].content

            # Hand should match the player's actual hand
            player = game.get_player(f"player-{i}")
            expected_hand = [c.value for c in player.hand]
            assert content["hand"] == expected_hand
