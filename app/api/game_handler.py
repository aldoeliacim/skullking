"""Game logic handler for WebSocket commands."""

import logging
from typing import TYPE_CHECKING, Any

from app.api.responses import Command, ServerMessage
from app.models.card import CardId, get_card
from app.models.enums import GameState
from app.models.game import Game
from app.models.trick import Trick

if TYPE_CHECKING:
    from app.api.websocket import ConnectionManager

logger = logging.getLogger(__name__)


class GameHandler:
    """
    Handles game logic for WebSocket commands.

    Processes client commands (BID, PICK) and generates
    appropriate server responses.
    """

    def __init__(self, manager: "ConnectionManager") -> None:
        """Initialize handler with connection manager."""
        self.manager = manager

    async def handle_command(
        self, game: Game, player_id: str, command: str, content: dict[str, Any]
    ) -> None:
        """
        Route incoming command to appropriate handler.

        Args:
            game: Game instance
            player_id: ID of player who sent command
            command: Command type
            content: Command payload
        """
        handlers = {
            "BID": self._handle_bid,
            "PICK": self._handle_pick,
            "START_GAME": self._handle_start_game,
            "SYNC_STATE": self._handle_sync_state,
        }

        handler = handlers.get(command)
        if handler:
            await handler(game, player_id, content)
        else:
            logger.warning("Unknown command: %s", command)

    async def _handle_start_game(
        self, game: Game, player_id: str, _content: dict[str, Any]
    ) -> None:
        """Handle START_GAME command - initiates the game."""
        if game.state != GameState.PENDING:
            await self._send_error(game.id, player_id, "Game already started")
            return

        if not game.can_start():
            await self._send_error(game.id, player_id, "Not enough players")
            return

        # Start the game
        game.state = GameState.DEALING

        # Broadcast game started
        await self.manager.broadcast_to_game(
            ServerMessage(
                command=Command.STARTED,
                game_id=game.id,
                content={"player_count": len(game.players)},
            ),
            game.id,
        )

        # Start first round
        await self._start_new_round(game)

    async def _handle_bid(self, game: Game, player_id: str, content: dict[str, Any]) -> None:
        """
        Handle BID command from a player.

        Args:
            game: Game instance
            player_id: ID of bidding player
            content: Must contain 'bid' key with bid amount
        """
        if game.state != GameState.BIDDING:
            await self._send_error(game.id, player_id, "Not in bidding phase")
            return

        player = game.get_player(player_id)
        if not player:
            await self._send_error(game.id, player_id, "Player not found")
            return

        if player.bid is not None:
            await self._send_error(game.id, player_id, "Already placed bid")
            return

        bid_amount = content.get("bid", 0)
        current_round = game.get_current_round()

        # Validate bid
        max_bid = current_round.number if current_round else 0
        if not current_round or not (0 <= bid_amount <= max_bid):
            await self._send_error(game.id, player_id, f"Invalid bid: must be 0-{max_bid}")
            return

        # Record bid
        player.bid = bid_amount
        if current_round:
            current_round.bids[player_id] = bid_amount

        logger.info("Player %s bid %d in game %s", player_id, bid_amount, game.id)

        # Broadcast bid to all players
        await self.manager.broadcast_to_game(
            ServerMessage(
                command=Command.BADE,
                game_id=game.id,
                content={"player_id": player_id, "bid": bid_amount},
            ),
            game.id,
        )

        # Check if all players have bid
        if all(p.bid is not None for p in game.players):
            await self._end_bidding_phase(game)

    async def _handle_pick(self, game: Game, player_id: str, content: dict[str, Any]) -> None:
        """
        Handle PICK command from a player.

        Args:
            game: Game instance
            player_id: ID of picking player
            content: Must contain 'card_id' key with card ID
        """
        if game.state != GameState.PICKING:
            await self._send_error(game.id, player_id, "Not in picking phase")
            return

        current_round = game.get_current_round()
        if not current_round or not current_round.tricks:
            await self._send_error(game.id, player_id, "No active trick")
            return

        current_trick = current_round.tricks[-1]

        # Verify it's this player's turn
        if current_trick.picking_player_id != player_id:
            await self._send_error(game.id, player_id, "Not your turn")
            return

        player = game.get_player(player_id)
        if not player:
            await self._send_error(game.id, player_id, "Player not found")
            return

        card_id_raw = content.get("card_id")
        try:
            card_id = CardId(card_id_raw)
        except (ValueError, TypeError):
            await self._send_error(game.id, player_id, "Invalid card ID")
            return

        # Verify player has this card
        if card_id not in player.hand:
            await self._send_error(game.id, player_id, "Card not in hand")
            return

        # Play the card
        player.hand.remove(card_id)
        current_trick.add_card(player_id, card_id)

        logger.info(
            "Player %s played card %s in game %s",
            player_id,
            get_card(card_id),
            game.id,
        )

        # Broadcast pick to all players
        await self.manager.broadcast_to_game(
            ServerMessage(
                command=Command.PICKED,
                game_id=game.id,
                content={"player_id": player_id, "card_id": card_id.value},
            ),
            game.id,
        )

        # Check if trick is complete
        if current_trick.is_complete(len(game.players)):
            await self._complete_trick(game, current_round, current_trick)
        else:
            # Move to next player
            await self._advance_to_next_player(game, current_round, current_trick)

    async def _start_new_round(self, game: Game) -> None:
        """Start a new round in the game."""
        current_round = game.start_new_round()
        game.deal_cards()

        logger.info("Starting round %d in game %s", current_round.number, game.id)

        # Send dealt cards to each player
        for player in game.players:
            await self.manager.send_personal_message(
                ServerMessage(
                    command=Command.DEAL,
                    game_id=game.id,
                    content={
                        "round": current_round.number,
                        "cards": [c.value for c in player.hand],
                    },
                ),
                game.id,
                player.id,
            )

        # Start bidding phase
        game.state = GameState.BIDDING
        await self.manager.broadcast_to_game(
            ServerMessage(
                command=Command.START_BIDDING,
                game_id=game.id,
                content={"round": current_round.number},
            ),
            game.id,
        )

    async def _end_bidding_phase(self, game: Game) -> None:
        """End bidding phase and start card picking."""
        current_round = game.get_current_round()
        if not current_round:
            return

        logger.info("Bidding complete for round %d in game %s", current_round.number, game.id)

        # Broadcast end of bidding with all bids
        all_bids = [{"player_id": p.id, "bid": p.bid} for p in game.players]
        await self.manager.broadcast_to_game(
            ServerMessage(
                command=Command.END_BIDDING,
                game_id=game.id,
                content={"bids": all_bids},
            ),
            game.id,
        )

        # Start first trick
        await self._start_new_trick(game, current_round)

    async def _start_new_trick(self, game: Game, current_round: Any) -> None:
        """Start a new trick in the current round."""
        trick_number = len(current_round.tricks) + 1
        starter_index = current_round.starter_player_index

        # Adjust starter for subsequent tricks (winner leads)
        if current_round.tricks:
            last_trick = current_round.tricks[-1]
            if last_trick.winner_player_id:
                winner = game.get_player(last_trick.winner_player_id)
                if winner:
                    starter_index = winner.index

        trick = Trick(
            number=trick_number,
            starter_player_index=starter_index,
        )

        # Set first player to pick
        starter_player = game.get_player_by_index(starter_index)
        if starter_player:
            trick.picking_player_id = starter_player.id

        current_round.tricks.append(trick)
        game.state = GameState.PICKING

        logger.info("Starting trick %d in round %d", trick_number, current_round.number)

        # Broadcast start of picking
        await self.manager.broadcast_to_game(
            ServerMessage(
                command=Command.START_PICKING,
                game_id=game.id,
                content={
                    "trick": trick_number,
                    "picking_player_id": trick.picking_player_id,
                },
            ),
            game.id,
        )

    async def _advance_to_next_player(self, game: Game, _current_round: Any, trick: Trick) -> None:
        """Advance to the next player in the trick."""
        # Find current player index
        current_player = game.get_player(trick.picking_player_id)
        if not current_player:
            return

        # Get next player
        next_index = (current_player.index + 1) % len(game.players)
        next_player = game.get_player_by_index(next_index)

        if next_player:
            trick.picking_player_id = next_player.id

            # Notify all players of who picks next
            await self.manager.broadcast_to_game(
                ServerMessage(
                    command=Command.NEXT_TRICK,
                    game_id=game.id,
                    content={"picking_player_id": next_player.id},
                ),
                game.id,
            )

    async def _complete_trick(self, game: Game, current_round: Any, trick: Trick) -> None:
        """Complete a trick and determine winner."""
        winner_card_id, winner_player_id = trick.determine_winner()
        bonus_points = trick.calculate_bonus_points()

        logger.info(
            "Trick %d won by %s (bonus: %d)",
            trick.number,
            winner_player_id,
            bonus_points,
        )

        # Update player's tricks won
        if winner_player_id:
            winner = game.get_player(winner_player_id)
            if winner:
                winner.tricks_won += 1

        # Broadcast trick winner
        await self.manager.broadcast_to_game(
            ServerMessage(
                command=Command.ANNOUNCE_TRICK_WINNER,
                game_id=game.id,
                content={
                    "trick": trick.number,
                    "winner_player_id": winner_player_id,
                    "winner_card_id": winner_card_id.value if winner_card_id else None,
                    "bonus_points": bonus_points,
                },
            ),
            game.id,
        )

        # Check if round is complete
        if current_round.is_complete():
            await self._complete_round(game, current_round)
        else:
            # Start next trick
            await self._start_new_trick(game, current_round)

    async def _complete_round(self, game: Game, current_round: Any) -> None:
        """Complete a round and calculate scores."""
        current_round.calculate_scores()

        logger.info("Round %d complete in game %s", current_round.number, game.id)

        # Build score update for all players
        scores = []
        for player in game.players:
            tricks_won = current_round.get_tricks_won(player.id)
            bid = player.bid if player.bid is not None else 0

            # Calculate score delta
            if tricks_won == bid:
                if bid == 0:
                    delta = current_round.number * 10  # Bonus for zero bid success
                else:
                    delta = bid * 20  # Points per trick + bonus
            else:
                delta = -abs(tricks_won - bid) * 10  # Penalty for missing bid

            player.score += delta

            scores.append(
                {
                    "player_id": player.id,
                    "bid": bid,
                    "tricks_won": tricks_won,
                    "score_delta": delta,
                    "total_score": player.score,
                }
            )

        # Broadcast scores
        await self.manager.broadcast_to_game(
            ServerMessage(
                command=Command.ANNOUNCE_SCORES,
                game_id=game.id,
                content={"round": current_round.number, "scores": scores},
            ),
            game.id,
        )

        # Check if game is complete
        if game.is_game_complete():
            await self._end_game(game)
        else:
            # Start next round
            await self._start_new_round(game)

    async def _end_game(self, game: Game) -> None:
        """End the game and announce winner."""
        game.state = GameState.ENDED
        leaderboard = game.get_leaderboard()

        logger.info("Game %s ended. Winner: %s", game.id, leaderboard[0]["player_id"])

        await self.manager.broadcast_to_game(
            ServerMessage(
                command=Command.END_GAME,
                game_id=game.id,
                content={"leaderboard": leaderboard},
            ),
            game.id,
        )

    async def _send_error(self, game_id: str, player_id: str, message: str) -> None:
        """Send error message to a specific player."""
        await self.manager.send_personal_message(
            ServerMessage(
                command=Command.REPORT_ERROR,
                game_id=game_id,
                content={"error": message},
            ),
            game_id,
            player_id,
        )

    async def _handle_sync_state(
        self, game: Game, player_id: str, _content: dict[str, Any]
    ) -> None:
        """Handle SYNC_STATE command - sends full game state to requesting player."""
        state = self._build_game_state(game, player_id)
        await self.manager.send_personal_message(
            ServerMessage(
                command=Command.GAME_STATE,
                game_id=game.id,
                content=state,
            ),
            game.id,
            player_id,
        )

    async def send_game_state(self, game: Game, player_id: str) -> None:
        """Send full game state to a player (for connect/reconnect)."""
        state = self._build_game_state(game, player_id)
        await self.manager.send_personal_message(
            ServerMessage(
                command=Command.GAME_STATE,
                game_id=game.id,
                content=state,
            ),
            game.id,
            player_id,
        )

    def _build_game_state(self, game: Game, player_id: str) -> dict[str, Any]:
        """
        Build complete game state for a player.

        Includes all public information plus player's private hand.
        """
        current_round = game.get_current_round()
        current_trick = current_round.get_current_trick() if current_round else None

        # Build player info
        players = []
        for p in game.players:
            player_info = {
                "id": p.id,
                "username": p.username,
                "avatar_id": p.avatar_id,
                "score": p.score,
                "index": p.index,
                "is_bot": p.is_bot,
                "bid": p.bid,
                "tricks_won": p.tricks_won,
            }
            players.append(player_info)

        # Build round info
        round_info = None
        if current_round:
            # Build bids (only show revealed bids based on game state)
            bids: list[dict[str, Any]] = []
            if game.state in (GameState.PICKING, GameState.ENDED):
                # All bids revealed after bidding ends
                bids = [{"player_id": pid, "bid": bid} for pid, bid in current_round.bids.items()]
            elif game.state == GameState.BIDDING:
                # Only show who has bid, not the amount
                bids = [{"player_id": pid, "has_bid": True} for pid in current_round.bids]

            # Build trick info
            tricks = []
            for trick in current_round.tricks:
                trick_info = {
                    "number": trick.number,
                    "cards": [
                        {"player_id": pc.player_id, "card_id": pc.card_id.value}
                        for pc in trick.picked_cards
                    ],
                    "winner_player_id": trick.winner_player_id,
                    "winner_card_id": (
                        trick.winner_card_id.value if trick.winner_card_id else None
                    ),
                }
                tricks.append(trick_info)

            round_info = {
                "number": current_round.number,
                "bids": bids,
                "tricks": tricks,
                "starter_player_index": current_round.starter_player_index,
            }

        # Get player's hand (private info)
        player = game.get_player(player_id)
        hand = [c.value for c in player.hand] if player else []

        # Determine whose turn it is
        picking_player_id = None
        if current_trick and game.state == GameState.PICKING:
            picking_player_id = current_trick.picking_player_id

        return {
            "game_id": game.id,
            "slug": game.slug,
            "state": game.state.value,
            "players": players,
            "current_round": round_info,
            "hand": hand,
            "picking_player_id": picking_player_id,
        }
