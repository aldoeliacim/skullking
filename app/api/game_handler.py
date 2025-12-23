"""Game logic handler for WebSocket commands."""

import asyncio
import logging
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.api.responses import Command, ServerMessage
from app.bots import RandomBot, RuleBasedBot
from app.bots.base_bot import BaseBot, BotDifficulty
from app.models.card import CardId, get_card
from app.models.enums import GameState
from app.models.game import Game
from app.models.player import Player
from app.models.trick import Trick

if TYPE_CHECKING:
    from app.api.websocket import ConnectionManager

logger = logging.getLogger(__name__)

# Module-level cache for RL model (avoids global statement)
_rl_cache: dict[str, Any] = {"model": None}
_rl_model_path = Path(__file__).parent.parent.parent / "models/masked_ppo/masked_ppo_final.zip"


def _load_rl_model():
    """Load the trained RL model if available."""
    if _rl_cache["model"] is None and _rl_model_path.exists():
        try:
            from sb3_contrib import MaskablePPO

            _rl_cache["model"] = MaskablePPO.load(str(_rl_model_path))
            logger.info("Loaded RL model from %s", _rl_model_path)
        except Exception as e:
            logger.warning("Could not load RL model: %s", e)
    return _rl_cache["model"]


class GameHandler:
    """
    Handles game logic for WebSocket commands.

    Processes client commands (BID, PICK) and generates
    appropriate server responses.
    """

    def __init__(self, manager: "ConnectionManager") -> None:
        """Initialize handler with connection manager."""
        self.manager = manager
        self.bots: dict[str, dict[str, BaseBot]] = {}  # game_id -> player_id -> bot
        self._bot_processing_locks: dict[str, asyncio.Lock] = {}  # game_id -> lock

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
            "ADD_BOT": self._handle_add_bot,
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

        # Process bot actions (bids)
        await self._process_bot_actions(game)

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

        if "bid" not in content:
            await self._send_error(game.id, player_id, "Missing bid value")
            return

        bid_amount = content.get("bid")
        if not isinstance(bid_amount, int):
            await self._send_error(game.id, player_id, "Bid must be a number")
            return

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
        else:
            # Process bot bids if any bots haven't bid yet
            await self._process_bot_actions(game)

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
        if not current_trick.add_card(player_id, card_id):
            # Card was rejected (player already picked)
            player.hand.append(card_id)
            await self._send_error(game.id, player_id, "Already played in this trick")
            return

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
            # Process bot picks if next player is a bot
            await self._process_bot_actions(game)

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
        # Process bot picks if first player is a bot
        await self._process_bot_actions(game)

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
            # Process bot picks if first player is a bot
            await self._process_bot_actions(game)

    async def _complete_round(self, game: Game, current_round: Any) -> None:
        """Complete a round and calculate scores."""
        current_round.calculate_scores()

        logger.info("Round %d complete in game %s", current_round.number, game.id)

        # Build score update for all players using the round's calculated scores
        scores = []
        for player in game.players:
            tricks_won = current_round.get_tricks_won(player.id)
            bid = player.bid if player.bid is not None else 0

            # Use the round's calculated score which includes bonus points
            delta = current_round.scores.get(player.id, 0)
            bonus_points = current_round.get_bonus_points(player.id) if tricks_won == bid else 0

            player.score += delta

            scores.append(
                {
                    "player_id": player.id,
                    "bid": bid,
                    "tricks_won": tricks_won,
                    "score_delta": delta,
                    "bonus_points": bonus_points,
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
            # Process bot bids for new round
            await self._process_bot_actions(game)

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

    async def _handle_add_bot(self, game: Game, player_id: str, content: dict[str, Any]) -> None:
        """Handle ADD_BOT command - adds an AI opponent to the game."""
        if game.state != GameState.PENDING:
            await self._send_error(game.id, player_id, "Cannot add bot after game started")
            return

        if len(game.players) >= 8:
            await self._send_error(game.id, player_id, "Game is full")
            return

        # Get bot type from content (default to rl if model exists, else rule_based)
        bot_type = content.get("bot_type", "rl" if _load_rl_model() else "rule_based")
        difficulty = content.get("difficulty", "hard")

        # Create bot player
        bot_id = f"bot-{uuid.uuid4().hex[:8]}"
        bot_names = [
            "Captain Hook",
            "Blackbeard",
            "Anne Bonny",
            "Davy Jones",
            "Calico Jack",
            "Red Beard",
        ]
        bot_name = bot_names[len(game.players) % len(bot_names)]

        player = Player(
            id=bot_id,
            username=f"{bot_name} (AI)",
            avatar_id=len(game.players),
            game_id=game.id,
            index=len(game.players),
            is_bot=True,
        )
        game.add_player(player)

        # Create bot instance
        difficulty_enum = BotDifficulty[difficulty.upper()]
        if bot_type == "rl" and _load_rl_model():
            # Use RL bot with trained model
            from app.bots.rl_bot import RLBot

            bot = RLBot(bot_id, model=_load_rl_model(), difficulty=difficulty_enum)
            logger.info("Added RL bot %s to game %s", bot_id, game.id)
        elif bot_type == "random":
            bot = RandomBot(bot_id, difficulty=difficulty_enum)
            logger.info("Added random bot %s to game %s", bot_id, game.id)
        else:
            bot = RuleBasedBot(bot_id, difficulty=difficulty_enum)
            logger.info("Added rule-based bot %s to game %s", bot_id, game.id)

        # Store bot
        if game.id not in self.bots:
            self.bots[game.id] = {}
        self.bots[game.id][bot_id] = bot

        # Broadcast player joined
        await self.manager.broadcast_to_game(
            ServerMessage(
                command=Command.JOINED,
                game_id=game.id,
                content={
                    "player_id": bot_id,
                    "username": player.username,
                    "is_bot": True,
                    "bot_type": bot_type,
                },
            ),
            game.id,
        )

        # Broadcast updated game state so lobby refreshes
        await self.manager.broadcast_to_game(
            ServerMessage(
                command=Command.GAME_STATE,
                game_id=game.id,
                content={
                    "id": game.id,
                    "slug": game.slug,
                    "state": game.state.value,
                    "players": [
                        {
                            "id": p.id,
                            "username": p.username,
                            "score": p.score,
                            "index": p.index,
                            "is_bot": p.is_bot,
                            "is_connected": p.is_connected,
                        }
                        for p in game.players
                    ],
                },
            ),
            game.id,
        )

    def _get_bot_lock(self, game_id: str) -> asyncio.Lock:
        """Get or create the bot processing lock for a game."""
        if game_id not in self._bot_processing_locks:
            self._bot_processing_locks[game_id] = asyncio.Lock()
        return self._bot_processing_locks[game_id]

    async def _process_bot_actions(self, game: Game) -> None:
        """Process any pending bot actions (bids or card plays)."""
        if game.id not in self.bots:
            return

        # Use lock to prevent concurrent bot processing
        lock = self._get_bot_lock(game.id)
        if lock.locked():
            # Already processing bots for this game, skip
            return

        async with lock:
            await asyncio.sleep(0.5)  # Small delay for natural feel

            if game.state == GameState.BIDDING:
                await self._process_bot_bids(game)
            elif game.state == GameState.PICKING:
                await self._process_bot_picks(game)

    async def _process_bot_bids(self, game: Game) -> None:
        """Process bids for all bots that haven't bid yet."""
        if game.id not in self.bots:
            return

        current_round = game.get_current_round()
        if not current_round:
            return

        for bot_id, bot in self.bots.get(game.id, {}).items():
            player = game.get_player(bot_id)
            if player and player.bid is None:
                # Bot makes bid
                hand = list(player.hand)
                bid = bot.make_bid(game, current_round.number, hand)
                bid = max(0, min(current_round.number, bid))

                # Record bid
                player.bid = bid
                current_round.bids[bot_id] = bid

                logger.info("Bot %s bids %d", bot_id, bid)

                # Broadcast bid
                await self.manager.broadcast_to_game(
                    ServerMessage(
                        command=Command.BADE,
                        game_id=game.id,
                        content={"player_id": bot_id, "bid": bid},
                    ),
                    game.id,
                )

                await asyncio.sleep(0.3)  # Delay between bot bids

        # Check if all players have bid
        if current_round.all_bids_placed(len(game.players)):
            await self._end_bidding_phase(game)

    async def _process_bot_picks(self, game: Game) -> None:
        """Process card pick for the current bot if it's their turn."""
        if game.id not in self.bots:
            return

        # Guard: ensure we're still in picking state
        if game.state != GameState.PICKING:
            return

        current_round = game.get_current_round()
        if not current_round:
            return

        trick = current_round.get_current_trick()
        if not trick:
            return

        # Guard: don't process if trick is already complete
        if trick.is_complete(len(game.players)):
            return

        picking_player_id = trick.picking_player_id
        if picking_player_id not in self.bots.get(game.id, {}):
            return  # Not a bot's turn

        # Guard: check if this player already played in this trick
        if any(pc.player_id == picking_player_id for pc in trick.picked_cards):
            logger.warning("Bot %s already played in trick %d", picking_player_id, trick.number)
            return

        bot = self.bots[game.id][picking_player_id]
        player = game.get_player(picking_player_id)
        if not player or not player.hand:
            return

        # Get valid cards
        cards_in_trick = [pc.card_id for pc in trick.picked_cards]
        valid_cards = trick.get_valid_cards(player.hand, cards_in_trick)

        # Bot picks card
        card_id = bot.pick_card(game, player.hand, cards_in_trick, valid_cards)
        card = get_card(card_id)

        # Remove from hand
        if card_id in player.hand:
            player.hand.remove(card_id)

        # Final guard: re-check trick isn't complete before adding
        if trick.is_complete(len(game.players)):
            logger.warning("Trick completed before bot %s could play", picking_player_id)
            # Put card back in hand
            player.hand.append(card_id)
            return

        # Add to trick
        if not trick.add_card(picking_player_id, card_id):
            # Card was rejected (player already picked) - shouldn't happen with guards above
            logger.error(
                "Bot %s card rejected - already picked in trick %d", picking_player_id, trick.number
            )
            player.hand.append(card_id)
            return

        logger.info("Bot %s plays %s", picking_player_id, card)

        # Broadcast card played
        await self.manager.broadcast_to_game(
            ServerMessage(
                command=Command.PICKED,
                game_id=game.id,
                content={"player_id": picking_player_id, "card_id": card_id.value},
            ),
            game.id,
        )

        # Check if trick is complete
        if trick.is_complete(len(game.players)):
            await self._complete_trick(game, current_round, trick)
        else:
            await self._advance_to_next_player(game, current_round, trick)
            # Continue processing if next player is also a bot
            await asyncio.sleep(0.5)
            await self._process_bot_picks(game)
