"""Game logic handler for WebSocket commands."""

import asyncio
import logging
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.api.responses import Command, ServerMessage
from app.bots import RandomBot, RuleBasedBot
from app.bots.base_bot import BaseBot, BotDifficulty
from app.bots.rl_bot import RLBot
from app.config import settings
from app.constants import HARRY_THE_GIANT_DISCARD_COUNT, MAX_PLAYERS
from app.models.card import CardId, get_card
from app.models.enums import GameState
from app.models.game import Game
from app.models.pirate_ability import AbilityType, PendingAbility, get_card_ability, get_pirate_type
from app.models.player import Player
from app.models.trick import TigressChoice, Trick
from app.services.event_recorder import event_recorder

if TYPE_CHECKING:
    from app.api.websocket import ConnectionManager
    from app.models.round import Round

# Optional RL imports - may not be available
try:
    from sb3_contrib import MaskablePPO

    _MASKABLE_PPO_AVAILABLE = True
except ImportError:
    MaskablePPO = None  # type: ignore[misc, assignment]
    _MASKABLE_PPO_AVAILABLE = False

logger = logging.getLogger(__name__)

# Module-level cache for RL model (avoids global statement)
_rl_cache: dict[str, Any] = {"model": None}


def _get_rl_model_path() -> Path:
    """Get the RL model path from config."""
    model_path = Path(settings.rl_model_path)
    if not model_path.is_absolute():
        # Relative paths are from project root
        model_path = Path(__file__).parent.parent.parent / model_path
    return model_path


def _load_rl_model() -> "MaskablePPO | None":
    """Load the trained RL model if available."""
    if not _MASKABLE_PPO_AVAILABLE:
        return None
    if _rl_cache["model"] is None:
        model_path = _get_rl_model_path()
        if model_path.exists():
            try:
                _rl_cache["model"] = MaskablePPO.load(str(model_path))
                logger.info("Loaded RL model from %s", model_path)
            except (OSError, ValueError, RuntimeError) as e:
                logger.warning("Could not load RL model: %s", e)
    return _rl_cache["model"]


class GameHandler:
    """Handles game logic for WebSocket commands.

    Processes client commands (BID, PICK) and generates
    appropriate server responses.
    """

    def __init__(self, manager: "ConnectionManager") -> None:
        """Initialize handler with connection manager."""
        self.manager = manager
        self.bots: dict[str, dict[str, BaseBot]] = {}  # game_id -> player_id -> bot

    async def handle_command(
        self, game: Game, player_id: str, command: str, content: dict[str, Any]
    ) -> None:
        """Route incoming command to appropriate handler.

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
            "REMOVE_BOT": self._handle_remove_bot,
            # Pirate ability handlers
            "RESOLVE_ROSIE": self._handle_resolve_rosie,
            "RESOLVE_BENDT": self._handle_resolve_bendt,
            "RESOLVE_ROATAN": self._handle_resolve_roatan,
            "RESOLVE_JADE": self._handle_resolve_jade,
            "RESOLVE_HARRY": self._handle_resolve_harry,
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

        # Record game start event
        event_recorder.start_game(game)

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
        """Handle BID command from a player.

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

        # Record event for replay
        event_recorder.record_bid(game, player_id, bid_amount)

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

    def _validate_pick_request(
        self, game: Game, player_id: str, current_round: "Round | None", current_trick: Trick | None
    ) -> Player | None:
        """Validate basic pick request conditions.

        Returns:
            Player object if all validations pass, None otherwise (error already sent).

        """
        if game.state != GameState.PICKING:
            return None

        if not current_round or not current_trick:
            return None

        if current_trick.picking_player_id != player_id:
            return None

        return game.get_player(player_id)

    def _parse_and_validate_card(
        self, player: Player, content: dict[str, Any]
    ) -> tuple[CardId, TigressChoice | None] | None:
        """Parse card ID and Tigress choice from content.

        Returns:
            Tuple of (card_id, tigress_choice) if valid, None otherwise.

        """
        card_id_raw = content.get("card_id")
        try:
            card_id = CardId(card_id_raw)
        except (ValueError, TypeError):
            return None

        if card_id not in player.hand:
            return None

        tigress_choice: TigressChoice | None = None
        card = get_card(card_id)
        if card.is_tigress():
            choice_raw = content.get("tigress_choice")
            if choice_raw not in ("pirate", "escape"):
                return None
            tigress_choice = TigressChoice(choice_raw)

        return (card_id, tigress_choice)

    async def _execute_pick(
        self,
        game: Game,
        player: Player,
        current_trick: Trick,
        card_id: CardId,
        tigress_choice: TigressChoice | None,
    ) -> bool:
        """Execute the card pick and broadcast to players.

        Returns:
            True if successful, False if card was rejected.

        """
        player.hand.remove(card_id)
        if not current_trick.add_card(player.id, card_id, tigress_choice):
            player.hand.append(card_id)
            return False

        card = get_card(card_id)
        logger.info(
            "Player %s played card %s%s in game %s",
            player.id,
            card,
            f" as {tigress_choice.value}" if tigress_choice else "",
            game.id,
        )

        event_recorder.record_card_played(
            game, player.id, card_id.value, tigress_choice.value if tigress_choice else None
        )

        pick_content: dict[str, Any] = {"player_id": player.id, "card_id": card_id.value}
        if tigress_choice:
            pick_content["tigress_choice"] = tigress_choice.value

        await self.manager.broadcast_to_game(
            ServerMessage(
                command=Command.PICKED,
                game_id=game.id,
                content=pick_content,
            ),
            game.id,
        )
        return True

    async def _handle_pick(self, game: Game, player_id: str, content: dict[str, Any]) -> None:
        """Handle PICK command from a player.

        Args:
            game: Game instance
            player_id: ID of picking player
            content: Must contain 'card_id' key with card ID, optional 'tigress_choice'

        """
        current_round = game.get_current_round()
        current_trick = current_round.tricks[-1] if current_round and current_round.tricks else None

        player = self._validate_pick_request(game, player_id, current_round, current_trick)
        if not player:
            if game.state != GameState.PICKING:
                await self._send_error(game.id, player_id, "Not in picking phase")
            elif not current_round or not current_trick:
                await self._send_error(game.id, player_id, "No active trick")
            elif current_trick.picking_player_id != player_id:
                await self._send_error(game.id, player_id, "Not your turn")
            else:
                await self._send_error(game.id, player_id, "Player not found")
            return

        card_data = self._parse_and_validate_card(player, content)
        if not card_data:
            card_id_raw = content.get("card_id")
            try:
                card_id = CardId(card_id_raw)
                if card_id not in player.hand:
                    await self._send_error(game.id, player_id, "Card not in hand")
                else:
                    await self._send_error(
                        game.id, player_id, "Tigress requires choice: pirate or escape"
                    )
            except (ValueError, TypeError):
                await self._send_error(game.id, player_id, "Invalid card ID")
            return

        card_id, tigress_choice = card_data

        # current_round and current_trick are guaranteed to be non-None here due to validation
        if not await self._execute_pick(game, player, current_trick, card_id, tigress_choice):
            await self._send_error(game.id, player_id, "Already played in this trick")
            return

        # Check if trick is complete
        if current_trick.is_complete(len(game.players)):
            await self._complete_trick(game, current_round, current_trick)
        else:
            await self._advance_to_next_player(game, current_round, current_trick)
            await self._process_bot_actions(game)

    async def _start_new_round(self, game: Game) -> None:
        """Start a new round in the game."""
        current_round = game.start_new_round()
        game.deal_cards()

        # Record round start with dealt cards for replay
        event_recorder.record_round_start(game)

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

    async def _start_new_trick(self, game: Game, current_round: "Round") -> None:
        """Start a new trick in the current round."""
        trick_number = len(current_round.tricks) + 1
        starter_index = current_round.starter_player_index
        default_starter_id: str | None = None

        logger.info(
            "[START_NEW_TRICK] Starting trick %d. Round starter_index=%d, tricks so far=%d",
            trick_number,
            current_round.starter_player_index,
            len(current_round.tricks),
        )

        # Adjust starter for subsequent tricks (winner leads)
        if current_round.tricks:
            last_trick = current_round.tricks[-1]
            logger.info(
                "[START_NEW_TRICK] Last trick %d: winner_player_id=%s, winner_card_id=%s",
                last_trick.number,
                last_trick.winner_player_id,
                last_trick.winner_card_id,
            )
            if last_trick.winner_player_id:
                winner = game.get_player(last_trick.winner_player_id)
                logger.info(
                    "[START_NEW_TRICK] Looking up winner %s: found=%s",
                    last_trick.winner_player_id,
                    winner is not None,
                )
                if winner:
                    starter_index = winner.index
                    default_starter_id = winner.id
                    logger.info(
                        "[START_NEW_TRICK] Winner %s has index %d, setting as starter",
                        winner.username,
                        winner.index,
                    )
                else:
                    logger.error(
                        "[START_NEW_TRICK] Could not find winner player %s! Players: %s",
                        last_trick.winner_player_id,
                        [p.id for p in game.players],
                    )
            else:
                logger.warning("[START_NEW_TRICK] Last trick has no winner (Kraken?)")
        else:
            logger.info("[START_NEW_TRICK] First trick of round, using round starter")

        # Check for Rosie's ability override
        if default_starter_id:
            actual_starter_id = current_round.get_next_trick_starter(default_starter_id)
            actual_starter = game.get_player(actual_starter_id)
            if actual_starter:
                starter_index = actual_starter.index
                logger.info(
                    "[START_NEW_TRICK] After Rosie check: actual starter is %s (index %d)",
                    actual_starter.username,
                    starter_index,
                )

        trick = Trick(
            number=trick_number,
            starter_player_index=starter_index,
        )

        # Set first player to pick
        starter_player = game.get_player_by_index(starter_index)
        if starter_player:
            trick.picking_player_id = starter_player.id
            logger.info(
                "[START_NEW_TRICK] Final starter: %s (id=%s, index=%d)",
                starter_player.username,
                starter_player.id,
                starter_index,
            )
        else:
            logger.error(
                "[START_NEW_TRICK] Could not find starter player at index %d!",
                starter_index,
            )

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

    async def _advance_to_next_player(
        self, game: Game, _current_round: "Round", trick: Trick
    ) -> None:
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

    async def _complete_trick(self, game: Game, current_round: "Round", trick: Trick) -> None:
        """Complete a trick and determine winner."""
        winner_card_id, winner_player_id = trick.determine_winner()
        bonus_points = trick.calculate_bonus_points()

        logger.info(
            "[COMPLETE_TRICK] Trick %d: winner_player_id=%s, winner_card_id=%s, bonus=%d",
            trick.number,
            winner_player_id,
            winner_card_id,
            bonus_points,
        )
        logger.info(
            "[COMPLETE_TRICK] Trick object winner_player_id after determine_winner: %s",
            trick.winner_player_id,
        )

        # Record trick won event for replay
        if winner_player_id and winner_card_id:
            event_recorder.record_trick_won(
                game, winner_player_id, winner_card_id.value, bonus_points
            )

        # Update player's tricks won
        if winner_player_id:
            winner = game.get_player(winner_player_id)
            if winner:
                winner.tricks_won += 1

        # Check for pirate ability trigger
        pending_ability: PendingAbility | None = None
        if winner_card_id and winner_player_id:
            ability_type = get_card_ability(winner_card_id)
            if ability_type:
                pending_ability = current_round.ability_state.trigger_ability(
                    winner_player_id, winner_card_id, trick.number
                )
                if pending_ability:
                    logger.info(
                        "Pirate ability triggered: %s for player %s",
                        ability_type.value,
                        winner_player_id,
                    )

        # Build announce content
        winner = game.get_player(winner_player_id) if winner_player_id else None
        announce_content: dict[str, Any] = {
            "trick": trick.number,
            "winner_player_id": winner_player_id,
            "winner_name": winner.username if winner else None,
            "winner_card_id": winner_card_id.value if winner_card_id else None,
            "bonus_points": bonus_points,
        }

        # Add ability info if triggered
        if pending_ability:
            pirate_type = get_pirate_type(winner_card_id) if winner_card_id else None
            announce_content["ability_triggered"] = {
                "ability_type": pending_ability.ability_type.value,
                "pirate_type": pirate_type.value if pirate_type else None,
                "player_id": winner_player_id,
            }

        # Broadcast trick winner
        await self.manager.broadcast_to_game(
            ServerMessage(
                command=Command.ANNOUNCE_TRICK_WINNER,
                game_id=game.id,
                content=announce_content,
            ),
            game.id,
        )

        # Handle pirate abilities that need immediate resolution
        if pending_ability:
            await self._handle_ability_trigger(game, current_round, pending_ability, trick)
            return  # Wait for ability resolution before continuing

        # Prompt for continue confirmation before proceeding
        await self._prompt_continue(game)

    def _is_bot_player(self, game: Game, player_id: str) -> bool:
        """Check if a player is a bot."""
        player = game.get_player(player_id)
        return player_id in self.bots.get(game.id, {}) or (player is not None and player.is_bot)

    async def _handle_rosie_ability(
        self, game: Game, current_round: "Round", ability: PendingAbility, trick: Trick
    ) -> None:
        """Handle Rosie's choose starter ability."""
        if self._is_bot_player(game, ability.player_id):
            current_round.ability_state.resolve_rosie(ability.player_id, ability.player_id)
            await self._ability_resolved(game, current_round, ability, trick)
        else:
            await self._send_ability_prompt(
                game,
                ability,
                {"options": [{"player_id": p.id, "username": p.username} for p in game.players]},
            )

    async def _handle_bendt_ability(
        self, game: Game, current_round: "Round", ability: PendingAbility, trick: Trick
    ) -> None:
        """Handle Bendt's draw and discard ability."""
        drawn_cards = self._draw_cards_from_deck(game, 2)
        ability.drawn_cards = drawn_cards

        if self._is_bot_player(game, ability.player_id):
            player_obj = game.get_player(ability.player_id)
            if player_obj:
                player_obj.hand.extend(drawn_cards)
                discard = (
                    player_obj.hand[:HARRY_THE_GIANT_DISCARD_COUNT]
                    if len(player_obj.hand) >= HARRY_THE_GIANT_DISCARD_COUNT
                    else player_obj.hand[:]
                )
                for card in discard:
                    player_obj.hand.remove(card)
                current_round.ability_state.resolve_bendt(ability.player_id, drawn_cards, discard)
            await self._ability_resolved(game, current_round, ability, trick)
        else:
            player = game.get_player(ability.player_id)
            if player:
                player.hand.extend(drawn_cards)
            await self._send_ability_prompt(
                game,
                ability,
                {
                    "drawn_cards": [c.value for c in drawn_cards],
                    "must_discard": min(2, len(drawn_cards)),
                },
            )

    async def _handle_roatan_ability(
        self, game: Game, current_round: "Round", ability: PendingAbility, trick: Trick
    ) -> None:
        """Handle Roatán's extra bet ability."""
        if self._is_bot_player(game, ability.player_id):
            current_round.ability_state.resolve_roatan(ability.player_id, 10)
            await self._ability_resolved(game, current_round, ability, trick)
        else:
            await self._send_ability_prompt(game, ability, {"options": [0, 10, 20]})

    async def _handle_jade_ability(
        self, game: Game, current_round: "Round", ability: PendingAbility, trick: Trick
    ) -> None:
        """Handle Jade's view deck ability."""
        undealt = self._get_undealt_cards(game)
        current_round.ability_state.resolve_jade(ability.player_id)

        if not self._is_bot_player(game, ability.player_id):
            await self.manager.send_personal_message(
                ServerMessage(
                    command=Command.SHOW_DECK,
                    game_id=game.id,
                    content={"undealt_cards": [c.value for c in undealt]},
                ),
                game.id,
                ability.player_id,
            )

        await self._ability_resolved(game, current_round, ability, trick)

    async def _handle_ability_trigger(
        self, game: Game, current_round: "Round", ability: PendingAbility, trick: Trick
    ) -> None:
        """Handle a triggered pirate ability."""
        if ability.ability_type == AbilityType.CHOOSE_STARTER:
            await self._handle_rosie_ability(game, current_round, ability, trick)
        elif ability.ability_type == AbilityType.DRAW_DISCARD:
            await self._handle_bendt_ability(game, current_round, ability, trick)
        elif ability.ability_type == AbilityType.EXTRA_BET:
            await self._handle_roatan_ability(game, current_round, ability, trick)
        elif ability.ability_type == AbilityType.VIEW_DECK:
            await self._handle_jade_ability(game, current_round, ability, trick)
        elif ability.ability_type == AbilityType.MODIFY_BID:
            # Harry - handled at end of round, just mark as armed
            await self._ability_resolved(game, current_round, ability, trick)

    async def _send_ability_prompt(
        self, game: Game, ability: PendingAbility, extra_data: dict[str, Any]
    ) -> None:
        """Send an ability prompt to a player."""
        await self.manager.send_personal_message(
            ServerMessage(
                command=Command.ABILITY_TRIGGERED,
                game_id=game.id,
                content={
                    "ability_type": ability.ability_type.value,
                    "pirate_type": ability.pirate_type.value,
                    **extra_data,
                },
            ),
            game.id,
            ability.player_id,
        )

    async def _ability_resolved(
        self, game: Game, current_round: "Round", ability: PendingAbility, _trick: Trick
    ) -> None:
        """Handle ability resolution and continue game flow."""
        # Notify all players that ability was resolved
        await self.manager.broadcast_to_game(
            ServerMessage(
                command=Command.ABILITY_RESOLVED,
                game_id=game.id,
                content={
                    "ability_type": ability.ability_type.value,
                    "player_id": ability.player_id,
                },
            ),
            game.id,
        )

        # Continue game flow
        if current_round.is_complete():
            await self._complete_round(game, current_round)
        else:
            await self._start_new_trick(game, current_round)
            await self._process_bot_actions(game)

    def _draw_cards_from_deck(self, game: Game, count: int) -> list[CardId]:
        """Draw cards from the deck (undealt cards)."""
        undealt = self._get_undealt_cards(game)
        return undealt[:count]

    def _get_undealt_cards(self, game: Game) -> list[CardId]:
        """Get cards that weren't dealt this round."""
        current_round = game.get_current_round()
        if not current_round:
            return []

        # All dealt cards
        dealt: set[CardId] = set()
        for cards in current_round.dealt_cards.values():
            dealt.update(cards)

        # Return cards not in dealt set
        return [card_id for card_id in game.deck.cards if card_id not in dealt]

    async def _complete_round(self, game: Game, current_round: "Round") -> None:
        """Complete a round and calculate scores."""
        # Check for Harry's ability before scoring
        players_with_harry = [
            player_id
            for player_id in current_round.ability_state.harry_armed
            if current_round.ability_state.harry_armed[player_id]
        ]

        for player_id in players_with_harry:
            player = game.get_player(player_id)
            is_bot = player_id in self.bots.get(game.id, {}) or (
                player is not None and player.is_bot
            )
            if is_bot:
                # Bot decides: adjust bid to match tricks won if possible
                if player:
                    tricks_won = current_round.get_tricks_won(player_id)
                    bid = player.bid if player.bid is not None else 0
                    diff = tricks_won - bid
                    if diff == 1:
                        modifier = 1
                    elif diff == -1:
                        modifier = -1
                    else:
                        modifier = 0
                    current_round.ability_state.resolve_harry(player_id, modifier)
            else:
                # Send prompt to human player - for now auto-resolve with 0
                # In a full implementation, we'd wait for player input
                current_round.ability_state.resolve_harry(player_id, 0)

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

        # Record scores for replay
        event_recorder.record_scores(game, scores)

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

        # Record game end and create history
        event_recorder.end_game(game)

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
        """Build complete game state for a player.

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

        if len(game.players) >= MAX_PLAYERS:
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

    async def _handle_remove_bot(
        self, game: Game, player_id: str, content: dict[str, Any]
    ) -> None:
        """Handle REMOVE_BOT command - removes an AI opponent from the game."""
        if game.state != GameState.PENDING:
            await self._send_error(game.id, player_id, "Cannot remove bot after game started")
            return

        bot_id = content.get("bot_id")
        if not bot_id:
            await self._send_error(game.id, player_id, "Missing bot_id")
            return

        # Check if bot exists
        bot_player = game.get_player(bot_id)
        if not bot_player or not bot_player.is_bot:
            await self._send_error(game.id, player_id, "Bot not found")
            return

        # Remove from game
        game.remove_player(bot_id)

        # Remove bot instance
        if game.id in self.bots and bot_id in self.bots[game.id]:
            del self.bots[game.id][bot_id]

        logger.info("Removed bot %s from game %s", bot_id, game.id)

        # Broadcast player left
        await self.manager.broadcast_to_game(
            ServerMessage(
                command=Command.LEFT,
                game_id=game.id,
                content={
                    "player_id": bot_id,
                    "username": bot_player.username,
                },
            ),
            game.id,
        )

        # Broadcast updated game state
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

    async def _process_bot_actions(self, game: Game) -> None:
        """Process bot actions sequentially.

        This is called after any game state change to check if a bot should take an action.
        No locks needed because this is a sequential turn-based game - only one action
        happens at a time.
        """
        if game.id not in self.bots:
            return

        await asyncio.sleep(0.5)  # Small delay for natural feel

        if game.state == GameState.BIDDING:
            await self._process_single_bot_bid(game)
        elif game.state == GameState.PICKING:
            await self._process_single_bot_pick(game)

    async def _process_single_bot_bid(self, game: Game) -> None:
        """Process ONE bot bid, then recursively check for more."""
        if game.id not in self.bots:
            return

        if game.state != GameState.BIDDING:
            return

        current_round = game.get_current_round()
        if not current_round:
            return

        if current_round.all_bids_placed(len(game.players)):
            return

        # Find first bot that hasn't bid
        for bot_id, bot in self.bots.get(game.id, {}).items():
            player = game.get_player(bot_id)
            if player and player.bid is None:
                # Bot makes bid
                hand = list(player.hand)
                bid = bot.make_bid(game, current_round.number, hand)
                bid = max(0, min(current_round.number, bid))

                player.bid = bid
                current_round.bids[bot_id] = bid

                event_recorder.record_bid(game, bot_id, bid)
                logger.info("Bot %s bids %d", bot_id, bid)

                await self.manager.broadcast_to_game(
                    ServerMessage(
                        command=Command.BADE,
                        game_id=game.id,
                        content={"player_id": bot_id, "bid": bid},
                    ),
                    game.id,
                )

                # Check if all bids placed after this one
                if current_round.all_bids_placed(len(game.players)):
                    await self._end_bidding_phase(game)
                    return

                # Process next bot bid after a delay
                await asyncio.sleep(0.3)
                await self._process_single_bot_bid(game)
                return  # Exit after processing chain

    def _validate_bot_pick_state(
        self, game: Game, trick: Trick | None
    ) -> tuple[str, BaseBot, Player] | None:
        """Validate that we can process a bot pick.

        Returns:
            Tuple of (picking_player_id, bot, player) if valid, None otherwise.

        """
        if not trick or trick.is_complete(len(game.players)):
            return None

        picking_player_id = trick.picking_player_id
        if picking_player_id not in self.bots.get(game.id, {}):
            return None

        if any(pc.player_id == picking_player_id for pc in trick.picked_cards):
            logger.warning("Bot %s already played in trick %d", picking_player_id, trick.number)
            return None

        bot = self.bots[game.id][picking_player_id]
        player = game.get_player(picking_player_id)
        if not player or not player.hand:
            return None

        return (picking_player_id, bot, player)

    def _choose_bot_card(
        self,
        game: Game,
        current_round: "Round",
        bot: BaseBot,
        player: Player,
        trick: Trick,
    ) -> tuple[CardId, TigressChoice | None]:
        """Choose a card for the bot to play.

        Returns:
            Tuple of (card_id, tigress_choice).

        """
        cards_in_trick = [pc.card_id for pc in trick.picked_cards]
        valid_cards = trick.get_valid_cards(player.hand, cards_in_trick)

        card_id = bot.pick_card(game, player.hand, cards_in_trick, valid_cards)
        card = get_card(card_id)

        tigress_choice: TigressChoice | None = None
        if card.is_tigress():
            tricks_won = current_round.get_tricks_won(player.id)
            bid = player.bid if player.bid is not None else 0
            need_more_wins = tricks_won < bid
            tigress_choice = TigressChoice.PIRATE if need_more_wins else TigressChoice.ESCAPE

        return (card_id, tigress_choice)

    async def _execute_bot_pick(
        self,
        game: Game,
        player: Player,
        trick: Trick,
        card_id: CardId,
        tigress_choice: TigressChoice | None,
    ) -> bool:
        """Execute the bot's card pick and broadcast.

        Returns:
            True if successful, False otherwise.

        """
        if card_id in player.hand:
            player.hand.remove(card_id)

        if trick.is_complete(len(game.players)):
            logger.warning("Trick completed before bot %s could play", player.id)
            player.hand.append(card_id)
            return False

        if not trick.add_card(player.id, card_id, tigress_choice):
            logger.error(
                "Bot %s card rejected - already picked in trick %d", player.id, trick.number
            )
            player.hand.append(card_id)
            return False

        card = get_card(card_id)
        logger.info(
            "Bot %s plays %s%s",
            player.id,
            card,
            f" as {tigress_choice.value}" if tigress_choice else "",
        )

        event_recorder.record_card_played(
            game, player.id, card_id.value, tigress_choice.value if tigress_choice else None
        )

        pick_content: dict[str, Any] = {"player_id": player.id, "card_id": card_id.value}
        if tigress_choice:
            pick_content["tigress_choice"] = tigress_choice.value

        await self.manager.broadcast_to_game(
            ServerMessage(
                command=Command.PICKED,
                game_id=game.id,
                content=pick_content,
            ),
            game.id,
        )
        return True

    async def _process_single_bot_pick(self, game: Game) -> None:
        """Process ONE bot card pick.

        Does NOT recursively continue - the game flow handles continuation
        through _prompt_continue or _advance_to_next_player.
        """
        if game.id not in self.bots or game.state != GameState.PICKING:
            return

        current_round = game.get_current_round()
        if not current_round:
            return

        trick = current_round.get_current_trick()
        bot_info = self._validate_bot_pick_state(game, trick)
        if not bot_info or not trick:
            return

        _picking_player_id, bot, player = bot_info
        card_id, tigress_choice = self._choose_bot_card(game, current_round, bot, player, trick)

        if not await self._execute_bot_pick(game, player, trick, card_id, tigress_choice):
            return

        # Handle what comes next - NO recursive calls here
        if trick.is_complete(len(game.players)):
            await self._complete_trick(game, current_round, trick)
        else:
            await self._advance_to_next_player(game, current_round, trick)
            next_player_id = trick.picking_player_id
            if next_player_id in self.bots.get(game.id, {}):
                await asyncio.sleep(0.5)
                await self._process_single_bot_pick(game)

    # Pirate ability resolution handlers

    async def _handle_resolve_rosie(
        self, game: Game, player_id: str, content: dict[str, Any]
    ) -> None:
        """Handle RESOLVE_ROSIE command - choose who starts next trick."""
        current_round = game.get_current_round()
        if not current_round:
            await self._send_error(game.id, player_id, "No active round")
            return

        chosen_player_id = content.get("chosen_player_id")
        if not chosen_player_id or not game.get_player(chosen_player_id):
            await self._send_error(game.id, player_id, "Invalid player chosen")
            return

        if not current_round.ability_state.resolve_rosie(player_id, chosen_player_id):
            await self._send_error(game.id, player_id, "Cannot resolve Rosie ability")
            return

        logger.info("Player %s chose %s to start next trick (Rosie)", player_id, chosen_player_id)

        # Get the pending ability and last trick
        ability = None
        for ab in current_round.ability_state.pending_abilities:
            if ab.player_id == player_id and ab.ability_type == AbilityType.CHOOSE_STARTER:
                ability = ab
                break

        trick = current_round.tricks[-1] if current_round.tricks else None
        if ability and trick:
            await self._ability_resolved(game, current_round, ability, trick)

    async def _handle_resolve_bendt(
        self, game: Game, player_id: str, content: dict[str, Any]
    ) -> None:
        """Handle RESOLVE_BENDT command - discard cards after drawing."""
        current_round = game.get_current_round()
        if not current_round:
            await self._send_error(game.id, player_id, "No active round")
            return

        discard_ids = content.get("discard_cards", [])
        try:
            discard_cards = [CardId(card_id) for card_id in discard_ids]
        except (ValueError, TypeError):
            await self._send_error(game.id, player_id, "Invalid card IDs")
            return

        # Get the pending ability
        ability = current_round.ability_state.get_pending_ability(player_id)
        if not ability or ability.ability_type != AbilityType.DRAW_DISCARD:
            await self._send_error(game.id, player_id, "No pending Bendt ability")
            return

        # Verify player has these cards
        player = game.get_player(player_id)
        if not player:
            await self._send_error(game.id, player_id, "Player not found")
            return

        for card_id in discard_cards:
            if card_id not in player.hand:
                await self._send_error(game.id, player_id, "Card not in hand")
                return

        # Remove discarded cards from hand
        for card_id in discard_cards:
            player.hand.remove(card_id)

        if not current_round.ability_state.resolve_bendt(
            player_id, ability.drawn_cards, discard_cards
        ):
            await self._send_error(game.id, player_id, "Cannot resolve Bendt ability")
            return

        logger.info("Player %s discarded %d cards (Bendt)", player_id, len(discard_cards))

        trick = current_round.tricks[-1] if current_round.tricks else None
        if trick:
            await self._ability_resolved(game, current_round, ability, trick)

    async def _handle_resolve_roatan(
        self, game: Game, player_id: str, content: dict[str, Any]
    ) -> None:
        """Handle RESOLVE_ROATAN command - declare extra bet."""
        current_round = game.get_current_round()
        if not current_round:
            await self._send_error(game.id, player_id, "No active round")
            return

        extra_bet = content.get("extra_bet")
        if extra_bet not in (0, 10, 20):
            await self._send_error(game.id, player_id, "Invalid bet amount (must be 0, 10, or 20)")
            return

        if not current_round.ability_state.resolve_roatan(player_id, extra_bet):
            await self._send_error(game.id, player_id, "Cannot resolve Roatán ability")
            return

        logger.info("Player %s declared extra bet of %d (Roatán)", player_id, extra_bet)

        # Get the pending ability
        ability = None
        for ab in current_round.ability_state.pending_abilities:
            if ab.player_id == player_id and ab.ability_type == AbilityType.EXTRA_BET:
                ability = ab
                break

        trick = current_round.tricks[-1] if current_round.tricks else None
        if ability and trick:
            await self._ability_resolved(game, current_round, ability, trick)

    async def _handle_resolve_jade(
        self, game: Game, player_id: str, _content: dict[str, Any]
    ) -> None:
        """Handle RESOLVE_JADE command - acknowledge deck view."""
        current_round = game.get_current_round()
        if not current_round:
            await self._send_error(game.id, player_id, "No active round")
            return

        # Jade is auto-resolved, but player can acknowledge
        logger.info("Player %s acknowledged deck view (Jade)", player_id)

    async def _handle_resolve_harry(
        self, game: Game, player_id: str, content: dict[str, Any]
    ) -> None:
        """Handle RESOLVE_HARRY command - modify bid at end of round."""
        current_round = game.get_current_round()
        if not current_round:
            await self._send_error(game.id, player_id, "No active round")
            return

        modifier = content.get("modifier")
        if modifier not in (-1, 0, 1):
            await self._send_error(game.id, player_id, "Invalid modifier (must be -1, 0, or 1)")
            return

        if not current_round.ability_state.resolve_harry(player_id, modifier):
            await self._send_error(game.id, player_id, "Cannot resolve Harry ability")
            return

        logger.info("Player %s modified bid by %d (Harry)", player_id, modifier)

    async def _prompt_continue(self, game: Game) -> None:
        """Auto-continue to the next trick or round after a brief delay."""
        logger.info(
            "[AUTO-CONTINUE] Continuing game %s, round %d",
            game.id,
            game.current_round_number,
        )

        # Brief delay to let players see the trick result
        await asyncio.sleep(1.5)

        # Continue game flow
        current_round = game.get_current_round()
        if not current_round:
            logger.warning("[AUTO-CONTINUE] No current round for game %s", game.id)
            return

        # Determine what to do next based on game state
        logger.info(
            "[AUTO-CONTINUE] Round %d: %d/%d tricks complete",
            current_round.number,
            len(current_round.tricks),
            current_round.number,
        )
        if current_round.is_complete():
            logger.info("[AUTO-CONTINUE] Round complete, moving to next round")
            await self._complete_round(game, current_round)
        else:
            logger.info("[AUTO-CONTINUE] Starting new trick")
            await self._start_new_trick(game, current_round)
            await self._process_bot_actions(game)
