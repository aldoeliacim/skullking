"""Pirate ability resolution handlers.

This module handles the resolution of pirate abilities:
- Rosie: Choose who starts the next trick
- Bendt: Draw 2 cards, discard 2 cards
- Roatan: Declare extra bet (0, 10, or 20)
- Jade: View undealt cards (auto-resolved)
- Harry: Modify bid by +1 or -1 at end of round
"""

import logging
from typing import TYPE_CHECKING, Any, Protocol

from app.api.responses import Command, ErrorCode, ServerMessage
from app.models.card import CardId
from app.models.pirate_ability import AbilityType, PendingAbility

if TYPE_CHECKING:
    from app.api.websocket import ConnectionManager
    from app.models.game import Game
    from app.models.round import Round
    from app.models.trick import Trick

logger = logging.getLogger(__name__)


class GameFlowCallback(Protocol):
    """Protocol for game flow callbacks after ability resolution."""

    async def complete_round(self, game: "Game", current_round: "Round") -> None:
        """Complete the current round."""
        ...

    async def start_new_trick(self, game: "Game", current_round: "Round") -> None:
        """Start a new trick."""
        ...

    async def process_bot_actions(self, game: "Game") -> None:
        """Process bot actions after ability resolution."""
        ...


class AbilityHandlers:
    """Handles pirate ability resolution for WebSocket commands.

    Processes ability resolution commands (RESOLVE_ROSIE, RESOLVE_BENDT, etc.)
    and broadcasts results to connected players.
    """

    def __init__(
        self,
        manager: "ConnectionManager",
        game_flow: GameFlowCallback,
    ) -> None:
        """Initialize handler with dependencies.

        Args:
            manager: WebSocket connection manager for broadcasting
            game_flow: Callback interface for game flow continuation
        """
        self.manager = manager
        self.game_flow = game_flow

    async def _send_error(self, game_id: str, player_id: str, message: str) -> None:
        """Send error message to a specific player."""
        await self.manager.send_to_player(
            ServerMessage(
                command=Command.ERROR,
                game_id=game_id,
                content={"message": message},
            ),
            game_id,
            player_id,
        )

    async def _ability_resolved(
        self, game: "Game", current_round: "Round", ability: PendingAbility, _trick: "Trick"
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
            await self.game_flow.complete_round(game, current_round)
        else:
            await self.game_flow.start_new_trick(game, current_round)
            await self.game_flow.process_bot_actions(game)

    async def handle_resolve_rosie(
        self, game: "Game", player_id: str, content: dict[str, Any]
    ) -> None:
        """Handle RESOLVE_ROSIE command - choose who starts next trick."""
        current_round = game.get_current_round()
        if not current_round:
            await self._send_error(game.id, player_id, ErrorCode.NO_ACTIVE_ROUND)
            return

        chosen_player_id = content.get("chosen_player_id")
        if not chosen_player_id or not game.get_player(chosen_player_id):
            await self._send_error(game.id, player_id, ErrorCode.INVALID_PLAYER_CHOSEN)
            return

        if not current_round.ability_state.resolve_rosie(player_id, chosen_player_id):
            await self._send_error(game.id, player_id, ErrorCode.CANNOT_RESOLVE_ABILITY)
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

    async def handle_resolve_bendt(
        self, game: "Game", player_id: str, content: dict[str, Any]
    ) -> None:
        """Handle RESOLVE_BENDT command - discard cards after drawing."""
        current_round = game.get_current_round()
        if not current_round:
            await self._send_error(game.id, player_id, ErrorCode.NO_ACTIVE_ROUND)
            return

        discard_ids = content.get("discard_cards", [])
        try:
            discard_cards = [CardId(card_id) for card_id in discard_ids]
        except (ValueError, TypeError):
            await self._send_error(game.id, player_id, ErrorCode.INVALID_CARD_IDS)
            return

        # Get the pending ability
        ability = current_round.ability_state.get_pending_ability(player_id)
        if not ability or ability.ability_type != AbilityType.DRAW_DISCARD:
            await self._send_error(game.id, player_id, ErrorCode.NO_PENDING_ABILITY)
            return

        # Verify player has these cards
        player = game.get_player(player_id)
        if not player:
            await self._send_error(game.id, player_id, ErrorCode.PLAYER_NOT_FOUND)
            return

        for card_id in discard_cards:
            if card_id not in player.hand:
                await self._send_error(game.id, player_id, ErrorCode.CARD_NOT_IN_HAND)
                return

        # Remove discarded cards from hand
        for card_id in discard_cards:
            player.hand.remove(card_id)

        if not current_round.ability_state.resolve_bendt(
            player_id, ability.drawn_cards, discard_cards
        ):
            await self._send_error(game.id, player_id, ErrorCode.CANNOT_RESOLVE_ABILITY)
            return

        logger.info("Player %s discarded %d cards (Bendt)", player_id, len(discard_cards))

        trick = current_round.tricks[-1] if current_round.tricks else None
        if trick:
            await self._ability_resolved(game, current_round, ability, trick)

    async def handle_resolve_roatan(
        self, game: "Game", player_id: str, content: dict[str, Any]
    ) -> None:
        """Handle RESOLVE_ROATAN command - declare extra bet."""
        current_round = game.get_current_round()
        if not current_round:
            await self._send_error(game.id, player_id, ErrorCode.NO_ACTIVE_ROUND)
            return

        extra_bet = content.get("extra_bet")
        if extra_bet not in (0, 10, 20):
            await self._send_error(game.id, player_id, ErrorCode.INVALID_BET_AMOUNT)
            return

        if not current_round.ability_state.resolve_roatan(player_id, extra_bet):
            await self._send_error(game.id, player_id, ErrorCode.CANNOT_RESOLVE_ABILITY)
            return

        logger.info("Player %s declared extra bet of %d (Roatan)", player_id, extra_bet)

        # Get the pending ability
        ability = None
        for ab in current_round.ability_state.pending_abilities:
            if ab.player_id == player_id and ab.ability_type == AbilityType.EXTRA_BET:
                ability = ab
                break

        trick = current_round.tricks[-1] if current_round.tricks else None
        if ability and trick:
            await self._ability_resolved(game, current_round, ability, trick)

    async def handle_resolve_jade(
        self, game: "Game", player_id: str, _content: dict[str, Any]
    ) -> None:
        """Handle RESOLVE_JADE command - acknowledge deck view."""
        current_round = game.get_current_round()
        if not current_round:
            await self._send_error(game.id, player_id, ErrorCode.NO_ACTIVE_ROUND)
            return

        # Get the pending ability
        ability = current_round.ability_state.get_pending_ability(player_id)
        if not ability or ability.ability_type != AbilityType.VIEW_DECK:
            await self._send_error(game.id, player_id, ErrorCode.NO_PENDING_ABILITY)
            return

        # Resolve and continue game
        if not current_round.ability_state.resolve_jade(player_id):
            await self._send_error(game.id, player_id, ErrorCode.CANNOT_RESOLVE_ABILITY)
            return

        logger.info("Player %s acknowledged deck view (Jade)", player_id)

        trick = current_round.tricks[-1] if current_round.tricks else None
        if trick:
            await self._ability_resolved(game, current_round, ability, trick)

    async def handle_resolve_harry(
        self, game: "Game", player_id: str, content: dict[str, Any]
    ) -> None:
        """Handle RESOLVE_HARRY command - modify bid at end of round."""
        current_round = game.get_current_round()
        if not current_round:
            await self._send_error(game.id, player_id, ErrorCode.NO_ACTIVE_ROUND)
            return

        modifier = content.get("modifier")
        if modifier not in (-1, 0, 1):
            await self._send_error(game.id, player_id, ErrorCode.INVALID_MODIFIER)
            return

        if not current_round.ability_state.resolve_harry(player_id, modifier):
            await self._send_error(game.id, player_id, ErrorCode.CANNOT_RESOLVE_ABILITY)
            return

        logger.info("Player %s modified bid by %d (Harry)", player_id, modifier)

        # Continue round completion after Harry resolution
        await self.game_flow.complete_round(game, current_round)
