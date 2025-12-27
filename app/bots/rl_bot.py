"""Reinforcement Learning bot interface.

This bot wraps a trained MaskablePPO model to play Skull King.
Uses the same observation encoding as the training environment
to ensure compatibility.

Note: Requires sb3_contrib for MaskablePPO support.
"""

import logging
from typing import TYPE_CHECKING, Any

from app.bots.base_bot import BaseBot, BotDifficulty
from app.bots.rule_based_bot import RuleBasedBot
from app.models.card import CardId

if TYPE_CHECKING:
    from app.models.game import Game

logger = logging.getLogger(__name__)


class RLBot(BaseBot):
    """Bot that uses a trained MaskablePPO model.

    This bot interfaces with RL models trained using the Gymnasium environment.
    It falls back to RuleBasedBot (HARD) when the model is unavailable.

    The observation building is delegated to a gym environment instance
    to ensure observations match exactly what the model was trained on.
    """

    def __init__(
        self,
        player_id: str,
        model: Any | None = None,
        difficulty: BotDifficulty = BotDifficulty.HARD,
    ) -> None:
        """Initialize RL bot.

        Args:
            player_id: Player ID for this bot
            model: Trained MaskablePPO model (or None to use fallback)
            difficulty: Difficulty level (used by fallback)

        """
        super().__init__(player_id, difficulty)
        self.model = model
        self._fallback = RuleBasedBot(player_id, BotDifficulty.HARD)
        self._gym_env: Any = None
        self._gym_env_available = True  # Track if import succeeded

        if model is None:
            logger.warning("RLBot initialized without model, using rule-based fallback")

    def _get_gym_env(self) -> Any | None:
        """Lazily create gym environment for observation building.

        Returns:
            SkullKingEnvMasked instance or None if import failed

        """
        if not self._gym_env_available:
            return None

        if self._gym_env is None:
            try:
                # Late import to avoid circular dependency and heavy torch import
                from app.gym_env import SkullKingEnvMasked  # noqa: PLC0415

                self._gym_env = SkullKingEnvMasked(
                    num_opponents=3,
                    opponent_bot_type="rule_based",
                    opponent_difficulty="hard",
                )
            except ImportError:
                logger.warning("Could not import SkullKingEnvMasked")
                self._gym_env_available = False
                return None
        return self._gym_env

    def make_bid(self, game: "Game", round_number: int, hand: list[CardId]) -> int:
        """Make a bid using the RL model or fallback.

        Args:
            game: Current game state
            round_number: Current round number (1-10)
            hand: Bot's cards for this round

        Returns:
            Bid amount (0 to round_number)

        """
        if self.model is None:
            return self._fallback.make_bid(game, round_number, hand)

        env = self._get_gym_env()
        if env is None:
            return self._fallback.make_bid(game, round_number, hand)

        try:
            # Sync env with current game state using public interface
            env.sync_game_state(game, self.player_id)

            # Get observation and action mask
            obs = env.get_observation()
            mask = env.action_masks()

            # Predict action
            action, _ = self.model.predict(obs, deterministic=True, action_masks=mask)
            bid = int(action.item() if hasattr(action, "item") else action)
            return max(0, min(round_number, bid))

        except (ValueError, IndexError, AttributeError) as e:
            logger.warning("RL bid failed: %s, using fallback", e)
            return self._fallback.make_bid(game, round_number, hand)

    def pick_card(
        self,
        game: "Game",
        hand: list[CardId],
        cards_in_trick: list[CardId],
        valid_cards: list[CardId] | None = None,
    ) -> CardId:
        """Pick a card using the RL model or fallback.

        Args:
            game: Current game state
            hand: Bot's remaining cards
            cards_in_trick: Cards already played in this trick
            valid_cards: Valid cards to play (optional)

        Returns:
            Card ID to play

        """
        if self.model is None:
            return self._fallback.pick_card(game, hand, cards_in_trick, valid_cards)

        env = self._get_gym_env()
        if env is None:
            return self._fallback.pick_card(game, hand, cards_in_trick, valid_cards)

        try:
            # Sync env with current game state using public interface
            env.sync_game_state(game, self.player_id)

            # Get observation and action mask
            obs = env.get_observation()
            mask = env.action_masks()

            # Predict action (card index)
            action, _ = self.model.predict(obs, deterministic=True, action_masks=mask)
            card_idx = int(action.item() if hasattr(action, "item") else action)

            # Map action to card
            if 0 <= card_idx < len(hand):
                return hand[card_idx]

            # Fallback if action is out of range
            logger.warning("RL action %d out of range, using fallback", card_idx)

        except (ValueError, IndexError, AttributeError) as e:
            logger.warning("RL pick failed: %s, using fallback", e)

        return self._fallback.pick_card(game, hand, cards_in_trick, valid_cards)

    def choose_tigress_mode(
        self, game: "Game", hand: list[CardId], cards_in_trick: list[CardId]
    ) -> str:
        """Choose Tigress mode (pirate or escape).

        Currently delegates to fallback as this requires special handling.
        """
        return self._fallback.choose_tigress_mode(game, hand, cards_in_trick)
