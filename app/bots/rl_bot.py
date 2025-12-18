"""Reinforcement Learning bot interface."""

from typing import List, Optional

import numpy as np

from app.bots.base_bot import BaseBot, BotDifficulty
from app.models.card import CardId
from app.models.game import Game


class RLBot(BaseBot):
    """
    Bot that uses a trained reinforcement learning model.

    This is an interface for RL agents trained using the Gymnasium environment.
    The actual model should be loaded and provided to this bot.

    Attributes:
        model: The trained RL model (should have predict() method)
        observation_builder: Function to convert game state to observation
    """

    def __init__(
        self,
        player_id: str,
        model: Optional[any] = None,
        difficulty: BotDifficulty = BotDifficulty.HARD,
    ):
        """
        Initialize RL bot.

        Args:
            player_id: Player ID
            model: Trained model with predict(observation) -> action method
            difficulty: Difficulty level (affects exploration)
        """
        super().__init__(player_id, difficulty)
        self.model = model

    def make_bid(self, game: Game, round_number: int, hand: List[CardId]) -> int:
        """
        Make a bid using the RL model.

        Args:
            game: Current game state
            round_number: Current round number
            hand: Bot's cards

        Returns:
            Model's predicted bid
        """
        if self.model is None:
            # Fallback to simple heuristic
            return self._fallback_bid(hand, round_number)

        # Build observation for bidding phase
        observation = self._build_bid_observation(game, hand, round_number)

        # Get model prediction
        action = self.model.predict(observation)

        # Action should be bid amount (0 to round_number)
        bid = int(action) if isinstance(action, (int, np.integer)) else int(action[0])
        bid = max(0, min(round_number, bid))

        return bid

    def pick_card(
        self,
        game: Game,
        hand: List[CardId],
        cards_in_trick: List[CardId],
        valid_cards: Optional[List[CardId]] = None,
    ) -> CardId:
        """
        Pick a card using the RL model.

        Args:
            game: Current game state
            hand: Bot's remaining cards
            cards_in_trick: Cards played in trick
            valid_cards: Valid cards to play

        Returns:
            Model's predicted card
        """
        playable = self._get_valid_cards(hand, valid_cards)
        if not playable:
            playable = hand

        if not playable:
            raise ValueError("No cards to play")

        if self.model is None:
            # Fallback
            return self._fallback_pick(playable)

        # Build observation for card picking phase
        observation = self._build_pick_observation(game, hand, cards_in_trick)

        # Get model prediction
        action = self.model.predict(observation)

        # Action should be index into valid cards
        if isinstance(action, (list, np.ndarray)):
            card_index = int(action[0])
        else:
            card_index = int(action)

        # Ensure valid index
        card_index = max(0, min(len(playable) - 1, card_index))

        return playable[card_index]

    def _build_bid_observation(self, game: Game, hand: List[CardId], round_number: int) -> np.ndarray:
        """
        Build observation vector for bidding phase.

        This should match the observation space in the Gymnasium environment.
        """
        # Placeholder - should match gym_env observation space
        obs = np.zeros(100)  # Adjust size as needed
        # TODO: Implement proper observation encoding
        return obs

    def _build_pick_observation(
        self, game: Game, hand: List[CardId], cards_in_trick: List[CardId]
    ) -> np.ndarray:
        """
        Build observation vector for card picking phase.

        This should match the observation space in the Gymnasium environment.
        """
        # Placeholder - should match gym_env observation space
        obs = np.zeros(200)  # Adjust size as needed
        # TODO: Implement proper observation encoding
        return obs

    def _fallback_bid(self, hand: List[CardId], round_number: int) -> int:
        """Simple fallback bidding when no model available."""
        from app.bots.rule_based_bot import RuleBasedBot
        from app.models.game import Game

        # Use rule-based logic as fallback
        temp_game = Game(id="temp", slug="temp")
        rule_bot = RuleBasedBot(self.player_id)
        return rule_bot.make_bid(temp_game, round_number, hand)

    def _fallback_pick(self, playable: List[CardId]) -> CardId:
        """Simple fallback card picking when no model available."""
        import random

        return random.choice(playable)
