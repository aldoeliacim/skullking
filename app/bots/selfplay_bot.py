"""
Self-play bot that loads a trained PPO model to use as an opponent.

This enables advanced training where the agent trains against copies of itself,
leading to emergent strategies and continuous improvement.
"""

from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from app.bots.base_bot import BaseBot, BotDifficulty
from app.models.card import CardId
from app.models.game import Game


class SelfPlayBot(BaseBot):
    """
    Bot that uses a trained PPO model to play.

    Used for self-play training where the agent competes against
    past versions of itself.
    """

    def __init__(
        self,
        player_id: str,
        model_path: str,
        difficulty: BotDifficulty = BotDifficulty.HARD,
        deterministic: bool = False,
    ):
        """
        Initialize self-play bot.

        Args:
            player_id: Player ID for this bot
            model_path: Path to trained PPO model
            difficulty: Difficulty level (affects randomness)
            deterministic: If True, always picks best action
        """
        super().__init__(player_id, difficulty)

        if not Path(model_path).exists():
            raise ValueError(f"Model not found: {model_path}")

        self.model = PPO.load(model_path)
        self.deterministic = deterministic
        self.last_observation: np.ndarray | None = None

    def make_bid(self, game: Game, round_number: int, hand: list[CardId]) -> int:
        """
        Make a bid using the trained model.

        Note: This is simplified - ideally we'd create a full observation
        from the game state. For now, we use a heuristic similar to rule-based.
        """
        # Simple fallback: use rule-based strategy for bidding
        # Full implementation would require reconstructing the observation
        expected_tricks = 0.0

        for card_id in hand:
            from app.models.card import get_card

            card = get_card(card_id)

            if card.is_king():
                expected_tricks += 0.85
            elif card.is_pirate():
                expected_tricks += 0.55
            elif card.is_mermaid():
                expected_tricks += 0.4
            elif card.is_roger() and card.number >= 10:
                expected_tricks += 0.65
            elif card.is_roger() and card.number >= 6:
                expected_tricks += 0.35
            elif card.is_suit() and card.number >= 12:
                expected_tricks += 0.25

        # Add slight randomness based on difficulty
        if self.difficulty == BotDifficulty.EASY:
            expected_tricks += np.random.uniform(-1.0, 1.0)
        elif self.difficulty == BotDifficulty.MEDIUM:
            expected_tricks += np.random.uniform(-0.5, 0.5)
        else:
            expected_tricks += np.random.uniform(-0.2, 0.2)

        bid = round(expected_tricks)
        return max(0, min(round_number, bid))

    def pick_card(
        self,
        game: Game,
        hand: list[CardId],
        cards_in_trick: list[CardId],
        valid_cards: list[CardId] | None = None,
    ) -> CardId:
        """
        Pick a card using the trained model.

        Note: Simplified implementation using rule-based fallback.
        Full implementation would require observation reconstruction.
        """
        playable = self._get_valid_cards(hand, valid_cards)
        if not playable:
            playable = hand

        if not playable:
            raise ValueError("No cards to play")

        # Simple strategy: play medium-strength cards
        # Full model-based play would require proper observation
        from app.models.card import get_card

        strengths = []
        for card_id in playable:
            card = get_card(card_id)
            if card.is_king():
                strength = 1.0
            elif card.is_pirate():
                strength = 0.85
            elif card.is_mermaid():
                strength = 0.75
            elif card.is_roger():
                strength = 0.4 + (card.number / 14) * 0.4
            elif card.is_suit():
                strength = 0.1 + (card.number / 14) * 0.3
            else:
                strength = 0.1

            strengths.append((card_id, strength))

        # Sort and pick middle card (balanced strategy)
        strengths.sort(key=lambda x: x[1])
        middle_index = len(strengths) // 2

        # Add some randomness for variety
        if not self.deterministic:
            offset = np.random.randint(-1, 2)
            middle_index = max(0, min(len(strengths) - 1, middle_index + offset))

        return strengths[middle_index][0]
