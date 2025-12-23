"""Rule-based bot with heuristic strategy."""

import random

from app.bots.base_bot import BaseBot, BotDifficulty
from app.models.card import CardId, get_card
from app.models.game import Game


class RuleBasedBot(BaseBot):
    """
    Bot that uses heuristics and game rules to make intelligent decisions.

    Bidding Strategy:
    - Count strong cards (high numbers, special cards)
    - Estimate tricks based on card strength distribution

    Playing Strategy:
    - If winning: Play high cards when likely to win
    - If losing: Play low cards or escapes to avoid winning
    - Handle special cards intelligently
    """

    def __init__(self, player_id: str, difficulty: BotDifficulty = BotDifficulty.MEDIUM):
        """Initialize rule-based bot."""
        super().__init__(player_id, difficulty)

    def make_bid(self, game: Game, round_number: int, hand: list[CardId]) -> int:
        """
        Make a bid based on hand strength.

        Strategy:
        - Count cards likely to win tricks
        - Skull King: almost always wins
        - Pirates: often win
        - High Jolly Rogers: likely to win
        - High standard suits: might win
        - Escapes/Low cards: unlikely to win

        Args:
            game: Current game state
            round_number: Current round number
            hand: Bot's cards

        Returns:
            Estimated bid
        """
        expected_tricks = 0.0

        for card_id in hand:
            card = get_card(card_id)

            # Estimate probability of winning with this card
            if card.is_king():
                expected_tricks += 0.9  # Very likely to win
            elif card.is_pirate():
                expected_tricks += 0.6  # Often wins
            elif card.is_whale() or card.is_kraken():
                expected_tricks += 0.3  # Special outcomes
            elif card.is_mermaid():
                expected_tricks += 0.4  # Can beat King
            elif card.is_roger():
                # Jolly Roger (trump) strength based on number
                if card.number >= 10:
                    expected_tricks += 0.7
                elif card.number >= 6:
                    expected_tricks += 0.4
                else:
                    expected_tricks += 0.2
            elif card.is_suit():
                # Standard suits - less likely
                if card.number >= 12:
                    expected_tricks += 0.3
                elif card.number >= 8:
                    expected_tricks += 0.15
            # Escapes: 0 probability

        # Apply difficulty modifier
        if self.difficulty == BotDifficulty.EASY:
            # Easy mode: less accurate bidding
            expected_tricks += random.uniform(-1.0, 1.0)
        elif self.difficulty == BotDifficulty.HARD:
            # Hard mode: more accurate
            expected_tricks += random.uniform(-0.2, 0.2)
        else:
            # Medium mode
            expected_tricks += random.uniform(-0.5, 0.5)

        # Round to nearest integer and clamp
        bid = round(expected_tricks)
        bid = max(0, min(round_number, bid))

        return bid

    def pick_card(
        self,
        game: Game,
        hand: list[CardId],
        cards_in_trick: list[CardId],
        valid_cards: list[CardId] | None = None,
    ) -> CardId:
        """
        Pick a card strategically.

        Strategy:
        - Calculate if we want to win this trick
        - If we want to win: play strong cards
        - If we want to lose: play weak cards or escapes

        Args:
            game: Current game state
            hand: Bot's remaining cards
            cards_in_trick: Cards played so far
            valid_cards: Valid cards to play

        Returns:
            CardId to play
        """
        playable = self._get_valid_cards(hand, valid_cards)
        if not playable:
            playable = hand

        if not playable:
            raise ValueError("No cards to play")

        # Determine strategy: should we try to win this trick?
        player = game.get_player(self.player_id)
        if not player or player.bid is None:
            # No bid yet, play conservatively
            return self._play_medium_card(playable, cards_in_trick)

        current_round = game.get_current_round()
        if not current_round:
            return random.choice(playable)

        tricks_won = current_round.get_tricks_won(self.player_id)
        tricks_remaining = current_round.number - len(current_round.tricks)
        tricks_needed = player.bid - tricks_won

        # Decide strategy
        if tricks_needed > tricks_remaining:
            # Can't possibly make bid, play conservatively
            strategy = "lose"
        elif tricks_needed == 0:
            # Already made bid, avoid winning more
            strategy = "lose"
        elif tricks_needed == tricks_remaining:
            # Must win all remaining tricks
            strategy = "win"
        elif tricks_won < player.bid:
            # Need to win some tricks
            strategy = "win" if tricks_needed > tricks_remaining // 2 else "medium"
        else:
            strategy = "lose"

        # Apply difficulty to strategy consistency
        if self.difficulty == BotDifficulty.EASY and random.random() < 0.3:
            # Easy bots make mistakes
            strategy = random.choice(["win", "lose", "medium"])

        # Execute strategy
        if strategy == "win":
            return self._play_strong_card(playable, cards_in_trick)
        if strategy == "lose":
            return self._play_weak_card(playable, cards_in_trick)
        return self._play_medium_card(playable, cards_in_trick)

    def _evaluate_card_strength(self, card_id: CardId, hand: list[CardId]) -> float:
        """
        Evaluate the strength of a card (0.0 to 1.0).

        Args:
            card_id: Card to evaluate
            hand: Full hand for context

        Returns:
            Strength score
        """
        card = get_card(card_id)

        if card.is_escape():
            return 0.0
        if card.is_king():
            return 1.0
        if card.is_pirate():
            return 0.85
        if card.is_mermaid():
            return 0.75
        if card.is_whale():
            return 0.5  # Special case
        if card.is_kraken():
            return 0.3  # Nobody wins
        if card.is_roger():
            # Trump suit
            return 0.4 + (card.number / 14) * 0.4
        if card.is_suit():
            # Standard suit
            return 0.1 + (card.number / 14) * 0.3
        return 0.2

    def _play_strong_card(self, playable: list[CardId], cards_in_trick: list[CardId]) -> CardId:
        """Play the strongest available card."""
        strengths = [
            (card_id, self._evaluate_card_strength(card_id, playable)) for card_id in playable
        ]
        strengths.sort(key=lambda x: x[1], reverse=True)
        return strengths[0][0]

    def _play_weak_card(self, playable: list[CardId], cards_in_trick: list[CardId]) -> CardId:
        """Play the weakest available card."""
        strengths = [
            (card_id, self._evaluate_card_strength(card_id, playable)) for card_id in playable
        ]
        strengths.sort(key=lambda x: x[1])
        return strengths[0][0]

    def _play_medium_card(self, playable: list[CardId], cards_in_trick: list[CardId]) -> CardId:
        """Play a medium-strength card."""
        strengths = [
            (card_id, self._evaluate_card_strength(card_id, playable)) for card_id in playable
        ]
        strengths.sort(key=lambda x: x[1])

        # Pick middle card
        middle_index = len(strengths) // 2
        return strengths[middle_index][0]
