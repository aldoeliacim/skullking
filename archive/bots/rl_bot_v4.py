"""Reinforcement Learning bot interface."""

import logging
import random
from collections.abc import Callable
from typing import Any

import numpy as np

from app.bots.base_bot import BaseBot, BotDifficulty
from app.constants import (
    CARD_ENCODING_DIMENSIONS,
    HIGH_CARD_THRESHOLD,
    HIGH_STRENGTH_THRESHOLD,
    LATE_ROUND_THRESHOLD,
    MAX_HAND_SIZE,
    MAX_OPPONENTS_TRACKED,
    MAX_PLAYERS_IN_TRICK,
    MIN_PIRATES_FOR_BONUS,
    MIN_SUIT_COUNT,
    OPPONENT_PATTERN_DIMENSIONS,
    SUIT_CONCENTRATION_FACTOR,
)
from app.models.card import Card, CardId, get_card
from app.models.enums import GameState
from app.models.game import Game

logger = logging.getLogger(__name__)


class RLBot(BaseBot):
    """Bot that uses a trained reinforcement learning model.

    This is an interface for RL agents trained using the Gymnasium environment.
    The actual model should be loaded and provided to this bot.

    Attributes:
        model: The trained RL model (should have predict() method)
        observation_builder: Function to convert game state to observation

    """

    def __init__(
        self,
        player_id: str,
        model: Any | None = None,
        difficulty: BotDifficulty = BotDifficulty.HARD,
    ) -> None:
        """Initialize RL bot.

        Args:
            player_id: Player ID
            model: Trained model with predict(observation) -> action method
            difficulty: Difficulty level (affects exploration)

        """
        super().__init__(player_id, difficulty)
        self.model = model

    def make_bid(self, game: Game, round_number: int, hand: list[CardId]) -> int:
        """Make a bid using the RL model.

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
        # predict() returns (action, states) tuple in stable-baselines3
        result = self.model.predict(observation, deterministic=True)
        action = result[0] if isinstance(result, tuple) else result

        # Action should be bid amount (0 to round_number)
        if isinstance(action, np.ndarray):
            bid = int(action.item()) if action.ndim == 0 else int(action[0])
        elif isinstance(action, int | np.integer):
            bid = int(action)
        else:
            bid = int(action[0])
        return max(0, min(round_number, bid))

    def pick_card(
        self,
        game: Game,
        hand: list[CardId],
        cards_in_trick: list[CardId],
        valid_cards: list[CardId] | None = None,
    ) -> CardId:
        """Pick a card using the RL model.

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
            msg = "No cards to play"
            raise ValueError(msg)

        if self.model is None:
            # Fallback
            return self._fallback_pick(playable)

        # Build observation for card picking phase
        observation = self._build_pick_observation(game, hand, cards_in_trick)

        # Try deterministic prediction first
        card = self._try_model_prediction(observation, hand, playable, deterministic=True)
        if card is not None:
            return card

        # Model chose invalid card - log warning with details for debugging
        logger.warning(
            "[RL_BOT_INVALID_CHOICE] Model chose invalid card! "
            "hand=%s, valid_cards=%s, cards_in_trick=%s. "
            "This indicates a bug in action masking or model training.",
            [c.value for c in hand],
            [c.value for c in playable],
            [c.value for c in cards_in_trick],
        )

        # Try stochastic sampling - might pick a different valid card
        card = self._retry_with_stochastic(observation, hand, playable)
        if card is not None:
            return card

        # All retries failed - use heuristic fallback
        logger.error(
            "[RL_BOT_FALLBACK] All inference attempts failed! "
            "Using heuristic fallback. playable=%s",
            [c.value for c in playable],
        )
        return self._heuristic_pick(playable, cards_in_trick)

    def _try_model_prediction(
        self,
        observation: np.ndarray,
        hand: list[CardId],
        playable: list[CardId],
        *,
        deterministic: bool = True,
    ) -> CardId | None:
        """Try to get a valid card from model prediction."""
        if self.model is None:
            return None
        result = self.model.predict(observation, deterministic=deterministic)
        action = result[0] if isinstance(result, tuple) else result
        card_index = self._extract_action_index(action)

        if 0 <= card_index < len(hand):
            chosen_card = hand[card_index]
            if chosen_card in playable:
                return chosen_card
        return None

    def _extract_action_index(self, action: Any) -> int:
        """Extract integer index from model action output."""
        if isinstance(action, np.ndarray):
            return int(action.item()) if action.ndim == 0 else int(action[0])
        if isinstance(action, list):
            return int(action[0])
        return int(action)

    def _retry_with_stochastic(
        self, observation: np.ndarray, hand: list[CardId], playable: list[CardId]
    ) -> CardId | None:
        """Retry prediction with stochastic sampling."""
        if self.model is None:
            return None
        for retry in range(3):
            card = self._try_model_prediction(observation, hand, playable, deterministic=False)
            if card is not None:
                logger.info("[RL_BOT_RETRY_SUCCESS] Stochastic retry %d succeeded", retry + 1)
                return card
        return None

    def _heuristic_pick(self, playable: list[CardId], cards_in_trick: list[CardId]) -> CardId:
        """Pick a card using simple heuristics when model fails.

        Strategy: If leading, play a medium card. If following, play lowest valid.
        """
        if not cards_in_trick:
            # Leading - play a medium strength card
            sorted_cards = sorted(playable, key=self._card_strength)
            mid_idx = len(sorted_cards) // 2
            return sorted_cards[mid_idx]
        # Following - play lowest valid card to minimize risk
        return min(playable, key=self._card_strength)

    def _card_strength(self, card_id: CardId) -> int:
        """Estimate card strength for heuristic picking."""
        card = get_card(card_id)
        # Map card types to strength values
        strength_map = {
            "king": 1000,
            "pirate": 900,
            "mermaid": 850,
            "kraken": 800,
            "whale": 800,
            "tigress": 100,
            "escape": 0,
            "loot": 0,
        }
        card_type = card.card_type.value
        if card_type in strength_map:
            return strength_map[card_type]
        # Suit cards - roger (trump) is strongest
        number = card.get_number() or 0
        if card_type == "roger":
            return 500 + number
        return number

    def _build_bid_observation(
        self, game: Game, hand: list[CardId], _round_number: int
    ) -> np.ndarray:
        """Build observation vector for bidding phase.

        Matches the enhanced observation space (171 dims).
        """
        return self._build_observation(game, hand, [], GameState.BIDDING)

    def _build_pick_observation(
        self, game: Game, hand: list[CardId], cards_in_trick: list[CardId]
    ) -> np.ndarray:
        """Build observation vector for card picking phase.

        Matches the enhanced observation space (171 dims).
        """
        return self._build_observation(game, hand, cards_in_trick, GameState.PICKING)

    def _build_observation(
        self, game: Game, hand: list[CardId], cards_in_trick: list[CardId], state: GameState
    ) -> np.ndarray:
        """Build complete observation vector (171 dims) matching the gym environment."""
        obs: list[float] = []
        current_round = game.get_current_round()
        player = game.get_player(self.player_id)

        # 1. GAME PHASE (4 dims)
        obs.extend(self._encode_game_phase(state))

        # 2. COMPACT HAND ENCODING (90 dims: 10 cards x 9 features)
        obs.extend(self._encode_hand(hand))

        # 3. TRICK STATE (36 dims: 4 players x 9 features)
        obs.extend(self._encode_trick_state(cards_in_trick))

        # 4. BIDDING CONTEXT (8 dims)
        obs.extend(self._encode_bidding_context(current_round, player, hand))

        # 5. OPPONENT STATE (9 dims: 3 opponents x 3 features)
        obs.extend(self._encode_opponent_state(game))

        # 6. HAND STRENGTH BREAKDOWN (4 dims)
        obs.extend(self._encode_hand_strength_breakdown(hand))

        # 7. TRICK POSITION (4 dims)
        obs.extend(self._encode_trick_position(cards_in_trick))

        # 8. OPPONENT PATTERNS (6 dims)
        obs.extend(self._encode_opponent_patterns(game))

        # 9. CARDS PLAYED COUNT (5 dims)
        obs.extend(self._encode_cards_played(game))

        # 10. ROUND PROGRESSION (1 dim)
        obs.append(self._encode_round_progression(current_round))

        # 11. BID PRESSURE (1 dim)
        obs.append(self._calculate_bid_pressure(game))

        # 12. POSITION ADVANTAGE (1 dim)
        obs.append(self._calculate_position_advantage(game))

        # 13. TRUMP STRENGTH (2 dims)
        obs.extend(self._encode_trump_strength(game, hand))

        return np.array(obs, dtype=np.float32)

    def _encode_game_phase(self, state: GameState) -> list[float]:
        """Encode game phase as one-hot vector (4 dims)."""
        phase = [0.0] * 4
        state_map = {
            GameState.PENDING: 0,
            GameState.BIDDING: 1,
            GameState.PICKING: 2,
            GameState.ENDED: 3,
        }
        phase[state_map.get(state, 0)] = 1.0
        return phase

    def _encode_hand(self, hand: list[CardId]) -> list[float]:
        """Encode hand cards (90 dims: 10 cards x 9 features)."""
        obs: list[float] = []
        for i in range(MAX_HAND_SIZE):
            if i < len(hand):
                card = get_card(hand[i])
                obs.extend(self._encode_card_compact(card))
            else:
                obs.extend([0.0] * CARD_ENCODING_DIMENSIONS)
        return obs

    def _encode_trick_state(self, cards_in_trick: list[CardId]) -> list[float]:
        """Encode trick state (36 dims: 4 players x 9 features)."""
        obs: list[float] = []
        for i in range(MAX_PLAYERS_IN_TRICK):
            if i < len(cards_in_trick) and cards_in_trick[i]:
                card = get_card(cards_in_trick[i])
                obs.extend(self._encode_card_compact(card))
            else:
                obs.extend([0.0] * CARD_ENCODING_DIMENSIONS)
        return obs

    def _encode_bidding_context(
        self,
        current_round: Any,
        player: Any,
        hand: list[CardId],
    ) -> list[float]:
        """Encode bidding context (8 dims)."""
        if current_round and player:
            tricks_won = current_round.get_tricks_won(self.player_id)
            tricks_remaining = current_round.number - len(current_round.tricks)
            bid = player.bid if player.bid is not None else 0
            tricks_needed = bid - tricks_won

            round_num = current_round.number if current_round else 1
            hand_strength = self._estimate_hand_strength(hand, round_num) / 10.0
            return [
                current_round.number / 10.0,
                len(hand) / 10.0,
                bid / 10.0,
                tricks_won / 10.0,
                max(tricks_needed, -10) / 10.0,
                tricks_remaining / 10.0,
                1.0 if tricks_needed <= tricks_remaining else 0.0,
                hand_strength,
            ]
        return [0.0] * 8

    def _encode_opponent_state(self, game: Game) -> list[float]:
        """Encode opponent state (9 dims: 3 opponents x 3 features)."""
        obs: list[float] = []
        opponent_count = 0
        for p in game.players:
            if p.id != self.player_id and opponent_count < MAX_OPPONENTS_TRACKED:
                obs.extend(
                    [
                        p.bid / 10.0 if p.bid is not None else 0.0,
                        p.score / 100.0,
                        p.tricks_won / 10.0,
                    ]
                )
                opponent_count += 1
        while opponent_count < MAX_OPPONENTS_TRACKED:
            obs.extend([0.0] * 3)
            opponent_count += 1
        return obs

    def _encode_hand_strength_breakdown(self, hand: list[CardId]) -> list[float]:
        """Encode hand strength breakdown (4 dims)."""
        high_standard = self._count_card_type(
            hand,
            lambda c: c.is_standard_suit()
            and c.number is not None
            and c.number >= HIGH_CARD_THRESHOLD,
        )
        return [
            self._count_card_type(hand, lambda c: c.is_pirate()) / 5.0,
            self._count_card_type(hand, lambda c: c.is_king()) / 4.0,
            self._count_card_type(hand, lambda c: c.is_mermaid()) / 2.0,
            high_standard / 10.0,
        ]

    def _encode_round_progression(self, current_round: Any) -> float:
        """Encode round progression (1 dim)."""
        if current_round:
            return float(len(current_round.tricks)) / float(max(current_round.number, 1))
        return 0.0

    def _encode_card_compact(self, card: Card) -> list[float]:
        """Encode card with 9 features. Handles all 74 cards."""
        encoding = []

        # Card type one-hot (5 dims) - grouped for feature efficiency
        # [0] = Standard suits (has number)
        # [1] = Pirate-like (Pirate, Tigress as pirate)
        # [2] = King (Skull King)
        # [3] = Mermaid
        # [4] = Escape-like (Escape, Loot, Tigress as escape)
        # Note: Beasts (Whale, Kraken) handled via special flags
        card_type_vec = [0.0] * 5
        if card.is_standard_suit() or card.is_roger():
            card_type_vec[0] = 1.0
        elif card.is_pirate():
            card_type_vec[1] = 1.0
        elif card.is_tigress():
            # Tigress can be pirate or escape - encode as both
            card_type_vec[1] = 0.5  # Partial pirate
            card_type_vec[4] = 0.5  # Partial escape
        elif card.is_king():
            card_type_vec[2] = 1.0
        elif card.is_mermaid():
            card_type_vec[3] = 1.0
        elif card.is_escape() or card.is_loot():
            card_type_vec[4] = 1.0
        elif card.is_whale() or card.is_kraken():
            # Beasts get their own partial encoding
            card_type_vec[2] = 0.5  # Beast power similar to king
        encoding.extend(card_type_vec)

        # Number (1 dim)
        encoding.append(card.number / 14.0 if card.number else 0.0)

        # Special flags (3 dims) - expanded meaning:
        # [0] = Is high-power card (Pirate, Tigress, Beast)
        # [1] = Is King or Beast
        # [2] = Is Mermaid or counter-card
        encoding.extend(
            [
                1.0 if (card.is_pirate() or card.is_tigress() or card.is_beast()) else 0.0,
                1.0 if (card.is_king() or card.is_beast()) else 0.0,
                1.0 if card.is_mermaid() else 0.0,
            ]
        )

        return encoding

    def _count_card_type(self, hand: list[CardId], predicate: Callable[[Card], bool]) -> int:
        """Count cards matching a predicate."""
        return sum(1 for card_id in hand if predicate(get_card(card_id)))

    def _evaluate_card_strength(self, card: Card) -> float:
        """Evaluate card strength (0.0 to 1.0).

        Handles all 74 cards using a priority-based evaluation.
        """
        # Special cards with fixed values
        special_cards = [
            (card.is_king, 0.95),
            (card.is_pirate, 0.8),
            (card.is_tigress, 0.55),  # Flexible - can be pirate (0.8) or escape (0.1)
            (card.is_whale, 0.4),  # Unpredictable - highest suit wins
            (card.is_mermaid, 0.35),
            (card.is_kraken, 0.25),  # Nobody wins - disruptive
            (card.is_loot, 0.08),  # Acts like escape but alliance bonus
            (card.is_escape, 0.05),
        ]

        for check_func, strength in special_cards:
            if check_func():
                return strength

        # Number-based cards
        if card.is_roger():
            # Trump suit - higher value
            return 0.5 + (card.number / 14.0) * 0.35 if card.number else 0.5
        if card.is_standard_suit():
            return 0.2 + (card.number / 14.0) * 0.35 if card.number else 0.2

        # Default for other cards
        return 0.3

    def _estimate_hand_strength(self, hand: list[CardId], round_number: int) -> float:
        """Enhanced hand strength estimation."""
        if not hand:
            return 0.0

        card_objects = [get_card(cid) for cid in hand]
        base_strength = sum(self._evaluate_card_strength(c) for c in card_objects)

        # Suit distribution bonus
        suit_counts = self._count_cards_by_suit(hand)
        max_suit_count = max(suit_counts.values()) if suit_counts else 0
        if max_suit_count >= max(round_number * SUIT_CONCENTRATION_FACTOR, MIN_SUIT_COUNT):
            base_strength += 0.5

        # Special card synergies
        has_mermaid = any(c.is_mermaid() for c in card_objects)
        high_cards = sum(
            1 for c in card_objects if self._evaluate_card_strength(c) > HIGH_STRENGTH_THRESHOLD
        )
        if has_mermaid and high_cards >= MIN_PIRATES_FOR_BONUS:
            base_strength -= 0.5

        # Pirate strength in later rounds
        pirates = sum(1 for c in card_objects if c.is_pirate())
        if round_number >= LATE_ROUND_THRESHOLD and pirates >= MIN_PIRATES_FOR_BONUS:
            base_strength += 0.3

        # Escape penalty
        escapes = sum(1 for c in card_objects if c.is_escape())
        if escapes > 0:
            base_strength -= escapes * 0.4

        # King bonus
        kings = sum(1 for c in card_objects if c.is_king())
        if kings >= 1:
            base_strength += 0.2

        return max(0, round(base_strength))

    def _count_cards_by_suit(self, hand: list[CardId]) -> dict[str, int]:
        """Count cards by suit."""
        suit_counts: dict[str, int] = {}
        for card_id in hand:
            card = get_card(card_id)
            if card.is_standard_suit() and hasattr(card, "card_type"):
                suit = card.card_type.name
                suit_counts[suit] = suit_counts.get(suit, 0) + 1
        return suit_counts

    def _encode_trick_position(self, cards_in_trick: list[CardId]) -> list[float]:
        """Encode trick position (4 dims)."""
        position = [0.0] * MAX_PLAYERS_IN_TRICK
        num_played = len([c for c in cards_in_trick if c])
        if num_played < MAX_PLAYERS_IN_TRICK:
            position[num_played] = 1.0
        return position

    def _encode_opponent_patterns(self, game: Game) -> list[float]:
        """Encode opponent bidding patterns (6 dims)."""
        patterns: list[float] = []

        for p in game.players:
            if p.id != self.player_id and len(patterns) < OPPONENT_PATTERN_DIMENSIONS:
                bids = []
                errors = []
                for round_obj in game.rounds:
                    if p.id in round_obj.bids:
                        bid = round_obj.bids[p.id]
                        tricks_won = round_obj.get_tricks_won(p.id)
                        bids.append(bid)
                        errors.append(abs(bid - tricks_won))

                avg_bid = float(np.mean(bids)) if bids else 0.0
                avg_error = float(np.mean(errors)) if errors else 0.0
                patterns.append(avg_bid / 10.0)
                patterns.append(avg_error / 5.0)

        while len(patterns) < OPPONENT_PATTERN_DIMENSIONS:
            patterns.append(0.0)
        return patterns[:OPPONENT_PATTERN_DIMENSIONS]

    def _encode_cards_played(self, game: Game) -> list[float]:
        """Encode cards played this round (5 dims)."""
        current_round = game.get_current_round()
        if not current_round:
            return [0.0] * 5

        counts = self._count_card_types_in_round(current_round)
        total = current_round.number * len(game.players)
        return [count / max(total, 1) for count in counts]

    def _count_card_types_in_round(self, current_round: Any) -> list[int]:
        """Count different card types played in the round."""
        pirates = kings = mermaids = escapes = high_cards = 0

        for trick in current_round.tricks:
            for card_id in trick.get_all_card_ids():
                if not card_id:
                    continue
                card = get_card(card_id)
                if card.is_pirate():
                    pirates += 1
                elif card.is_king():
                    kings += 1
                elif card.is_mermaid():
                    mermaids += 1
                elif card.is_escape():
                    escapes += 1
                elif self._is_high_standard_card(card):
                    high_cards += 1

        return [pirates, kings, mermaids, escapes, high_cards]

    def _is_high_standard_card(self, card: Card) -> bool:
        """Check if card is a high-value standard suit card."""
        return (
            card.is_standard_suit()
            and card.number is not None
            and card.number >= HIGH_CARD_THRESHOLD
        )

    def _calculate_bid_pressure(self, game: Game) -> float:
        """Calculate bid pressure (1 dim)."""
        current_round = game.get_current_round()
        player = game.get_player(self.player_id)
        if not current_round or not player or player.bid is None:
            return 0.0

        tricks_won = current_round.get_tricks_won(self.player_id)
        tricks_needed = player.bid - tricks_won
        tricks_remaining = current_round.number - len(current_round.tricks)

        if tricks_needed <= 0:
            return -0.5
        if tricks_remaining == 0:
            return 1.0 if tricks_needed > 0 else 0.0
        return min(tricks_needed / max(tricks_remaining, 1), 1.0)

    def _calculate_position_advantage(self, game: Game) -> float:
        """Calculate position advantage (1 dim)."""
        current_round = game.get_current_round()
        if not current_round or not current_round.tricks:
            return 0.0

        last_count = 0
        total = 0

        for trick in current_round.tricks:
            if len(trick.picked_cards) == len(game.players):
                total += 1
                if trick.picked_cards and trick.picked_cards[-1].player_id == self.player_id:
                    last_count += 1

        return last_count / max(total, 1)

    def _encode_trump_strength(self, game: Game, hand: list[CardId]) -> list[float]:
        """Encode trump strength (2 dims)."""
        if not hand:
            return [0.0, 0.0]

        our_strengths = [self._evaluate_card_strength(get_card(cid)) for cid in hand]
        our_best = max(our_strengths) if our_strengths else 0.0
        our_avg = float(np.mean(our_strengths)) if our_strengths else 0.0

        current_round = game.get_current_round()
        if current_round:
            seen_strengths = [
                self._evaluate_card_strength(get_card(card_id))
                for trick in current_round.tricks
                for card_id in trick.get_all_card_ids()
                if card_id
            ]

            if seen_strengths:
                seen_avg = float(np.mean(seen_strengths))
                return [our_best - seen_avg, our_avg - seen_avg]

        return [our_best, our_avg]

    def _fallback_bid(self, hand: list[CardId], round_number: int) -> int:
        """Provide fallback bidding when no model available."""
        # Import here to avoid circular dependency
        from app.bots.rule_based_bot import RuleBasedBot  # noqa: PLC0415

        # Use rule-based logic as fallback
        temp_game = Game(id="temp", slug="temp")
        rule_bot = RuleBasedBot(self.player_id)
        return rule_bot.make_bid(temp_game, round_number, hand)

    def _fallback_pick(self, playable: list[CardId]) -> CardId:
        """Pick fallback card when no model available."""
        return random.choice(playable)  # noqa: S311 - game AI, not security-sensitive
