"""Reinforcement Learning bot interface."""

from collections.abc import Callable
from typing import Any

import numpy as np

from app.bots.base_bot import BaseBot, BotDifficulty
from app.models.card import Card, CardId, get_card
from app.models.enums import GameState
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
        model: Any | None = None,
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

    def make_bid(self, game: Game, round_number: int, hand: list[CardId]) -> int:
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
        # predict() returns (action, states) tuple in stable-baselines3
        result = self.model.predict(observation, deterministic=True)
        if isinstance(result, tuple):
            action = result[0]
        else:
            action = result

        # Action should be bid amount (0 to round_number)
        if isinstance(action, np.ndarray):
            bid = int(action.item()) if action.ndim == 0 else int(action[0])
        elif isinstance(action, int | np.integer):
            bid = int(action)
        else:
            bid = int(action[0])
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
        # predict() returns (action, states) tuple in stable-baselines3
        result = self.model.predict(observation, deterministic=True)
        if isinstance(result, tuple):
            action = result[0]  # First element is the action
        else:
            action = result

        # Action should be index into valid cards
        if isinstance(action, np.ndarray):
            card_index = int(action.item()) if action.ndim == 0 else int(action[0])
        elif isinstance(action, list):
            card_index = int(action[0])
        else:
            card_index = int(action)

        # Ensure valid index
        card_index = max(0, min(len(playable) - 1, card_index))

        return playable[card_index]

    def _build_bid_observation(
        self, game: Game, hand: list[CardId], _round_number: int
    ) -> np.ndarray:
        """
        Build observation vector for bidding phase.
        Matches the enhanced observation space (171 dims).
        """
        return self._build_observation(game, hand, [], GameState.BIDDING)

    def _build_pick_observation(
        self, game: Game, hand: list[CardId], cards_in_trick: list[CardId]
    ) -> np.ndarray:
        """
        Build observation vector for card picking phase.
        Matches the enhanced observation space (171 dims).
        """
        return self._build_observation(game, hand, cards_in_trick, GameState.PICKING)

    def _build_observation(
        self, game: Game, hand: list[CardId], cards_in_trick: list[CardId], state: GameState
    ) -> np.ndarray:
        """
        Build complete observation vector (171 dims) matching the gym environment.
        """
        obs: list[float] = []
        current_round = game.get_current_round()
        player = game.get_player(self.player_id)

        # 1. GAME PHASE (4 dims)
        phase = [0.0] * 4
        state_map = {
            GameState.PENDING: 0,
            GameState.BIDDING: 1,
            GameState.PICKING: 2,
            GameState.ENDED: 3,
        }
        phase[state_map.get(state, 0)] = 1.0
        obs.extend(phase)

        # 2. COMPACT HAND ENCODING (90 dims: 10 cards × 9 features)
        for i in range(10):
            if i < len(hand):
                card = get_card(hand[i])
                obs.extend(self._encode_card_compact(card))
            else:
                obs.extend([0.0] * 9)

        # 3. TRICK STATE (36 dims: 4 players × 9 features)
        for i in range(4):
            if i < len(cards_in_trick) and cards_in_trick[i]:
                card = get_card(cards_in_trick[i])
                obs.extend(self._encode_card_compact(card))
            else:
                obs.extend([0.0] * 9)

        # 4. BIDDING CONTEXT (8 dims)
        if current_round and player:
            tricks_won = current_round.get_tricks_won(self.player_id)
            tricks_remaining = current_round.number - len(current_round.tricks)
            bid = player.bid if player.bid is not None else 0
            tricks_needed = bid - tricks_won

            round_num = current_round.number if current_round else 1
            hand_strength = self._estimate_hand_strength(hand, round_num) / 10.0
            obs.extend(
                [
                    current_round.number / 10.0,
                    len(hand) / 10.0,
                    bid / 10.0,
                    tricks_won / 10.0,
                    max(tricks_needed, -10) / 10.0,
                    tricks_remaining / 10.0,
                    1.0 if tricks_needed <= tricks_remaining else 0.0,
                    hand_strength,
                ]
            )
        else:
            obs.extend([0.0] * 8)

        # 5. OPPONENT STATE (9 dims: 3 opponents × 3 features)
        opponent_count = 0
        for p in game.players:
            if p.id != self.player_id and opponent_count < 3:
                obs.extend(
                    [
                        p.bid / 10.0 if p.bid is not None else 0.0,
                        p.score / 100.0,
                        p.tricks_won / 10.0,
                    ]
                )
                opponent_count += 1
        while opponent_count < 3:
            obs.extend([0.0] * 3)
            opponent_count += 1

        # 6. HAND STRENGTH BREAKDOWN (4 dims)
        high_standard = self._count_card_type(
            hand, lambda c: c.is_standard_suit() and c.number and c.number >= 10
        )
        obs.extend(
            [
                self._count_card_type(hand, lambda c: c.is_pirate()) / 5.0,
                self._count_card_type(hand, lambda c: c.is_king()) / 4.0,
                self._count_card_type(hand, lambda c: c.is_mermaid()) / 2.0,
                high_standard / 10.0,
            ]
        )

        # 7. TRICK POSITION (4 dims)
        obs.extend(self._encode_trick_position(game, cards_in_trick))

        # 8. OPPONENT PATTERNS (6 dims)
        obs.extend(self._encode_opponent_patterns(game))

        # 9. CARDS PLAYED COUNT (5 dims)
        obs.extend(self._encode_cards_played(game))

        # 10. ROUND PROGRESSION (1 dim)
        if current_round:
            round_progress = len(current_round.tricks) / max(current_round.number, 1)
        else:
            round_progress = 0.0
        obs.append(round_progress)

        # 11. BID PRESSURE (1 dim)
        obs.append(self._calculate_bid_pressure(game))

        # 12. POSITION ADVANTAGE (1 dim)
        obs.append(self._calculate_position_advantage(game))

        # 13. TRUMP STRENGTH (2 dims)
        obs.extend(self._encode_trump_strength(game, hand))

        return np.array(obs, dtype=np.float32)

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
        """Evaluate card strength (0.0 to 1.0). Handles all 74 cards."""
        if card.is_king():
            return 0.95
        if card.is_pirate():
            return 0.8
        if card.is_tigress():
            # Flexible - can be pirate (0.8) or escape (0.1)
            return 0.55
        if card.is_whale():
            # Unpredictable - highest suit wins
            return 0.4
        if card.is_kraken():
            # Nobody wins - disruptive
            return 0.25
        if card.is_mermaid():
            return 0.35
        if card.is_loot():
            # Acts like escape but alliance bonus
            return 0.08
        if card.is_escape():
            return 0.05
        if card.is_roger():
            # Trump suit - higher value
            return 0.5 + (card.number / 14.0) * 0.35 if card.number else 0.5
        if card.is_standard_suit():
            return 0.2 + (card.number / 14.0) * 0.35 if card.number else 0.2
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
        if max_suit_count >= max(round_number * 0.6, 3):
            base_strength += 0.5

        # Special card synergies
        has_mermaid = any(c.is_mermaid() for c in card_objects)
        high_cards = sum(1 for c in card_objects if self._evaluate_card_strength(c) > 0.7)
        if has_mermaid and high_cards >= 2:
            base_strength -= 0.5

        # Pirate strength in later rounds
        pirates = sum(1 for c in card_objects if c.is_pirate())
        if round_number >= 5 and pirates >= 2:
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

    def _count_cards_by_suit(self, hand: list[CardId]) -> dict:
        """Count cards by suit."""
        suit_counts: dict = {}
        for card_id in hand:
            card = get_card(card_id)
            if card.is_standard_suit() and hasattr(card, "card_type"):
                suit = card.card_type.name
                suit_counts[suit] = suit_counts.get(suit, 0) + 1
        return suit_counts

    def _encode_trick_position(self, game: Game, cards_in_trick: list[CardId]) -> list[float]:
        """Encode trick position (4 dims)."""
        position = [0.0] * 4
        num_played = len([c for c in cards_in_trick if c])
        if num_played < 4:
            position[num_played] = 1.0
        return position

    def _encode_opponent_patterns(self, game: Game) -> list[float]:
        """Encode opponent bidding patterns (6 dims)."""
        patterns: list[float] = []

        for p in game.players:
            if p.id != self.player_id and len(patterns) < 6:
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

        while len(patterns) < 6:
            patterns.append(0.0)
        return patterns[:6]

    def _encode_cards_played(self, game: Game) -> list[float]:
        """Encode cards played this round (5 dims)."""
        current_round = game.get_current_round()
        if not current_round:
            return [0.0] * 5

        pirates = kings = mermaids = escapes = high_cards = 0

        for trick in current_round.tricks:
            for card_id in trick.get_all_card_ids():
                if card_id:
                    card = get_card(card_id)
                    if card.is_pirate():
                        pirates += 1
                    elif card.is_king():
                        kings += 1
                    elif card.is_mermaid():
                        mermaids += 1
                    elif card.is_escape():
                        escapes += 1
                    elif card.is_standard_suit() and card.number and card.number >= 10:
                        high_cards += 1

        total = current_round.number * len(game.players)
        return [
            pirates / max(total, 1),
            kings / max(total, 1),
            mermaids / max(total, 1),
            escapes / max(total, 1),
            high_cards / max(total, 1),
        ]

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
        """Simple fallback bidding when no model available."""
        from app.bots.rule_based_bot import RuleBasedBot
        from app.models.game import Game

        # Use rule-based logic as fallback
        temp_game = Game(id="temp", slug="temp")
        rule_bot = RuleBasedBot(self.player_id)
        return rule_bot.make_bid(temp_game, round_number, hand)

    def _fallback_pick(self, playable: list[CardId]) -> CardId:
        """Simple fallback card picking when no model available."""
        import random

        return random.choice(playable)
