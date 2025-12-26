"""Hierarchical Gymnasium environment for Skull King.

V8 Hierarchical RL: Separates bidding (Manager) from card-playing (Worker) policies.

Manager Policy:
- Activated: Once per round (during bidding phase)
- Input: Full hand, game state, opponent history, pirate abilities in hand
- Output: Bid value (0 to round_number)
- Reward: Round completion accuracy

Worker Policy:
- Activated: Every trick (during picking phase)
- Input: Hand, trick state, TARGET BID, tricks won so far, ability state
- Output: Card index to play
- Reward: Trick outcome relative to goal

Pirate Ability Observations:
1. Rosie (PIRATE1) - Choose trick starter: positional advantage
2. Bendt (PIRATE2) - Draw/discard: hand improvement potential
3. Roatán (PIRATE3) - Extra bet: bonus scoring opportunity
4. Jade (PIRATE4) - View deck: information advantage
5. Harry (PIRATE5) - Modify bid ±1: bid flexibility (CRITICAL for Manager)

Benefits:
- Clear credit assignment (bid vs play decisions)
- Shorter effective horizon per policy
- Specialized learning for each task
- 2-3x expected sample efficiency improvement
- Pirate ability awareness for strategic decisions
"""

import logging
import uuid
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from sb3_contrib.common.wrappers import ActionMasker

from app.bots import RandomBot, RuleBasedBot
from app.bots.base_bot import BaseBot, BotDifficulty
from app.models.card import Card, CardId, get_card
from app.models.enums import MAX_ROUNDS, GameState
from app.models.game import Game
from app.models.player import Player
from app.models.pirate_ability import (
    AbilityState,
    AbilityType,
    PirateType,
    PIRATE_IDENTITY,
    get_pirate_type,
)
from app.models.round import Round
from app.models.trick import TigressChoice, Trick

logger = logging.getLogger(__name__)


class ManagerEnv(gym.Env[np.ndarray, int]):
    """Environment for the bidding (Manager) policy.

    Manager decides the bid at the start of each round.
    Episode = one round (not full game).
    Reward = bid accuracy at round end.

    Observation breakdown (168 dims):
    - Hand encoding (90): 10 cards x 9 features
    - Game context (28): round info, hand strength, etc.
    - Pirate abilities in hand (10): which pirates we hold + strategic value
    - Opponent context (24): bids, patterns, history
    - Round one-hot (10): explicit round encoding
    - Alliance features (6): loot alliance status
    """

    metadata = {"render_modes": ["ansi"]}

    # Observation dimensions breakdown
    HAND_DIM = 90  # 10 cards x 9 features
    GAME_CONTEXT_DIM = 28  # round info, hand strength
    PIRATE_ABILITY_DIM = 10  # pirates in hand + strategic value
    OPPONENT_DIM = 24  # 3 opponents x 8 features
    ROUND_ONEHOT_DIM = 10  # explicit round
    ALLIANCE_DIM = 6  # loot alliance features

    OBS_DIM = HAND_DIM + GAME_CONTEXT_DIM + PIRATE_ABILITY_DIM + OPPONENT_DIM + ROUND_ONEHOT_DIM + ALLIANCE_DIM  # 168

    def __init__(
        self,
        worker_policy: Any = None,
        num_opponents: int = 3,
        opponent_bot_type: str = "rule_based",
        opponent_difficulty: str = "medium",
    ) -> None:
        """Initialize Manager environment.

        Args:
            worker_policy: Pre-trained worker policy for simulating round
            num_opponents: Number of bot opponents
            opponent_bot_type: Type of opponent bot
            opponent_difficulty: Difficulty level
        """
        super().__init__()
        self.worker_policy = worker_policy
        self.num_players = num_opponents + 1
        self.opponent_bot_type = opponent_bot_type
        self.opponent_difficulty = self._parse_difficulty(opponent_difficulty)

        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(self.OBS_DIM,), dtype=np.float32
        )
        # Bid 0-10
        self.action_space = spaces.Discrete(11)

        self.game: Game | None = None
        self.agent_player_id: str = ""
        self.bots: list[tuple[str, BaseBot]] = []
        self.current_round_num = 1

    def _parse_difficulty(self, difficulty: str) -> BotDifficulty:
        difficulty_map = {
            "easy": BotDifficulty.EASY,
            "medium": BotDifficulty.MEDIUM,
            "hard": BotDifficulty.HARD,
        }
        return difficulty_map.get(difficulty.lower(), BotDifficulty.MEDIUM)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset to start of a new round."""
        super().reset(seed=seed)

        # Determine round number (can be set via options for curriculum)
        if options and "round_number" in options:
            self.current_round_num = options["round_number"]
        else:
            # Random round for diverse training
            self.current_round_num = self.np_random.integers(1, MAX_ROUNDS + 1)

        # Create new game
        game_id = str(uuid.uuid4())
        self.game = Game(id=game_id, slug=game_id[:4].upper())
        self.agent_player_id = "agent"
        agent_player = Player(id=self.agent_player_id, username="Agent", game_id=game_id)
        self.game.add_player(agent_player)

        # Add opponents
        self.bots = []
        for i in range(self.num_players - 1):
            player_id = f"bot_{i}"
            bot_player = Player(id=player_id, username=f"Bot {i}", game_id=game_id)
            self.game.add_player(bot_player)
            bot = self._create_bot(player_id)
            self.bots.append((player_id, bot))

        # Fast-forward to target round
        for r in range(1, self.current_round_num):
            self._simulate_round(r, random_bid=True)

        # Start target round
        self.game.start_new_round()

        # Make bot bids
        self._make_bot_bids()

        return self._get_manager_obs(), {}

    def _create_bot(self, player_id: str) -> BaseBot:
        if self.opponent_bot_type == "random":
            return RandomBot(player_id)
        return RuleBasedBot(player_id, self.opponent_difficulty)

    def _simulate_round(self, round_num: int, random_bid: bool = False) -> None:
        """Simulate a complete round with random/rule-based play."""
        self.game.start_new_round()
        current_round = self.game.get_current_round()
        if not current_round:
            return

        # Make all bids (including agent)
        for player_id, bot in self.bots:
            player = self.game.get_player(player_id)
            if player and player.hand:
                bid = bot.make_bid(self.game, round_num, player.hand)
                current_round.place_bid(player_id, bid)

        # Agent bids randomly or based on hand
        agent = self.game.get_player(self.agent_player_id)
        if agent and agent.hand:
            if random_bid:
                bid = self.np_random.integers(0, round_num + 1)
            else:
                bid = self._estimate_bid(agent.hand)
            current_round.place_bid(self.agent_player_id, bid)

        # Play all tricks
        for _ in range(round_num):
            self._play_trick_with_bots(current_round)

        current_round.calculate_scores()

    def _estimate_bid(self, hand: list[CardId]) -> int:
        """Simple bid estimation based on hand strength."""
        strength = 0
        for card_id in hand:
            card = get_card(card_id)
            if card.is_pirate() or card.is_king():
                strength += 1
            elif card.is_mermaid():
                strength += 0.7
            elif card.is_standard_suit() and card.number and card.number >= 12:
                strength += 0.5
        return min(int(strength), len(hand))

    def _make_bot_bids(self) -> None:
        """Make bids for all bots."""
        current_round = self.game.get_current_round()
        if not current_round:
            return

        for player_id, bot in self.bots:
            player = self.game.get_player(player_id)
            if player and player.hand:
                bid = bot.make_bid(self.game, self.current_round_num, player.hand)
                current_round.place_bid(player_id, bid)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute bid and simulate round with worker policy.

        Args:
            action: Bid value (0-10)

        Returns:
            obs, reward, terminated, truncated, info
        """
        current_round = self.game.get_current_round()
        if not current_round:
            return self._get_manager_obs(), 0.0, True, False, {}

        # Clamp bid to valid range
        bid = min(action, self.current_round_num)

        # Place agent's bid
        current_round.place_bid(self.agent_player_id, bid)

        # Simulate round with worker policy (or rule-based fallback)
        tricks_won = self._simulate_card_play(current_round)

        # Calculate reward based on bid accuracy
        reward = self._calculate_manager_reward(bid, tricks_won)

        info = {
            "bid": bid,
            "tricks_won": tricks_won,
            "round": self.current_round_num,
            "bid_accuracy": 1.0 if bid == tricks_won else 0.0,
        }

        # Episode ends after one round decision
        return self._get_manager_obs(), reward, True, False, info

    def _simulate_card_play(self, current_round: Round) -> int:
        """Simulate card play phase and return tricks won by agent."""
        for trick_num in range(self.current_round_num):
            self._play_trick_with_worker(current_round)

        current_round.calculate_scores()
        return current_round.get_tricks_won(self.agent_player_id)

    def _play_trick_with_worker(self, current_round: Round) -> None:
        """Play a single trick using worker policy or fallback."""
        trick = current_round.start_trick()
        if not trick:
            return

        play_order = current_round.get_play_order()

        for player_id in play_order:
            player = self.game.get_player(player_id)
            if not player or not player.hand:
                continue

            if player_id == self.agent_player_id:
                # Use worker policy or rule-based fallback
                if self.worker_policy is not None:
                    card_id = self._worker_pick_card(player, trick, current_round)
                else:
                    # Fallback: use rule-based strategy
                    fallback = RuleBasedBot(player_id, BotDifficulty.HARD)
                    card_id = fallback.pick_card(
                        self.game, player.hand, trick.cards_played()
                    )
            else:
                # Bot plays
                bot = next((b for pid, b in self.bots if pid == player_id), None)
                if bot:
                    card_id = bot.pick_card(
                        self.game, player.hand, trick.cards_played()
                    )
                else:
                    card_id = player.hand[0]

            # Handle Tigress choice
            tigress_choice = None
            card = get_card(card_id)
            if card.is_tigress():
                tigress_choice = TigressChoice.ESCAPE

            trick.play_card(player_id, card_id, tigress_choice)
            player.hand.remove(card_id)

        trick.determine_winner()

    def _worker_pick_card(
        self, player: Player, trick: Trick, current_round: Round
    ) -> CardId:
        """Use worker policy to pick a card."""
        # Build worker observation
        obs = self._get_worker_obs_for_manager(player, trick, current_round)

        # Get action mask
        valid_cards = current_round.get_valid_cards(
            self.agent_player_id, player.hand, trick.cards_played()
        )
        mask = np.zeros(11, dtype=bool)
        for i, card_id in enumerate(player.hand):
            if i < 11 and card_id in valid_cards:
                mask[i] = True

        # Get worker action
        action, _ = self.worker_policy.predict(obs, action_masks=mask, deterministic=True)

        # Map to card
        if 0 <= action < len(player.hand):
            return player.hand[action]
        return valid_cards[0] if valid_cards else player.hand[0]

    def _get_worker_obs_for_manager(
        self, player: Player, trick: Trick, current_round: Round
    ) -> np.ndarray:
        """Build worker observation for simulating card play."""
        # Simplified obs for worker - full implementation in WorkerEnv
        obs = np.zeros(WorkerEnv.OBS_DIM, dtype=np.float32)
        # This would be filled with proper worker observation
        # For now, return zeros (worker will use action mask)
        return obs

    def _play_trick_with_bots(self, current_round: Round) -> None:
        """Play trick with all bots (for fast-forward simulation)."""
        trick = current_round.start_trick()
        if not trick:
            return

        play_order = current_round.get_play_order()

        for player_id in play_order:
            player = self.game.get_player(player_id)
            if not player or not player.hand:
                continue

            if player_id == self.agent_player_id:
                # Random card for agent during fast-forward
                card_id = self.np_random.choice(player.hand)
            else:
                bot = next((b for pid, b in self.bots if pid == player_id), None)
                if bot:
                    card_id = bot.pick_card(
                        self.game, player.hand, trick.cards_played()
                    )
                else:
                    card_id = player.hand[0]

            tigress_choice = None
            card = get_card(card_id)
            if card.is_tigress():
                tigress_choice = TigressChoice.ESCAPE

            trick.play_card(player_id, card_id, tigress_choice)
            player.hand.remove(card_id)

        trick.determine_winner()

    def _calculate_manager_reward(self, bid: int, tricks_won: int) -> float:
        """Calculate reward for manager based on bid accuracy."""
        diff = abs(bid - tricks_won)

        if diff == 0:
            # Perfect bid
            if bid == 0:
                # Zero bid bonus scales with round
                return 5.0 + self.current_round_num * 0.5
            return 5.0
        elif diff == 1:
            return 1.0
        elif diff == 2:
            return -1.0
        else:
            return -2.0 - diff * 0.5

    def _get_manager_obs(self) -> np.ndarray:
        """Build observation for manager (bidding decision).

        Comprehensive observation including pirate abilities for strategic bidding.
        """
        obs = np.zeros(self.OBS_DIM, dtype=np.float32)

        if not self.game:
            return obs

        agent = self.game.get_player(self.agent_player_id)
        if not agent or not agent.hand:
            return obs

        idx = 0

        # === HAND ENCODING (90 dims: 10 cards x 9 features) ===
        for i, card_id in enumerate(agent.hand[:10]):
            card = get_card(card_id)
            obs[idx:idx + 9] = self._encode_card(card)
            idx += 9

        # Pad remaining card slots
        idx = self.HAND_DIM  # 90

        # === GAME CONTEXT (28 dims) ===
        # Round number (normalized)
        obs[idx] = self.current_round_num / MAX_ROUNDS
        idx += 1

        # Hand size (normalized)
        obs[idx] = len(agent.hand) / MAX_ROUNDS
        idx += 1

        # Hand strength breakdown (10 dims)
        pirates = sum(1 for c in agent.hand if get_card(c).is_pirate())
        kings = sum(1 for c in agent.hand if get_card(c).is_king())
        mermaids = sum(1 for c in agent.hand if get_card(c).is_mermaid())
        escapes = sum(1 for c in agent.hand if get_card(c).is_escape())
        tigress_count = sum(1 for c in agent.hand if get_card(c).is_tigress())
        high_cards = sum(
            1 for c in agent.hand
            if get_card(c).is_standard_suit() and get_card(c).number and get_card(c).number >= 10
        )
        loot_cards = sum(1 for c in agent.hand if get_card(c).is_loot())
        specials = pirates + kings + mermaids + tigress_count

        obs[idx] = pirates / 5.0
        obs[idx + 1] = kings / 1.0
        obs[idx + 2] = mermaids / 2.0
        obs[idx + 3] = escapes / 5.0
        obs[idx + 4] = high_cards / 10.0
        obs[idx + 5] = loot_cards / 2.0
        obs[idx + 6] = specials / 8.0
        obs[idx + 7] = self._estimate_hand_strength(agent.hand)
        obs[idx + 8] = tigress_count / 1.0  # Tigress is flexible (pirate or escape)
        obs[idx + 9] = min(self._count_trump_cards(agent.hand) / 4.0, 1.0)  # Trump density
        idx += 10

        # Trump suit breakdown (4 dims) - black suits are trump
        black_count = sum(
            1 for c in agent.hand
            if get_card(c).is_standard_suit() and get_card(c).suit and get_card(c).suit.value in ("black",)
        )
        obs[idx] = black_count / MAX_ROUNDS
        obs[idx + 1] = (black_count / max(len(agent.hand), 1))  # Trump density in hand
        # Best trump value
        best_trump = max(
            (get_card(c).number or 0 for c in agent.hand
             if get_card(c).is_standard_suit() and get_card(c).suit and get_card(c).suit.value == "black"),
            default=0
        )
        obs[idx + 2] = best_trump / 14.0
        # Average hand value
        avg_value = np.mean([get_card(c).number or 7 for c in agent.hand if get_card(c).is_standard_suit()] or [0])
        obs[idx + 3] = avg_value / 14.0
        idx += 4

        # Suit distribution (4 dims) - how balanced is the hand
        suit_counts = self._count_suits(agent.hand)
        for i, count in enumerate(suit_counts[:4]):
            obs[idx + i] = count / MAX_ROUNDS
        idx += 4

        # Bid pressure features (4 dims)
        total_opponent_bids = 0
        current_round = self.game.get_current_round()
        if current_round:
            for player_id, _ in self.bots:
                player = self.game.get_player(player_id)
                if player and player.bid is not None:
                    total_opponent_bids += player.bid

        obs[idx] = total_opponent_bids / (self.current_round_num * 3)  # Avg opponent bid
        obs[idx + 1] = max(0, self.current_round_num - total_opponent_bids) / MAX_ROUNDS  # Available tricks
        obs[idx + 2] = 1.0 if total_opponent_bids > self.current_round_num else 0.0  # Overbid signal
        obs[idx + 3] = 1.0 if total_opponent_bids == 0 else 0.0  # All zeros signal
        idx += 4

        # === PIRATE ABILITIES IN HAND (10 dims) ===
        # Critical for bidding strategy!
        pirate_obs = self._encode_pirate_abilities(agent.hand)
        obs[idx:idx + 10] = pirate_obs
        idx += 10

        # === OPPONENT CONTEXT (24 dims: 3 opponents x 8 features) ===
        for i, (player_id, _) in enumerate(self.bots[:3]):
            player = self.game.get_player(player_id)
            opp_idx = idx + i * 8

            if player:
                # Bid (if available)
                obs[opp_idx] = (player.bid or 0) / MAX_ROUNDS
                obs[opp_idx + 1] = 1.0 if player.bid is not None else 0.0

                # Historical performance (from previous rounds)
                total_score = player.score or 0
                obs[opp_idx + 2] = total_score / 500.0  # Normalized score

                # Hand size (info about remaining cards)
                obs[opp_idx + 3] = len(player.hand) / MAX_ROUNDS

                # Bidding tendencies (aggressive vs conservative)
                # This would ideally track historical data
                obs[opp_idx + 4] = 0.5  # Neutral for now

                # Position relative to agent
                obs[opp_idx + 5] = (i + 1) / 4.0  # Normalized position

                # Check if opponent has special cards visible (from previous tricks)
                obs[opp_idx + 6] = 0.0  # Would track visible specials
                obs[opp_idx + 7] = 0.0  # Reserved

        idx += 24

        # === ROUND ONE-HOT (10 dims) ===
        if 1 <= self.current_round_num <= MAX_ROUNDS:
            obs[idx + self.current_round_num - 1] = 1.0
        idx += 10

        # === ALLIANCE FEATURES (6 dims) ===
        # Loot card alliance potential
        obs[idx] = 1.0 if loot_cards > 0 else 0.0  # Has loot
        obs[idx + 1] = loot_cards / 2.0  # Loot count
        obs[idx + 2] = 0.0  # Alliance status (would track active alliances)
        obs[idx + 3] = 0.0  # Ally bid accuracy
        obs[idx + 4] = 0.0  # Alliance potential
        obs[idx + 5] = 0.0  # Reserved
        idx += 6

        return obs

    def _encode_pirate_abilities(self, hand: list[CardId]) -> np.ndarray:
        """Encode pirate abilities in hand for strategic bidding.

        Returns 10-dim vector:
        - [0]: Has Harry (can modify bid by ±1) - CRITICAL for bidding
        - [1]: Harry strategic value (1.0 = very valuable, depends on hand uncertainty)
        - [2]: Has Rosie (can choose trick starter)
        - [3]: Has Bendt (can draw/discard to improve hand)
        - [4]: Has Roatán (can make extra bet for bonus)
        - [5]: Has Jade (can view undealt cards)
        - [6]: Total pirates in hand (normalized)
        - [7]: Pirate win potential (how likely to win tricks with pirates)
        - [8]: Bid flexibility score (how much ± adjustment possible)
        - [9]: Expected ability bonus (estimated bonus from abilities)
        """
        features = np.zeros(10, dtype=np.float32)

        # Track which pirates are in hand
        has_harry = False
        has_rosie = False
        has_bendt = False
        has_roatan = False
        has_jade = False
        pirate_count = 0

        for card_id in hand:
            pirate_type = get_pirate_type(card_id)
            if pirate_type:
                pirate_count += 1
                if pirate_type == PirateType.HARRY:
                    has_harry = True
                elif pirate_type == PirateType.ROSIE:
                    has_rosie = True
                elif pirate_type == PirateType.BENDT:
                    has_bendt = True
                elif pirate_type == PirateType.ROATAN:
                    has_roatan = True
                elif pirate_type == PirateType.JADE:
                    has_jade = True

        # Binary indicators
        features[0] = 1.0 if has_harry else 0.0
        features[2] = 1.0 if has_rosie else 0.0
        features[3] = 1.0 if has_bendt else 0.0
        features[4] = 1.0 if has_roatan else 0.0
        features[5] = 1.0 if has_jade else 0.0

        # Harry strategic value: higher when hand strength is uncertain
        # If we have Harry, we can adjust bid by ±1, so we can bid more aggressively
        if has_harry:
            # Calculate hand uncertainty (variance in expected tricks)
            hand_strength = self._estimate_hand_strength(hand)
            # Harry is more valuable when we're uncertain (mid-range hand strength)
            uncertainty = 1.0 - abs(hand_strength - 0.5) * 2  # Max at 0.5 strength
            features[1] = 0.5 + uncertainty * 0.5  # Range: 0.5 to 1.0
        else:
            features[1] = 0.0

        # Pirate count (normalized)
        features[6] = min(pirate_count / 5.0, 1.0)

        # Pirate win potential: pirates beat regular cards
        # More pirates = more guaranteed trick wins
        features[7] = min(pirate_count * 0.2, 1.0)

        # Bid flexibility: how much we can adjust bid at end
        # +1 for Harry, slight bonus for Bendt (can improve hand)
        flexibility = 0.0
        if has_harry:
            flexibility += 1.0  # ±1 bid adjustment
        if has_bendt:
            flexibility += 0.3  # Can improve hand
        features[8] = min(flexibility / 1.5, 1.0)

        # Expected ability bonus
        expected_bonus = 0.0
        if has_roatan:
            expected_bonus += 0.15  # Average of 0/10/20 scaled
        if has_jade:
            expected_bonus += 0.05  # Information advantage
        if has_rosie:
            expected_bonus += 0.1  # Positional advantage
        features[9] = min(expected_bonus, 1.0)

        return features

    def _count_trump_cards(self, hand: list[CardId]) -> int:
        """Count trump cards (black suit) in hand."""
        count = 0
        for card_id in hand:
            card = get_card(card_id)
            if card.is_standard_suit() and card.suit and card.suit.value == "black":
                count += 1
        return count

    def _count_suits(self, hand: list[CardId]) -> list[int]:
        """Count cards in each suit."""
        suit_counts = [0, 0, 0, 0]  # yellow, green, purple, black
        suit_map = {"yellow": 0, "green": 1, "purple": 2, "black": 3}
        for card_id in hand:
            card = get_card(card_id)
            if card.is_standard_suit() and card.suit:
                suit_idx = suit_map.get(card.suit.value, 0)
                suit_counts[suit_idx] += 1
        return suit_counts

    def _encode_card(self, card: Card) -> np.ndarray:
        """Encode a card as 9 features."""
        features = np.zeros(9, dtype=np.float32)

        # Card type (one-hot-ish)
        if card.is_escape():
            features[0] = 1.0
        elif card.is_pirate():
            features[1] = 1.0
        elif card.is_king():
            features[2] = 1.0
        elif card.is_mermaid():
            features[3] = 1.0
        elif card.is_tigress():
            features[4] = 1.0
        elif card.is_standard_suit():
            features[5] = 1.0
            if card.number:
                features[6] = card.number / 14.0
        elif card.is_loot():
            features[7] = 1.0

        # Estimated strength
        features[8] = self._card_strength(card)

        return features

    def _card_strength(self, card: Card) -> float:
        """Estimate card strength for observations."""
        if card.is_king():
            return 0.95
        if card.is_pirate():
            return 0.8
        if card.is_mermaid():
            return 0.35
        if card.is_tigress():
            return 0.55
        if card.is_standard_suit() and card.number:
            return 0.2 + (card.number / 14.0) * 0.35
        if card.is_escape():
            return 0.05
        return 0.3

    def _estimate_hand_strength(self, hand: list[CardId]) -> float:
        """Estimate overall hand strength (0-1)."""
        if not hand:
            return 0.0

        total = 0.0
        for card_id in hand:
            card = get_card(card_id)
            total += self._card_strength(card)

        return min(total / len(hand), 1.0)

    def action_masks(self) -> np.ndarray:
        """Return valid bid mask."""
        mask = np.zeros(11, dtype=bool)
        # Valid bids are 0 to round_number
        for i in range(self.current_round_num + 1):
            mask[i] = True
        return mask


class WorkerEnv(gym.Env[np.ndarray, int]):
    """Environment for the card-playing (Worker) policy.

    Worker plays cards to achieve the bid goal set by Manager.
    Episode = one round of card play (after bidding).
    Reward = progress toward goal + trick outcomes.

    Observation breakdown (200 dims):
    - Hand encoding (90): 10 cards x 9 features
    - Trick state (36): 4 players x 9 features per card
    - Goal context (20): bid goal, progress, needs
    - Pirate abilities (18): abilities in hand + ability state
    - Game state (20): round, position, opponent info
    - Round one-hot (10): explicit round encoding
    - Alliance features (6): loot alliance status
    """

    metadata = {"render_modes": ["ansi"]}

    # Observation dimensions breakdown
    HAND_DIM = 90  # 10 cards x 9 features
    TRICK_DIM = 36  # 4 players x 9 features
    GOAL_DIM = 20  # goal context
    PIRATE_ABILITY_DIM = 18  # abilities in hand + ability state
    GAME_STATE_DIM = 20  # round, position, opponents
    ROUND_ONEHOT_DIM = 10  # explicit round
    ALLIANCE_DIM = 6  # loot alliance

    OBS_DIM = HAND_DIM + TRICK_DIM + GOAL_DIM + PIRATE_ABILITY_DIM + GAME_STATE_DIM + ROUND_ONEHOT_DIM + ALLIANCE_DIM  # 200

    def __init__(
        self,
        num_opponents: int = 3,
        opponent_bot_type: str = "rule_based",
        opponent_difficulty: str = "medium",
        fixed_goal: int | None = None,
    ) -> None:
        """Initialize Worker environment.

        Args:
            num_opponents: Number of bot opponents
            opponent_bot_type: Type of opponent bot
            opponent_difficulty: Difficulty level
            fixed_goal: If set, use this bid goal (for pre-training)
        """
        super().__init__()
        self.num_players = num_opponents + 1
        self.opponent_bot_type = opponent_bot_type
        self.opponent_difficulty = self._parse_difficulty(opponent_difficulty)
        self.fixed_goal = fixed_goal

        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(self.OBS_DIM,), dtype=np.float32
        )
        # Card index 0-10
        self.action_space = spaces.Discrete(11)

        self.game: Game | None = None
        self.agent_player_id: str = ""
        self.bots: list[tuple[str, BaseBot]] = []
        self.current_round_num = 1
        self.goal_bid = 0
        self.tricks_won = 0

    def _parse_difficulty(self, difficulty: str) -> BotDifficulty:
        difficulty_map = {
            "easy": BotDifficulty.EASY,
            "medium": BotDifficulty.MEDIUM,
            "hard": BotDifficulty.HARD,
        }
        return difficulty_map.get(difficulty.lower(), BotDifficulty.MEDIUM)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset to start of card-play phase."""
        super().reset(seed=seed)

        # Get round number and goal from options
        if options:
            self.current_round_num = options.get("round_number", self.np_random.integers(1, MAX_ROUNDS + 1))
            self.goal_bid = options.get("goal_bid", self.fixed_goal or 0)
        else:
            self.current_round_num = self.np_random.integers(1, MAX_ROUNDS + 1)
            self.goal_bid = self.fixed_goal if self.fixed_goal is not None else self.np_random.integers(0, self.current_round_num + 1)

        self.tricks_won = 0

        # Create game and fast-forward to bidding complete
        self._setup_game()

        return self._get_worker_obs(), {"goal_bid": self.goal_bid}

    def _setup_game(self) -> None:
        """Set up game state for card-play phase."""
        game_id = str(uuid.uuid4())
        self.game = Game(id=game_id, slug=game_id[:4].upper())
        self.agent_player_id = "agent"
        agent_player = Player(id=self.agent_player_id, username="Agent", game_id=game_id)
        self.game.add_player(agent_player)

        self.bots = []
        for i in range(self.num_players - 1):
            player_id = f"bot_{i}"
            bot_player = Player(id=player_id, username=f"Bot {i}", game_id=game_id)
            self.game.add_player(bot_player)
            bot = self._create_bot(player_id)
            self.bots.append((player_id, bot))

        # Fast-forward to target round
        for r in range(1, self.current_round_num):
            self._simulate_full_round(r)

        # Start target round and complete bidding
        self.game.start_new_round()
        self._complete_bidding()

    def _create_bot(self, player_id: str) -> BaseBot:
        if self.opponent_bot_type == "random":
            return RandomBot(player_id)
        return RuleBasedBot(player_id, self.opponent_difficulty)

    def _simulate_full_round(self, round_num: int) -> None:
        """Simulate a complete round."""
        self.game.start_new_round()
        current_round = self.game.get_current_round()
        if not current_round:
            return

        # All players bid
        for player_id, bot in self.bots:
            player = self.game.get_player(player_id)
            if player and player.hand:
                bid = bot.make_bid(self.game, round_num, player.hand)
                current_round.place_bid(player_id, bid)

        agent = self.game.get_player(self.agent_player_id)
        if agent and agent.hand:
            bid = self.np_random.integers(0, round_num + 1)
            current_round.place_bid(self.agent_player_id, bid)

        # Play all tricks
        for _ in range(round_num):
            self._play_random_trick(current_round)

        current_round.calculate_scores()

    def _play_random_trick(self, current_round: Round) -> None:
        """Play trick with random agent moves."""
        trick = current_round.start_trick()
        if not trick:
            return

        for player_id in current_round.get_play_order():
            player = self.game.get_player(player_id)
            if not player or not player.hand:
                continue

            if player_id == self.agent_player_id:
                card_id = self.np_random.choice(player.hand)
            else:
                bot = next((b for pid, b in self.bots if pid == player_id), None)
                if bot:
                    card_id = bot.pick_card(self.game, player.hand, trick.cards_played())
                else:
                    card_id = player.hand[0]

            tigress_choice = None
            if get_card(card_id).is_tigress():
                tigress_choice = TigressChoice.ESCAPE

            trick.play_card(player_id, card_id, tigress_choice)
            player.hand.remove(card_id)

        trick.determine_winner()

    def _complete_bidding(self) -> None:
        """Complete bidding phase."""
        current_round = self.game.get_current_round()
        if not current_round:
            return

        # Bot bids
        for player_id, bot in self.bots:
            player = self.game.get_player(player_id)
            if player and player.hand:
                bid = bot.make_bid(self.game, self.current_round_num, player.hand)
                current_round.place_bid(player_id, bid)

        # Agent bid (from goal)
        current_round.place_bid(self.agent_player_id, self.goal_bid)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Play a card and return result.

        Args:
            action: Card index (0-10)

        Returns:
            obs, reward, terminated, truncated, info
        """
        current_round = self.game.get_current_round()
        if not current_round:
            return self._get_worker_obs(), 0.0, True, False, {}

        agent = self.game.get_player(self.agent_player_id)
        if not agent or not agent.hand:
            return self._get_worker_obs(), 0.0, True, False, {}

        # Get or create current trick
        trick = current_round.get_current_trick()
        if not trick or trick.is_complete():
            trick = current_round.start_trick()

        if not trick:
            return self._get_worker_obs(), 0.0, True, False, {}

        # Play cards up to agent's turn
        self._play_until_agent_turn(current_round, trick)

        # Play agent's card
        valid_cards = current_round.get_valid_cards(
            self.agent_player_id, agent.hand, trick.cards_played()
        )

        # Map action to card
        if 0 <= action < len(agent.hand):
            card_id = agent.hand[action]
            if card_id not in valid_cards:
                card_id = valid_cards[0] if valid_cards else agent.hand[0]
        else:
            card_id = valid_cards[0] if valid_cards else agent.hand[0]

        # Handle Tigress
        tigress_choice = None
        card = get_card(card_id)
        if card.is_tigress():
            # Choose based on goal progress
            need_wins = self.goal_bid - self.tricks_won
            tigress_choice = TigressChoice.PIRATE if need_wins > 0 else TigressChoice.ESCAPE

        trick.play_card(self.agent_player_id, card_id, tigress_choice)
        agent.hand.remove(card_id)

        # Complete trick with remaining players
        self._complete_trick(current_round, trick)

        # Determine winner and update state
        trick.determine_winner()
        won_trick = trick.winner == self.agent_player_id
        if won_trick:
            self.tricks_won += 1

        # Calculate reward
        reward = self._calculate_worker_reward(won_trick, trick)

        # Check if round is complete
        round_complete = len(current_round.tricks) >= self.current_round_num and trick.is_complete()

        if round_complete:
            current_round.calculate_scores()

        info = {
            "won_trick": won_trick,
            "tricks_won": self.tricks_won,
            "goal_bid": self.goal_bid,
            "tricks_remaining": self.current_round_num - len(current_round.tricks),
            "goal_achieved": self.tricks_won == self.goal_bid if round_complete else None,
        }

        return self._get_worker_obs(), reward, round_complete, False, info

    def _play_until_agent_turn(self, current_round: Round, trick: Trick) -> None:
        """Play bot cards until it's agent's turn."""
        play_order = current_round.get_play_order()
        played_ids = {pid for pid, _ in trick.plays}

        for player_id in play_order:
            if player_id in played_ids:
                continue
            if player_id == self.agent_player_id:
                break

            player = self.game.get_player(player_id)
            if not player or not player.hand:
                continue

            bot = next((b for pid, b in self.bots if pid == player_id), None)
            if bot:
                card_id = bot.pick_card(self.game, player.hand, trick.cards_played())
            else:
                card_id = player.hand[0]

            tigress_choice = None
            if get_card(card_id).is_tigress():
                tigress_choice = TigressChoice.ESCAPE

            trick.play_card(player_id, card_id, tigress_choice)
            player.hand.remove(card_id)

    def _complete_trick(self, current_round: Round, trick: Trick) -> None:
        """Complete trick with remaining players after agent."""
        play_order = current_round.get_play_order()
        played_ids = {pid for pid, _ in trick.plays}

        agent_played = False
        for player_id in play_order:
            if player_id == self.agent_player_id:
                agent_played = True
                continue
            if not agent_played:
                continue
            if player_id in played_ids:
                continue

            player = self.game.get_player(player_id)
            if not player or not player.hand:
                continue

            bot = next((b for pid, b in self.bots if pid == player_id), None)
            if bot:
                card_id = bot.pick_card(self.game, player.hand, trick.cards_played())
            else:
                card_id = player.hand[0]

            tigress_choice = None
            if get_card(card_id).is_tigress():
                tigress_choice = TigressChoice.ESCAPE

            trick.play_card(player_id, card_id, tigress_choice)
            player.hand.remove(card_id)

    def _calculate_worker_reward(self, won_trick: bool, trick: Trick) -> float:
        """Calculate reward for card play decision."""
        tricks_needed = self.goal_bid - self.tricks_won
        tricks_remaining = self.current_round_num - len(self.game.get_current_round().tricks)

        if won_trick:
            if tricks_needed > 0:
                # Needed this win
                return 3.0
            elif tricks_needed == 0:
                # Already at goal, didn't need win
                return -1.5
            else:
                # Over goal
                return -2.0
        else:
            # Lost trick
            if tricks_needed <= tricks_remaining:
                # Still achievable
                return 0.5
            elif tricks_needed > tricks_remaining + 1:
                # Goal now impossible
                return -1.0
            else:
                # Close but risky
                return 0.0

    def _get_worker_obs(self) -> np.ndarray:
        """Build observation for worker (card-play decision).

        Comprehensive observation including pirate ability state for strategic card play.
        """
        obs = np.zeros(self.OBS_DIM, dtype=np.float32)

        if not self.game:
            return obs

        agent = self.game.get_player(self.agent_player_id)
        current_round = self.game.get_current_round()
        if not agent or not current_round:
            return obs

        idx = 0

        # === HAND ENCODING (90 dims: 10 cards x 9 features) ===
        for i, card_id in enumerate(agent.hand[:10]):
            card = get_card(card_id)
            obs[idx:idx + 9] = self._encode_card(card)
            idx += 9
        idx = self.HAND_DIM  # 90

        # === TRICK STATE (36 dims: 4 players x 9 features) ===
        trick = current_round.get_current_trick()
        if trick:
            for i, (player_id, card_id) in enumerate(trick.plays[:4]):
                card = get_card(card_id)
                obs[idx + i * 9:idx + i * 9 + 9] = self._encode_card(card)
        idx += self.TRICK_DIM  # 36

        # === GOAL CONTEXT (20 dims) ===
        tricks_needed = self.goal_bid - self.tricks_won
        tricks_remaining = self.current_round_num - len(current_round.tricks)

        obs[idx] = self.goal_bid / MAX_ROUNDS  # Normalized goal
        obs[idx + 1] = self.tricks_won / MAX_ROUNDS  # Tricks won
        obs[idx + 2] = tricks_needed / MAX_ROUNDS  # Needed
        obs[idx + 3] = tricks_remaining / MAX_ROUNDS  # Remaining

        # Goal achievability features
        obs[idx + 4] = 1.0 if tricks_needed <= tricks_remaining else 0.0  # Still achievable
        obs[idx + 5] = 1.0 if tricks_needed == 0 else 0.0  # Goal met
        obs[idx + 6] = 1.0 if tricks_needed < 0 else 0.0  # Over goal (bad)
        obs[idx + 7] = min(tricks_needed / max(tricks_remaining, 1), 1.0)  # Win urgency

        # Goal one-hot (11 dims for bid 0-10)
        if 0 <= self.goal_bid <= 10:
            obs[idx + 8 + self.goal_bid] = 1.0
        idx += self.GOAL_DIM  # 20

        # === PIRATE ABILITIES (18 dims) ===
        pirate_obs = self._encode_pirate_ability_state(agent.hand, current_round)
        obs[idx:idx + 18] = pirate_obs
        idx += self.PIRATE_ABILITY_DIM  # 18

        # === GAME STATE (20 dims) ===
        obs[idx] = self.current_round_num / MAX_ROUNDS
        obs[idx + 1] = len(agent.hand) / MAX_ROUNDS
        obs[idx + 2] = len(current_round.tricks) / MAX_ROUNDS  # Tricks played

        # Position in trick (4 dims one-hot)
        if trick:
            pos = len(trick.plays)
            if pos < 4:
                obs[idx + 3 + pos] = 1.0
        idx += 7

        # Opponent state (3 opponents x 4 features = 12 dims)
        for i, (player_id, _) in enumerate(self.bots[:3]):
            player = self.game.get_player(player_id)
            opp_idx = idx + i * 4
            if player:
                obs[opp_idx] = (player.bid or 0) / MAX_ROUNDS
                obs[opp_idx + 1] = current_round.get_tricks_won(player_id) / MAX_ROUNDS
                # Opponent goal progress
                opp_needed = (player.bid or 0) - current_round.get_tricks_won(player_id)
                obs[opp_idx + 2] = opp_needed / MAX_ROUNDS
                # Is opponent on track?
                obs[opp_idx + 3] = 1.0 if opp_needed <= tricks_remaining else 0.0
        idx += 12

        # Reserved (1 dim)
        idx += 1

        # === ROUND ONE-HOT (10 dims) ===
        if 1 <= self.current_round_num <= MAX_ROUNDS:
            obs[idx + self.current_round_num - 1] = 1.0
        idx += self.ROUND_ONEHOT_DIM  # 10

        # === ALLIANCE FEATURES (6 dims) ===
        loot_cards = sum(1 for c in agent.hand if get_card(c).is_loot())
        obs[idx] = 1.0 if loot_cards > 0 else 0.0
        obs[idx + 1] = loot_cards / 2.0
        obs[idx + 2] = 0.0  # Alliance active
        obs[idx + 3] = 0.0  # Ally progress
        obs[idx + 4] = 0.0  # Alliance potential
        obs[idx + 5] = 0.0  # Reserved
        idx += self.ALLIANCE_DIM  # 6

        return obs

    def _encode_pirate_ability_state(
        self, hand: list[CardId], current_round: Round
    ) -> np.ndarray:
        """Encode pirate ability state for card-play decisions.

        Returns 18-dim vector:
        Pirates in hand (6 dims):
        - [0]: Has Harry (can modify bid at end)
        - [1]: Has Rosie (can choose starter if wins)
        - [2]: Has Bendt (can draw/discard if wins)
        - [3]: Has Roatán (can make extra bet if wins)
        - [4]: Has Jade (can view deck if wins)
        - [5]: Total pirates in hand

        Ability state (6 dims):
        - [6]: Harry armed (won with Harry earlier, can adjust bid)
        - [7]: Roatán bets accumulated (extra stakes)
        - [8]: Has pending abilities (waiting for resolution)
        - [9]: Rosie override active (someone controls next starter)
        - [10]: Cards drawn via Bendt (hand was improved)
        - [11]: Number of abilities used this round

        Strategic context (6 dims):
        - [12]: Pirate win potential (can win this trick with pirate?)
        - [13]: Ability opportunity (would winning trigger useful ability?)
        - [14]: Harry urgency (should win with Harry if close to goal?)
        - [15]: Roatán risk/reward (extra stakes assessment)
        - [16]: Specials remaining in deck (estimate)
        - [17]: Pirate advantage in current trick
        """
        features = np.zeros(18, dtype=np.float32)

        # Track pirates in hand
        has_harry = False
        has_rosie = False
        has_bendt = False
        has_roatan = False
        has_jade = False
        pirate_count = 0

        for card_id in hand:
            pirate_type = get_pirate_type(card_id)
            if pirate_type:
                pirate_count += 1
                if pirate_type == PirateType.HARRY:
                    has_harry = True
                elif pirate_type == PirateType.ROSIE:
                    has_rosie = True
                elif pirate_type == PirateType.BENDT:
                    has_bendt = True
                elif pirate_type == PirateType.ROATAN:
                    has_roatan = True
                elif pirate_type == PirateType.JADE:
                    has_jade = True

        # Pirates in hand (6 dims)
        features[0] = 1.0 if has_harry else 0.0
        features[1] = 1.0 if has_rosie else 0.0
        features[2] = 1.0 if has_bendt else 0.0
        features[3] = 1.0 if has_roatan else 0.0
        features[4] = 1.0 if has_jade else 0.0
        features[5] = min(pirate_count / 5.0, 1.0)

        # Ability state (6 dims)
        ability_state = current_round.ability_state

        # Harry armed - can adjust bid at end
        features[6] = 1.0 if ability_state.has_armed_harry(self.agent_player_id) else 0.0

        # Roatán bets accumulated
        roatan_bets = ability_state.roatan_bets.get(self.agent_player_id, 0)
        features[7] = min(roatan_bets / 40.0, 1.0)  # Max 40 (two 20s)

        # Pending abilities
        features[8] = 1.0 if ability_state.has_pending_abilities(self.agent_player_id) else 0.0

        # Rosie override active
        features[9] = 1.0 if ability_state.rosie_next_starter is not None else 0.0

        # Cards drawn via Bendt
        features[10] = min(len(ability_state.bendt_drawn_cards) / 4.0, 1.0)

        # Abilities used this round
        resolved_count = sum(1 for ab in ability_state.pending_abilities if ab.resolved)
        features[11] = min(resolved_count / 5.0, 1.0)

        # Strategic context (6 dims)
        tricks_needed = self.goal_bid - self.tricks_won

        # Pirate win potential - do we have a pirate that could win?
        features[12] = min(pirate_count * 0.25, 1.0)

        # Ability opportunity - would winning with a pirate help?
        # Harry is useful when we might miss our bid by 1
        if has_harry and abs(tricks_needed) <= 2:
            features[13] = 0.8
        elif has_roatan and tricks_needed >= 0:
            features[13] = 0.6  # Extra bet opportunity
        elif has_rosie or has_bendt:
            features[13] = 0.4
        else:
            features[13] = 0.0

        # Harry urgency - should prioritize winning with Harry?
        if has_harry:
            # High urgency if we're close to goal but might miss
            if tricks_needed == 1 or tricks_needed == -1:
                features[14] = 0.9  # Very useful to have ±1 flexibility
            elif tricks_needed == 0:
                features[14] = 0.3  # Still useful as insurance
            else:
                features[14] = 0.5
        else:
            features[14] = 0.0

        # Roatán risk/reward
        if has_roatan:
            # More valuable when we're confident we'll make our bid
            if tricks_needed <= 0:
                features[15] = 0.8  # Good position, can risk extra bet
            elif tricks_needed == 1:
                features[15] = 0.5  # Risky
            else:
                features[15] = 0.2  # Very risky
        else:
            features[15] = 0.0

        # Specials remaining estimate (rough)
        features[16] = 0.5  # Would need to track played specials

        # Pirate advantage in current trick
        trick = current_round.get_current_trick()
        if trick and pirate_count > 0:
            # Check if any pirate in trick
            trick_has_pirate = any(
                get_card(card_id).is_pirate() for _, card_id in trick.plays
            )
            if not trick_has_pirate:
                features[17] = 0.8  # Our pirate would dominate
            else:
                features[17] = 0.3  # Pirate already played, less advantage

        return features

    def _encode_card(self, card: Card) -> np.ndarray:
        """Encode card as 9 features."""
        features = np.zeros(9, dtype=np.float32)

        if card.is_escape():
            features[0] = 1.0
        elif card.is_pirate():
            features[1] = 1.0
        elif card.is_king():
            features[2] = 1.0
        elif card.is_mermaid():
            features[3] = 1.0
        elif card.is_tigress():
            features[4] = 1.0
        elif card.is_standard_suit():
            features[5] = 1.0
            if card.number:
                features[6] = card.number / 14.0
        elif card.is_loot():
            features[7] = 1.0

        features[8] = self._card_strength(card)
        return features

    def _card_strength(self, card: Card) -> float:
        """Estimate card strength."""
        if card.is_king():
            return 0.95
        if card.is_pirate():
            return 0.8
        if card.is_mermaid():
            return 0.35
        if card.is_tigress():
            return 0.55
        if card.is_standard_suit() and card.number:
            return 0.2 + (card.number / 14.0) * 0.35
        if card.is_escape():
            return 0.05
        return 0.3

    def action_masks(self) -> np.ndarray:
        """Return valid card mask."""
        mask = np.zeros(11, dtype=bool)

        if not self.game:
            mask[0] = True
            return mask

        agent = self.game.get_player(self.agent_player_id)
        current_round = self.game.get_current_round()

        if not agent or not agent.hand or not current_round:
            mask[0] = True
            return mask

        trick = current_round.get_current_trick()
        cards_in_trick = trick.cards_played() if trick else []

        valid_cards = current_round.get_valid_cards(
            self.agent_player_id, agent.hand, cards_in_trick
        )

        for i, card_id in enumerate(agent.hand):
            if i < 11 and card_id in valid_cards:
                mask[i] = True

        if not mask.any():
            mask[0] = True

        return mask


def mask_fn_manager(env: ManagerEnv) -> np.ndarray:
    """Extract action masks for Manager."""
    return env.action_masks()


def mask_fn_worker(env: WorkerEnv) -> np.ndarray:
    """Extract action masks for Worker."""
    return env.action_masks()


def create_manager_env(
    opponent_type: str = "rule_based",
    difficulty: str = "medium",
    worker_policy: Any = None,
) -> ActionMasker:
    """Create masked Manager environment."""
    env = ManagerEnv(
        worker_policy=worker_policy,
        opponent_bot_type=opponent_type,
        opponent_difficulty=difficulty,
    )
    return ActionMasker(env, mask_fn_manager)


def create_worker_env(
    opponent_type: str = "rule_based",
    difficulty: str = "medium",
    fixed_goal: int | None = None,
) -> ActionMasker:
    """Create masked Worker environment."""
    env = WorkerEnv(
        opponent_bot_type=opponent_type,
        opponent_difficulty=difficulty,
        fixed_goal=fixed_goal,
    )
    return ActionMasker(env, mask_fn_worker)
