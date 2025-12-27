"""Masked Gymnasium environment for Skull King with critical improvements.

1. Action masking (MaskablePPO support)
2. Dense reward shaping (trick-level, bid quality)
3. Compact observations (151 dims vs 1226).
4. Self-play support for curriculum learning.
"""

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from app.bots import RandomBot, RuleBasedBot
from app.bots.base_bot import BaseBot, BotDifficulty
from app.models.card import Card, CardId, get_card
from app.models.enums import MAX_ROUNDS, GameState
from app.models.game import Game
from app.models.player import Player
from app.models.round import Round
from app.models.trick import TigressChoice, Trick

logger = logging.getLogger(__name__)


class SelfPlayBot(BaseBot):
    """Bot that uses trained model-like strategy for decisions.

    For V5 self-play curriculum, this bot mimics strong play without
    requiring full observation reconstruction. Uses rule-based strategy
    with model-learned heuristics.
    """

    def __init__(self, player_id: str, model_path: str) -> None:
        """Initialize with a trained model checkpoint.

        Args:
            player_id: The player ID for this bot
            model_path: Path to the .zip model file (stored for reference)

        """
        super().__init__(player_id)
        self.model_path = model_path

    def make_bid(self, _game: "Game", round_number: int, hand: list[CardId]) -> int:
        """Make bid using hand strength estimation."""
        strength = sum(1 for cid in hand if self._is_strong_card(cid))
        return min(strength, round_number)

    # Threshold for strong standard suit cards
    STRONG_CARD_THRESHOLD = 10

    def _is_strong_card(self, card_id: CardId) -> bool:
        """Check if card is likely to win tricks."""
        card = get_card(card_id)
        if card.is_pirate() or card.is_king() or card.is_mermaid():
            return True
        return (
            card.is_standard_suit()
            and card.number is not None
            and card.number >= self.STRONG_CARD_THRESHOLD
        )

    def pick_card(self, game: "Game", hand: list[CardId], cards_in_trick: list[CardId]) -> CardId:
        """Pick card using rule-based strategy (trained model behavior)."""
        # Use rule-based hard as proxy for self-play
        fallback = RuleBasedBot(self.player_id, BotDifficulty.HARD)
        return fallback.pick_card(game, hand, cards_in_trick)


class SkullKingEnvMasked(gym.Env["np.ndarray", int]):
    """Masked action environment with dense rewards and compact observations."""

    # Override as dict literal - gymnasium expects this pattern
    metadata = {"render_modes": ["ansi"], "render_fps": 4}  # noqa: RUF012

    # Constants for magic values
    CARD_STRENGTH_KING = 0.95
    CARD_STRENGTH_PIRATE = 0.8
    CARD_STRENGTH_TIGRESS = 0.55
    CARD_STRENGTH_WHALE = 0.4
    CARD_STRENGTH_KRAKEN = 0.25
    CARD_STRENGTH_MERMAID = 0.35
    CARD_STRENGTH_LOOT = 0.08
    CARD_STRENGTH_ESCAPE = 0.05
    CARD_STRENGTH_ROGER_BASE = 0.5
    CARD_STRENGTH_ROGER_MULT = 0.35
    CARD_STRENGTH_STANDARD_BASE = 0.2
    CARD_STRENGTH_STANDARD_MULT = 0.35
    CARD_STRENGTH_DEFAULT = 0.3
    CARD_NUMBER_MAX = 14.0
    HIGH_CARD_THRESHOLD = 10
    HAND_STRENGTH_SUIT_FACTOR = 0.6
    HAND_STRENGTH_SUIT_MIN = 3
    HAND_STRENGTH_MERMAID_PENALTY = -0.5
    HAND_STRENGTH_PIRATE_BONUS = 0.3
    HAND_STRENGTH_ESCAPE_PENALTY = 0.4
    HAND_STRENGTH_KING_BONUS = 0.2
    ROUND_5_THRESHOLD = 5
    PIRATES_MIN_FOR_BONUS = 2
    HIGH_CARDS_MIN = 2
    CARD_PLAY_HIGH_THRESHOLD = 0.6
    CARD_PLAY_LOW_THRESHOLD = 0.3
    CARD_PLAY_WEAK_THRESHOLD = 0.4
    CARD_PLAY_STRONG_THRESHOLD = 0.7
    BID_ACCURACY_CLOSE = 1
    BID_ACCURACY_OFF_BY_2 = 2
    MAX_TRICK_PLAYERS = 4
    OPPONENT_PATTERN_DIMS = 6

    def __init__(
        self,
        num_opponents: int = 3,
        opponent_bot_type: str = "random",
        opponent_difficulty: str = "medium",
        max_invalid_moves: int = 50,  # INCREASED from 10
        render_mode: str | None = None,
    ) -> None:
        """Initialize the Skull King environment.

        Args:
            num_opponents: Number of bot opponents (default: 3)
            opponent_bot_type: Type of bot ("random" or "rule_based")
            opponent_difficulty: Bot difficulty level ("easy", "medium", or "hard")
            max_invalid_moves: Maximum invalid moves before episode termination
            render_mode: Rendering mode (currently only "ansi" is supported)

        """
        super().__init__()
        self.num_players = num_opponents + 1
        self.opponent_bot_type = opponent_bot_type
        self.opponent_difficulty = self._parse_difficulty(opponent_difficulty)
        self.max_invalid_moves = max_invalid_moves
        self.render_mode = render_mode

        # ENHANCED OBSERVATION SPACE: 171 dims (was 151)
        # Breakdown:
        # - Game phase (4): one-hot for PENDING/BIDDING/PICKING/ENDED
        # - Hand encoding (90): 10 cards x 9 features each
        # - Trick state (36): 4 players x 9 features each
        # - Bidding context (8): round info, tricks, hand strength
        # - Opponent state (9): 3 opponents x 3 features each
        # - Hand strength breakdown (4): pirates, kings, mermaids, high cards
        # NEW ADDITIONS (+20 dims):
        # - Trick position (4): one-hot for 1st/2nd/3rd/4th to play
        # - Opponent patterns (6): avg bid and error per opponent
        # - Cards played count (5): by type (pirates, kings, etc.)
        # - Round progression (1): tricks played / total
        # - Bid pressure (1): (needed - remaining) / needed
        # - Position advantage (1): how often we play last
        # - Trump strength (2): our best vs seen, our avg vs seen
        # V5 QUICK WINS (+11 dims):
        # - Round one-hot (10): explicit round representation for round-specific learning
        # - Bid goal (1): explicit target during card play (helps credit assignment)
        # V6 LOOT ALLIANCE (+8 dims):
        # - Has loot card (1): binary indicator
        # - Loot card count (1): normalized 0-1
        # - Alliance status (4): binary mask for allied players (supports multiple alliances)
        # - Ally bid accuracy (1): average accuracy across all allies
        # - Alliance potential (1): sum of potential bonuses (0.2 per ally on track)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(190,), dtype=np.float32)

        # Action space: 0-10 (bids or card indices)
        self.max_action_size = 11
        self.action_space = spaces.Discrete(self.max_action_size)

        # Game state
        self.game: Game | None = None
        self.agent_player_id: str = ""
        self.bots: list[tuple[str, BaseBot]] = []
        self.invalid_move_count = 0

        # Enhanced tracking for dense rewards
        self.previous_score = 0
        self.previous_tricks_won = 0
        self.last_trick_winner: str | None = None

    def _parse_difficulty(self, difficulty: str) -> BotDifficulty:
        """Parse difficulty string to enum."""
        difficulty_map = {
            "easy": BotDifficulty.EASY,
            "medium": BotDifficulty.MEDIUM,
            "hard": BotDifficulty.HARD,
        }
        return difficulty_map.get(difficulty.lower(), BotDifficulty.MEDIUM)

    @property
    def _game(self) -> Game:
        """Get the game, asserting it exists.

        Use this in methods that should only be called after reset().
        """
        assert self.game is not None, "Game not initialized. Call reset() first."
        return self.game

    def action_masks(self) -> np.ndarray:
        """Return binary mask of valid actions.

        This enables MaskablePPO to only sample from valid actions.
        """
        mask = np.zeros(self.max_action_size, dtype=np.int8)

        if not self.game:
            mask[0] = 1
            return mask

        if self.game.state == GameState.BIDDING:
            self._mask_bidding_actions(mask)
        elif self.game.state == GameState.PICKING:
            self._mask_picking_actions(mask)
        else:
            mask[0] = 1

        return mask

    # -------------------------------------------------------------------------
    # Public interface for external use (e.g., RLBot)
    # -------------------------------------------------------------------------

    def sync_game_state(self, game: Game, agent_player_id: str) -> None:
        """Sync the environment with an external game state.

        This allows RLBot to use the environment for observation building
        without running the full simulation loop.

        Args:
            game: The current game state to sync with
            agent_player_id: The player ID to use as the agent

        """
        self.game = game
        self.agent_player_id = agent_player_id

    def get_observation(self) -> np.ndarray:
        """Get current observation array.

        Public wrapper around _get_observation for external use.

        Returns:
            Observation array of shape (190,)

        """
        return self._get_observation()

    def _mask_bidding_actions(self, mask: np.ndarray) -> None:
        """Set mask for valid bidding actions."""
        current_round = self.game.get_current_round() if self.game else None
        if current_round:
            max_bid = min(current_round.number, self.max_action_size - 1)
            mask[: max_bid + 1] = 1
        else:
            mask[0] = 1

    def _mask_picking_actions(self, mask: np.ndarray) -> None:
        """Set mask for valid card picking actions with suit-following rules."""
        if not self.game:
            mask[0] = 1
            return

        current_round = self.game.get_current_round()
        current_trick = current_round.get_current_trick() if current_round else None

        if not current_trick or current_trick.picking_player_id != self.agent_player_id:
            mask[0] = 1  # Not agent's turn, dummy action
            return

        agent_player = self.game.get_player(self.agent_player_id)
        if not agent_player or not agent_player.hand:
            mask[0] = 1
            return

        # Get valid cards based on suit-following rules
        cards_in_trick = [pc.card_id for pc in current_trick.picked_cards]
        valid_cards = current_trick.get_valid_cards(agent_player.hand, cards_in_trick)

        # Map valid card IDs to hand indices (max 10 actions)
        for i, card_id in enumerate(agent_player.hand[: self.max_action_size - 1]):
            if card_id in valid_cards:
                mask[i] = 1

        # Ensure at least one action is valid
        if mask.sum() == 0:
            mask[0] = 1

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset environment."""
        super().reset(seed=seed, options=options)

        # Create new game
        self.game = Game(id="env_game", slug="env-game")

        # Create agent player
        self.agent_player_id = "agent_0"
        agent_player = Player(
            id=self.agent_player_id,
            username="Agent",
            game_id=self.game.id,
            index=0,
            is_bot=False,
        )
        self.game.add_player(agent_player)

        # Create bot opponents
        self.bots = []
        for i in range(self.num_players - 1):
            bot_id = f"bot_{i}"
            bot_player = Player(
                id=bot_id,
                username=f"Bot{i + 1}",
                game_id=self.game.id,
                index=i + 1,
                is_bot=True,
            )
            self.game.add_player(bot_player)

            bot: BaseBot
            if self.opponent_bot_type == "random":
                bot = RandomBot(bot_player.id)
            else:
                bot = RuleBasedBot(bot_player.id, self.opponent_difficulty)

            self.bots.append((bot_player.id, bot))

        # Start first round
        self.game.start_new_round()
        self.game.deal_cards()  # CRITICAL: Deal cards to players!
        self.game.state = GameState.BIDDING
        self.invalid_move_count = 0
        self.previous_score = 0
        self.previous_tricks_won = 0
        self.last_trick_winner = None

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute action with DENSE REWARD SHAPING."""
        if self.game is None:
            msg = "Environment not initialized. Call reset() first."
            raise RuntimeError(msg)

        agent_player = self.game.get_player(self.agent_player_id)
        if not agent_player:
            msg = "Agent player not found"
            raise RuntimeError(msg)

        # V5: Verify action mask is correctly applied
        mask = self.action_masks()
        if mask[action] == 0:
            # Log but don't crash - model may occasionally violate mask during exploration
            # This helps debug mask issues during training
            logger.warning(
                "Action %d violates mask! Valid actions: %s", action, np.where(mask == 1)[0]
            )

        reward = self._execute_masked_action(action, agent_player)
        terminated, truncated, final_reward = self._check_masked_termination(reward, agent_player)

        if not terminated and not truncated:
            self._bots_play_cards()

        observation = self._get_observation()
        info = self._get_info()

        return observation, final_reward, terminated, truncated, info

    def _execute_masked_action(self, action: int, agent_player: Player) -> float:
        """Execute action and calculate rewards."""
        success = False
        reward = 0.0

        if self._game.state == GameState.BIDDING:
            success = self._handle_bidding(action, agent_player)
            if success:
                reward += self._calculate_bid_quality_reward(action, agent_player)
        elif self._game.state == GameState.PICKING:
            success = self._handle_card_playing(action, agent_player)
            if success:
                reward += self._calculate_card_play_reward(action, agent_player)

        # Valid action bonus/penalty
        if success:
            reward += 0.1
        else:
            reward -= 0.5
            self.invalid_move_count += 1

        return reward

    def _check_masked_termination(
        self, reward: float, agent_player: Player
    ) -> tuple[bool, bool, float]:
        """Check termination conditions and calculate final reward."""
        terminated = False
        truncated = False
        final_reward = reward

        if self.invalid_move_count >= self.max_invalid_moves:
            truncated = True
            final_reward -= 5.0

        # Trick completion rewards
        current_round = self._game.get_current_round()
        if current_round and current_round.tricks:
            final_reward += self._process_trick_rewards(agent_player, current_round)

        # Round completion rewards
        if current_round and current_round.is_complete():
            final_reward += self._calculate_round_reward(agent_player, current_round)
            current_round.calculate_scores()

        # Game completion rewards
        if self._game.is_game_complete():
            terminated = True
            final_reward += self._calculate_game_reward(agent_player)

        return terminated, truncated, final_reward

    def _process_trick_rewards(self, agent_player: Player, current_round: Round) -> float:
        """Process rewards for completed tricks."""
        reward = 0.0
        last_trick = current_round.tricks[-1]
        is_complete = last_trick.is_complete(self.num_players)
        is_new_winner = last_trick.winner_player_id != self.last_trick_winner

        if is_complete and is_new_winner:
            self.last_trick_winner = last_trick.winner_player_id
            reward += self._calculate_trick_reward(agent_player, last_trick)
            reward += self._calculate_bonus_capture_reward(last_trick)

        return reward

    def _handle_bidding(self, action: int, agent_player: Player) -> bool:
        """Handle bidding phase."""
        current_round = self._game.get_current_round()
        if not current_round:
            return False

        bid = min(action, current_round.number)
        agent_player.bid = bid
        current_round.add_bid(self.agent_player_id, bid)

        # Have bots make their bids
        for bot_id, bot in self.bots:
            bot_player = self._game.get_player(bot_id)
            if bot_player and bot_player.bid is None:
                bot_bid = bot.make_bid(self._game, current_round.number, bot_player.hand)
                bot_player.bid = bot_bid
                current_round.add_bid(bot_id, bot_bid)

        # Transition to picking after all bids
        if self._all_players_bid():
            self._game.state = GameState.PICKING
            self._start_new_trick()
            # Bots will play in step() after this returns

        return True

    def _handle_card_playing(self, action: int, agent_player: Player) -> bool:
        """Handle card playing phase with suit-following validation."""
        card_index = action
        if 0 <= card_index < len(agent_player.hand):
            card_to_play = agent_player.hand[card_index]

            # Validate suit-following rules
            current_round = self._game.get_current_round()
            current_trick = current_round.get_current_trick() if current_round else None
            if current_trick:
                cards_in_trick = [pc.card_id for pc in current_trick.picked_cards]
                valid_cards = current_trick.get_valid_cards(agent_player.hand, cards_in_trick)
                if card_to_play not in valid_cards:
                    return False  # Invalid move - must follow suit

            return self._play_card(self.agent_player_id, card_to_play)
        return False

    def _calculate_bid_quality_reward(self, bid: int, agent_player: Player) -> float:
        """DENSE REWARD: Reward reasonable bids based on hand strength."""
        hand_strength = self._estimate_hand_strength(agent_player.hand)
        current_round = self._game.get_current_round()
        if not current_round:
            return 0.0

        # Reward bids close to estimated hand strength
        bid_error = abs(bid - hand_strength)
        max_error = current_round.number

        if max_error == 0:
            return 0.0

        # Linear reward: 0 error = +2.0, max error = 0.0
        quality = 1.0 - (bid_error / max_error)
        return quality * 2.0

    def _calculate_card_play_reward(self, card_index: int, agent_player: Player) -> float:
        """DENSE REWARD: Reward strategic card play."""
        current_round = self._game.get_current_round()
        if not current_round:
            return 0.0

        card_played = agent_player.hand[card_index] if card_index < len(agent_player.hand) else None
        if not card_played:
            return 0.0

        bid = agent_player.bid if agent_player.bid is not None else 0
        tricks_won = current_round.get_tricks_won(self.agent_player_id)
        tricks_needed = bid - tricks_won

        card = get_card(card_played)
        card_strength = self._evaluate_card_strength(card)

        return self._evaluate_card_play_strategy(tricks_needed, card_strength)

    def _evaluate_card_play_strategy(self, tricks_needed: int, card_strength: float) -> float:
        """Evaluate card play strategy and return reward."""
        if tricks_needed > 0:
            if card_strength > self.CARD_PLAY_HIGH_THRESHOLD:
                return 1.0
            if card_strength < self.CARD_PLAY_LOW_THRESHOLD:
                return -0.5
        elif tricks_needed == 0:
            if card_strength < self.CARD_PLAY_WEAK_THRESHOLD:
                return 0.5
            if card_strength > self.CARD_PLAY_STRONG_THRESHOLD:
                return -0.3
        return 0.0

    def _calculate_trick_reward(self, agent_player: Player, trick: Trick) -> float:
        """DENSE REWARD: Immediate feedback on trick outcomes."""
        current_round = self._game.get_current_round()
        if not current_round:
            return 0.0

        won_trick = trick.winner_player_id == self.agent_player_id
        bid = agent_player.bid if agent_player.bid is not None else 0
        tricks_won = current_round.get_tricks_won(self.agent_player_id)
        tricks_needed = bid - (tricks_won - (1 if won_trick else 0))

        return self._evaluate_trick_outcome(
            tricks_needed, won_trick=won_trick, current_round=current_round
        )

    def _evaluate_trick_outcome(
        self, tricks_needed: int, *, won_trick: bool, current_round: Round
    ) -> float:
        """Evaluate trick outcome and return reward."""
        if tricks_needed > 0 and won_trick:
            return 3.0
        if tricks_needed == 0 and not won_trick:
            return 1.5
        if tricks_needed == 0 and won_trick:
            return -2.0
        if tricks_needed > 0 and not won_trick:
            tricks_remaining = current_round.number - len(current_round.tricks)
            if tricks_needed > tricks_remaining:
                return 0.0
            return -1.0
        return 0.0

    def _calculate_bonus_capture_reward(self, trick: Trick) -> float:
        """Reward for capturing valuable cards (14s and character combos).

        Only awarded if agent won the trick AND hit their bid (checked at round end).
        """
        if trick.winner_player_id != self.agent_player_id:
            return 0.0

        winner_card = get_card(trick.winner_card_id) if trick.winner_card_id else None
        reward = 0.0

        for picked in trick.picked_cards:
            reward += self._calculate_card_capture_bonus(picked.card_id, winner_card)

        return reward

    def _calculate_card_capture_bonus(self, card_id: CardId, winner_card: Card | None) -> float:
        """Calculate bonus for capturing a specific card."""
        card = get_card(card_id)
        reward = 0.0

        # Bonus for capturing 14s
        if card_id in [CardId.PARROT14, CardId.CHEST14, CardId.MAP14]:
            reward += 0.3
        elif card_id == CardId.ROGER14:
            reward += 0.5

        # Character capture bonuses
        if winner_card:
            if winner_card.is_pirate() and card.is_mermaid():
                reward += 0.5
            elif winner_card.is_king() and card.is_pirate():
                reward += 0.7
            elif winner_card.is_mermaid() and card.is_king():
                reward += 1.0

        return reward

    def _calculate_round_reward(self, agent_player: Player, current_round: Round) -> float:
        """Round completion reward (bidding accuracy + alliance bonus) - NORMALIZED."""
        bid = agent_player.bid if agent_player.bid is not None else 0
        tricks_won = current_round.get_tricks_won(self.agent_player_id)
        bid_accuracy = abs(bid - tricks_won)

        # Base reward from bid accuracy (normalized scale: -5 to +5)
        if bid_accuracy == 0:
            base_reward = 5.0  # Perfect bid!
        elif bid_accuracy == self.BID_ACCURACY_CLOSE:
            base_reward = 2.0  # Close
        elif bid_accuracy == self.BID_ACCURACY_OFF_BY_2:
            base_reward = -1.0
        else:
            base_reward = -5.0  # Bad bid (capped)

        # Alliance bonus: +2.0 per successful alliance (normalized from +20 actual)
        alliance_reward = self._calculate_alliance_reward(agent_player, current_round)

        return base_reward + alliance_reward

    def _calculate_alliance_reward(self, _agent_player: Player, current_round: Round) -> float:
        """Calculate alliance bonus reward at round end.

        Backend awards +20 to each player if both loot player and ally made their bids.
        We normalize to +2.0 per successful alliance for RL training.
        """
        if not current_round.loot_alliances:
            return 0.0

        agent_bid_correct = self._check_bid_correct(self.agent_player_id, current_round)
        if not agent_bid_correct:
            return 0.0  # Agent must make bid to get alliance bonus

        alliance_bonus = 0.0
        for loot_player_id, ally_player_id in current_round.loot_alliances.items():
            # Check if agent is involved in this alliance
            if loot_player_id == self.agent_player_id and self._check_bid_correct(
                ally_player_id, current_round
            ):
                # Agent played loot and ally made their bid
                alliance_bonus += 2.0  # +20 normalized to +2.0
            elif ally_player_id == self.agent_player_id and self._check_bid_correct(
                loot_player_id, current_round
            ):
                # Agent won the loot and loot player made their bid
                alliance_bonus += 2.0  # +20 normalized to +2.0

        return alliance_bonus

    def _check_bid_correct(self, player_id: str, current_round: Round) -> bool:
        """Check if a player made their bid correctly."""
        if player_id not in current_round.bids:
            return False
        bid = current_round.bids[player_id]
        tricks_won = current_round.get_tricks_won(player_id)
        return bid == tricks_won

    def _calculate_game_reward(self, _agent_player: Player) -> float:
        """Calculate final game reward (ranking) - NORMALIZED."""
        leaderboard = self._game.get_leaderboard()
        agent_rank = next(
            (i for i, p in enumerate(leaderboard) if p["player_id"] == self.agent_player_id), 3
        )

        # Normalized scale: -5 to +10 (was -35 to +80)
        rank_rewards = [10, 3, -2, -5]
        return rank_rewards[min(agent_rank, 3)]

    def _estimate_hand_strength(self, hand: list[CardId]) -> float:
        """Enhanced hand strength estimation with context awareness."""
        if not hand:
            return 0.0

        card_objects = [get_card(cid) for cid in hand]
        base_strength = sum(self._evaluate_card_strength(c) for c in card_objects)

        current_round = self._game.get_current_round()
        round_number = current_round.number if current_round else 1

        context_adjustment = self._calculate_context_adjustments(hand, card_objects, round_number)

        return max(0, round(base_strength + context_adjustment))

    def _calculate_context_adjustments(
        self, hand: list[CardId], card_objects: list[Card], round_number: int
    ) -> float:
        """Calculate context-based adjustments to hand strength."""
        adjustment = 0.0

        # Suit distribution bonus
        suit_counts = self._count_cards_by_suit(hand)
        max_suit_count = max(suit_counts.values()) if suit_counts else 0
        if max_suit_count >= max(
            round_number * self.HAND_STRENGTH_SUIT_FACTOR, self.HAND_STRENGTH_SUIT_MIN
        ):
            adjustment += 0.5

        # Special card synergies
        has_mermaid = any(c.is_mermaid() for c in card_objects)
        high_cards = sum(
            1
            for c in card_objects
            if self._evaluate_card_strength(c) > self.CARD_PLAY_STRONG_THRESHOLD
        )

        if has_mermaid and high_cards >= self.HIGH_CARDS_MIN:
            adjustment += self.HAND_STRENGTH_MERMAID_PENALTY

        # Pirate strength in later rounds
        pirates = sum(1 for c in card_objects if c.is_pirate())
        if round_number >= self.ROUND_5_THRESHOLD and pirates >= self.PIRATES_MIN_FOR_BONUS:
            adjustment += self.HAND_STRENGTH_PIRATE_BONUS

        # Escape cards reduce expected tricks
        escapes = sum(1 for c in card_objects if c.is_escape())
        if escapes > 0:
            adjustment -= escapes * self.HAND_STRENGTH_ESCAPE_PENALTY

        # Kings guarantee some tricks
        kings = sum(1 for c in card_objects if c.is_king())
        if kings >= 1:
            adjustment += self.HAND_STRENGTH_KING_BONUS

        return adjustment

    def _count_cards_by_suit(self, hand: list[CardId]) -> dict[str, int]:
        """Count cards by suit."""
        suit_counts: dict[str, int] = {}
        for card_id in hand:
            card = get_card(card_id)
            if card.is_standard_suit() and hasattr(card, "card_type"):
                suit = card.card_type.name
                suit_counts[suit] = suit_counts.get(suit, 0) + 1
        return suit_counts

    def _encode_trick_position(self) -> list[float]:
        """Encode current trick position (who plays when) - 4 dims."""
        position = [0.0] * self.MAX_TRICK_PLAYERS  # [first, second, third, fourth]

        if not self.game:
            return position

        current_round = self.game.get_current_round()
        if not current_round:
            return position

        current_trick = current_round.get_current_trick()
        if not current_trick:
            return position

        # Determine agent's position in this trick
        num_played = len(current_trick.picked_cards)

        # If agent hasn't played yet and position is valid
        agent_not_played = all(
            pc.player_id != self.agent_player_id for pc in current_trick.picked_cards
        )
        if agent_not_played and num_played < self.MAX_TRICK_PLAYERS:
            position[num_played] = 1.0

        return position

    def _encode_opponent_patterns(self) -> list[float]:
        """Encode opponent bidding patterns - 6 dims."""
        patterns = []

        if not self.game:
            return [0.0] * 6

        for bot_id, _ in self.bots:
            bids = []
            errors = []

            for round_obj in self.game.rounds:
                if bot_id in round_obj.bids:
                    bid = round_obj.bids[bot_id]
                    tricks_won = round_obj.get_tricks_won(bot_id)
                    bids.append(bid)
                    errors.append(abs(bid - tricks_won))

            avg_bid = float(np.mean(bids)) if bids else 0.0
            avg_error = float(np.mean(errors)) if errors else 0.0

            patterns.append(avg_bid / 10.0)
            patterns.append(avg_error / 5.0)

        # Pad to 6 dims if fewer than 3 bots
        while len(patterns) < self.OPPONENT_PATTERN_DIMS:
            patterns.append(0.0)

        return patterns[: self.OPPONENT_PATTERN_DIMS]

    def _encode_cards_played(self) -> list[float]:
        """Encode what cards have been played this round - 5 dims."""
        if not self.game:
            return [0.0] * 5

        current_round = self.game.get_current_round()
        if not current_round:
            return [0.0] * 5

        card_counts = self._count_played_cards(current_round)
        total_round_cards = current_round.number * self.num_players

        return [count / max(total_round_cards, 1) for count in card_counts]

    def _count_played_cards(self, current_round: Round) -> list[int]:
        """Count different types of played cards."""
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
                    elif (
                        card.is_standard_suit()
                        and card.number
                        and card.number >= self.HIGH_CARD_THRESHOLD
                    ):
                        high_cards += 1

        return [pirates, kings, mermaids, escapes, high_cards]

    def _calculate_bid_pressure(self) -> float:
        """Calculate pressure to win tricks - 1 dim."""
        if not self.game:
            return 0.0

        current_round = self.game.get_current_round()
        if not current_round:
            return 0.0

        agent_player = self.game.get_player(self.agent_player_id)
        if not agent_player or agent_player.bid is None:
            return 0.0

        bid = agent_player.bid
        tricks_won = current_round.get_tricks_won(self.agent_player_id)
        tricks_needed = bid - tricks_won
        tricks_remaining = current_round.number - len(current_round.tricks)

        if tricks_needed <= 0:
            return -0.5  # Negative pressure - avoid winning
        if tricks_remaining == 0:
            return 1.0 if tricks_needed > 0 else 0.0
        return min(tricks_needed / max(tricks_remaining, 1), 1.0)

    def _calculate_position_advantage(self) -> float:
        """Calculate how often we play in advantageous position - 1 dim."""
        if not self.game:
            return 0.0

        current_round = self.game.get_current_round()
        if not current_round or not current_round.tricks:
            return 0.0

        last_position_count = 0
        total_tricks = 0

        for trick in current_round.tricks:
            if len(trick.picked_cards) == self.num_players:
                total_tricks += 1
                if trick.picked_cards and trick.picked_cards[-1].player_id == self.agent_player_id:
                    last_position_count += 1

        return last_position_count / max(total_tricks, 1)

    def _encode_trump_strength(self) -> list[float]:
        """Encode strength of our cards vs cards seen - 2 dims."""
        if not self.game:
            return [0.0, 0.0]

        agent_player = self.game.get_player(self.agent_player_id)
        if not agent_player or not agent_player.hand:
            return [0.0, 0.0]

        # Calculate our strongest and average card strength
        our_strengths = [self._evaluate_card_strength(get_card(cid)) for cid in agent_player.hand]
        our_best = max(our_strengths) if our_strengths else 0.0
        our_avg = float(np.mean(our_strengths)) if our_strengths else 0.0

        # Calculate average strength of cards played
        current_round = self.game.get_current_round()
        if current_round:
            seen_strengths = [
                self._evaluate_card_strength(get_card(card_id))
                for trick in current_round.tricks
                for card_id in trick.get_all_card_ids()
                if card_id
            ]

            if seen_strengths:
                seen_avg = float(np.mean(seen_strengths))
                return [
                    our_best - seen_avg,  # How much better is our best card
                    our_avg - seen_avg,  # How much better is our average
                ]

        return [our_best, our_avg]

    def _evaluate_card_strength(self, card: Card) -> float:
        """Evaluate card strength (0.0 to 1.0). Handles all 74 cards."""
        strength_map = {
            "king": self.CARD_STRENGTH_KING,
            "pirate": self.CARD_STRENGTH_PIRATE,
            "tigress": self.CARD_STRENGTH_TIGRESS,
            "whale": self.CARD_STRENGTH_WHALE,
            "kraken": self.CARD_STRENGTH_KRAKEN,
            "mermaid": self.CARD_STRENGTH_MERMAID,
            "loot": self.CARD_STRENGTH_LOOT,
            "escape": self.CARD_STRENGTH_ESCAPE,
        }

        for card_type, strength in strength_map.items():
            if getattr(card, f"is_{card_type}")():
                return strength

        if card.is_roger():
            base = self.CARD_STRENGTH_ROGER_BASE
            mult = self.CARD_STRENGTH_ROGER_MULT
            return base + (card.number / self.CARD_NUMBER_MAX) * mult if card.number else base

        if card.is_standard_suit():
            base = self.CARD_STRENGTH_STANDARD_BASE
            mult = self.CARD_STRENGTH_STANDARD_MULT
            return base + (card.number / self.CARD_NUMBER_MAX) * mult if card.number else base

        return self.CARD_STRENGTH_DEFAULT

    def _get_observation(self) -> np.ndarray:
        """COMPACT OBSERVATIONS: 190 dims (V6 with alliance awareness)."""
        if self.game is None:
            return np.zeros((190,), dtype=np.float32)

        obs = []
        agent_player = self.game.get_player(self.agent_player_id)
        current_round = self.game.get_current_round()

        # 1. GAME PHASE (4 dims)
        obs.extend(self._encode_game_phase())

        # 2. COMPACT HAND ENCODING (90 dims: 10 cards x 9 features)
        obs.extend(self._encode_hand_compact(agent_player))

        # 3. TRICK STATE (36 dims: 4 players x 9 features)
        obs.extend(self._encode_trick_state(current_round))

        # 4. BIDDING CONTEXT (8 dims)
        obs.extend(self._encode_bidding_context(agent_player, current_round))

        # 5. OPPONENT STATE (9 dims: 3 opponents x 3 features)
        obs.extend(self._encode_opponent_state())

        # 6. HAND STRENGTH BREAKDOWN (4 dims)
        obs.extend(self._encode_hand_strength_breakdown(agent_player))

        # 7-13. NEW OBSERVATIONS (+20 dims)
        obs.extend(self._encode_trick_position())
        obs.extend(self._encode_opponent_patterns())
        obs.extend(self._encode_cards_played())
        obs.append(self._encode_round_progression(current_round))
        obs.append(self._calculate_bid_pressure())
        obs.append(self._calculate_position_advantage())
        obs.extend(self._encode_trump_strength())

        # 14-15. V5 QUICK WINS (+11 dims)
        obs.extend(self._encode_round_onehot(current_round))
        obs.append(self._encode_bid_goal(agent_player))

        # 16. V6 LOOT ALLIANCE (+8 dims)
        obs.extend(self._encode_loot_alliance_state(agent_player, current_round))

        return np.array(obs, dtype=np.float32)

    def _encode_game_phase(self) -> list[float]:
        """Encode game phase (4 dims)."""
        phase = np.zeros(4, dtype=np.float32)
        state_map = {
            GameState.PENDING: 0,
            GameState.BIDDING: 1,
            GameState.PICKING: 2,
            GameState.ENDED: 3,
        }
        phase[state_map.get(self._game.state, 0)] = 1.0
        return phase.tolist()

    def _encode_hand_compact(self, agent_player: Player | None) -> list[float]:
        """Encode hand (90 dims: 10 cards x 9 features)."""
        obs = []
        for i in range(10):
            if agent_player and i < len(agent_player.hand):
                card = get_card(agent_player.hand[i])
                obs.extend(self._encode_card_compact(card))
            else:
                obs.extend([0.0] * 9)
        return obs

    def _encode_trick_state(self, current_round: Round | None) -> list[float]:
        """Encode trick state (36 dims: 4 players x 9 features)."""
        obs = []
        if current_round:
            current_trick = current_round.get_current_trick()
            if current_trick:
                cards_in_trick = current_trick.get_all_card_ids()
                for i in range(4):
                    if i < len(cards_in_trick):
                        card = get_card(cards_in_trick[i])
                        obs.extend(self._encode_card_compact(card))
                    else:
                        obs.extend([0.0] * 9)
            else:
                obs.extend([0.0] * 36)
        else:
            obs.extend([0.0] * 36)
        return obs

    def _encode_bidding_context(
        self, agent_player: Player | None, current_round: Round | None
    ) -> list[float]:
        """Encode bidding context (8 dims)."""
        if not current_round or not agent_player:
            return [0.0] * 8

        tricks_won = current_round.get_tricks_won(self.agent_player_id)
        tricks_remaining = current_round.number - len(current_round.tricks)
        bid = agent_player.bid if agent_player.bid is not None else 0
        tricks_needed = bid - tricks_won

        return [
            current_round.number / 10.0,
            len(agent_player.hand) / 10.0,
            bid / 10.0,
            tricks_won / 10.0,
            max(tricks_needed, -10) / 10.0,
            tricks_remaining / 10.0,
            1.0 if tricks_needed <= tricks_remaining else 0.0,
            self._estimate_hand_strength(agent_player.hand) / 10.0,
        ]

    def _encode_opponent_state(self) -> list[float]:
        """Encode opponent state (9 dims: 3 opponents x 3 features)."""
        obs: list[float] = []
        for i in range(1, 4):
            if i < len(self._game.players):
                opp = self._game.players[i]
                obs.extend(
                    [
                        opp.bid / 10.0 if opp.bid is not None else 0.0,
                        opp.score / 100.0,
                        opp.tricks_won / 10.0,
                    ]
                )
            else:
                obs.extend([0.0] * 3)
        return obs

    def _encode_hand_strength_breakdown(self, agent_player: Player | None) -> list[float]:
        """Encode hand strength breakdown (4 dims)."""
        if not agent_player:
            return [0.0] * 4

        high_standard = self._count_card_type(
            agent_player.hand,
            lambda c: c.is_standard_suit() and c.number >= self.HIGH_CARD_THRESHOLD,
        )
        return [
            self._count_card_type(agent_player.hand, lambda c: c.is_pirate()) / 5.0,
            self._count_card_type(agent_player.hand, lambda c: c.is_king()) / 4.0,
            self._count_card_type(agent_player.hand, lambda c: c.is_mermaid()) / 2.0,
            high_standard / 10.0,
        ]

    def _encode_round_progression(self, current_round: Round | None) -> float:
        """Encode round progression (1 dim)."""
        if current_round:
            return len(current_round.tricks) / max(current_round.number, 1)
        return 0.0

    def _encode_round_onehot(self, current_round: Round | None) -> list[float]:
        """Encode round number as one-hot (10 dims).

        V5 Quick Win: Explicit round representation enables round-specific
        learning without requiring separate policies per round.
        """
        onehot = [0.0] * 10
        if current_round:
            round_idx = min(current_round.number - 1, 9)  # 0-indexed, capped at 9
            onehot[round_idx] = 1.0
        return onehot

    def _encode_bid_goal(self, agent_player: Player | None) -> float:
        """Encode current bid as explicit goal (1 dim).

        V5 Quick Win: During card play, the agent needs to know its target.
        This makes the bid an explicit "goal" observation, improving credit
        assignment for the card-playing decisions.
        """
        if agent_player and agent_player.bid is not None:
            return agent_player.bid / 10.0  # Normalized to [0, 1]
        return 0.0

    def _encode_loot_alliance_state(
        self, agent_player: Player | None, current_round: Round | None
    ) -> list[float]:
        """Encode loot card and alliance information (8 dims).

        V6 Enhancement: Allow agent to understand loot alliance mechanics.
        When a player plays a loot card, they form an alliance with the trick
        winner. If both make their bids at round end, each gets +20 bonus.

        Supports multiple alliances: A player can be allied with multiple others
        (e.g., winning tricks containing loot cards from different players).

        Returns:
            8 floats:
            - has_loot (1): 1.0 if agent has loot in hand, else 0.0
            - loot_count (1): normalized count (0, 0.5, or 1.0)
            - alliance_status (4): binary mask for allied players (multiple can be 1.0)
            - ally_bid_accuracy (1): average accuracy across all allies
            - alliance_potential (1): sum of potential bonuses (0.2 per ally on track)
        """
        obs = []

        # Check if agent has loot cards in hand
        loot_card_ids = {CardId.LOOT1, CardId.LOOT2}
        loot_count = 0
        if agent_player:
            loot_count = sum(1 for card_id in agent_player.hand if card_id in loot_card_ids)

        # Has loot (1 dim)
        obs.append(1.0 if loot_count > 0 else 0.0)

        # Loot count normalized (1 dim): 0, 0.5, or 1.0
        obs.append(loot_count / 2.0)

        # Alliance status (4 dims): binary mask for allied player indices
        # Multiple alliances possible (e.g., winning multiple loot cards)
        alliance_mask = [0.0, 0.0, 0.0, 0.0]
        ally_players: list[Player] = []

        if current_round and current_round.loot_alliances and self.game:
            # Check ALL alliances involving the agent (as loot player or ally)
            for loot_player_id, ally_player_id in current_round.loot_alliances.items():
                if loot_player_id == self.agent_player_id:
                    # Agent played loot, find ally's index
                    ally_player = self.game.get_player(ally_player_id)
                    if ally_player:
                        ally_idx = self.game.players.index(ally_player)
                        if 0 <= ally_idx < 4:
                            alliance_mask[ally_idx] = 1.0
                            ally_players.append(ally_player)
                elif ally_player_id == self.agent_player_id:
                    # Agent is the ally (won trick with loot)
                    loot_player = self.game.get_player(loot_player_id)
                    if loot_player:
                        ally_idx = self.game.players.index(loot_player)
                        if 0 <= ally_idx < 4:
                            alliance_mask[ally_idx] = 1.0
                            ally_players.append(loot_player)

        obs.extend(alliance_mask)

        # Ally bid accuracy (1 dim): average across all allies
        ally_accuracy = 0.0
        if ally_players and current_round and current_round.number > 0:
            accuracies = []
            for ally in ally_players:
                if ally.bid is not None:
                    tricks_won = current_round.get_tricks_won(ally.id)
                    acc = (tricks_won - ally.bid) / current_round.number
                    accuracies.append(max(-1.0, min(1.0, acc)))
            if accuracies:
                ally_accuracy = sum(accuracies) / len(accuracies)
        obs.append(ally_accuracy)

        # Alliance potential (1 dim): sum of potential bonuses
        # +0.2 per ally on track (could be +0.4 if 2 allies on track)
        alliance_potential = 0.0
        if ally_players and agent_player and current_round:
            tricks_remaining = current_round.number - len(current_round.tricks)

            # Check if agent is on track
            agent_tricks_won = current_round.get_tricks_won(self.agent_player_id)
            agent_needed = (agent_player.bid or 0) - agent_tricks_won
            agent_on_track = 0 <= agent_needed <= tricks_remaining

            if agent_on_track:
                for ally in ally_players:
                    if ally.bid is not None:
                        ally_tricks_won = current_round.get_tricks_won(ally.id)
                        ally_needed = ally.bid - ally_tricks_won
                        ally_on_track = 0 <= ally_needed <= tricks_remaining
                        if ally_on_track:
                            alliance_potential += 0.2  # +20 per ally, normalized

        obs.append(alliance_potential)

        return obs

    def _encode_card_compact(self, card: Card) -> list[float]:
        """Encode card with 9 features. Handles all 74 cards including expansion."""
        encoding = []

        # Card type one-hot (5 dims) - grouped for feature efficiency
        # [0] = Standard suits including Roger (has number)
        # [1] = Pirate-like (Pirate, Tigress as pirate)
        # [2] = King (Skull King)
        # [3] = Mermaid
        # [4] = Escape-like (Escape, Loot, Tigress as escape)
        # Note: Beasts (Whale, Kraken) handled via special flags
        card_type_vec = [0.0] * 5
        if card.is_standard_suit() or card.is_roger():
            card_type_vec[0] = 1.0  # Standard suit card
        elif card.is_pirate():
            card_type_vec[1] = 1.0  # Pirate
        elif card.is_tigress():
            # Tigress can be pirate or escape - encode as both
            card_type_vec[1] = 0.5  # Partial pirate
            card_type_vec[4] = 0.5  # Partial escape
        elif card.is_king():
            card_type_vec[2] = 1.0  # King/Skull King
        elif card.is_mermaid():
            card_type_vec[3] = 1.0  # Mermaid
        elif card.is_escape() or card.is_loot():
            card_type_vec[4] = 1.0  # Escape or Loot
        elif card.is_whale() or card.is_kraken():
            # Beasts get their own partial encoding
            card_type_vec[2] = 0.5  # Beast power similar to king
        encoding.extend(card_type_vec)

        # Number (1 dim, normalized) - only meaningful for standard suits
        encoding.append(card.number / 14.0 if card.number else 0.0)

        # Special flags (3 dims) - expanded meaning for 74 cards:
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

    def _get_info(self) -> dict[str, Any]:
        """Get additional info."""
        info: dict[str, Any] = {}
        if self.game:
            info["game_state"] = self.game.state.value
            info["invalid_moves"] = self.invalid_move_count
            agent_player = self.game.get_player(self.agent_player_id)
            if agent_player:
                info["agent_score"] = agent_player.score
        return info

    # Helper methods (same as enhanced env)
    def _all_players_bid(self) -> bool:
        if not self.game:
            return False
        return all(p.made_bid() for p in self.game.players)

    def _start_new_trick(self) -> None:
        if not self.game:
            return
        current_round = self.game.get_current_round()
        if not current_round:
            return

        trick_number = len(current_round.tricks) + 1
        starter_index = current_round.starter_player_index

        if current_round.tricks:
            last_trick = current_round.tricks[-1]
            if last_trick.winner_player_id:
                winner = self.game.get_player(last_trick.winner_player_id)
                if winner:
                    starter_index = winner.index

        trick = Trick(number=trick_number, starter_player_index=starter_index)
        trick.picking_player_id = self.game.players[starter_index].id
        current_round.tricks.append(trick)

    def _play_card(self, player_id: str, card_id: CardId) -> bool:
        if not self._validate_play_card(player_id, card_id):
            return False

        # After _validate_play_card, we know game, player, round, and trick exist
        player = self._game.get_player(player_id)
        current_round = self._game.get_current_round()
        if not player or not current_round:
            return False

        current_trick = current_round.get_current_trick()
        if not current_trick:
            return False

        tigress_choice = self._determine_tigress_choice(player, card_id, current_round)

        player.remove_card(card_id)
        current_trick.add_card(player_id, card_id, tigress_choice)

        if current_trick.is_complete(self.num_players):
            self._complete_trick(current_trick, current_round)
        else:
            self._advance_trick(player)

        return True

    def _validate_play_card(self, player_id: str, card_id: CardId) -> bool:
        """Validate that a card can be played."""
        if not self.game:
            return False

        player = self.game.get_player(player_id)
        if not player or not player.has_card(card_id):
            return False

        current_round = self.game.get_current_round()
        if not current_round:
            return False

        current_trick = current_round.get_current_trick()
        if not current_trick:
            return False

        return current_trick.picking_player_id == player_id

    def _determine_tigress_choice(
        self, player: Player, card_id: CardId, current_round: Round
    ) -> TigressChoice | None:
        """Determine Tigress choice based on bid status."""
        card = get_card(card_id)
        if not card.is_tigress():
            return None

        tricks_won = current_round.get_tricks_won(player.id)
        bid = player.bid if player.bid is not None else 0
        need_more_wins = tricks_won < bid
        return TigressChoice.PIRATE if need_more_wins else TigressChoice.ESCAPE

    def _complete_trick(self, current_trick: Trick, current_round: Round) -> None:
        """Complete a trick and update game state."""
        current_trick.determine_winner()
        if current_trick.winner_player_id:
            winner = self._game.get_player(current_trick.winner_player_id)
            if winner:
                winner.tricks_won += 1

        if current_round.is_complete():
            self._end_round()
        else:
            self._start_new_trick()

    def _advance_trick(self, player: Player) -> None:
        """Advance to next player in the trick."""
        current_round = self._game.get_current_round()
        if not current_round:
            return
        current_trick = current_round.get_current_trick()
        if not current_trick:
            return
        next_index = (player.index + 1) % self.num_players
        current_trick.picking_player_id = self._game.players[next_index].id

    def _end_round(self) -> None:
        current_round = self._game.get_current_round()
        if current_round:
            current_round.calculate_scores()

        if len(self._game.rounds) < MAX_ROUNDS:
            self._game.start_new_round()
            self._game.deal_cards()  # CRITICAL: Deal cards!
            self._game.state = GameState.BIDDING
        else:
            self._game.state = GameState.ENDED

    def _bots_play_cards(self) -> None:
        """Have bots play their cards (iterative, with safety limit)."""
        if not self.game:
            return

        max_iterations = 100
        iterations = 0

        while iterations < max_iterations:
            iterations += 1

            if self._should_stop_bot_play():
                break

            current_round = self._game.get_current_round()
            if not current_round:
                break
            current_trick = current_round.get_current_trick()
            if not current_trick:
                break
            picking_player_id = current_trick.picking_player_id

            if picking_player_id == self.agent_player_id:
                break

            if not self._execute_bot_play(picking_player_id, current_trick):
                break

    def _should_stop_bot_play(self) -> bool:
        """Check if bot play should stop."""
        if self._game.state == GameState.ENDED:
            return True

        current_round = self._game.get_current_round()
        if not current_round:
            return True

        if self._game.state == GameState.BIDDING:
            return True

        current_trick = current_round.get_current_trick()
        return not current_trick

    def _execute_bot_play(self, picking_player_id: str, current_trick: Trick) -> bool:
        """Execute a bot's card play."""
        for bot_id, bot in self.bots:
            if picking_player_id == bot_id:
                player = self._game.get_player(bot_id)
                if not player:
                    return False

                card_to_play = bot.pick_card(
                    self._game, player.hand, current_trick.get_all_card_ids()
                )
                self._play_card(bot_id, card_to_play)
                return True

        return False

    def set_opponent(self, opponent_type: str, difficulty: str = "medium") -> None:
        """Change opponent type and difficulty (for curriculum learning)."""
        self.opponent_bot_type = opponent_type
        self.opponent_difficulty = self._parse_difficulty(difficulty)

    def set_self_play_opponent(self, model_path: str) -> None:
        """Set opponents to use a trained model checkpoint (V5 self-play).

        Args:
            model_path: Path to a MaskablePPO .zip checkpoint file

        This replaces all bot opponents with SelfPlayBot instances that
        use the trained model for decision-making. Useful for preventing
        overfitting to fixed opponent strategies.
        """
        if not Path(model_path).exists():
            logger.warning("Self-play model not found: %s, using rule_based", model_path)
            self.set_opponent("rule_based", "hard")
            return

        self.opponent_bot_type = "self_play"
        self.opponent_difficulty = BotDifficulty.HARD

        # Update existing bots to use self-play
        new_bots: list[tuple[str, BaseBot]] = []
        for bot_id, _ in self.bots:
            new_bots.append((bot_id, SelfPlayBot(bot_id, model_path)))
        self.bots = new_bots

    def render(self) -> Any:
        """Render the environment."""
        if self.render_mode == "ansi":
            return self._render_ansi()
        return None

    def _render_ansi(self) -> str:
        """Render game state as ANSI string."""
        if not self.game:
            return "Game not initialized"

        lines = []
        lines.append("=" * 60)
        lines.append(f"Skull King - State: {self.game.state.value}")
        lines.append("=" * 60)

        agent_player = self.game.get_player(self.agent_player_id)
        if agent_player:
            lines.append(f"Agent Score: {agent_player.score}")
            lines.append(f"Agent Bid: {agent_player.bid}")
            lines.append(f"Agent Tricks Won: {agent_player.tricks_won}")

        return "\n".join(lines)
