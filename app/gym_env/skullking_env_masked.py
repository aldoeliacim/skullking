"""
Masked Gymnasium environment for Skull King with critical improvements:
1. Action masking (MaskablePPO support)
2. Dense reward shaping (trick-level, bid quality)
3. Compact observations (151 dims vs 1226)
"""

from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from app.bots import RandomBot, RuleBasedBot
from app.bots.base_bot import BotDifficulty
from app.models.card import Card, CardId, get_card
from app.models.enums import MAX_ROUNDS, GameState
from app.models.game import Game
from app.models.player import Player
from app.models.trick import TigressChoice, Trick


class SkullKingEnvMasked(gym.Env):
    """Masked action environment with dense rewards and compact observations."""

    metadata: ClassVar[dict[str, Any]] = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(
        self,
        num_opponents: int = 3,
        opponent_bot_type: str = "random",
        opponent_difficulty: str = "medium",
        max_invalid_moves: int = 50,  # INCREASED from 10
        render_mode: str | None = None,
    ):
        super().__init__()
        self.num_players = num_opponents + 1
        self.opponent_bot_type = opponent_bot_type
        self.opponent_difficulty = self._parse_difficulty(opponent_difficulty)
        self.max_invalid_moves = max_invalid_moves
        self.render_mode = render_mode

        # ENHANCED OBSERVATION SPACE: 171 dims (was 151)
        # Breakdown:
        # - Game phase (4): one-hot for PENDING/BIDDING/PICKING/ENDED
        # - Hand encoding (90): 10 cards × 9 features each
        # - Trick state (36): 4 players × 9 features each
        # - Bidding context (8): round info, tricks, hand strength
        # - Opponent state (9): 3 opponents × 3 features each
        # - Hand strength breakdown (4): pirates, kings, mermaids, high cards
        # NEW ADDITIONS (+20 dims):
        # - Trick position (4): one-hot for 1st/2nd/3rd/4th to play
        # - Opponent patterns (6): avg bid and error per opponent
        # - Cards played count (5): by type (pirates, kings, etc.)
        # - Round progression (1): tricks played / total
        # - Bid pressure (1): (needed - remaining) / needed
        # - Position advantage (1): how often we play last
        # - Trump strength (2): our best vs seen, our avg vs seen
        self.observation_space = spaces.Box(low=-1, high=1, shape=(171,), dtype=np.float32)

        # Action space: 0-10 (bids or card indices)
        self.action_space = spaces.Discrete(11)

        # Game state
        self.game: Game | None = None
        self.agent_player_id: str = ""
        self.bots: list[tuple[str, Any]] = []
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

    def action_masks(self) -> np.ndarray:
        """
        CRITICAL: Return binary mask of valid actions.
        This enables MaskablePPO to only sample from valid actions.
        """
        mask = np.zeros(11, dtype=np.int8)

        if not self.game:
            mask[0] = 1  # Default: allow action 0
            return mask

        if self.game.state == GameState.BIDDING:
            # Valid bids: 0 to current_round.number
            current_round = self.game.get_current_round()
            if current_round:
                max_bid = min(current_round.number, 10)
                mask[: max_bid + 1] = 1
            else:
                mask[0] = 1

        elif self.game.state == GameState.PICKING:
            # Valid cards: 0 to len(hand)-1, but ONLY if it's agent's turn
            current_round = self.game.get_current_round()
            current_trick = current_round.get_current_trick() if current_round else None

            # Check if it's the agent's turn
            if current_trick and current_trick.picking_player_id == self.agent_player_id:
                agent_player = self.game.get_player(self.agent_player_id)
                if agent_player:
                    hand_size = min(len(agent_player.hand), 10)
                    if hand_size > 0:
                        mask[:hand_size] = 1
                    else:
                        mask[0] = 1
                else:
                    mask[0] = 1
            else:
                # Not agent's turn - no valid actions (bots will play)
                # But we need at least one valid action for the policy
                mask[0] = 1  # Dummy action (will be handled by step)

        else:
            # Default state: allow action 0
            mask[0] = 1

        return mask

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset environment."""
        super().reset(seed=seed)

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
            raise RuntimeError("Environment not initialized. Call reset() first.")

        reward = 0.0
        terminated = False
        truncated = False

        agent_player = self.game.get_player(self.agent_player_id)
        if not agent_player:
            raise RuntimeError("Agent player not found")

        # Execute action based on game state
        success = False

        if self.game.state == GameState.BIDDING:
            success = self._handle_bidding(action, agent_player)
            if success:
                # DENSE REWARD: Bid quality
                reward += self._calculate_bid_quality_reward(action, agent_player)

        elif self.game.state == GameState.PICKING:
            success = self._handle_card_playing(action, agent_player)
            if success:
                # DENSE REWARD: Card play strategy
                reward += self._calculate_card_play_reward(action, agent_player)

        # DENSE REWARD: Valid action bonus
        if success:
            reward += 0.1  # Small positive reinforcement
        else:
            reward -= 0.5  # Invalid action penalty
            self.invalid_move_count += 1

        # Check for invalid move threshold
        if self.invalid_move_count >= self.max_invalid_moves:
            truncated = True
            reward -= 5.0  # Reduced from -10

        # DENSE REWARD: Trick completion rewards
        current_round = self.game.get_current_round()
        if current_round and current_round.tricks:
            last_trick = current_round.tricks[-1]
            is_complete = last_trick.is_complete(self.num_players)
            is_new_winner = last_trick.winner_player_id != self.last_trick_winner
            if is_complete and is_new_winner:
                self.last_trick_winner = last_trick.winner_player_id
                reward += self._calculate_trick_reward(agent_player, last_trick)
                # BONUS CAPTURE: Reward for capturing valuable cards
                reward += self._calculate_bonus_capture_reward(last_trick)

        # DENSE REWARD: Round completion rewards
        if current_round and current_round.is_complete():
            reward += self._calculate_round_reward(agent_player, current_round)
            current_round.calculate_scores()

        # Check if game is over
        if self.game.is_game_complete():
            terminated = True
            reward += self._calculate_game_reward(agent_player)

        # Advance to agent's next turn (let bots play)
        if not terminated and not truncated:
            self._bots_play_cards()

        observation = self._get_observation()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _handle_bidding(self, action: int, agent_player: Player) -> bool:
        """Handle bidding phase."""
        current_round = self.game.get_current_round()
        if not current_round:
            return False

        bid = min(action, current_round.number)
        agent_player.bid = bid
        current_round.add_bid(self.agent_player_id, bid)

        # Have bots make their bids
        for bot_id, bot in self.bots:
            bot_player = self.game.get_player(bot_id)
            if bot_player and bot_player.bid is None:
                bot_bid = bot.make_bid(self.game, current_round.number, bot_player.hand)
                bot_player.bid = bot_bid
                current_round.add_bid(bot_id, bot_bid)

        # Transition to picking after all bids
        if self._all_players_bid():
            self.game.state = GameState.PICKING
            self._start_new_trick()
            # Bots will play in step() after this returns

        return True

    def _handle_card_playing(self, action: int, agent_player: Player) -> bool:
        """Handle card playing phase."""
        card_index = action
        if 0 <= card_index < len(agent_player.hand):
            card_to_play = agent_player.hand[card_index]
            success = self._play_card(self.agent_player_id, card_to_play)
            return success
        return False

    def _calculate_bid_quality_reward(self, bid: int, agent_player: Player) -> float:
        """DENSE REWARD: Reward reasonable bids based on hand strength."""
        hand_strength = self._estimate_hand_strength(agent_player.hand)
        current_round = self.game.get_current_round()
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
        current_round = self.game.get_current_round()
        if not current_round:
            return 0.0

        card_played = agent_player.hand[card_index] if card_index < len(agent_player.hand) else None
        if not card_played:
            return 0.0

        # Determine if agent needs to win tricks
        bid = agent_player.bid if agent_player.bid is not None else 0
        tricks_won = current_round.get_tricks_won(self.agent_player_id)
        tricks_needed = bid - tricks_won

        card = get_card(card_played)
        card_strength = self._evaluate_card_strength(card)

        # Reward playing strong cards when needing to win
        if tricks_needed > 0:
            if card_strength > 0.6:
                return 1.0  # Good strategic choice
            if card_strength < 0.3:
                return -0.5  # Poor choice
        # Reward playing weak cards when not needing to win
        elif tricks_needed == 0:
            if card_strength < 0.4:
                return 0.5  # Good strategic choice
            if card_strength > 0.7:
                return -0.3  # Wasteful

        return 0.0

    def _calculate_trick_reward(self, agent_player: Player, trick: Trick) -> float:
        """DENSE REWARD: Immediate feedback on trick outcomes."""
        current_round = self.game.get_current_round()
        if not current_round:
            return 0.0

        won_trick = trick.winner_player_id == self.agent_player_id
        bid = agent_player.bid if agent_player.bid is not None else 0
        tricks_won = current_round.get_tricks_won(self.agent_player_id)
        tricks_needed = bid - (tricks_won - (1 if won_trick else 0))  # Before this trick

        # Reward correct strategic outcomes
        if tricks_needed > 0 and won_trick:
            return 3.0  # Good! Needed to win and did
        if tricks_needed == 0 and not won_trick:
            return 1.5  # Good! Avoided overbidding
        if tricks_needed == 0 and won_trick:
            return -2.0  # Bad! Overbidding
        if tricks_needed > 0 and not won_trick:
            tricks_remaining = current_round.number - len(current_round.tricks)
            if tricks_needed > tricks_remaining:
                return 0.0  # Can't make bid anyway
            return -1.0  # Missed needed trick

        return 0.0

    def _calculate_bonus_capture_reward(self, trick: Trick) -> float:
        """
        Reward for capturing valuable cards (14s and character combos).
        Only awarded if agent won the trick AND hit their bid (checked at round end).
        """
        if trick.winner_player_id != self.agent_player_id:
            return 0.0

        reward = 0.0
        winner_card = get_card(trick.winner_card_id) if trick.winner_card_id else None

        for picked in trick.picked_cards:
            card = get_card(picked.card_id)

            # Bonus for capturing 14s
            if picked.card_id in [CardId.PARROT14, CardId.CHEST14, CardId.MAP14]:
                reward += 0.3  # +10 points = 0.3 reward
            elif picked.card_id == CardId.ROGER14:
                reward += 0.5  # +20 points = 0.5 reward

            # Character capture bonuses (only if we won with the right card)
            if winner_card:
                if winner_card.is_pirate() and card.is_mermaid():
                    reward += 0.5  # +20 points
                elif winner_card.is_king() and card.is_pirate():
                    reward += 0.7  # +30 points
                elif winner_card.is_mermaid() and card.is_king():
                    reward += 1.0  # +40 points

        return reward

    def _calculate_round_reward(self, agent_player: Player, current_round) -> float:
        """Round completion reward (bidding accuracy) - NORMALIZED."""
        bid = agent_player.bid if agent_player.bid is not None else 0
        tricks_won = current_round.get_tricks_won(self.agent_player_id)
        bid_accuracy = abs(bid - tricks_won)

        # Normalized scale: -5 to +5 (was -80 to +20)
        if bid_accuracy == 0:
            return 5.0  # Perfect bid!
        if bid_accuracy == 1:
            return 2.0  # Close
        if bid_accuracy == 2:
            return -1.0
        return -5.0  # Bad bid (capped)

    def _calculate_game_reward(self, agent_player: Player) -> float:
        """Final game reward (ranking) - NORMALIZED."""
        leaderboard = self.game.get_leaderboard()
        agent_rank = next(
            (i for i, p in enumerate(leaderboard) if p["player_id"] == self.agent_player_id), 3
        )

        # Normalized scale: -5 to +10 (was -35 to +80)
        rank_rewards = [10, 3, -2, -5]
        reward = rank_rewards[min(agent_rank, 3)]

        return reward

    def _estimate_hand_strength(self, hand: list[CardId]) -> float:
        """Enhanced hand strength estimation with context awareness."""
        if not hand:
            return 0.0

        # Base strength from individual cards
        card_objects = [get_card(cid) for cid in hand]
        base_strength = sum(self._evaluate_card_strength(c) for c in card_objects)

        # Context adjustments
        current_round = self.game.get_current_round()
        round_number = current_round.number if current_round else 1

        # 1. Suit distribution bonus - strong in one suit helps win tricks
        suit_counts = self._count_cards_by_suit(hand)
        max_suit_count = max(suit_counts.values()) if suit_counts else 0
        if max_suit_count >= max(round_number * 0.6, 3):
            base_strength += 0.5

        # 2. Special card synergies
        has_mermaid = any(c.is_mermaid() for c in card_objects)
        high_cards = sum(1 for c in card_objects if self._evaluate_card_strength(c) > 0.7)

        if has_mermaid and high_cards >= 2:
            base_strength -= 0.5  # Mermaid liability with high cards

        # 3. Pirate strength in later rounds (more tricks = more chances)
        pirates = sum(1 for c in card_objects if c.is_pirate())
        if round_number >= 5 and pirates >= 2:
            base_strength += 0.3

        # 4. Escape cards reduce expected tricks
        escapes = sum(1 for c in card_objects if c.is_escape())
        if escapes > 0:
            base_strength -= escapes * 0.4

        # 5. Kings guarantee some tricks
        kings = sum(1 for c in card_objects if c.is_king())
        if kings >= 1:
            base_strength += 0.2

        return max(0, round(base_strength))

    def _count_cards_by_suit(self, hand: list[CardId]) -> dict:
        """Count cards by suit."""
        suit_counts = {}
        for card_id in hand:
            card = get_card(card_id)
            if card.is_standard_suit() and hasattr(card, "card_type"):
                suit = card.card_type.name
                suit_counts[suit] = suit_counts.get(suit, 0) + 1
        return suit_counts

    def _encode_trick_position(self) -> list[float]:
        """Encode current trick position (who plays when) - 4 dims."""
        position = [0.0] * 4  # [first, second, third, fourth]

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
        if agent_not_played and num_played < 4:
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
        while len(patterns) < 6:
            patterns.append(0.0)

        return patterns[:6]

    def _encode_cards_played(self) -> list[float]:
        """Encode what cards have been played this round - 5 dims."""
        if not self.game:
            return [0.0] * 5

        current_round = self.game.get_current_round()
        if not current_round:
            return [0.0] * 5

        pirates_played = 0
        kings_played = 0
        mermaids_played = 0
        escapes_played = 0
        high_cards_played = 0

        for trick in current_round.tricks:
            for card_id in trick.get_all_card_ids():
                if card_id:
                    card = get_card(card_id)
                    if card.is_pirate():
                        pirates_played += 1
                    elif card.is_king():
                        kings_played += 1
                    elif card.is_mermaid():
                        mermaids_played += 1
                    elif card.is_escape():
                        escapes_played += 1
                    elif card.is_standard_suit() and card.number and card.number >= 10:
                        high_cards_played += 1

        total_round_cards = current_round.number * self.num_players

        return [
            pirates_played / max(total_round_cards, 1),
            kings_played / max(total_round_cards, 1),
            mermaids_played / max(total_round_cards, 1),
            escapes_played / max(total_round_cards, 1),
            high_cards_played / max(total_round_cards, 1),
        ]

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
        if card.is_king():
            return 0.95  # King (Skull King) - highest
        if card.is_pirate():
            return 0.8
        if card.is_tigress():
            # Flexible - can be pirate (0.8) or escape (0.05)
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
            # Suited cards: value-based (1-14)
            return 0.2 + (card.number / 14.0) * 0.35 if card.number else 0.2
        return 0.3

    def _get_observation(self) -> np.ndarray:
        """COMPACT OBSERVATIONS: 151 dims."""
        if self.game is None:
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        obs = []
        agent_player = self.game.get_player(self.agent_player_id)
        current_round = self.game.get_current_round()

        # 1. GAME PHASE (4 dims)
        phase = np.zeros(4, dtype=np.float32)
        state_map = {
            GameState.PENDING: 0,
            GameState.BIDDING: 1,
            GameState.PICKING: 2,
            GameState.ENDED: 3,
        }
        phase[state_map.get(self.game.state, 0)] = 1.0
        obs.extend(phase)

        # 2. COMPACT HAND ENCODING (90 dims: 10 cards × 9 features)
        for i in range(10):
            if agent_player and i < len(agent_player.hand):
                card = get_card(agent_player.hand[i])
                obs.extend(self._encode_card_compact(card))
            else:
                obs.extend([0.0] * 9)

        # 3. TRICK STATE (36 dims: 4 players × 9 features)
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

        # 4. BIDDING CONTEXT (8 dims)
        if current_round and agent_player:
            tricks_won = current_round.get_tricks_won(self.agent_player_id)
            tricks_remaining = current_round.number - len(current_round.tricks)
            bid = agent_player.bid if agent_player.bid is not None else 0
            tricks_needed = bid - tricks_won

            obs.extend(
                [
                    current_round.number / 10.0,
                    len(agent_player.hand) / 10.0,
                    bid / 10.0,
                    tricks_won / 10.0,
                    max(tricks_needed, -10) / 10.0,  # Can be negative
                    tricks_remaining / 10.0,
                    1.0 if tricks_needed <= tricks_remaining else 0.0,
                    self._estimate_hand_strength(agent_player.hand) / 10.0,
                ]
            )
        else:
            obs.extend([0.0] * 8)

        # 5. OPPONENT STATE (9 dims: 3 opponents × 3 features)
        for i in range(1, 4):
            if i < len(self.game.players):
                opp = self.game.players[i]
                obs.extend(
                    [
                        opp.bid / 10.0 if opp.bid is not None else 0.0,
                        opp.score / 100.0,
                        opp.tricks_won / 10.0,
                    ]
                )
            else:
                obs.extend([0.0] * 3)

        # 6. HAND STRENGTH BREAKDOWN (4 dims)
        if agent_player:
            high_standard = self._count_card_type(
                agent_player.hand, lambda c: c.is_standard_suit() and c.number >= 10
            )
            obs.extend(
                [
                    self._count_card_type(agent_player.hand, lambda c: c.is_pirate()) / 5.0,
                    self._count_card_type(agent_player.hand, lambda c: c.is_king()) / 4.0,
                    self._count_card_type(agent_player.hand, lambda c: c.is_mermaid()) / 2.0,
                    high_standard / 10.0,
                ]
            )
        else:
            obs.extend([0.0] * 4)

        # === NEW OBSERVATIONS (+20 dims) ===

        # 7. TRICK POSITION (4 dims) - one-hot for 1st/2nd/3rd/4th to play
        obs.extend(self._encode_trick_position())

        # 8. OPPONENT PATTERNS (6 dims) - avg bid and error per opponent
        obs.extend(self._encode_opponent_patterns())

        # 9. CARDS PLAYED COUNT (5 dims) - by type
        obs.extend(self._encode_cards_played())

        # 10. ROUND PROGRESSION (1 dim)
        if current_round:
            round_progress = len(current_round.tricks) / max(current_round.number, 1)
        else:
            round_progress = 0.0
        obs.append(round_progress)

        # 11. BID PRESSURE (1 dim)
        obs.append(self._calculate_bid_pressure())

        # 12. POSITION ADVANTAGE (1 dim)
        obs.append(self._calculate_position_advantage())

        # 13. TRUMP STRENGTH (2 dims)
        obs.extend(self._encode_trump_strength())

        return np.array(obs, dtype=np.float32)

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

    def _count_card_type(self, hand: list[CardId], predicate) -> int:
        """Count cards matching a predicate."""
        return sum(1 for card_id in hand if predicate(get_card(card_id)))

    def _get_info(self) -> dict[str, Any]:
        """Get additional info."""
        info = {}
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

        if current_trick.picking_player_id != player_id:
            return False

        # Handle Tigress choice - decide based on bid status
        tigress_choice: TigressChoice | None = None
        card = get_card(card_id)
        if card.is_tigress():
            tricks_won = current_round.get_tricks_won(player_id)
            bid = player.bid if player.bid is not None else 0
            need_more_wins = tricks_won < bid
            tigress_choice = TigressChoice.PIRATE if need_more_wins else TigressChoice.ESCAPE

        player.remove_card(card_id)
        current_trick.add_card(player_id, card_id, tigress_choice)

        if current_trick.is_complete(self.num_players):
            current_trick.determine_winner()
            if current_trick.winner_player_id:
                winner = self.game.get_player(current_trick.winner_player_id)
                if winner:
                    winner.tricks_won += 1

            if current_round.is_complete():
                self._end_round()
            else:
                self._start_new_trick()
        else:
            next_index = (player.index + 1) % self.num_players
            current_trick.picking_player_id = self.game.players[next_index].id

        return True

    def _end_round(self) -> None:
        if not self.game:
            return

        current_round = self.game.get_current_round()
        if current_round:
            current_round.calculate_scores()

        if len(self.game.rounds) < MAX_ROUNDS:
            self.game.start_new_round()
            self.game.deal_cards()  # CRITICAL: Deal cards!
            self.game.state = GameState.BIDDING
        else:
            self.game.state = GameState.ENDED

    def _bots_play_cards(self) -> None:
        """Have bots play their cards (iterative, with safety limit)."""
        if not self.game:
            return

        max_iterations = 100
        iterations = 0

        while iterations < max_iterations:
            iterations += 1

            if self.game.state == GameState.ENDED:
                break

            current_round = self.game.get_current_round()
            if not current_round:
                break

            if self.game.state == GameState.BIDDING:
                break

            current_trick = current_round.get_current_trick()
            if not current_trick:
                break

            picking_player_id = current_trick.picking_player_id

            if picking_player_id == self.agent_player_id:
                break

            bot_found = False
            for bot_id, bot in self.bots:
                if picking_player_id == bot_id:
                    player = self.game.get_player(bot_id)
                    if not player:
                        break

                    card_to_play = bot.pick_card(
                        self.game, player.hand, current_trick.get_all_card_ids()
                    )
                    self._play_card(bot_id, card_to_play)
                    bot_found = True
                    break

            if not bot_found:
                break

    def set_opponent(self, opponent_type: str, difficulty: str = "medium"):
        """Change opponent type and difficulty (for curriculum learning)."""
        self.opponent_bot_type = opponent_type
        self.opponent_difficulty = self._parse_difficulty(difficulty)

    def render(self) -> str | None:
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
