"""
Masked Gymnasium environment for Skull King with critical improvements:
1. Action masking (MaskablePPO support)
2. Dense reward shaping (trick-level, bid quality)
3. Compact observations (151 dims vs 1226)
"""

from typing import Any, Dict, List, Optional, Tuple
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from app.bots import RandomBot, RuleBasedBot
from app.bots.base_bot import BotDifficulty
from app.models.card import CardId, get_card, Card
from app.models.enums import GameState, MAX_PLAYERS, MAX_ROUNDS
from app.models.game import Game
from app.models.player import Player
from app.models.trick import Trick


class SkullKingEnvMasked(gym.Env):
    """Masked action environment with dense rewards and compact observations."""

    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(
        self,
        num_opponents: int = 3,
        opponent_bot_type: str = "random",
        opponent_difficulty: str = "medium",
        max_invalid_moves: int = 50,  # INCREASED from 10
        render_mode: Optional[str] = None,
    ):
        super().__init__()
        self.num_players = num_opponents + 1
        self.opponent_bot_type = opponent_bot_type
        self.opponent_difficulty = self._parse_difficulty(opponent_difficulty)
        self.max_invalid_moves = max_invalid_moves
        self.render_mode = render_mode

        # COMPACT OBSERVATION SPACE: 151 dims (vs 1226)
        # Breakdown:
        # - Game phase (4): one-hot for PENDING/BIDDING/PICKING/ENDED
        # - Hand encoding (90): 10 cards × 9 features each
        # - Trick state (36): 4 players × 9 features each
        # - Bidding context (8): round info, tricks, hand strength
        # - Opponent state (9): 3 opponents × 3 features each
        # - Hand strength breakdown (4): pirates, kings, mermaids, high cards
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(151,), dtype=np.float32
        )

        # Action space: 0-10 (bids or card indices)
        self.action_space = spaces.Discrete(11)

        # Game state
        self.game: Optional[Game] = None
        self.agent_player_id: str = ""
        self.bots: List[Tuple[str, Any]] = []
        self.invalid_move_count = 0

        # Enhanced tracking for dense rewards
        self.previous_score = 0
        self.previous_tricks_won = 0
        self.last_trick_winner: Optional[str] = None

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
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
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
                username=f"Bot{i+1}",
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

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
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
            if last_trick.is_complete(self.num_players) and last_trick.winner_player_id != self.last_trick_winner:
                self.last_trick_winner = last_trick.winner_player_id
                reward += self._calculate_trick_reward(agent_player, last_trick)

        # DENSE REWARD: Round completion rewards
        if current_round and current_round.is_complete():
            reward += self._calculate_round_reward(agent_player, current_round)
            current_round.update_scores()

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
            self._bots_play_cards()

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
        tricks_remaining = current_round.number - len(current_round.tricks) + 1

        card = get_card(card_played)
        card_strength = self._evaluate_card_strength(card)

        # Reward playing strong cards when needing to win
        if tricks_needed > 0:
            if card_strength > 0.6:
                return 1.0  # Good strategic choice
            elif card_strength < 0.3:
                return -0.5  # Poor choice
        # Reward playing weak cards when not needing to win
        elif tricks_needed == 0:
            if card_strength < 0.4:
                return 0.5  # Good strategic choice
            elif card_strength > 0.7:
                return -0.3  # Wasteful

        return 0.0

    def _calculate_trick_reward(self, agent_player: Player, trick: Trick) -> float:
        """DENSE REWARD: Immediate feedback on trick outcomes."""
        current_round = self.game.get_current_round()
        if not current_round:
            return 0.0

        won_trick = (trick.winner_player_id == self.agent_player_id)
        bid = agent_player.bid if agent_player.bid is not None else 0
        tricks_won = current_round.get_tricks_won(self.agent_player_id)
        tricks_needed = bid - (tricks_won - (1 if won_trick else 0))  # Before this trick

        # Reward correct strategic outcomes
        if tricks_needed > 0 and won_trick:
            return 3.0  # Good! Needed to win and did
        elif tricks_needed == 0 and not won_trick:
            return 1.5  # Good! Avoided overbidding
        elif tricks_needed == 0 and won_trick:
            return -2.0  # Bad! Overbidding
        elif tricks_needed > 0 and not won_trick:
            tricks_remaining = current_round.number - len(current_round.tricks)
            if tricks_needed > tricks_remaining:
                return 0.0  # Can't make bid anyway
            else:
                return -1.0  # Missed needed trick

        return 0.0

    def _calculate_round_reward(self, agent_player: Player, current_round) -> float:
        """Round completion reward (bidding accuracy)."""
        bid = agent_player.bid if agent_player.bid is not None else 0
        tricks_won = current_round.get_tricks_won(self.agent_player_id)
        bid_accuracy = abs(bid - tricks_won)

        if bid_accuracy == 0:
            return 20.0  # Perfect bid!
        elif bid_accuracy == 1:
            return 8.0  # Close
        elif bid_accuracy == 2:
            return -3.0
        else:
            return -8.0 * bid_accuracy

    def _calculate_game_reward(self, agent_player: Player) -> float:
        """Final game reward (ranking)."""
        leaderboard = self.game.get_leaderboard()
        agent_rank = next(
            (i for i, p in enumerate(leaderboard) if p["player_id"] == self.agent_player_id),
            3
        )

        rank_rewards = [50, 15, -10, -35]
        reward = rank_rewards[min(agent_rank, 3)]

        if agent_rank == 0:
            reward += 30  # Win bonus

        return reward

    def _estimate_hand_strength(self, hand: List[CardId]) -> float:
        """Estimate expected tricks from hand."""
        strength = 0.0
        for card_id in hand:
            card = get_card(card_id)
            strength += self._evaluate_card_strength(card)
        return round(strength)

    def _evaluate_card_strength(self, card: Card) -> float:
        """Evaluate card strength (0.0 to 1.0)."""
        if card.is_skull_king():
            return 1.0
        elif card.is_pirate():
            return 0.8
        elif card.is_king():
            return 0.7
        elif card.is_mermaid():
            return 0.3
        elif card.is_escape():
            return 0.1
        else:
            # Suited cards: value-based (1-14)
            return 0.3 + (card.value / 14.0) * 0.4

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

            obs.extend([
                current_round.number / 10.0,
                len(agent_player.hand) / 10.0,
                bid / 10.0,
                tricks_won / 10.0,
                max(tricks_needed, -10) / 10.0,  # Can be negative
                tricks_remaining / 10.0,
                1.0 if tricks_needed <= tricks_remaining else 0.0,
                self._estimate_hand_strength(agent_player.hand) / 10.0,
            ])
        else:
            obs.extend([0.0] * 8)

        # 5. OPPONENT STATE (9 dims: 3 opponents × 3 features)
        for i in range(1, 4):
            if i < len(self.game.players):
                opp = self.game.players[i]
                obs.extend([
                    opp.bid / 10.0 if opp.bid is not None else 0.0,
                    opp.score / 100.0,
                    opp.tricks_won / 10.0,
                ])
            else:
                obs.extend([0.0] * 3)

        # 6. HAND STRENGTH BREAKDOWN (4 dims)
        if agent_player:
            obs.extend([
                self._count_card_type(agent_player.hand, lambda c: c.is_pirate()) / 5.0,
                self._count_card_type(agent_player.hand, lambda c: c.is_king()) / 4.0,
                self._count_card_type(agent_player.hand, lambda c: c.is_mermaid()) / 2.0,
                self._count_card_type(agent_player.hand, lambda c: not c.is_special() and c.value >= 10) / 10.0,
            ])
        else:
            obs.extend([0.0] * 4)

        return np.array(obs, dtype=np.float32)

    def _encode_card_compact(self, card: Card) -> List[float]:
        """Encode card with 9 features: suit (5) + value (1) + special flags (3)."""
        encoding = []

        # Suit one-hot (5 dims)
        suits = ['black', 'green', 'purple', 'yellow', 'escape']
        suit_vec = [0.0] * 5
        if card.suit and card.suit.value in suits:
            suit_vec[suits.index(card.suit.value)] = 1.0
        encoding.extend(suit_vec)

        # Value (1 dim, normalized)
        encoding.append(card.value / 14.0 if card.value else 0.0)

        # Special flags (3 dims)
        encoding.extend([
            1.0 if card.is_pirate() else 0.0,
            1.0 if card.is_king() or card.is_skull_king() else 0.0,
            1.0 if card.is_mermaid() else 0.0,
        ])

        return encoding

    def _count_card_type(self, hand: List[CardId], predicate) -> int:
        """Count cards matching a predicate."""
        return sum(1 for card_id in hand if predicate(get_card(card_id)))

    def _get_info(self) -> Dict[str, Any]:
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

        player.remove_card(card_id)
        current_trick.add_card(player_id, card_id)

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
            current_round.update_scores()

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

    def render(self) -> Optional[str]:
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
