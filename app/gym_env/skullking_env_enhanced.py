"""Enhanced Gymnasium environment for Skull King with improved reward shaping.

Key improvements:
1. Reward for bidding accuracy (matching bid exactly)
2. Trick-level rewards for strategic play
3. Penalties for overbidding/underbidding
4. Better observation encoding for bid tracking
"""

from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from app.bots import RandomBot, RuleBasedBot
from app.bots.base_bot import BotDifficulty
from app.models.card import CardId
from app.models.enums import MAX_PLAYERS, MAX_ROUNDS, GameState
from app.models.game import Game
from app.models.player import Player
from app.models.round import Round
from app.models.trick import Trick


class SkullKingEnvEnhanced(gym.Env):
    """Enhanced Gymnasium environment for Skull King with better reward shaping.

    Improvements over base environment:
    - Bidding accuracy rewards: +10 for exact match, scaled penalties for misses
    - Trick-level strategic rewards: +2 for winning when needed, +1 for losing when ahead
    - Better bid tracking in observations
    - Progressive opponent difficulty option
    """

    metadata: ClassVar[dict[str, Any]] = {"render_modes": ["human", "ansi"], "render_fps": 1}

    # Constants
    MAX_INVALID_MOVES = 10
    PERFECT_BID_REWARD = 10.0
    CLOSE_BID_REWARD = 3.0
    BID_ERROR_2_PENALTY = -2.0
    TRICK_WIN_NEEDED_REWARD = 2.0
    TRICK_LOSE_NEEDED_REWARD = 1.0
    TRICK_CANT_MAKE_BID_REWARD = 0.5
    TRICK_OVERBID_PENALTY = -1.5
    INVALID_CARD_PENALTY = -0.5
    INVALID_MOVE_PENALTY = -1.0
    TRUNCATION_PENALTY = -10.0
    FIRST_RANK_REWARD = 40
    SECOND_RANK_REWARD = 10
    THIRD_RANK_PENALTY = -10
    FOURTH_RANK_PENALTY = -30
    WINNER_BONUS = 20
    BID_ACCURACY_1 = 1
    BID_ACCURACY_2 = 2

    def __init__(
        self,
        num_opponents: int = 3,
        opponent_bot_type: str = "rule_based",
        opponent_difficulty: str = "medium",
        render_mode: str | None = None,
    ) -> None:
        """Initialize enhanced environment."""
        super().__init__()

        self.num_players = num_opponents + 1
        self.opponent_bot_type = opponent_bot_type
        self.render_mode = render_mode

        # Convert difficulty string to enum
        difficulty_map = {
            "easy": BotDifficulty.EASY,
            "medium": BotDifficulty.MEDIUM,
            "hard": BotDifficulty.HARD,
        }
        self.opponent_difficulty = difficulty_map.get(opponent_difficulty, BotDifficulty.MEDIUM)

        # Observation space: hand + trick + bids + scores + bid tracking
        # Enhanced with explicit bid tracking features
        hand_size = 10 * 71  # 10 cards x 71 card types (one-hot)
        trick_size = MAX_PLAYERS * 71  # Cards in current trick
        bid_size = MAX_PLAYERS  # All players' bids
        score_size = MAX_PLAYERS  # All players' scores
        metadata_size = 5  # round, tricks_won, tricks_remaining, tricks_needed, bid_accuracy

        obs_size = hand_size + trick_size + bid_size + score_size + metadata_size
        self.observation_space = spaces.Box(low=-100, high=100, shape=(obs_size,), dtype=np.float32)

        # Action space: bid (0-10) or pick card (0-9)
        self.action_space = spaces.Discrete(11)  # Max 10 cards + 1 for bidding

        # Game state
        self.game: Game | None = None
        self.agent_player_id: str = ""
        self.bots: list[tuple[str, Any]] = []
        self.invalid_move_count = 0

        # Enhanced tracking
        self.previous_score = 0
        self.previous_tricks_won = 0

    def reset(
        self, seed: int | None = None, _options: dict[str, Any] | None = None
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
        self.game.state = GameState.BIDDING  # CRITICAL FIX: Set state to BIDDING
        self.invalid_move_count = 0
        self.previous_score = 0
        self.previous_tricks_won = 0

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute action and return result with enhanced rewards."""
        if self.game is None:
            msg = "Environment not initialized. Call reset() first."
            raise RuntimeError(msg)

        agent_player = self.game.get_player(self.agent_player_id)
        if not agent_player:
            msg = "Agent player not found"
            raise RuntimeError(msg)

        reward = self._execute_enhanced_action(action, agent_player)
        terminated, truncated, final_reward = self._check_enhanced_termination(reward, agent_player)

        observation = self._get_observation()
        info = self._get_info()

        return observation, final_reward, terminated, truncated, info

    def _execute_enhanced_action(self, action: int, agent_player: Player) -> float:
        """Execute action and return enhanced reward."""
        if self.game.state == GameState.BIDDING:
            return self._handle_enhanced_bidding(action, agent_player)
        if self.game.state == GameState.PICKING:
            return self._handle_enhanced_picking(action, agent_player)
        return 0.0

    def _handle_enhanced_bidding(self, action: int, agent_player: Player) -> float:
        """Handle bidding phase with enhanced rewards."""
        current_round = self.game.get_current_round()
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

        return 0.0

    def _handle_enhanced_picking(self, action: int, agent_player: Player) -> float:
        """Handle card picking phase with enhanced rewards."""
        card_index = action
        if not 0 <= card_index < len(agent_player.hand):
            self.invalid_move_count += 1
            return self.INVALID_MOVE_PENALTY

        card_to_play = agent_player.hand[card_index]
        current_round = self.game.get_current_round()
        tricks_won_before = current_round.get_tricks_won(self.agent_player_id)

        success = self._play_card(self.agent_player_id, card_to_play)
        if not success:
            self.invalid_move_count += 1
            return self.INVALID_CARD_PENALTY

        return self._calculate_trick_reward(agent_player, current_round, tricks_won_before)

    def _calculate_trick_reward(
        self, agent_player: Player, current_round: Round, tricks_won_before: int
    ) -> float:
        """Calculate reward for trick completion."""
        current_trick = current_round.get_current_trick()
        if not current_trick or not current_trick.is_complete(self.num_players):
            return 0.0

        tricks_won_after = current_round.get_tricks_won(self.agent_player_id)
        won_trick = tricks_won_after > tricks_won_before

        bid = agent_player.bid if agent_player.bid is not None else 0
        tricks_needed = bid - tricks_won_before
        tricks_remaining = current_round.number - len(current_round.tricks) + 1

        # Reward for strategic play
        if tricks_needed > 0 and won_trick:
            return self.TRICK_WIN_NEEDED_REWARD
        if tricks_needed == 0 and not won_trick:
            return self.TRICK_LOSE_NEEDED_REWARD
        if tricks_needed > tricks_remaining and not won_trick:
            return self.TRICK_CANT_MAKE_BID_REWARD
        if tricks_needed == 0 and won_trick:
            return self.TRICK_OVERBID_PENALTY

        return 0.0

    def _check_enhanced_termination(
        self, reward: float, agent_player: Player
    ) -> tuple[bool, bool, float]:
        """Check termination with enhanced rewards."""
        terminated = False
        truncated = False
        final_reward = reward

        if self.invalid_move_count >= self.MAX_INVALID_MOVES:
            truncated = True
            final_reward = self.TRUNCATION_PENALTY

        # Check if round ended (for bidding accuracy reward)
        current_round = self.game.get_current_round()
        if current_round and current_round.is_complete():
            final_reward += self._calculate_bidding_accuracy_reward(agent_player, current_round)
            current_round.update_scores()

        # Check if game is over
        if self.game.is_game_complete():
            terminated = True
            final_reward += self._calculate_game_ranking_reward()

        return terminated, truncated, final_reward

    def _calculate_bidding_accuracy_reward(
        self, agent_player: Player, current_round: Round
    ) -> float:
        """Calculate reward based on bidding accuracy."""
        bid = agent_player.bid if agent_player.bid is not None else 0
        tricks_won = current_round.get_tricks_won(self.agent_player_id)
        bid_accuracy = abs(bid - tricks_won)

        if bid_accuracy == 0:
            return self.PERFECT_BID_REWARD
        if bid_accuracy == self.BID_ACCURACY_1:
            return self.CLOSE_BID_REWARD
        if bid_accuracy == self.BID_ACCURACY_2:
            return self.BID_ERROR_2_PENALTY
        return -5.0 * bid_accuracy

    def _calculate_game_ranking_reward(self) -> float:
        """Calculate final ranking reward."""
        leaderboard = self.game.get_leaderboard()
        agent_rank = next(
            i for i, p in enumerate(leaderboard) if p["player_id"] == self.agent_player_id
        )

        rank_rewards = [
            self.FIRST_RANK_REWARD,
            self.SECOND_RANK_REWARD,
            self.THIRD_RANK_PENALTY,
            self.FOURTH_RANK_PENALTY,
        ]
        reward = (
            rank_rewards[agent_rank] if agent_rank < len(rank_rewards) else self.FOURTH_RANK_PENALTY
        )

        if agent_rank == 0:
            reward += self.WINNER_BONUS

        return reward

    def _get_observation(self) -> np.ndarray:  # noqa: C901
        """Build enhanced observation with bid tracking."""
        if self.game is None:
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        obs = []

        agent_player = self.game.get_player(self.agent_player_id)
        current_round = self.game.get_current_round()

        # Hand encoding (10 cards x 71 card types)
        hand_encoding = np.zeros((10, 71), dtype=np.float32)
        if agent_player:
            for i, card_id in enumerate(agent_player.hand[:10]):
                hand_encoding[i, card_id - 1] = 1.0
        obs.extend(hand_encoding.flatten())

        # Trick encoding (7 players x 71 card types)
        trick_encoding = np.zeros((MAX_PLAYERS, 71), dtype=np.float32)
        if current_round:
            current_trick = current_round.get_current_trick()
            if current_trick:
                for i, card_id in enumerate(current_trick.get_all_card_ids()[:MAX_PLAYERS]):
                    trick_encoding[i, card_id - 1] = 1.0
        obs.extend(trick_encoding.flatten())

        # Bids (normalized to 0-1)
        bids = np.zeros(MAX_PLAYERS, dtype=np.float32)
        for i, player in enumerate(self.game.players[:MAX_PLAYERS]):
            if player.bid is not None:
                bids[i] = player.bid / 10.0
        obs.extend(bids)

        # Player scores normalized to 0-1 range
        scores = np.zeros(MAX_PLAYERS, dtype=np.float32)
        for i, player in enumerate(self.game.players[:MAX_PLAYERS]):
            scores[i] = player.score / 100.0
        obs.extend(scores)

        # ENHANCED: Metadata with bid tracking
        metadata = np.zeros(5, dtype=np.float32)
        if current_round:
            metadata[0] = current_round.number / 10.0  # Round number

            if agent_player and agent_player.bid is not None:
                tricks_won = current_round.get_tricks_won(self.agent_player_id)
                tricks_remaining = current_round.number - len(current_round.tricks)
                tricks_needed = agent_player.bid - tricks_won

                metadata[1] = tricks_won / 10.0  # Tricks won
                metadata[2] = tricks_remaining / 10.0  # Tricks remaining
                metadata[3] = tricks_needed / 10.0  # Tricks needed (can be negative)
                metadata[4] = 1.0 if tricks_needed <= tricks_remaining else 0.0  # Can make bid

        obs.extend(metadata)

        return np.array(obs, dtype=np.float32)

    def _get_info(self) -> dict[str, Any]:
        """Get additional information."""
        info = {}
        if self.game:
            info["game_state"] = self.game.state.name
            info["round_number"] = (
                self.game.get_current_round().number if self.game.get_current_round() else 0
            )
            agent_player = self.game.get_player(self.agent_player_id)
            if agent_player:
                info["agent_score"] = agent_player.score
        return info

    # Include all the helper methods from the original environment
    def _all_players_bid(self) -> bool:
        """Check if all players have made their bids."""
        if not self.game:
            return False
        return all(p.made_bid() for p in self.game.players)

    def _start_new_trick(self) -> None:
        """Start a new trick."""
        if not self.game:
            return

        current_round = self.game.get_current_round()
        if not current_round:
            return

        trick_number = len(current_round.tricks) + 1
        starter_index = current_round.starter_player_index

        # If not first trick, winner of last trick starts
        if current_round.tricks:
            last_trick = current_round.tricks[-1]
            if last_trick.winner_player_id:
                winner = self.game.get_player(last_trick.winner_player_id)
                if winner:
                    starter_index = winner.index

        trick = Trick(
            number=trick_number,
            starter_player_index=starter_index,
        )
        trick.picking_player_id = self.game.players[starter_index].id
        current_round.tricks.append(trick)

    def _play_card(self, player_id: str, card_id: CardId) -> bool:
        """Play a card in the current trick."""
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

        # Play the card
        player.remove_card(card_id)
        current_trick.add_card(player_id, card_id)

        # Check if trick is complete
        if current_trick.is_complete(self.num_players):
            # Determine winner
            current_trick.determine_winner()
            if current_trick.winner_player_id:
                winner = self.game.get_player(current_trick.winner_player_id)
                if winner:
                    winner.tricks_won += 1

            # Check if round is complete
            if current_round.is_complete():
                self._end_round()
            else:
                self._start_new_trick()
        else:
            # Next player's turn
            next_index = (player.index + 1) % self.num_players
            current_trick.picking_player_id = self.game.players[next_index].id

        return True

    def _end_round(self) -> None:
        """End the current round and start next one."""
        if not self.game:
            return

        current_round = self.game.get_current_round()
        if current_round:
            current_round.update_scores()

        # Start next round or end game
        if len(self.game.rounds) < MAX_ROUNDS:
            self.game.start_new_round()
            self.game.state = GameState.BIDDING
        else:
            self.game.state = GameState.ENDED

    def _bots_play_cards(self) -> None:  # noqa: C901
        """Have bots play their cards until it's the agent's turn (iterative, no recursion)."""
        if not self.game:
            return

        # Safety limit to prevent infinite loops
        max_iterations = 100
        iterations = 0

        while iterations < max_iterations:
            iterations += 1

            # Check game state
            if self.game.state == GameState.ENDED:
                break

            current_round = self.game.get_current_round()
            if not current_round:
                break

            # If we're in bidding state, stop (agent needs to bid)
            if self.game.state == GameState.BIDDING:
                break

            current_trick = current_round.get_current_trick()
            if not current_trick:
                break

            # Check whose turn it is
            picking_player_id = current_trick.picking_player_id

            # If it's the agent's turn, stop
            if picking_player_id == self.agent_player_id:
                break

            # Find the bot whose turn it is
            bot_found = False
            for bot_id, bot in self.bots:
                if picking_player_id == bot_id:
                    player = self.game.get_player(bot_id)
                    if not player:
                        break

                    # Bot picks and plays a card
                    card_to_play = bot.pick_card(
                        self.game, player.hand, current_trick.get_all_card_ids()
                    )
                    self._play_card(bot_id, card_to_play)
                    bot_found = True
                    break

            # If no bot was found for current player, something's wrong
            # Break to avoid infinite loop
            if not bot_found:
                break

    def render(self) -> str | None:
        """Render the environment."""
        if self.render_mode == "ansi":
            return self._render_ansi()
        return None

    def set_opponent(self, opponent_type: str, difficulty: str = "medium") -> None:
        """Change opponent type and difficulty (for curriculum learning).

        Args:
            opponent_type: "random" or "rule_based"
            difficulty: "easy", "medium", or "hard"

        """
        self.opponent_bot_type = opponent_type

        difficulty_map = {
            "easy": BotDifficulty.EASY,
            "medium": BotDifficulty.MEDIUM,
            "hard": BotDifficulty.HARD,
        }
        self.opponent_difficulty = difficulty_map.get(difficulty, BotDifficulty.MEDIUM)

        # Note: This will take effect on next reset()
        # Current bots will continue with their current settings

    def _render_ansi(self) -> str:
        """Render game state as ANSI string."""
        if not self.game:
            return "Game not started"

        lines = []
        lines.append(f"Game State: {self.game.state.name}")

        if current_round := self.game.get_current_round():
            lines.append(f"Round: {current_round.number}")

        for player in self.game.players:
            is_agent = "(Agent)" if player.id == self.agent_player_id else ""
            lines.append(
                f"{player.name} {is_agent}: Score={player.score}, "
                f"Bid={player.bid}, Tricks={player.tricks_won}"
            )

        return "\n".join(lines)
