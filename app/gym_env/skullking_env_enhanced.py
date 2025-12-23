"""
Enhanced Gymnasium environment for Skull King with improved reward shaping.

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
from app.models.trick import Trick


class SkullKingEnvEnhanced(gym.Env):
    """
    Enhanced Gymnasium environment for Skull King with better reward shaping.

    Improvements over base environment:
    - Bidding accuracy rewards: +10 for exact match, scaled penalties for misses
    - Trick-level strategic rewards: +2 for winning when needed, +1 for losing when ahead
    - Better bid tracking in observations
    - Progressive opponent difficulty option
    """

    metadata: ClassVar[dict[str, Any]] = {"render_modes": ["human", "ansi"], "render_fps": 1}

    def __init__(
        self,
        num_opponents: int = 3,
        opponent_bot_type: str = "rule_based",
        opponent_difficulty: str = "medium",
        render_mode: str | None = None,
    ):
        """Initialize enhanced environment."""
        super().__init__()

        self.num_players = num_opponents + 1
        self.opponent_bot_type = opponent_bot_type
        self.render_mode = render_mode
        self.max_invalid_moves = 10

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
            raise RuntimeError("Environment not initialized. Call reset() first.")

        reward = 0.0
        terminated = False
        truncated = False

        agent_player = self.game.get_player(self.agent_player_id)
        if not agent_player:
            raise RuntimeError("Agent player not found")

        # Execute action based on game state
        if self.game.state == GameState.BIDDING:
            # Action is bid amount
            current_round = self.game.get_current_round()
            bid = min(action, current_round.number)
            agent_player.bid = bid
            current_round.add_bid(self.agent_player_id, bid)

            # Have bots make their bids
            for bot_id, bot in self.bots:
                bot_player = self.game.get_player(bot_id)
                if bot_player and bot_player.bid is None:
                    bot_bid = bot.make_bid(
                        self.game,
                        current_round.number,
                        bot_player.hand,
                    )
                    bot_player.bid = bot_bid
                    current_round.add_bid(bot_id, bot_bid)

            # Transition to picking after all bids
            if self._all_players_bid():
                self.game.state = GameState.PICKING
                self._start_new_trick()

                # Have bots play if agent isn't first
                self._bots_play_cards()

        elif self.game.state == GameState.PICKING:
            # Action is card index
            card_index = action
            if 0 <= card_index < len(agent_player.hand):
                card_to_play = agent_player.hand[card_index]

                # Track state before playing
                current_round = self.game.get_current_round()
                tricks_won_before = current_round.get_tricks_won(self.agent_player_id)

                success = self._play_card(self.agent_player_id, card_to_play)

                if not success:
                    reward = -0.5
                    self.invalid_move_count += 1
                else:
                    # ENHANCED: Trick-level strategic reward
                    current_trick = current_round.get_current_trick()
                    if current_trick and current_trick.is_complete(self.num_players):
                        tricks_won_after = current_round.get_tricks_won(self.agent_player_id)
                        won_trick = tricks_won_after > tricks_won_before

                        bid = agent_player.bid if agent_player.bid is not None else 0
                        tricks_needed = bid - tricks_won_before
                        tricks_remaining = current_round.number - len(current_round.tricks) + 1

                        # Reward for strategic play
                        if tricks_needed > 0 and won_trick:
                            # Needed to win and did win
                            reward += 2.0
                        elif tricks_needed == 0 and not won_trick:
                            # Didn't need to win and didn't win
                            reward += 1.0
                        elif tricks_needed > tricks_remaining and not won_trick:
                            # Can't make bid anyway, good to avoid winning
                            reward += 0.5
                        elif tricks_needed == 0 and won_trick:
                            # Overbidding penalty
                            reward -= 1.5

            else:
                # Invalid card index
                reward = -1.0
                self.invalid_move_count += 1

        # Check for invalid move threshold
        if self.invalid_move_count >= self.max_invalid_moves:
            truncated = True
            reward = -10.0

        # Check if round ended (for bidding accuracy reward)
        current_round = self.game.get_current_round()
        if current_round and current_round.is_complete():
            # ENHANCED: Bidding accuracy reward
            bid = agent_player.bid if agent_player.bid is not None else 0
            tricks_won = current_round.get_tricks_won(self.agent_player_id)
            bid_accuracy = abs(bid - tricks_won)

            if bid_accuracy == 0:
                # Perfect bid!
                reward += 10.0
            elif bid_accuracy == 1:
                # Close
                reward += 3.0
            elif bid_accuracy == 2:
                reward -= 2.0
            else:
                # Way off
                reward -= 5.0 * bid_accuracy

            # Update round score
            current_round.update_scores()

        # Check if game is over
        if self.game.is_game_complete():
            terminated = True

            # ENHANCED: Better final ranking reward
            leaderboard = self.game.get_leaderboard()
            agent_rank = next(
                i for i, p in enumerate(leaderboard) if p["player_id"] == self.agent_player_id
            )

            # Scaled ranking reward: 1st=+40, 2nd=+10, 3rd=-10, 4th=-30
            rank_rewards = [40, 10, -10, -30]
            if agent_rank < len(rank_rewards):
                reward += rank_rewards[agent_rank]

            # Bonus for winning
            if agent_rank == 0:
                reward += 20

        observation = self._get_observation()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
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

    def _bots_play_cards(self) -> None:
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

    def set_opponent(self, opponent_type: str, difficulty: str = "medium"):
        """
        Change opponent type and difficulty (for curriculum learning).

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
