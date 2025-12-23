"""
Gymnasium environment for Skull King card game.

This environment allows training reinforcement learning agents to play Skull King.
The agent controls one player, while other players can be controlled by bots or other agents.
"""

from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from app.bots import RandomBot, RuleBasedBot
from app.models.card import CardId, get_card
from app.models.enums import MAX_ROUNDS, GameState
from app.models.game import Game
from app.models.player import Player
from app.models.trick import Trick


class SkullKingEnv(gym.Env):
    """
    Gymnasium environment for Skull King.

    Observation Space:
        - Player's hand (one-hot encoding of cards)
        - Current trick cards
        - Bids made by all players
        - Current scores
        - Round number
        - Tricks won so far this round
        - Game metadata

    Action Space:
        During bidding: Discrete(round_number + 1) - bid 0 to round_number
        During picking: Discrete(hand_size) - pick a card from hand

    Rewards:
        - +score_delta when round ends
        - Small penalties for invalid moves
        - Bonus for winning the game

    Episode ends when:
        - Game completes (after 10 rounds)
        - Invalid move threshold reached
    """

    metadata: ClassVar[dict[str, Any]] = {"render_modes": ["human", "ansi"], "render_fps": 1}

    def __init__(
        self,
        num_opponents: int = 3,
        opponent_bot_type: str = "rule_based",
        render_mode: str | None = None,
    ):
        """
        Initialize the Skull King environment.

        Args:
            num_opponents: Number of opponent players (1-6)
            opponent_bot_type: Type of bot opponents ("random" or "rule_based")
            render_mode: Rendering mode ("human" or "ansi")
        """
        super().__init__()

        assert 1 <= num_opponents <= 6, "Must have 1-6 opponents"
        self.num_opponents = num_opponents
        self.num_players = num_opponents + 1  # +1 for the agent
        self.opponent_bot_type = opponent_bot_type
        self.render_mode = render_mode

        # Game state
        self.game: Game | None = None
        self.agent_player_id = "agent_0"
        self.bots: list[Any] = []

        # Observation space: vectorized game state
        # Cards: 63 physical cards; 71 one-hot indices (CardId values 1-71)
        # Player hand: 10 slots (max cards in round 10)
        # Trick cards: 7 slots (max players)
        # Bids: 7 players x 11 possible bids (0-10)
        # Scores: 7 players (normalized)
        # Metadata: round number, tricks won, etc.
        obs_size = (
            10 * 71  # Hand (10 cards x 71 card id indices)
            + 7 * 71  # Trick cards (7 players x 71 card id indices)
            + 7 * 11  # Bids (7 players x 11 bids)
            + 7  # Scores
            + 7  # Tricks won this round
            + 10  # Metadata (round, phase, etc.)
        )

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32)

        # Action space: either bid (0-10) or pick card (0-9 for max 10 cards)
        self.action_space = spaces.Discrete(11)  # Max action is bid 10 or pick card at index 10

        self.invalid_move_count = 0
        self.max_invalid_moves = 10

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Reset the environment for a new game.

        Args:
            seed: Random seed
            options: Additional options

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)

        if seed is not None:
            np.random.seed(seed)

        # Create new game
        self.game = Game(id="gym_game", slug="gym_game")

        # Add agent player
        agent_player = Player(
            id=self.agent_player_id,
            username="Agent",
            game_id=self.game.id,
            index=0,
            is_bot=False,
        )
        self.game.add_player(agent_player)

        # Add bot opponents
        self.bots = []
        for i in range(self.num_opponents):
            bot_id = f"bot_{i}"
            bot_player = Player(
                id=bot_id,
                username=f"Bot_{i}",
                game_id=self.game.id,
                index=i + 1,
                is_bot=True,
            )
            self.game.add_player(bot_player)

            # Create bot controller
            if self.opponent_bot_type == "random":
                bot = RandomBot(bot_id)
            else:
                bot = RuleBasedBot(bot_id)
            self.bots.append((bot_id, bot))

        # Start game
        self.game.state = GameState.DEALING
        self.game.start_new_round()
        self.game.deal_cards()
        self.game.state = GameState.BIDDING

        self.invalid_move_count = 0

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Take a step in the environment.

        Args:
            action: Action to take (bid or card index)

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        if self.game is None:
            raise RuntimeError("Must call reset() before step()")

        reward = 0.0
        terminated = False
        truncated = False

        agent_player = self.game.get_player(self.agent_player_id)
        if not agent_player:
            raise RuntimeError("Agent player not found")

        # Handle agent action based on game state
        if self.game.state == GameState.BIDDING:
            # Action is a bid
            bid = action
            current_round = self.game.get_current_round()
            if current_round and 0 <= bid <= current_round.number:
                agent_player.bid = bid
                current_round.add_bid(self.agent_player_id, bid)
            else:
                # Invalid bid
                reward = -1.0
                self.invalid_move_count += 1

            # Let bots make their bids
            self._bots_make_bids()

            # Check if all bids are in
            if self._all_players_bid():
                self.game.state = GameState.PICKING
                self._start_new_trick()

        elif self.game.state == GameState.PICKING:
            # Action is card index
            card_index = action
            if 0 <= card_index < len(agent_player.hand):
                card_to_play = agent_player.hand[card_index]
                success = self._play_card(self.agent_player_id, card_to_play)
                if not success:
                    reward = -0.5
                    self.invalid_move_count += 1
            else:
                # Invalid card index
                reward = -1.0
                self.invalid_move_count += 1

        # Check for invalid move threshold
        if self.invalid_move_count >= self.max_invalid_moves:
            truncated = True
            reward = -10.0

        # Check if game is over
        if self.game.is_game_complete():
            terminated = True
            # Reward based on final ranking
            leaderboard = self.game.get_leaderboard()
            agent_rank = next(
                i for i, p in enumerate(leaderboard) if p["player_id"] == self.agent_player_id
            )
            # First place: +50, last place: -50
            rank_reward = 50 * (1 - agent_rank / (self.num_players - 1)) - 25
            reward += rank_reward

        observation = self._get_observation()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Build observation vector from current game state."""
        if self.game is None:
            return np.zeros(self.observation_space.shape, dtype=np.float32)

        obs = []

        agent_player = self.game.get_player(self.agent_player_id)
        current_round = self.game.get_current_round()

        # Encode player's hand (10 cards x 71 card types)
        hand_encoding = np.zeros((10, 71), dtype=np.float32)
        if agent_player:
            for i, card_id in enumerate(agent_player.hand[:10]):
                hand_encoding[i, int(card_id) - 1] = 1.0
        obs.append(hand_encoding.flatten())

        # Encode trick cards (7 players x 71 card types)
        trick_encoding = np.zeros((7, 71), dtype=np.float32)
        if current_round:
            current_trick = current_round.get_current_trick()
            if current_trick:
                for i, picked_card in enumerate(current_trick.picked_cards[:7]):
                    trick_encoding[i, int(picked_card.card_id) - 1] = 1.0
        obs.append(trick_encoding.flatten())

        # Encode bids (7 players x 11 bids)
        bid_encoding = np.zeros((7, 11), dtype=np.float32)
        if current_round:
            for i, player in enumerate(self.game.players[:7]):
                if player.bid is not None:
                    bid_encoding[i, player.bid] = 1.0
        obs.append(bid_encoding.flatten())

        # Encode scores (normalized)
        scores = np.zeros(7, dtype=np.float32)
        for i, player in enumerate(self.game.players[:7]):
            scores[i] = player.score / 500.0  # Normalize
        obs.append(scores)

        # Encode tricks won this round
        tricks_won = np.zeros(7, dtype=np.float32)
        if current_round:
            for i, player in enumerate(self.game.players[:7]):
                tricks_won[i] = current_round.get_tricks_won(player.id) / 10.0
        obs.append(tricks_won)

        # Metadata
        metadata = np.zeros(10, dtype=np.float32)
        metadata[0] = self.game.current_round_number / MAX_ROUNDS
        metadata[1] = 1.0 if self.game.state == GameState.BIDDING else 0.0
        metadata[2] = 1.0 if self.game.state == GameState.PICKING else 0.0
        if current_round:
            metadata[3] = len(current_round.tricks) / 10.0
        obs.append(metadata)

        return np.concatenate(obs).astype(np.float32)

    def _get_info(self) -> dict[str, Any]:
        """Get additional info about current state."""
        info: dict[str, Any] = {}

        if self.game:
            info["round"] = self.game.current_round_number
            info["state"] = self.game.state.value
            info["invalid_moves"] = self.invalid_move_count

            agent_player = self.game.get_player(self.agent_player_id)
            if agent_player:
                info["agent_score"] = agent_player.score
                info["agent_hand_size"] = len(agent_player.hand)

        return info

    def _bots_make_bids(self) -> None:
        """Have all bots make their bids."""
        if not self.game:
            return

        current_round = self.game.get_current_round()
        if not current_round:
            return

        for bot_id, bot in self.bots:
            player = self.game.get_player(bot_id)
            if player and not player.made_bid():
                bid = bot.make_bid(self.game, current_round.number, player.hand)
                player.bid = bid
                current_round.add_bid(bot_id, bid)

    def _bots_play_cards(self) -> None:
        """Have bots play their cards in the current trick."""
        if not self.game:
            return

        current_round = self.game.get_current_round()
        if not current_round:
            return

        current_trick = current_round.get_current_trick()
        if not current_trick:
            return

        for bot_id, bot in self.bots:
            if current_trick.picking_player_id != bot_id:
                continue

            player = self.game.get_player(bot_id)
            if not player:
                continue

            # Bot picks a card
            card_to_play = bot.pick_card(self.game, player.hand, current_trick.get_all_card_ids())
            self._play_card(bot_id, card_to_play)
            return  # Exit after one bot plays to avoid recursion

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
        """
        Play a card in the current trick.

        Returns:
            True if successful, False if invalid move
        """
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

            # If next player is bot, have them play
            if current_trick.picking_player_id != self.agent_player_id:
                self._bots_play_cards()

        return True

    def _end_round(self) -> None:
        """End the current round and start next one."""
        if not self.game:
            return

        current_round = self.game.get_current_round()
        if not current_round:
            return

        # Calculate scores
        current_round.calculate_scores()

        # Update player scores
        for player_id, score_delta in current_round.scores.items():
            player = self.game.get_player(player_id)
            if player:
                player.update_score(score_delta)

        # Check if game is complete
        if self.game.is_game_complete():
            self.game.state = GameState.ENDED
        else:
            # Start next round
            self.game.start_new_round()
            self.game.deal_cards()
            self.game.state = GameState.BIDDING

    def render(self) -> str | None:
        """Render the environment."""
        if self.render_mode is None:
            return None

        if not self.game:
            return "No game in progress"

        output = []
        output.append(f"\n{'=' * 60}")
        output.append(f"Skull King - Round {self.game.current_round_number}")
        output.append(f"State: {self.game.state.value}")
        output.append(f"{'=' * 60}")

        # Player scores
        output.append("\nScores:")
        for player in self.game.players:
            bot_str = " [BOT]" if player.is_bot else " [AGENT]"
            output.append(f"  {player.username}{bot_str}: {player.score}")

        # Current round info
        current_round = self.game.get_current_round()
        if current_round:
            output.append(f"\nRound {current_round.number} Bids:")
            for player in self.game.players:
                bid_str = str(player.bid) if player.bid is not None else "?"
                tricks = current_round.get_tricks_won(player.id)
                output.append(f"  {player.username}: {bid_str} (won: {tricks})")

            # Current trick
            current_trick = current_round.get_current_trick()
            if current_trick and current_trick.picked_cards:
                output.append("\nCurrent Trick:")
                for picked_card in current_trick.picked_cards:
                    player = self.game.get_player(picked_card.player_id)
                    card = get_card(picked_card.card_id)
                    output.append(f"  {player.username if player else '?'}: {card}")

        # Agent's hand
        agent_player = self.game.get_player(self.agent_player_id)
        if agent_player and agent_player.hand:
            output.append("\nAgent's Hand:")
            for i, card_id in enumerate(agent_player.hand):
                card = get_card(card_id)
                output.append(f"  [{i}] {card}")

        output.append(f"{'=' * 60}\n")

        result = "\n".join(output)

        if self.render_mode == "human":
            print(result)

        return result

    def close(self) -> None:
        """Clean up resources."""
        self.game = None
        self.bots = []
