"""Gymnasium environment for Skull King card game.

This environment allows training reinforcement learning agents to play Skull King.
The agent controls one player, while other players can be controlled by bots or other agents.
"""

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from app.bots import RandomBot, RuleBasedBot
from app.bots.base_bot import BaseBot
from app.models.card import CardId, get_card
from app.models.enums import MAX_ROUNDS, GameState
from app.models.game import Game
from app.models.player import Player
from app.models.round import Round
from app.models.trick import Trick


class SkullKingEnv(gym.Env["np.ndarray", int]):
    """Gymnasium environment for Skull King.

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

    # Override as dict literal - gymnasium expects this pattern
    metadata = {"render_modes": ["human", "ansi"], "render_fps": 1}  # noqa: RUF012

    # Constants
    MIN_OPPONENTS = 1
    MAX_OPPONENTS = 6
    MAX_INVALID_MOVES = 10
    INVALID_BID_PENALTY = -1.0
    INVALID_CARD_PENALTY = -0.5
    TRUNCATION_PENALTY = -10.0
    FIRST_PLACE_BASE_REWARD = 50
    LAST_PLACE_PENALTY = -25
    MAX_HAND_SIZE = 10
    MAX_PLAYERS_COUNT = 8
    MAX_BID_OPTIONS = 11
    SCORE_NORMALIZATION = 500.0
    TRICKS_NORMALIZATION = 10.0

    def __init__(
        self,
        num_opponents: int = 3,
        opponent_bot_type: str = "rule_based",
        render_mode: str | None = None,
    ) -> None:
        """Initialize the Skull King environment.

        Args:
            num_opponents: Number of opponent players (1-6)
            opponent_bot_type: Type of bot opponents ("random" or "rule_based")
            render_mode: Rendering mode ("human" or "ansi")

        """
        super().__init__()

        if not self.MIN_OPPONENTS <= num_opponents <= self.MAX_OPPONENTS:
            msg = f"Must have {self.MIN_OPPONENTS}-{self.MAX_OPPONENTS} opponents"
            raise ValueError(msg)
        self.num_opponents = num_opponents
        self.num_players = num_opponents + 1  # +1 for the agent
        self.opponent_bot_type = opponent_bot_type
        self.render_mode = render_mode

        # Game state
        self.game: Game | None = None
        self.agent_player_id = "agent_0"
        self.bots: list[tuple[str, BaseBot]] = []
        self.rng = np.random.default_rng()

        # Observation space: vectorized game state
        # Cards: 63 physical cards; 71 one-hot indices (CardId values 1-71)
        # Player hand: 10 slots (max cards in round 10)
        # Trick cards: 8 slots (max players)
        # Bids: 8 players x 11 possible bids (0-10)
        # Scores: 8 players (normalized)
        # Metadata: round number, tricks won, etc.
        card_id_count = 71
        obs_size = (
            self.MAX_HAND_SIZE * card_id_count  # Hand (10 cards x 71 card id indices)
            + self.MAX_PLAYERS_COUNT * card_id_count  # Trick cards (8 players x 71 card id indices)
            + self.MAX_PLAYERS_COUNT * self.MAX_BID_OPTIONS  # Bids (8 players x 11 bids)
            + self.MAX_PLAYERS_COUNT  # Scores
            + self.MAX_PLAYERS_COUNT  # Tricks won this round
            + 10  # Metadata (round, phase, etc.)
        )

        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32)

        # Action space: either bid (0-10) or pick card (0-9 for max 10 cards)
        self.action_space = spaces.Discrete(self.MAX_BID_OPTIONS)

        self.invalid_move_count = 0

    @property
    def _game(self) -> Game:
        """Get the game, asserting it exists.

        Use this in methods that should only be called after reset().
        """
        assert self.game is not None, "Game not initialized. Call reset() first."
        return self.game

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment for a new game.

        Args:
            seed: Random seed
            options: Additional options (unused)

        Returns:
            Tuple of (observation, info)

        """
        del options  # Unused
        super().reset(seed=seed)

        if seed is not None:
            self.rng = np.random.default_rng(seed)

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
            bot = RandomBot(bot_id) if self.opponent_bot_type == "random" else RuleBasedBot(bot_id)
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
        """Take a step in the environment.

        Args:
            action: Action to take (bid or card index)

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)

        """
        if self.game is None:
            msg = "Must call reset() before step()"
            raise RuntimeError(msg)

        agent_player = self.game.get_player(self.agent_player_id)
        if not agent_player:
            msg = "Agent player not found"
            raise RuntimeError(msg)

        reward = self._execute_action(action, agent_player)
        terminated, truncated, final_reward = self._check_termination(reward)

        observation = self._get_observation()
        info = self._get_info()

        return observation, final_reward, terminated, truncated, info

    def _execute_action(self, action: int, agent_player: Player) -> float:
        """Execute the agent's action and return reward."""
        reward = 0.0

        if self._game.state == GameState.BIDDING:
            reward = self._handle_bidding_action(action, agent_player)
        elif self._game.state == GameState.PICKING:
            reward = self._handle_picking_action(action, agent_player)

        return reward

    def _handle_bidding_action(self, action: int, agent_player: Player) -> float:
        """Handle bidding phase action."""
        bid = action
        current_round = self._game.get_current_round()

        if current_round and 0 <= bid <= current_round.number:
            agent_player.bid = bid
            current_round.add_bid(self.agent_player_id, bid)
        else:
            self.invalid_move_count += 1
            return self.INVALID_BID_PENALTY

        self._bots_make_bids()

        if self._all_players_bid():
            self._game.state = GameState.PICKING
            self._start_new_trick()

        return 0.0

    def _handle_picking_action(self, action: int, agent_player: Player) -> float:
        """Handle card picking phase action."""
        card_index = action

        if 0 <= card_index < len(agent_player.hand):
            card_to_play = agent_player.hand[card_index]
            success = self._play_card(self.agent_player_id, card_to_play)
            if not success:
                self.invalid_move_count += 1
                return self.INVALID_CARD_PENALTY
        else:
            self.invalid_move_count += 1
            return self.INVALID_BID_PENALTY

        return 0.0

    def _check_termination(self, reward: float) -> tuple[bool, bool, float]:
        """Check if episode should terminate and calculate final reward."""
        terminated = False
        truncated = False
        final_reward = reward

        if self.invalid_move_count >= self.MAX_INVALID_MOVES:
            truncated = True
            final_reward = self.TRUNCATION_PENALTY

        if self._game.is_game_complete():
            terminated = True
            final_reward += self._calculate_final_ranking_reward()

        return terminated, truncated, final_reward

    def _calculate_final_ranking_reward(self) -> float:
        """Calculate reward based on final ranking."""
        leaderboard = self._game.get_leaderboard()
        agent_rank = next(
            i for i, p in enumerate(leaderboard) if p["player_id"] == self.agent_player_id
        )
        return (
            self.FIRST_PLACE_BASE_REWARD * (1 - agent_rank / (self.num_players - 1))
            + self.LAST_PLACE_PENALTY
        )

    def _get_observation(self) -> np.ndarray:
        """Build observation vector from current game state."""
        obs_size = 1226  # Pre-calculated observation size
        if self.game is None:
            return np.zeros((obs_size,), dtype=np.float32)

        agent_player = self.game.get_player(self.agent_player_id)
        current_round = self.game.get_current_round()

        obs = []
        obs.append(self._encode_hand(agent_player))
        obs.append(self._encode_trick_cards(current_round))
        obs.append(self._encode_bids(current_round))
        obs.append(self._encode_scores())
        obs.append(self._encode_tricks_won(current_round))
        obs.append(self._encode_metadata(current_round))

        return np.concatenate(obs).astype(np.float32)

    def _encode_hand(self, agent_player: Player | None) -> np.ndarray:
        """Encode player's hand (10 cards x 71 card types)."""
        card_id_count = 71
        hand_encoding = np.zeros((self.MAX_HAND_SIZE, card_id_count), dtype=np.float32)
        if agent_player:
            for i, card_id in enumerate(agent_player.hand[: self.MAX_HAND_SIZE]):
                hand_encoding[i, int(card_id) - 1] = 1.0
        return hand_encoding.flatten()

    def _encode_trick_cards(self, current_round: Round | None) -> np.ndarray:
        """Encode trick cards (8 players x 71 card types)."""
        card_id_count = 71
        trick_encoding = np.zeros((self.MAX_PLAYERS_COUNT, card_id_count), dtype=np.float32)
        if current_round:
            current_trick = current_round.get_current_trick()
            if current_trick:
                for i, picked_card in enumerate(
                    current_trick.picked_cards[: self.MAX_PLAYERS_COUNT]
                ):
                    trick_encoding[i, int(picked_card.card_id) - 1] = 1.0
        return trick_encoding.flatten()

    def _encode_bids(self, current_round: Round | None) -> np.ndarray:
        """Encode bids (8 players x 11 bids)."""
        bid_encoding = np.zeros((self.MAX_PLAYERS_COUNT, self.MAX_BID_OPTIONS), dtype=np.float32)
        if current_round:
            for i, player in enumerate(self._game.players[: self.MAX_PLAYERS_COUNT]):
                if player.bid is not None:
                    bid_encoding[i, player.bid] = 1.0
        return bid_encoding.flatten()

    def _encode_scores(self) -> np.ndarray:
        """Encode scores (normalized)."""
        scores = np.zeros(self.MAX_PLAYERS_COUNT, dtype=np.float32)
        for i, player in enumerate(self._game.players[: self.MAX_PLAYERS_COUNT]):
            scores[i] = player.score / self.SCORE_NORMALIZATION
        return scores

    def _encode_tricks_won(self, current_round: Round | None) -> np.ndarray:
        """Encode tricks won this round."""
        tricks_won = np.zeros(self.MAX_PLAYERS_COUNT, dtype=np.float32)
        if current_round:
            for i, player in enumerate(self._game.players[: self.MAX_PLAYERS_COUNT]):
                tricks_won[i] = current_round.get_tricks_won(player.id) / self.TRICKS_NORMALIZATION
        return tricks_won

    def _encode_metadata(self, current_round: Round | None) -> np.ndarray:
        """Encode metadata."""
        metadata = np.zeros(10, dtype=np.float32)
        metadata[0] = self._game.current_round_number / MAX_ROUNDS
        metadata[1] = 1.0 if self._game.state == GameState.BIDDING else 0.0
        metadata[2] = 1.0 if self._game.state == GameState.PICKING else 0.0
        if current_round:
            metadata[3] = len(current_round.tricks) / self.TRICKS_NORMALIZATION
        return metadata

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
        """Play a card in the current trick.

        Returns:
            True if successful, False if invalid move

        """
        if not self._validate_card_play(player_id, card_id):
            return False

        current_round = self._game.get_current_round()
        if not current_round:
            return False
        current_trick = current_round.get_current_trick()
        if not current_trick:
            return False
        player = self._game.get_player(player_id)
        if not player:
            return False

        # Play the card
        player.remove_card(card_id)
        current_trick.add_card(player_id, card_id)

        # Process trick completion or continue to next player
        if current_trick.is_complete(self.num_players):
            self._process_completed_trick(current_round, current_trick)
        else:
            self._advance_to_next_player(player)

        return True

    def _validate_card_play(self, player_id: str, card_id: CardId) -> bool:
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

    def _process_completed_trick(self, current_round: Round, current_trick: Trick) -> None:
        """Process a completed trick."""
        current_trick.determine_winner()
        if current_trick.winner_player_id:
            winner = self._game.get_player(current_trick.winner_player_id)
            if winner:
                winner.tricks_won += 1

        if current_round.is_complete():
            self._end_round()
        else:
            self._start_new_trick()

    def _advance_to_next_player(self, player: Player) -> None:
        """Advance to the next player's turn."""
        current_round = self._game.get_current_round()
        if not current_round:
            return
        current_trick = current_round.get_current_trick()
        if not current_trick:
            return

        next_index = (player.index + 1) % self.num_players
        current_trick.picking_player_id = self._game.players[next_index].id

        # If next player is bot, have them play
        if current_trick.picking_player_id != self.agent_player_id:
            self._bots_play_cards()

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

    def render(self) -> Any:
        """Render the environment."""
        if self.render_mode is None:
            return None

        if not self.game:
            return "No game in progress"

        output: list[str] = []
        self._render_header(output)
        self._render_scores(output)
        self._render_round_info(output)
        self._render_agent_hand(output)
        output.append(f"{'=' * 60}\n")

        return "\n".join(output)

    def _render_header(self, output: list[str]) -> None:
        """Render game header."""
        output.append(f"\n{'=' * 60}")
        output.append(f"Skull King - Round {self._game.current_round_number}")
        output.append(f"State: {self._game.state.value}")
        output.append(f"{'=' * 60}")

    def _render_scores(self, output: list[str]) -> None:
        """Render player scores."""
        output.append("\nScores:")
        for player in self._game.players:
            bot_str = " [BOT]" if player.is_bot else " [AGENT]"
            output.append(f"  {player.username}{bot_str}: {player.score}")

    def _render_round_info(self, output: list[str]) -> None:
        """Render current round information."""
        current_round = self._game.get_current_round()
        if not current_round:
            return

        output.append(f"\nRound {current_round.number} Bids:")
        for player in self._game.players:
            bid_str = str(player.bid) if player.bid is not None else "?"
            tricks = current_round.get_tricks_won(player.id)
            output.append(f"  {player.username}: {bid_str} (won: {tricks})")

        self._render_current_trick(output, current_round)

    def _render_current_trick(self, output: list[str], current_round: Round) -> None:
        """Render current trick."""
        current_trick = current_round.get_current_trick()
        if current_trick and current_trick.picked_cards:
            output.append("\nCurrent Trick:")
            for picked_card in current_trick.picked_cards:
                player = self._game.get_player(picked_card.player_id)
                card = get_card(picked_card.card_id)
                output.append(f"  {player.username if player else '?'}: {card}")

    def _render_agent_hand(self, output: list[str]) -> None:
        """Render agent's hand."""
        agent_player = self._game.get_player(self.agent_player_id)
        if agent_player and agent_player.hand:
            output.append("\nAgent's Hand:")
            for i, card_id in enumerate(agent_player.hand):
                card = get_card(card_id)
                output.append(f"  [{i}] {card}")

    def close(self) -> None:
        """Clean up resources."""
        self.game = None
        self.bots = []
