"""Training callbacks for MaskablePPO curriculum learning."""

import os
import random
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import sync_envs_normalization


class CurriculumCallback(BaseCallback):
    """Callback to change opponent difficulty during training.

    Implements curriculum learning by progressively increasing opponent
    difficulty as training progresses.

    Args:
        curriculum_schedule: List of (timestep, opponent_type, difficulty)
        vec_env: Vectorized environment to update
        verbose: Verbosity level

    """

    def __init__(
        self,
        curriculum_schedule: list[tuple[int, str, str]],
        vec_env,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.curriculum_schedule = sorted(curriculum_schedule, key=lambda x: x[0])
        self.vec_env = vec_env
        self.current_phase = 0

    def _on_step(self) -> bool:
        """Check if we should advance curriculum."""
        if self.current_phase < len(self.curriculum_schedule) - 1:
            next_step, next_type, next_diff = self.curriculum_schedule[
                self.current_phase + 1
            ]

            if self.num_timesteps >= next_step:
                print(f"\n{'=' * 60}")
                print(f"CURRICULUM ADVANCEMENT at {self.num_timesteps:,} steps")
                print(f"Phase {self.current_phase + 2}/{len(self.curriculum_schedule)}")
                print(f"Opponent: {next_type} ({next_diff})")
                print(f"{'=' * 60}\n")

                # Update all sub-environments
                for env_idx in range(self.vec_env.num_envs):
                    self.vec_env.env_method(
                        "set_opponent", next_type, next_diff, indices=[env_idx]
                    )

                self.current_phase += 1

        return True


class MixedOpponentEvalCallback(BaseCallback):
    """Evaluate against multiple opponent types for robust metrics.

    Rotates through different opponent configurations to get a more
    comprehensive view of agent performance.

    Args:
        eval_env: Evaluation environment
        opponent_configs: List of (opponent_type, difficulty) tuples
        n_eval_episodes: Total episodes across all opponents
        eval_freq: Steps between evaluations
        best_model_save_path: Path to save best model
        log_path: Path for evaluation logs
        deterministic: Use deterministic actions
        verbose: Verbosity level

    """

    def __init__(
        self,
        eval_env,
        opponent_configs: list[tuple[str, str]],
        n_eval_episodes: int = 21,
        eval_freq: int = 50_000,
        best_model_save_path: str | None = None,
        log_path: str | None = None,
        deterministic: bool = True,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.opponent_configs = opponent_configs
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_model_save_path = best_model_save_path
        self.log_path = log_path
        self.deterministic = deterministic
        self.best_mean_reward = -np.inf
        self.last_eval_timestep = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_eval_timestep >= self.eval_freq:
            self._evaluate()
            self.last_eval_timestep = self.num_timesteps
        return True

    def _evaluate(self) -> None:
        """Run evaluation against all opponent types."""
        all_rewards = []
        all_lengths = []

        episodes_per_opponent = max(1, self.n_eval_episodes // len(self.opponent_configs))

        for opp_type, opp_diff in self.opponent_configs:
            # Set opponent type
            self.eval_env.env_method("set_opponent", opp_type, opp_diff)
            sync_envs_normalization(self.training_env, self.eval_env)

            rewards = []
            lengths = []

            for _ in range(episodes_per_opponent):
                obs = self.eval_env.reset()
                done = False
                episode_reward = 0.0
                episode_length = 0

                while not done:
                    action_masks = self.eval_env.env_method("action_masks")[0]
                    action, _ = self.model.predict(
                        obs,
                        deterministic=self.deterministic,
                        action_masks=action_masks,
                    )
                    obs, reward, done, _info = self.eval_env.step(action)
                    episode_reward += reward[0]
                    episode_length += 1
                    done = done[0]

                rewards.append(episode_reward)
                lengths.append(episode_length)

            all_rewards.extend(rewards)
            all_lengths.extend(lengths)

        mean_reward = np.mean(all_rewards)
        std_reward = np.std(all_rewards)
        mean_length = np.mean(all_lengths)

        print(
            f"Eval num_timesteps={self.num_timesteps}, "
            f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}"
        )
        print(f"Episode length: {mean_length:.2f} +/- {np.std(all_lengths):.2f}")
        print(
            f"(Mixed eval: {len(self.opponent_configs)} opponent types, "
            f"{len(all_rewards)} total episodes)"
        )

        # Save best model
        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            if self.best_model_save_path:
                os.makedirs(self.best_model_save_path, exist_ok=True)
                self.model.save(f"{self.best_model_save_path}/best_model")
                print("New best mean reward!")


class SelfPlayCallback(BaseCallback):
    """Periodically train against past versions of the agent.

    Loads checkpoints from training and uses them as opponents
    to prevent overfitting to fixed opponent strategies.

    Args:
        checkpoint_dir: Directory containing model checkpoints
        vec_env: Vectorized training environment
        self_play_start: Timestep to start self-play
        self_play_freq: Steps between self-play activations
        verbose: Verbosity level

    """

    def __init__(
        self,
        checkpoint_dir: str,
        vec_env,
        self_play_start: int = 2_000_000,
        self_play_freq: int = 200_000,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.vec_env = vec_env
        self.self_play_start = self_play_start
        self.self_play_freq = self_play_freq
        self.last_self_play_step = 0
        self.in_self_play = False

    def _on_step(self) -> bool:
        # Only start self-play after reaching threshold
        if self.num_timesteps < self.self_play_start:
            return True

        # Check if it's time for self-play phase
        if self.num_timesteps - self.last_self_play_step >= self.self_play_freq:
            self._activate_self_play()
            self.last_self_play_step = self.num_timesteps

        return True

    def _activate_self_play(self) -> None:
        """Load a random past checkpoint as opponent."""
        checkpoints = list(self.checkpoint_dir.glob("*.zip"))
        if not checkpoints:
            if self.verbose:
                print("[SelfPlay] No checkpoints found, skipping")
            return

        # Select random checkpoint (prefer more recent ones)
        weights = [i + 1 for i in range(len(checkpoints))]
        checkpoint = random.choices(checkpoints, weights=weights, k=1)[0]

        print(f"\n{'=' * 60}")
        print(f"SELF-PLAY ACTIVATED at {self.num_timesteps:,} steps")
        print(f"Opponent checkpoint: {checkpoint.name}")
        print(f"{'=' * 60}\n")

        # Update environments to use self-play
        for env_idx in range(self.vec_env.num_envs):
            self.vec_env.env_method(
                "set_self_play_opponent",
                str(checkpoint),
                indices=[env_idx],
            )

        self.in_self_play = True
