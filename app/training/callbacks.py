"""Training callbacks for MaskablePPO curriculum learning."""

import json
import os
import random
import time
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
    comprehensive view of agent performance. Tracks wall-clock time
    for training efficiency analysis. Supports early stopping when
    training plateaus.

    Args:
        eval_env: Evaluation environment
        opponent_configs: List of (opponent_type, difficulty) tuples
        n_eval_episodes: Total episodes across all opponents
        eval_freq: Steps between evaluations
        best_model_save_path: Path to save best model
        log_path: Path for evaluation logs
        deterministic: Use deterministic actions
        verbose: Verbosity level
        early_stopping: Enable automatic early stopping on plateau
        plateau_window: Number of recent evals to check for plateau
        plateau_threshold: Min reward improvement to not be considered plateau
        min_evals_before_stopping: Minimum evals before early stopping can trigger
        reward_per_hour_threshold: Stop if Δreward/hour drops below this

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
        # Early stopping parameters
        early_stopping: bool = False,
        plateau_window: int = 10,
        plateau_threshold: float = 2.0,
        min_evals_before_stopping: int = 20,
        reward_per_hour_threshold: float = 10.0,
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

        # Early stopping config
        self.early_stopping = early_stopping
        self.plateau_window = plateau_window
        self.plateau_threshold = plateau_threshold
        self.min_evals_before_stopping = min_evals_before_stopping
        self.reward_per_hour_threshold = reward_per_hour_threshold
        self.stop_reason: str | None = None

        # Time tracking for training efficiency
        self.start_time: float | None = None
        self.eval_history: list[dict] = []

    def _on_training_start(self) -> None:
        """Record training start time."""
        self.start_time = time.time()

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_eval_timestep >= self.eval_freq:
            self._evaluate()
            self.last_eval_timestep = self.num_timesteps

            # Check for early stopping
            if self.early_stopping and self._should_stop():
                return False

        return True

    def _should_stop(self) -> bool:
        """Check if training should stop due to plateau or diminishing returns."""
        if len(self.eval_history) < self.min_evals_before_stopping:
            return False

        # Get recent evaluations
        recent = self.eval_history[-self.plateau_window:]
        if len(recent) < self.plateau_window:
            return False

        recent_rewards = [e["mean_reward"] for e in recent]
        current_reward = recent_rewards[-1]

        # Check 1: Plateau detection (reward not improving)
        reward_range = max(recent_rewards) - min(recent_rewards)
        best_in_window = max(recent_rewards)
        improvement_from_start = current_reward - self.eval_history[0]["mean_reward"]

        if reward_range < self.plateau_threshold and improvement_from_start > 10:
            self.stop_reason = (
                f"PLATEAU DETECTED: Reward range {reward_range:.2f} < {self.plateau_threshold} "
                f"over last {self.plateau_window} evals (rewards: {min(recent_rewards):.1f}-{max(recent_rewards):.1f})"
            )
            self._print_stop_analysis(recent)
            return True

        # Check 2: Diminishing returns (reward/hour too low)
        current_record = self.eval_history[-1]
        if current_record.get("reward_per_hour", 999) < self.reward_per_hour_threshold:
            # Confirm with trend - check if last 5 are all below threshold
            last_5_rates = [e.get("reward_per_hour", 999) for e in self.eval_history[-5:]]
            if all(r < self.reward_per_hour_threshold for r in last_5_rates):
                avg_rate = np.mean(last_5_rates)
                self.stop_reason = (
                    f"DIMINISHING RETURNS: Δreward/hour {avg_rate:.1f} < {self.reward_per_hour_threshold} "
                    f"for last 5 evals"
                )
                self._print_stop_analysis(recent)
                return True

        # Check 3: No improvement from best for extended period
        evals_since_best = 0
        for i in range(len(self.eval_history) - 1, -1, -1):
            if self.eval_history[i]["mean_reward"] >= self.best_mean_reward - 0.5:
                break
            evals_since_best += 1

        if evals_since_best >= self.plateau_window * 2:
            self.stop_reason = (
                f"NO IMPROVEMENT: {evals_since_best} evals since best reward "
                f"({self.best_mean_reward:.2f})"
            )
            self._print_stop_analysis(recent)
            return True

        return False

    def _print_stop_analysis(self, recent_evals: list[dict]) -> None:
        """Print detailed analysis when stopping."""
        print("\n" + "=" * 70)
        print("🛑 EARLY STOPPING TRIGGERED")
        print("=" * 70)
        print(f"Reason: {self.stop_reason}")
        print()

        # Training summary
        elapsed = time.time() - self.start_time if self.start_time else 0
        print(f"Training duration: {elapsed/60:.1f} minutes ({elapsed/3600:.2f} hours)")
        print(f"Total timesteps: {self.num_timesteps:,}")
        print(f"Total evaluations: {len(self.eval_history)}")
        print()

        # Reward trajectory
        first = self.eval_history[0]
        last = self.eval_history[-1]
        print(f"First eval reward: {first['mean_reward']:.2f}")
        print(f"Best eval reward:  {self.best_mean_reward:.2f}")
        print(f"Final eval reward: {last['mean_reward']:.2f}")
        print(f"Total improvement: {last['mean_reward'] - first['mean_reward']:+.2f}")
        print()

        # Recent trend
        print(f"Last {len(recent_evals)} evaluations:")
        for e in recent_evals[-5:]:
            print(
                f"  {e['elapsed_minutes']:>6.1f}m: reward={e['mean_reward']:.1f}, "
                f"Δ/hour={e.get('reward_per_hour', 0):+.1f}"
            )

        print()
        print("Recommendation: Try Hierarchical RL (V9) or stronger opponents")
        print("=" * 70 + "\n")

    def _evaluate(self) -> None:
        """Run evaluation against all opponent types."""
        eval_start = time.time()
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

        eval_duration = time.time() - eval_start
        mean_reward = np.mean(all_rewards)
        std_reward = np.std(all_rewards)
        mean_length = np.mean(all_lengths)

        # Calculate wall-clock metrics
        elapsed_time = time.time() - self.start_time if self.start_time else 0
        elapsed_minutes = elapsed_time / 60
        fps = self.num_timesteps / elapsed_time if elapsed_time > 0 else 0

        # Calculate reward improvement rate
        reward_per_hour = 0.0
        if len(self.eval_history) > 0 and elapsed_time > 0:
            first_eval = self.eval_history[0]
            reward_delta = mean_reward - first_eval["mean_reward"]
            time_delta_hours = (elapsed_time - first_eval["elapsed_seconds"]) / 3600
            if time_delta_hours > 0:
                reward_per_hour = reward_delta / time_delta_hours

        # Store eval record
        eval_record = {
            "timesteps": self.num_timesteps,
            "elapsed_seconds": elapsed_time,
            "elapsed_minutes": round(elapsed_minutes, 1),
            "mean_reward": round(mean_reward, 2),
            "std_reward": round(std_reward, 2),
            "mean_length": round(mean_length, 2),
            "fps": round(fps, 0),
            "eval_duration_seconds": round(eval_duration, 1),
            "reward_per_hour": round(reward_per_hour, 2),
        }
        self.eval_history.append(eval_record)

        # Print with time info
        print(
            f"Eval num_timesteps={self.num_timesteps}, "
            f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}"
        )
        print(f"Episode length: {mean_length:.2f} +/- {np.std(all_lengths):.2f}")
        print(
            f"(Mixed eval: {len(self.opponent_configs)} opponent types, "
            f"{len(all_rewards)} total episodes)"
        )
        print(
            f"⏱  Time: {elapsed_minutes:.1f}min | "
            f"FPS: {fps:,.0f} | "
            f"Δreward/hour: {reward_per_hour:+.1f}"
        )

        # Save best model
        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            if self.best_model_save_path:
                os.makedirs(self.best_model_save_path, exist_ok=True)
                self.model.save(f"{self.best_model_save_path}/best_model")
                print("New best mean reward!")

        # Save eval history to JSON
        if self.log_path:
            os.makedirs(self.log_path, exist_ok=True)
            history_file = Path(self.log_path) / "eval_history.json"
            with open(history_file, "w") as f:
                json.dump(self.eval_history, f, indent=2)

    def _on_training_end(self) -> None:
        """Print training efficiency summary."""
        if not self.eval_history:
            return

        total_time = time.time() - self.start_time if self.start_time else 0
        total_hours = total_time / 3600

        first_eval = self.eval_history[0]
        last_eval = self.eval_history[-1]
        total_reward_gain = last_eval["mean_reward"] - first_eval["mean_reward"]

        print("\n" + "=" * 60)
        print("TRAINING EFFICIENCY SUMMARY")
        print("=" * 60)
        print(f"Total time: {total_hours:.2f} hours ({total_time/60:.1f} minutes)")
        print(f"Total timesteps: {self.num_timesteps:,}")
        print(f"Average FPS: {self.num_timesteps / total_time:,.0f}")
        print(f"First eval reward: {first_eval['mean_reward']:.2f}")
        print(f"Final eval reward: {last_eval['mean_reward']:.2f}")
        print(f"Total reward gain: {total_reward_gain:+.2f}")
        print(f"Reward gain per hour: {total_reward_gain / total_hours:+.2f}")
        print("=" * 60)


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
