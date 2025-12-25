#!/usr/bin/env python3
"""Train MaskablePPO with V5 improvements:
1. Action masking (only sample valid actions)
2. Dense reward shaping (trick-level, bid quality)
3. Enhanced observations (171 dims with context awareness)
4. Refined curriculum with self-play phases
5. Mixed opponent evaluation for robust metrics
6. More eval episodes (20) for stable estimates
"""

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, sync_envs_normalization

from app.gym_env.skullking_env_masked import SkullKingEnvMasked


class CurriculumCallback(BaseCallback):
    """Callback to change opponent difficulty during training."""

    def __init__(self, curriculum_schedule, vec_env, verbose=0):
        """Args:
        curriculum_schedule: List of (timestep, opponent_type, difficulty)
        vec_env: Vectorized environment to update
        verbose: Verbosity level

        """
        super().__init__(verbose)
        self.curriculum_schedule = sorted(curriculum_schedule, key=lambda x: x[0])
        self.vec_env = vec_env
        self.current_phase = 0

    def _on_step(self) -> bool:
        """Check if we should advance curriculum."""
        if self.current_phase < len(self.curriculum_schedule) - 1:
            next_step, next_type, next_diff = self.curriculum_schedule[self.current_phase + 1]

            if self.num_timesteps >= next_step:
                print(f"\n{'=' * 60}")
                print(f"📈 CURRICULUM ADVANCEMENT at {self.num_timesteps:,} steps")
                print(f"Phase {self.current_phase + 2}/{len(self.curriculum_schedule)}")
                print(f"Opponent: {next_type} ({next_diff})")
                print(f"{'=' * 60}\n")

                # Update all sub-environments
                for env_idx in range(self.vec_env.num_envs):
                    self.vec_env.env_method("set_opponent", next_type, next_diff, indices=[env_idx])

                self.current_phase += 1

        return True


class MixedOpponentEvalCallback(BaseCallback):
    """Evaluate against multiple opponent types for robust metrics.

    Rotates through different opponent configurations to get a more
    comprehensive view of agent performance.
    """

    def __init__(
        self,
        eval_env,
        opponent_configs: list[tuple[str, str]],
        n_eval_episodes: int = 20,
        eval_freq: int = 50_000,
        best_model_save_path: str | None = None,
        log_path: str | None = None,
        deterministic: bool = True,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.opponent_configs = opponent_configs  # [(type, difficulty), ...]
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

    def _evaluate(self):
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
                    obs, reward, done, info = self.eval_env.step(action)
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

        print(f"Eval num_timesteps={self.num_timesteps}, episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
        print(f"Episode length: {mean_length:.2f} +/- {np.std(all_lengths):.2f}")
        print(f"(Mixed eval: {len(self.opponent_configs)} opponent types, {len(all_rewards)} total episodes)")

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

    def _activate_self_play(self):
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
        print(f"🎯 SELF-PLAY ACTIVATED at {self.num_timesteps:,} steps")
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


def mask_fn(env):
    """Wrapper to extract action masks from environment."""
    return env.action_masks()


def create_masked_env(opponent_type: str = "random", difficulty: str = "medium"):
    """Create masked environment instance with action masking wrapper."""
    env = SkullKingEnvMasked(
        num_opponents=3,
        opponent_bot_type=opponent_type,
        opponent_difficulty=difficulty,
    )
    # Wrap with ActionMasker for MaskablePPO
    return ActionMasker(env, mask_fn)


def train_masked_ppo(
    total_timesteps: int = 1_500_000,
    n_envs: int = 4,
    save_dir: str = "./models/masked_ppo",
    load_path: str | None = None,
):
    """Train MaskablePPO agent with optimized curriculum.

    Args:
        total_timesteps: Total training timesteps (default 1.5M)
        n_envs: Number of parallel environments
        save_dir: Directory to save models
        load_path: Optional path to load existing model

    """
    # Check GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    else:
        gpu_name = "N/A"
        gpu_mem = 0

    print("=" * 60)
    print("MASKABLE PPO TRAINING - Optimized Curriculum")
    print("=" * 60)
    print(
        f"Device: {device.upper()}"
        + (f" ({gpu_name}, {gpu_mem:.1f} GB)" if device == "cuda" else "")
    )
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Parallel envs: {n_envs}")
    print(f"Save directory: {save_dir}")
    print("=" * 60 + "\n")

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{save_dir}/checkpoints", exist_ok=True)

    # Define refined curriculum (based on ULTRATHINK_ANALYSIS recommendations)
    # Key changes: earlier intro to rule-based, more gradual progression
    curriculum = [
        (0, "random", "easy"),  # Phase 1: Faster basic learning (0-50k)
        (50_000, "random", "medium"),  # Phase 2: Extended core learning (50k-150k)
        (150_000, "random", "hard"),  # Phase 3: Challenge basics (150k-250k)
        (250_000, "rule_based", "easy"),  # Phase 4: Learn strategy (250k-400k)
        (400_000, "rule_based", "medium"),  # Phase 5: Refine strategy (400k-600k)
        (600_000, "rule_based", "hard"),  # Phase 6: Advanced tactics (600k-850k)
        (850_000, "rule_based", "medium"),  # Phase 7: Robustness check (850k-1.1M)
        (1_100_000, "rule_based", "hard"),  # Phase 8: Final mastery (1.1M-1.5M)
    ]

    print("Curriculum Schedule:")
    for step, opp_type, diff in curriculum:
        print(f"  {step:>10,} steps: {opp_type:12s} ({diff})")
    print()

    # Create vectorized environment with action masking
    # Start with random easy opponents (Phase 1)
    # Use DummyVecEnv for stability (SubprocVecEnv can have issues on some systems)
    print(f"Creating {n_envs} parallel masked environments...")
    vec_env = make_vec_env(
        lambda: create_masked_env("random", "easy"),
        n_envs=n_envs,
        vec_env_cls=DummyVecEnv,
    )

    # Create evaluation environment
    print("Creating evaluation environment...")
    eval_env = make_vec_env(
        lambda: create_masked_env("rule_based", "medium"),
        n_envs=1,
        vec_env_cls=DummyVecEnv,
    )

    # Curriculum callback
    curriculum_cb = CurriculumCallback(curriculum, vec_env, verbose=1)

    # Checkpoint callback (save every 100k steps)
    checkpoint_cb = CheckpointCallback(
        save_freq=100_000 // n_envs,
        save_path=f"{save_dir}/checkpoints",
        name_prefix="masked_ppo",
        save_vecnormalize=True,
    )

    # Mixed opponent evaluation callback (V5: more robust metrics)
    # Evaluates against easy, medium, and hard opponents
    opponent_configs = [
        ("rule_based", "easy"),
        ("rule_based", "medium"),
        ("rule_based", "hard"),
    ]
    eval_cb = MixedOpponentEvalCallback(
        eval_env,
        opponent_configs=opponent_configs,
        n_eval_episodes=21,  # 7 episodes per opponent type
        eval_freq=200_000 // n_envs,  # Evaluate every 200k steps (was 50k)
        best_model_save_path=f"{save_dir}/best_model",
        log_path=f"{save_dir}/eval_logs",
        deterministic=True,
    )

    # Self-play callback (V5: prevent overfitting to fixed opponents)
    # Activates after 2M steps, switches to past checkpoints periodically
    self_play_cb = SelfPlayCallback(
        checkpoint_dir=f"{save_dir}/checkpoints",
        vec_env=vec_env,
        self_play_start=2_000_000,
        self_play_freq=200_000,
    )

    # Create or load MaskablePPO model
    if load_path and os.path.exists(load_path):
        print(f"\nLoading existing model from {load_path}...")
        model = MaskablePPO.load(
            load_path,
            env=vec_env,
            tensorboard_log=f"{save_dir}/tensorboard",
        )
    else:
        print("\nCreating new MaskablePPO model...")
        print("Hyperparameters (v4 - MAXIMIZED for RTX 4080 SUPER):")
        print("  learning_rate: 3e-4")
        print("  n_steps: 4096 (large buffer for stability)")
        print("  batch_size: 1024 (maximum GPU utilization)")
        print("  n_epochs: 15 (balanced training)")
        print("  gamma: 0.995")
        print("  gae_lambda: 0.98")
        print("  ent_coef: 0.01")
        print("  vf_coef: 0.5")
        print()

        model = MaskablePPO(
            "MlpPolicy",
            vec_env,
            learning_rate=3e-4,  # Higher LR for faster learning
            n_steps=4096,  # Large buffer for stable updates
            batch_size=1024,  # Maximize GPU batch processing
            n_epochs=15,  # Slightly fewer epochs, more throughput
            gamma=0.995,
            gae_lambda=0.98,
            clip_range=0.2,  # Standard clip range
            ent_coef=0.01,  # Less exploration, more exploitation
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log=f"{save_dir}/tensorboard",
            device=device,
            policy_kwargs={"net_arch": [256, 256]},  # Larger network
        )

    print("\n🏋️ Starting training...\n")

    # Train with all callbacks (V5: added self-play)
    model.learn(
        total_timesteps=total_timesteps,
        callback=[curriculum_cb, checkpoint_cb, eval_cb, self_play_cb],
        progress_bar=True,
    )

    # Save final model
    final_path = f"{save_dir}/masked_ppo_final"
    model.save(final_path)
    vec_env.close()
    eval_env.close()

    print("\n✅ Training complete!")
    print(f"Final model saved to: {final_path}.zip")
    print(f"Best model saved to: {save_dir}/best_model/best_model.zip")
    print("\nTo view training progress:")
    print(f"  tensorboard --logdir {save_dir}/tensorboard")


def main():
    parser = argparse.ArgumentParser(description="Train MaskablePPO on Skull King")
    parser.add_argument("command", choices=["train", "resume"], help="Train new or resume existing")
    parser.add_argument("--timesteps", type=int, default=1_500_000, help="Total training timesteps")
    parser.add_argument("--envs", type=int, default=32, help="Number of parallel environments")
    parser.add_argument("--load", type=str, help="Path to load existing model")

    args = parser.parse_args()

    if args.command == "train":
        train_masked_ppo(
            total_timesteps=args.timesteps,
            n_envs=args.envs,
            save_dir="./models/masked_ppo",
            load_path=args.load,
        )
    elif args.command == "resume":
        if not args.load:
            print("Error: --load required for resume")
            sys.exit(1)
        train_masked_ppo(
            total_timesteps=args.timesteps,
            n_envs=args.envs,
            save_dir="./models/masked_ppo",
            load_path=args.load,
        )


if __name__ == "__main__":
    main()
