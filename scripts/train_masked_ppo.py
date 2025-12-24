#!/usr/bin/env python3
"""
Train MaskablePPO with critical improvements:
1. Action masking (only sample valid actions)
2. Dense reward shaping (trick-level, bid quality)
3. Enhanced observations (171 dims with context awareness)
4. Refined curriculum for faster learning
"""

import argparse
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from app.gym_env.skullking_env_masked import SkullKingEnvMasked


class CurriculumCallback(BaseCallback):
    """Callback to change opponent difficulty during training."""

    def __init__(self, curriculum_schedule, vec_env, verbose=0):
        """
        Args:
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
    """
    Train MaskablePPO agent with optimized curriculum.

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
    print(f"Creating {n_envs} parallel masked environments...")
    vec_env = make_vec_env(
        lambda: create_masked_env("random", "easy"),
        n_envs=n_envs,
        vec_env_cls=SubprocVecEnv,
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

    # Evaluation callback (evaluate every 50k steps)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=f"{save_dir}/best_model",
        log_path=f"{save_dir}/eval_logs",
        eval_freq=50_000 // n_envs,
        n_eval_episodes=5,
        deterministic=True,
        render=False,
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
        print("Hyperparameters (v3 - optimized for high-end hardware):")
        print("  learning_rate: 1e-4")
        print("  n_steps: 2048 (more frequent updates)")
        print("  batch_size: 512 (better GPU utilization)")
        print("  n_epochs: 20")
        print("  gamma: 0.995")
        print("  gae_lambda: 0.99")
        print("  ent_coef: 0.02")
        print("  vf_coef: 1.0")
        print()

        model = MaskablePPO(
            "MlpPolicy",
            vec_env,
            learning_rate=1e-4,
            n_steps=2048,  # Reduced for more frequent updates
            batch_size=512,  # Increased for better GPU utilization
            n_epochs=20,
            gamma=0.995,
            gae_lambda=0.99,
            clip_range=0.15,
            ent_coef=0.02,
            vf_coef=1.0,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log=f"{save_dir}/tensorboard",
            device=device,
        )

    print("\n🏋️ Starting training...\n")

    # Train with all callbacks
    model.learn(
        total_timesteps=total_timesteps,
        callback=[curriculum_cb, checkpoint_cb, eval_cb],
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
