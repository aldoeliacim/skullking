#!/usr/bin/env python3
"""Train MaskablePPO agent for Skull King.

V7 Training Features:
1. Action masking (only sample valid actions)
2. Dense reward shaping (trick-level, bid quality, alliance bonus)
3. Enhanced observations (190 dims with loot alliance awareness)
4. Curriculum learning with progressive difficulty
5. Mixed opponent evaluation for robust metrics
6. Self-play to prevent overfitting
7. SubprocVecEnv for multi-core parallelism (NEW)
8. Larger batch sizes for better GPU utilization (NEW)
9. torch.compile for optimized forward passes (NEW)

Usage:
    # Train with V7 max performance (128 envs, batch 4096, SubprocVecEnv, torch.compile)
    uv run python -m app.training.train_ppo train --timesteps 10000000

    # Resume training from checkpoint
    uv run python -m app.training.train_ppo resume --load models/masked_ppo/best_model/best_model.zip

    # Quick test
    uv run python -m app.training.train_ppo train --timesteps 100000 --envs 8 --no-subproc

See TRAINING_LOG.md for training history and hyperparameters.
"""

import argparse
import os
from pathlib import Path

import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from app.gym_env import SkullKingEnvMasked
from app.training.callbacks import (
    CurriculumCallback,
    MixedOpponentEvalCallback,
    SelfPlayCallback,
)

# Default training configuration (V7 max performance)
DEFAULT_TIMESTEPS = 10_000_000
DEFAULT_N_ENVS = 128  # V7: max parallelism
DEFAULT_BATCH_SIZE = 4096  # V7: max GPU utilization
DEFAULT_N_STEPS = 4096
DEFAULT_SAVE_DIR = "./models/masked_ppo"

# Curriculum schedule: (timestep, opponent_type, difficulty)
CURRICULUM = [
    (0, "random", "easy"),  # Phase 1: Basic learning
    (50_000, "random", "medium"),  # Phase 2: Core learning
    (150_000, "random", "hard"),  # Phase 3: Challenge basics
    (250_000, "rule_based", "easy"),  # Phase 4: Learn strategy
    (400_000, "rule_based", "medium"),  # Phase 5: Refine strategy
    (600_000, "rule_based", "hard"),  # Phase 6: Advanced tactics
    (850_000, "rule_based", "medium"),  # Phase 7: Robustness check
    (1_100_000, "rule_based", "hard"),  # Phase 8: Final mastery
]

# Mixed evaluation opponent configurations
EVAL_OPPONENTS = [
    ("rule_based", "easy"),
    ("rule_based", "medium"),
    ("rule_based", "hard"),
]


def mask_fn(env: SkullKingEnvMasked):
    """Extract action masks from environment for MaskablePPO."""
    return env.action_masks()


def create_masked_env(opponent_type: str = "random", difficulty: str = "medium"):
    """Create masked environment instance with action masking wrapper."""
    env = SkullKingEnvMasked(
        num_opponents=3,
        opponent_bot_type=opponent_type,
        opponent_difficulty=difficulty,
    )
    return ActionMasker(env, mask_fn)


def train(
    total_timesteps: int = DEFAULT_TIMESTEPS,
    n_envs: int = DEFAULT_N_ENVS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    n_steps: int = DEFAULT_N_STEPS,
    use_subproc: bool = True,
    use_compile: bool = False,
    save_dir: str = DEFAULT_SAVE_DIR,
    load_path: str | None = None,
) -> None:
    """Train MaskablePPO agent with curriculum learning.

    Args:
        total_timesteps: Total training timesteps
        n_envs: Number of parallel environments
        batch_size: Minibatch size for PPO updates
        n_steps: Steps per environment before update
        use_subproc: Use SubprocVecEnv for multi-core parallelism
        use_compile: Use torch.compile for optimized forward passes
        save_dir: Directory to save models and logs
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

    vec_env_type = "SubprocVecEnv" if use_subproc else "DummyVecEnv"

    print("=" * 60)
    print("SKULL KING - MaskablePPO Training (V7)")
    print("=" * 60)
    print(
        f"Device: {device.upper()}"
        + (f" ({gpu_name}, {gpu_mem:.1f} GB)" if device == "cuda" else "")
    )
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Parallel envs: {n_envs} ({vec_env_type})")
    print(f"Batch size: {batch_size}, n_steps: {n_steps}")
    print(f"torch.compile: {'enabled' if use_compile else 'disabled'}")
    print(f"Save directory: {save_dir}")
    print("=" * 60 + "\n")

    # Create directories
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    (save_path / "checkpoints").mkdir(exist_ok=True)
    (save_path / "best_model").mkdir(exist_ok=True)
    (save_path / "eval_logs").mkdir(exist_ok=True)

    # Print curriculum schedule
    print("Curriculum Schedule:")
    for step, opp_type, diff in CURRICULUM:
        print(f"  {step:>10,} steps: {opp_type:12s} ({diff})")
    print()

    # Select vectorized environment class
    vec_env_cls = SubprocVecEnv if use_subproc else DummyVecEnv

    # Create vectorized training environment
    print(f"Creating {n_envs} parallel environments ({vec_env_type})...")
    vec_env = make_vec_env(
        lambda: create_masked_env("random", "easy"),
        n_envs=n_envs,
        vec_env_cls=vec_env_cls,
    )

    # Create evaluation environment (always DummyVecEnv for simplicity)
    print("Creating evaluation environment...")
    eval_env = make_vec_env(
        lambda: create_masked_env("rule_based", "medium"),
        n_envs=1,
        vec_env_cls=DummyVecEnv,
    )

    # Setup callbacks
    curriculum_cb = CurriculumCallback(CURRICULUM, vec_env, verbose=1)

    checkpoint_cb = CheckpointCallback(
        save_freq=100_000 // n_envs,
        save_path=str(save_path / "checkpoints"),
        name_prefix="masked_ppo",
        save_vecnormalize=True,
    )

    eval_cb = MixedOpponentEvalCallback(
        eval_env,
        opponent_configs=EVAL_OPPONENTS,
        n_eval_episodes=21,  # 7 episodes per opponent type
        eval_freq=200_000 // n_envs,
        best_model_save_path=str(save_path / "best_model"),
        log_path=str(save_path / "eval_logs"),
        deterministic=True,
    )

    self_play_cb = SelfPlayCallback(
        checkpoint_dir=str(save_path / "checkpoints"),
        vec_env=vec_env,
        self_play_start=2_000_000,
        self_play_freq=200_000,
    )

    # Create or load model
    if load_path and os.path.exists(load_path):
        print(f"\nLoading existing model from {load_path}...")
        model = MaskablePPO.load(
            load_path,
            env=vec_env,
            tensorboard_log=str(save_path / "tensorboard"),
        )
    else:
        print("\nCreating new MaskablePPO model...")
        print("Hyperparameters (V7 - optimized for multi-core + GPU):")
        print("  learning_rate: 3e-4")
        print(f"  n_steps: {n_steps}")
        print(f"  batch_size: {batch_size}")
        print("  n_epochs: 15")
        print("  network: [256, 256]")
        print()

        model = MaskablePPO(
            "MlpPolicy",
            vec_env,
            learning_rate=3e-4,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=15,
            gamma=0.995,
            gae_lambda=0.98,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log=str(save_path / "tensorboard"),
            device=device,
            policy_kwargs={"net_arch": [256, 256]},
        )

    # Apply torch.compile for optimized forward passes (PyTorch 2.0+)
    if use_compile and hasattr(torch, "compile"):
        print("Applying torch.compile to policy network...")
        try:
            model.policy = torch.compile(model.policy, mode="reduce-overhead")
            print("torch.compile applied successfully!")
        except Exception as e:
            print(f"torch.compile failed (non-fatal): {e}")

    # Train
    print("\nStarting training...\n")
    model.learn(
        total_timesteps=total_timesteps,
        callback=[curriculum_cb, checkpoint_cb, eval_cb, self_play_cb],
        progress_bar=True,
    )

    # Save final model
    final_path = save_path / "masked_ppo_final"
    model.save(str(final_path))
    vec_env.close()
    eval_env.close()

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"Final model: {final_path}.zip")
    print(f"Best model: {save_path / 'best_model' / 'best_model.zip'}")
    print(f"\nView training progress:")
    print(f"  tensorboard --logdir {save_path / 'tensorboard'}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train MaskablePPO agent for Skull King",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # V7 max performance (default: 128 envs, batch 4096, SubprocVecEnv, torch.compile)
  uv run python -m app.training.train_ppo train --timesteps 10000000

  # Conservative settings (less RAM)
  uv run python -m app.training.train_ppo train --envs 64 --batch-size 2048

  # V6-style training (single-threaded, for debugging)
  uv run python -m app.training.train_ppo train --no-subproc --no-compile --envs 32 --batch-size 1024

  # Resume training
  uv run python -m app.training.train_ppo resume --load models/masked_ppo/best_model/best_model.zip

  # Quick test
  uv run python -m app.training.train_ppo train --timesteps 100000 --envs 8 --no-subproc
""",
    )
    parser.add_argument(
        "command",
        choices=["train", "resume"],
        help="train: Start new training, resume: Continue from checkpoint",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=DEFAULT_TIMESTEPS,
        help=f"Total training timesteps (default: {DEFAULT_TIMESTEPS:,})",
    )
    parser.add_argument(
        "--envs",
        type=int,
        default=DEFAULT_N_ENVS,
        help=f"Number of parallel environments (default: {DEFAULT_N_ENVS})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Minibatch size for PPO updates (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=DEFAULT_N_STEPS,
        help=f"Steps per env before update (default: {DEFAULT_N_STEPS})",
    )
    parser.add_argument(
        "--subproc",
        action="store_true",
        default=True,
        help="Use SubprocVecEnv for multi-core parallelism (default: enabled)",
    )
    parser.add_argument(
        "--no-subproc",
        action="store_true",
        help="Disable SubprocVecEnv (use DummyVecEnv instead)",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        default=True,
        help="Use torch.compile for optimized forward passes (default: enabled)",
    )
    parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable torch.compile",
    )
    parser.add_argument(
        "--load",
        type=str,
        help="Path to model checkpoint (required for resume)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=DEFAULT_SAVE_DIR,
        help=f"Directory to save models (default: {DEFAULT_SAVE_DIR})",
    )

    args = parser.parse_args()

    if args.command == "resume" and not args.load:
        parser.error("--load is required when using resume command")

    # Handle flag logic
    use_subproc = args.subproc and not args.no_subproc
    use_compile = args.compile and not args.no_compile

    train(
        total_timesteps=args.timesteps,
        n_envs=args.envs,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        use_subproc=use_subproc,
        use_compile=use_compile,
        save_dir=args.save_dir,
        load_path=args.load,
    )


if __name__ == "__main__":
    main()
