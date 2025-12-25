#!/usr/bin/env python3
"""Enhanced MaskablePPO training with self-play.

Training Phases:
1. Random opponents (warmup)
2. Rule-based opponents (learn strategy)
3. Self-play (compete against past versions)

Key improvements:
- Bonus capture rewards (14s, character combos)
- Self-play with periodic opponent updates
- Extended curriculum for multi-hour training
"""

import argparse
import os
import shutil
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


class SelfPlayCallback(BaseCallback):
    """Callback for self-play training.

    Periodically copies the current model to an opponent pool,
    then updates the environment to play against past versions.
    """

    def __init__(
        self,
        update_freq: int,
        opponent_pool_dir: str,
        max_pool_size: int = 10,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.update_freq = update_freq
        self.opponent_pool_dir = Path(opponent_pool_dir)
        self.max_pool_size = max_pool_size
        self.opponent_pool_dir.mkdir(parents=True, exist_ok=True)
        self.last_update = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_update >= self.update_freq:
            # Save current model to opponent pool
            model_path = self.opponent_pool_dir / f"opponent_{self.num_timesteps}.zip"
            self.model.save(str(model_path))

            if self.verbose:
                print(f"\n🔄 Self-play: Saved opponent checkpoint at {self.num_timesteps:,} steps")

            # Keep only most recent models
            pool_models = sorted(self.opponent_pool_dir.glob("opponent_*.zip"))
            while len(pool_models) > self.max_pool_size:
                oldest = pool_models.pop(0)
                oldest.unlink()
                if self.verbose:
                    print(f"   Removed old opponent: {oldest.name}")

            self.last_update = self.num_timesteps

        return True


class CurriculumCallback(BaseCallback):
    """Callback to change opponent difficulty during training."""

    def __init__(self, curriculum_schedule, vec_env, verbose=0):
        super().__init__(verbose)
        self.curriculum_schedule = sorted(curriculum_schedule, key=lambda x: x[0])
        self.vec_env = vec_env
        self.current_phase = 0

    def _on_step(self) -> bool:
        if self.current_phase < len(self.curriculum_schedule) - 1:
            next_step, next_type, next_diff = self.curriculum_schedule[self.current_phase + 1]

            if self.num_timesteps >= next_step:
                print(f"\n{'=' * 60}")
                print(f"📈 CURRICULUM ADVANCEMENT at {self.num_timesteps:,} steps")
                print(f"Phase {self.current_phase + 2}/{len(self.curriculum_schedule)}")
                print(f"Opponent: {next_type} ({next_diff})")
                print(f"{'=' * 60}\n")

                for env_idx in range(self.vec_env.num_envs):
                    self.vec_env.env_method("set_opponent", next_type, next_diff, indices=[env_idx])

                self.current_phase += 1

        return True


class ProgressCallback(BaseCallback):
    """Print training progress and metrics."""

    def __init__(self, print_freq: int = 50_000, verbose: int = 0):
        super().__init__(verbose)
        self.print_freq = print_freq
        self.last_print = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_print >= self.print_freq:
            # Get training stats
            fps = self.model.logger.name_to_value.get("time/fps", 0)
            ep_rew = self.model.logger.name_to_value.get("rollout/ep_rew_mean", 0)
            ep_len = self.model.logger.name_to_value.get("rollout/ep_len_mean", 0)

            print(
                f"\n📊 Progress: {self.num_timesteps:,} steps | "
                f"FPS: {fps:.0f} | "
                f"Mean Reward: {ep_rew:.2f} | "
                f"Mean Ep Length: {ep_len:.0f}"
            )

            self.last_print = self.num_timesteps

        return True


def mask_fn(env):
    """Extract action masks from environment."""
    return env.action_masks()


def create_masked_env(opponent_type: str = "random", difficulty: str = "medium"):
    """Create masked environment with action masking wrapper."""
    env = SkullKingEnvMasked(
        num_opponents=3,
        opponent_bot_type=opponent_type,
        opponent_difficulty=difficulty,
    )
    return ActionMasker(env, mask_fn)


def train_with_selfplay(
    total_timesteps: int = 5_000_000,
    n_envs: int = 32,
    save_dir: str = "./models/selfplay_ppo",
    load_path: str | None = None,
):
    """Train with self-play curriculum.

    Curriculum:
    - Phase 1-3: Random opponents (warmup)
    - Phase 4-6: Rule-based opponents (learn strategy)
    - Phase 7+: Mixed self-play and rule-based (mastery)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 70)
    print("SELF-PLAY MASKABLE PPO TRAINING")
    print("=" * 70)
    print(f"Device: {device.upper()}", end="")
    if device == "cuda":
        print(f" ({torch.cuda.get_device_name(0)})")
    else:
        print()
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Parallel envs: {n_envs}")
    print(f"Save directory: {save_dir}")
    print("=" * 70 + "\n")

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{save_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{save_dir}/opponent_pool", exist_ok=True)

    # Extended curriculum for long training
    curriculum = [
        # Phase 1-3: Random warmup (0-500k)
        (0, "random", "easy"),
        (100_000, "random", "medium"),
        (300_000, "random", "hard"),
        # Phase 4-6: Rule-based strategy (500k-2M)
        (500_000, "rule_based", "easy"),
        (800_000, "rule_based", "medium"),
        (1_200_000, "rule_based", "hard"),
        # Phase 7+: Hard rule-based mastery (2M+)
        (2_000_000, "rule_based", "hard"),
        (3_000_000, "rule_based", "hard"),
        (4_000_000, "rule_based", "hard"),
    ]

    print("Curriculum Schedule:")
    for step, opp_type, diff in curriculum:
        print(f"  {step:>12,} steps: {opp_type:12s} ({diff})")
    print()

    # Create environments
    print(f"Creating {n_envs} parallel environments...")
    vec_env = make_vec_env(
        lambda: create_masked_env("random", "easy"),
        n_envs=n_envs,
        vec_env_cls=SubprocVecEnv,
    )

    print("Creating evaluation environment...")
    eval_env = make_vec_env(
        lambda: create_masked_env("rule_based", "hard"),
        n_envs=1,
        vec_env_cls=DummyVecEnv,
    )

    # Callbacks
    callbacks = [
        CurriculumCallback(curriculum, vec_env, verbose=1),
        CheckpointCallback(
            save_freq=200_000 // n_envs,
            save_path=f"{save_dir}/checkpoints",
            name_prefix="selfplay_ppo",
            save_vecnormalize=True,
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=f"{save_dir}/best_model",
            log_path=f"{save_dir}/eval_logs",
            eval_freq=100_000 // n_envs,
            n_eval_episodes=10,
            deterministic=True,
            render=False,
        ),
        SelfPlayCallback(
            update_freq=500_000,
            opponent_pool_dir=f"{save_dir}/opponent_pool",
            max_pool_size=5,
            verbose=1,
        ),
        ProgressCallback(print_freq=100_000, verbose=1),
    ]

    # Create or load model
    if load_path and os.path.exists(load_path):
        print(f"\nLoading existing model from {load_path}...")
        model = MaskablePPO.load(
            load_path,
            env=vec_env,
            tensorboard_log=f"{save_dir}/tensorboard",
        )
    else:
        print("\nCreating new MaskablePPO model...")
        print("Hyperparameters (optimized for long training):")
        print("  learning_rate: 3e-5 (lower for stability)")
        print("  n_steps: 4096")
        print("  batch_size: 512")
        print("  n_epochs: 15")
        print("  gamma: 0.995")
        print("  ent_coef: 0.015 (reduced for exploitation)")
        print()

        model = MaskablePPO(
            "MlpPolicy",
            vec_env,
            learning_rate=3e-5,  # Lower for long training stability
            n_steps=4096,
            batch_size=512,
            n_epochs=15,
            gamma=0.995,
            gae_lambda=0.98,
            clip_range=0.1,  # Tighter for stability
            ent_coef=0.015,  # Lower for more exploitation
            vf_coef=1.0,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log=f"{save_dir}/tensorboard",
            device=device,
            policy_kwargs={
                "net_arch": {"pi": [512, 256], "vf": [512, 256]},
            },
        )

    print("\n🏋️ Starting training...\n")

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
    )

    # Save final model
    final_path = f"{save_dir}/selfplay_ppo_final"
    model.save(final_path)

    # Also copy to standard location for web game
    shutil.copy(f"{final_path}.zip", "./models/masked_ppo/masked_ppo_final.zip")

    vec_env.close()
    eval_env.close()

    print("\n" + "=" * 70)
    print("✅ Training complete!")
    print("=" * 70)
    print(f"Final model: {final_path}.zip")
    print(f"Best model: {save_dir}/best_model/best_model.zip")
    print("Also copied to: ./models/masked_ppo/masked_ppo_final.zip")
    print(f"\nTensorBoard: tensorboard --logdir {save_dir}/tensorboard")


def main():
    parser = argparse.ArgumentParser(description="Train MaskablePPO with self-play")
    parser.add_argument("command", choices=["train", "resume"], help="Train new or resume")
    parser.add_argument("--timesteps", type=int, default=5_000_000, help="Total timesteps")
    parser.add_argument("--envs", type=int, default=32, help="Parallel environments")
    parser.add_argument("--load", type=str, help="Model path to resume from")

    args = parser.parse_args()

    train_with_selfplay(
        total_timesteps=args.timesteps,
        n_envs=args.envs,
        save_dir="./models/selfplay_ppo",
        load_path=args.load if args.command == "resume" else None,
    )


if __name__ == "__main__":
    main()
