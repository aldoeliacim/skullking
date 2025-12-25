#!/usr/bin/env python3
"""Train MaskablePPO V2 with comprehensive improvements:

1. Reward normalization (VecNormalize) - Reduces variance
2. Self-play training - Better generalization
3. Frame stacking - Memory/history awareness
4. Mixed opponent pool - Robustness
5. Fine-tuning phase - Polish final performance
"""

import argparse
import os
import sys
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from app.gym_env.skullking_env_masked import SkullKingEnvMasked


class FrameStackWrapper(gym.Wrapper):
    """Wrapper to stack observations for pseudo-memory."""

    def __init__(self, env, n_frames: int = 4):
        super().__init__(env)
        self.n_frames = n_frames
        self.frames = deque(maxlen=n_frames)

        # Update observation space
        orig_shape = env.observation_space.shape[0]
        self.stacked_shape = orig_shape * n_frames

        self.observation_space = gym.spaces.Box(
            low=-1, high=1, shape=(self.stacked_shape,), dtype=np.float32
        )

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Fill buffer with initial observation
        for _ in range(self.n_frames):
            self.frames.append(obs)
        return self._get_stacked_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._get_stacked_obs(), reward, terminated, truncated, info

    def _get_stacked_obs(self):
        return np.concatenate(list(self.frames)).astype(np.float32)

    def action_masks(self):
        return self.env.action_masks()


class SelfPlayCallback(BaseCallback):
    """Callback for self-play training with opponent pool."""

    def __init__(
        self,
        save_freq: int = 100_000,
        pool_size: int = 5,
        save_path: str = "./models/opponent_pool",
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.pool_size = pool_size
        self.save_path = save_path
        self.opponent_pool: list[str] = []
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.num_timesteps % self.save_freq == 0 and self.num_timesteps > 0:
            # Save current model snapshot
            snapshot_path = f"{self.save_path}/opponent_{self.num_timesteps}.zip"
            self.model.save(snapshot_path)
            self.opponent_pool.append(snapshot_path)

            # Keep only recent snapshots
            if len(self.opponent_pool) > self.pool_size:
                old_path = self.opponent_pool.pop(0)
                if os.path.exists(old_path):
                    os.remove(old_path)

            if self.verbose:
                print(f"\n📸 Saved opponent snapshot: {snapshot_path}")
                print(f"   Pool size: {len(self.opponent_pool)}")

        return True


class MixedOpponentCallback(BaseCallback):
    """Callback to randomly switch opponents during training."""

    def __init__(
        self,
        vec_env,
        switch_freq: int = 10_000,
        opponent_weights: dict | None = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.vec_env = vec_env
        self.switch_freq = switch_freq
        self.opponent_configs = [
            ("random", "easy"),
            ("random", "medium"),
            ("random", "hard"),
            ("rule_based", "easy"),
            ("rule_based", "medium"),
            ("rule_based", "hard"),
        ]
        # Default weights favor harder opponents
        self.weights = opponent_weights or [0.05, 0.10, 0.15, 0.15, 0.25, 0.30]

    def _on_step(self) -> bool:
        if self.num_timesteps % self.switch_freq == 0 and self.num_timesteps > 0:
            # Random opponent for each env
            for env_idx in range(self.vec_env.num_envs):
                choice = np.random.choice(len(self.opponent_configs), p=self.weights)
                opp_type, diff = self.opponent_configs[choice]
                self.vec_env.env_method("set_opponent", opp_type, diff, indices=[env_idx])

            if self.verbose > 1:
                print(f"\n🔀 Shuffled opponents at {self.num_timesteps:,} steps")

        return True


class ProgressCallback(BaseCallback):
    """Enhanced progress tracking with ELO-like rating."""

    def __init__(self, eval_freq: int = 50_000, verbose: int = 1):
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.ratings: list[float] = []
        self.win_rates: list[float] = []

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq == 0 and self.num_timesteps > 0:
            # Get recent episode rewards
            if len(self.model.ep_info_buffer) > 0:
                recent_rewards = [ep["r"] for ep in self.model.ep_info_buffer]
                win_rate = np.mean([r > 0 for r in recent_rewards]) * 100
                avg_reward = np.mean(recent_rewards)

                self.win_rates.append(win_rate)
                self.ratings.append(avg_reward)

                if self.verbose:
                    trend = (
                        "📈" if len(self.ratings) > 1 and avg_reward > self.ratings[-2] else "📉"
                    )
                    print(f"\n{trend} Progress @ {self.num_timesteps:,} steps:")
                    print(f"   Win rate: {win_rate:.1f}%")
                    print(f"   Avg reward: {avg_reward:.1f}")
                    if len(self.ratings) > 5:
                        print(f"   Trend (last 5): {np.mean(self.ratings[-5:]):.1f}")

        return True


def mask_fn(env):
    """Extract action masks from environment."""
    return env.action_masks()


def create_env(
    opponent_type: str = "rule_based",
    difficulty: str = "medium",
    use_frame_stack: bool = True,
    n_frames: int = 4,
):
    """Create environment with optional frame stacking."""
    env = SkullKingEnvMasked(
        num_opponents=3,
        opponent_bot_type=opponent_type,
        opponent_difficulty=difficulty,
    )
    env = ActionMasker(env, mask_fn)

    if use_frame_stack:
        env = FrameStackWrapper(env, n_frames=n_frames)

    return env


def train_v2(
    total_timesteps: int = 5_000_000,
    n_envs: int = 32,
    save_dir: str = "./models/masked_ppo_v2",
    load_path: str | None = None,
    use_frame_stack: bool = True,
    use_self_play: bool = True,
    use_mixed_opponents: bool = True,
    fine_tune_steps: int = 500_000,
):
    """Train MaskablePPO V2 with all improvements."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_info = ""
    if device == "cuda":
        gpu_info = f" ({torch.cuda.get_device_name(0)})"

    print("=" * 70)
    print("MASKABLE PPO V2 TRAINING - COMPREHENSIVE IMPROVEMENTS")
    print("=" * 70)
    print(f"Device: {device.upper()}{gpu_info}")
    print(f"Total timesteps: {total_timesteps:,} + {fine_tune_steps:,} fine-tuning")
    print(f"Parallel envs: {n_envs}")
    print(f"Frame stacking: {'4 frames' if use_frame_stack else 'disabled'}")
    print(f"Self-play: {'enabled' if use_self_play else 'disabled'}")
    print(f"Mixed opponents: {'enabled' if use_mixed_opponents else 'disabled'}")
    print("=" * 70 + "\n")

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{save_dir}/checkpoints", exist_ok=True)

    # Create vectorized environments
    print(f"Creating {n_envs} parallel environments...")
    vec_env = DummyVecEnv(
        [lambda: create_env("rule_based", "medium", use_frame_stack) for _ in range(n_envs)]
    )

    # Add reward normalization
    print("Adding VecNormalize for reward normalization...")
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.995,
    )

    # Create evaluation environment (no normalization for consistent eval)
    print("Creating evaluation environment...")
    eval_env = DummyVecEnv(
        [lambda: create_env("rule_based", "hard", use_frame_stack) for _ in range(1)]
    )
    # Use same normalization stats
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, training=False)

    # Build callbacks
    callbacks = []

    # Checkpoint callback
    callbacks.append(
        CheckpointCallback(
            save_freq=100_000 // n_envs,
            save_path=f"{save_dir}/checkpoints",
            name_prefix="mppo_v2",
            save_vecnormalize=True,
        )
    )

    # Evaluation callback
    callbacks.append(
        EvalCallback(
            eval_env,
            best_model_save_path=f"{save_dir}/best_model",
            log_path=f"{save_dir}/eval_logs",
            eval_freq=50_000 // n_envs,
            n_eval_episodes=10,
            deterministic=True,
        )
    )

    # Self-play callback
    if use_self_play:
        callbacks.append(
            SelfPlayCallback(
                save_freq=200_000,
                pool_size=5,
                save_path=f"{save_dir}/opponent_pool",
                verbose=1,
            )
        )

    # Mixed opponent callback
    if use_mixed_opponents:
        # Get the underlying env for opponent switching
        callbacks.append(
            MixedOpponentCallback(
                vec_env.venv,  # Access underlying VecEnv
                switch_freq=25_000,
                verbose=1,
            )
        )

    # Progress tracking
    callbacks.append(ProgressCallback(eval_freq=100_000, verbose=1))

    # Create or load model
    obs_dim = 171 * 4 if use_frame_stack else 171  # Frame stacked observation size

    if load_path and os.path.exists(load_path):
        print(f"\nLoading existing model from {load_path}...")
        model = MaskablePPO.load(load_path, env=vec_env)
        # Load normalization stats if available
        norm_path = load_path.replace(".zip", "_vecnormalize.pkl")
        if os.path.exists(norm_path):
            vec_env = VecNormalize.load(norm_path, vec_env.venv)
    else:
        print("\nCreating new MaskablePPO V2 model...")
        print("Hyperparameters (V2 - Optimized for stability):")
        print(f"  observation_dim: {obs_dim}")
        print("  learning_rate: 3e-4 → 1e-5 (cosine decay)")
        print("  n_steps: 4096")
        print("  batch_size: 1024")
        print("  n_epochs: 20")
        print("  gamma: 0.995")
        print("  gae_lambda: 0.98")
        print("  ent_coef: 0.02 (more exploration early)")
        print("  net_arch: [512, 256, 256] (deeper network)")
        print()

        model = MaskablePPO(
            "MlpPolicy",
            vec_env,
            learning_rate=3e-4,
            n_steps=4096,
            batch_size=1024,
            n_epochs=20,
            gamma=0.995,
            gae_lambda=0.98,
            clip_range=0.2,
            ent_coef=0.02,  # More exploration
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log=f"{save_dir}/tensorboard",
            device=device,
            policy_kwargs={
                "net_arch": {"pi": [512, 256, 256], "vf": [512, 256, 256]},
            },
        )

    # Main training phase
    print("\n🏋️ Phase 1: Main Training...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
        reset_num_timesteps=False,
    )

    # Save main training model
    main_path = f"{save_dir}/mppo_v2_main"
    model.save(main_path)
    vec_env.save(f"{main_path}_vecnormalize.pkl")
    print(f"\n✅ Main training complete! Saved to {main_path}.zip")

    # Fine-tuning phase
    if fine_tune_steps > 0:
        print(f"\n🎯 Phase 2: Fine-tuning ({fine_tune_steps:,} steps)...")
        print("  Lowering learning rate to 1e-5")
        print("  Reducing entropy to 0.005")
        print("  Training against hard opponents only")

        # Update hyperparameters for fine-tuning
        model.learning_rate = 1e-5
        model.ent_coef = 0.005

        # Set all envs to hard opponents
        for env_idx in range(vec_env.venv.num_envs):
            vec_env.venv.env_method("set_opponent", "rule_based", "hard", indices=[env_idx])

        # Fine-tune
        model.learn(
            total_timesteps=fine_tune_steps,
            callback=[callbacks[0], callbacks[1]],  # Checkpoint and eval only
            progress_bar=True,
            reset_num_timesteps=False,
        )

    # Save final model
    final_path = f"{save_dir}/mppo_v2_final"
    model.save(final_path)
    vec_env.save(f"{final_path}_vecnormalize.pkl")

    vec_env.close()
    eval_env.close()

    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)
    print(f"Final model: {final_path}.zip")
    print(f"Best model: {save_dir}/best_model/best_model.zip")
    print(f"Normalization stats: {final_path}_vecnormalize.pkl")
    print(f"\nTensorboard: tensorboard --logdir {save_dir}/tensorboard")


def main():
    parser = argparse.ArgumentParser(description="Train MaskablePPO V2")
    parser.add_argument("command", choices=["train", "resume"], help="Train new or resume")
    parser.add_argument("--timesteps", type=int, default=5_000_000, help="Main training steps")
    parser.add_argument("--fine-tune", type=int, default=500_000, help="Fine-tuning steps")
    parser.add_argument("--envs", type=int, default=32, help="Parallel environments")
    parser.add_argument("--load", type=str, help="Path to load existing model")
    parser.add_argument("--no-frame-stack", action="store_true", help="Disable frame stacking")
    parser.add_argument("--no-self-play", action="store_true", help="Disable self-play")
    parser.add_argument("--no-mixed", action="store_true", help="Disable mixed opponents")

    args = parser.parse_args()

    train_v2(
        total_timesteps=args.timesteps,
        n_envs=args.envs,
        save_dir="./models/masked_ppo_v2",
        load_path=args.load if args.command == "resume" else None,
        use_frame_stack=not args.no_frame_stack,
        use_self_play=not args.no_self_play,
        use_mixed_opponents=not args.no_mixed,
        fine_tune_steps=args.fine_tune,
    )


if __name__ == "__main__":
    main()
