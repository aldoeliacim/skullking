#!/usr/bin/env python3
"""
Ultra-optimized PPO training with 4-phase curriculum and enhanced rewards.

Training phases:
1. Random opponents (200k) - Learn basic game rules
2. Rule-based Easy (400k) - Learn strategic concepts
3. Rule-based Medium (800k) - Refine strategy
4. Rule-based Hard (600k) - Master advanced tactics

Total: 2M timesteps
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from app.gym_env.skullking_env_enhanced import SkullKingEnvEnhanced


class CurriculumCallback(BaseCallback):
    """Callback to change opponent difficulty during training."""

    def __init__(
        self,
        curriculum_schedule: list[tuple[int, str, str]],
        vec_env,
        verbose: int = 1,
    ):
        """
        Initialize curriculum callback.

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
                print(f"\n{'='*60}")
                print(f"📈 CURRICULUM ADVANCEMENT at {self.num_timesteps:,} steps")
                print(f"Phase {self.current_phase + 2}/{len(self.curriculum_schedule)}")
                print(f"Opponent: {next_type} ({next_diff})")
                print(f"{'='*60}\n")

                # Update all sub-environments
                for env_idx in range(self.vec_env.num_envs):
                    self.vec_env.env_method(
                        "set_opponent",
                        next_type,
                        next_diff,
                        indices=[env_idx]
                    )

                self.current_phase += 1

        return True


def create_enhanced_env(opponent_type: str = "random", difficulty: str = "medium"):
    """Create enhanced environment instance."""
    return SkullKingEnvEnhanced(
        num_opponents=3,
        opponent_bot_type=opponent_type,
        opponent_difficulty=difficulty,
    )


def train_ultra_ppo(
    total_timesteps: int = 2_000_000,
    n_envs: int = 4,
    save_dir: str = "./models/ultra_ppo",
    load_path: Optional[str] = None,
):
    """
    Train PPO agent with 4-phase curriculum.

    Args:
        total_timesteps: Total training timesteps (default 2M)
        n_envs: Number of parallel environments
        save_dir: Directory to save models
        load_path: Optional path to load existing model
    """
    print("="*60)
    print("ULTRA PPO TRAINING - 4-Phase Curriculum")
    print("="*60)
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Parallel envs: {n_envs}")
    print(f"Save directory: {save_dir}")
    print("="*60 + "\n")

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(f"{save_dir}/checkpoints", exist_ok=True)

    # Define 4-phase curriculum
    curriculum = [
        (0, "random", "medium"),                  # Phase 1: Learn basics
        (200_000, "rule_based", "easy"),         # Phase 2: Learn strategy
        (600_000, "rule_based", "medium"),       # Phase 3: Refine
        (1_400_000, "rule_based", "hard"),       # Phase 4: Master
    ]

    print("Curriculum Schedule:")
    for step, opp_type, diff in curriculum:
        print(f"  {step:>10,} steps: {opp_type:12s} ({diff})")
    print()

    # Create vectorized environment
    print(f"Creating {n_envs} parallel environments...")
    vec_env = make_vec_env(
        lambda: create_enhanced_env("random", "medium"),
        n_envs=n_envs,
        vec_env_cls=SubprocVecEnv,
    )

    # Create evaluation environment
    print("Creating evaluation environment...")
    eval_env = make_vec_env(
        lambda: create_enhanced_env("rule_based", "medium"),
        n_envs=1,
        vec_env_cls=DummyVecEnv,
    )

    # Curriculum callback
    curriculum_cb = CurriculumCallback(curriculum, vec_env, verbose=1)

    # Checkpoint callback (save every 100k steps)
    checkpoint_cb = CheckpointCallback(
        save_freq=100_000 // n_envs,
        save_path=f"{save_dir}/checkpoints",
        name_prefix="ultra_ppo",
        save_vecnormalize=True,
    )

    # Evaluation callback (evaluate every 50k steps)
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=f"{save_dir}/best_model",
        log_path=f"{save_dir}/eval_logs",
        eval_freq=50_000 // n_envs,
        n_eval_episodes=5,  # Reduced from 10 for faster evaluation
        deterministic=True,
        render=False,
    )

    # Create or load PPO model
    if load_path and os.path.exists(load_path):
        print(f"\nLoading existing model from {load_path}...")
        model = PPO.load(
            load_path,
            env=vec_env,
            tensorboard_log=f"{save_dir}/tensorboard",
        )
    else:
        print("\nCreating new PPO model...")
        print("Hyperparameters:")
        print("  learning_rate: 3e-4")
        print("  n_steps: 2048")
        print("  batch_size: 64")
        print("  n_epochs: 10")
        print("  gamma: 0.99")
        print("  gae_lambda: 0.95")
        print("  ent_coef: 0.01 (entropy bonus for exploration)")
        print()

        model = PPO(
            "MlpPolicy",
            vec_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,  # Entropy coefficient for exploration
            verbose=1,
            tensorboard_log=f"{save_dir}/tensorboard",
        )

    print("\n🏋️ Starting training...\n")

    # Train with all callbacks
    model.learn(
        total_timesteps=total_timesteps,
        callback=[curriculum_cb, checkpoint_cb, eval_cb],
        progress_bar=True,
    )

    # Save final model
    final_path = f"{save_dir}/ultra_ppo_final"
    model.save(final_path)
    vec_env.close()
    eval_env.close()

    print(f"\n✅ Training complete!")
    print(f"Final model saved to: {final_path}.zip")
    print(f"Best model saved to: {save_dir}/best_model/best_model.zip")
    print(f"\nTo view training progress:")
    print(f"  tensorboard --logdir {save_dir}/tensorboard")


def evaluate_model(model_path: str, n_games: int = 50):
    """Evaluate trained model."""
    import numpy as np

    print(f"\n{'='*60}")
    print(f"Evaluating model: {model_path}")
    print(f"{'='*60}\n")

    model = PPO.load(model_path)

    for opponent_type, difficulty in [
        ("random", "medium"),
        ("rule_based", "easy"),
        ("rule_based", "medium"),
        ("rule_based", "hard"),
    ]:
        print(f"\nTesting vs {opponent_type} ({difficulty}):")
        print("-" * 40)

        env = create_enhanced_env(opponent_type, difficulty)
        rewards = []
        wins = 0
        rankings = []

        for game_num in range(n_games):
            obs, _ = env.reset()
            done = False
            total_reward = 0

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                done = terminated or truncated

            rewards.append(total_reward)

            # Get final ranking
            if env.game and env.game.players:
                agent_score = env.game.players[0].score
                opponent_scores = [p.score for p in env.game.players[1:]]
                ranking = 1 + sum(1 for s in opponent_scores if s > agent_score)
                rankings.append(ranking)

                if ranking == 1:
                    wins += 1

        env.close()

        print(f"  Games played: {n_games}")
        print(f"  Win rate: {100*wins/n_games:.1f}%")
        print(f"  Avg reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
        print(f"  Avg ranking: {np.mean(rankings):.2f}")
        print(f"  Rank distribution: 1st={sum(r==1 for r in rankings)}, "
              f"2nd={sum(r==2 for r in rankings)}, "
              f"3rd={sum(r==3 for r in rankings)}, "
              f"4th={sum(r==4 for r in rankings)}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Ultra PPO Training")

    subparsers = parser.add_subparsers(dest="command")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train agent")
    train_parser.add_argument(
        "--timesteps",
        type=int,
        default=2_000_000,
        help="Total timesteps",
    )
    train_parser.add_argument(
        "--envs",
        type=int,
        default=4,
        help="Number of parallel environments",
    )
    train_parser.add_argument(
        "--load-path",
        type=str,
        help="Path to load existing model",
    )

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate agent")
    eval_parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to model to evaluate",
    )
    eval_parser.add_argument(
        "--games",
        type=int,
        default=50,
        help="Number of games to play",
    )

    args = parser.parse_args()

    if args.command == "train":
        train_ultra_ppo(
            total_timesteps=args.timesteps,
            n_envs=args.envs,
            load_path=args.load_path,
        )
    elif args.command == "evaluate":
        evaluate_model(args.model_path, n_games=args.games)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
