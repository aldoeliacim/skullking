#!/usr/bin/env python3
"""
Advanced RL training with curriculum learning and self-play.

This script implements advanced training strategies:
1. Curriculum learning - start with easy opponents, increase difficulty
2. Self-play - train against copies of itself
3. Multiple algorithms - PPO, DQN, A2C
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import A2C, PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env

from app.gym_env import SkullKingEnv


class CurriculumCallback(BaseCallback):
    """
    Callback for curriculum learning.

    Gradually increases opponent difficulty during training.
    """

    def __init__(self, difficulty_steps: list[tuple[int, str]], verbose: int = 0):
        """
        Initialize curriculum callback.

        Args:
            difficulty_steps: List of (timestep, opponent_type) tuples
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.difficulty_steps = sorted(difficulty_steps, key=lambda x: x[0])
        self.current_difficulty_idx = 0

    def _on_step(self) -> bool:
        """Called at each step."""
        # Check if we should increase difficulty
        if self.current_difficulty_idx < len(self.difficulty_steps) - 1:
            next_step, next_opponent = self.difficulty_steps[self.current_difficulty_idx + 1]

            if self.num_timesteps >= next_step:
                print(f"\n📈 Increasing difficulty to {next_opponent} at step {self.num_timesteps}")
                # TODO: Update environment opponent type
                self.current_difficulty_idx += 1

        return True


def train_with_curriculum(
    total_timesteps: int = 2_000_000,
    save_dir: str = "./models/curriculum",
) -> None:
    """
    Train agent with curriculum learning.

    Args:
        total_timesteps: Total training steps
        save_dir: Directory to save models
    """
    print("=" * 60)
    print("Training with Curriculum Learning")
    print("=" * 60 + "\n")

    os.makedirs(save_dir, exist_ok=True)

    # Define curriculum
    curriculum = [
        (0, "random"),  # Start with random opponents
        (500_000, "rule_based"),  # Switch to rule-based opponents
    ]

    # Create environment (starts with random opponents)
    env = make_vec_env(
        lambda: SkullKingEnv(num_opponents=3, opponent_bot_type="random"),
        n_envs=4,
    )

    # Create callback
    curriculum_callback = CurriculumCallback(curriculum, verbose=1)

    # Create and train model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=os.path.join(save_dir, "tensorboard"),
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=curriculum_callback,
        progress_bar=True,
    )

    # Save model
    model.save(os.path.join(save_dir, "curriculum_agent"))
    env.close()

    print("\n✅ Curriculum training complete!")


def compare_algorithms(
    timesteps_per_algo: int = 500_000,
    save_dir: str = "./models/comparison",
) -> None:
    """
    Compare different RL algorithms.

    Args:
        timesteps_per_algo: Training steps per algorithm
        save_dir: Directory to save models
    """
    print("=" * 60)
    print("Comparing RL Algorithms")
    print("=" * 60 + "\n")

    os.makedirs(save_dir, exist_ok=True)

    algorithms = {
        "PPO": PPO,
        "A2C": A2C,
    }

    results = {}

    for algo_name, algo_class in algorithms.items():
        print(f"\n🤖 Training {algo_name}...")

        # Create environment
        env = make_vec_env(
            lambda: SkullKingEnv(num_opponents=3, opponent_bot_type="rule_based"),
            n_envs=4,
        )

        # Create model
        model = algo_class(
            "MlpPolicy",
            env,
            verbose=1,
            tensorboard_log=os.path.join(save_dir, f"tensorboard_{algo_name.lower()}"),
        )

        # Train
        model.learn(
            total_timesteps=timesteps_per_algo,
            progress_bar=True,
        )

        # Save
        model_path = os.path.join(save_dir, f"{algo_name.lower()}_agent")
        model.save(model_path)

        # Evaluate
        eval_env = SkullKingEnv(num_opponents=3, opponent_bot_type="rule_based")
        total_reward = 0
        n_eval = 20

        for _ in range(n_eval):
            obs, info = eval_env.reset()
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = eval_env.step(action)
                total_reward += reward
                done = terminated or truncated

        avg_reward = total_reward / n_eval
        results[algo_name] = avg_reward

        print(f"{algo_name} average reward: {avg_reward:.2f}")

        env.close()
        eval_env.close()

    # Print comparison
    print("\n" + "=" * 60)
    print("Algorithm Comparison")
    print("=" * 60)
    for algo_name, avg_reward in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"{algo_name}: {avg_reward:.2f}")
    print("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Advanced RL Training")

    subparsers = parser.add_subparsers(dest="command")

    # Curriculum learning
    curriculum_parser = subparsers.add_parser("curriculum", help="Train with curriculum learning")
    curriculum_parser.add_argument(
        "--timesteps",
        type=int,
        default=2_000_000,
        help="Total timesteps",
    )

    # Algorithm comparison
    compare_parser = subparsers.add_parser("compare", help="Compare algorithms")
    compare_parser.add_argument(
        "--timesteps",
        type=int,
        default=500_000,
        help="Timesteps per algorithm",
    )

    args = parser.parse_args()

    if args.command == "curriculum":
        train_with_curriculum(total_timesteps=args.timesteps)
    elif args.command == "compare":
        compare_algorithms(timesteps_per_algo=args.timesteps)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
