#!/usr/bin/env python3
"""
Train a reinforcement learning agent to play Skull King.

This script trains a PPO agent using Stable-Baselines3 on the
Skull King Gymnasium environment.
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from app.gym_env import SkullKingEnv


def create_env(opponent_type: str = "rule_based", num_opponents: int = 3):
    """Create Skull King environment."""
    return SkullKingEnv(
        num_opponents=num_opponents,
        opponent_bot_type=opponent_type,
        render_mode=None,  # No rendering during training
    )


def train_ppo_agent(
    total_timesteps: int = 1_000_000,
    n_envs: int = 4,
    opponent_type: str = "rule_based",
    save_dir: str = "./models",
    eval_freq: int = 10000,
) -> None:
    """
    Train a PPO agent.

    Args:
        total_timesteps: Total training steps
        n_envs: Number of parallel environments
        opponent_type: Type of opponent bots ("random" or "rule_based")
        save_dir: Directory to save models
        eval_freq: Frequency of evaluation
    """
    print("=" * 60)
    print("Training PPO Agent for Skull King")
    print("=" * 60)
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Parallel environments: {n_envs}")
    print(f"Opponent type: {opponent_type}")
    print(f"Save directory: {save_dir}")
    print("=" * 60 + "\n")

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Create vectorized environments
    env = make_vec_env(
        lambda: create_env(opponent_type),
        n_envs=n_envs,
        vec_env_cls=SubprocVecEnv,
    )

    # Create evaluation environment
    eval_env = make_vec_env(
        lambda: create_env(opponent_type),
        n_envs=1,
    )

    # Create callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=eval_freq,
        save_path=os.path.join(save_dir, "checkpoints"),
        name_prefix="skullking_ppo",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(save_dir, "best_model"),
        log_path=os.path.join(save_dir, "eval_logs"),
        eval_freq=eval_freq // n_envs,  # Adjust for parallel envs
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )

    # Create PPO agent
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=os.path.join(save_dir, "tensorboard"),
        # PPO hyperparameters
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
    )

    print("\n🏋️ Starting training...\n")

    # Train the agent
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True,
    )

    # Save final model
    final_model_path = os.path.join(save_dir, "skullking_ppo_final")
    model.save(final_model_path)

    print(f"\n✅ Training complete! Model saved to {final_model_path}")

    # Close environments
    env.close()
    eval_env.close()


def evaluate_agent(
    model_path: str,
    n_episodes: int = 100,
    opponent_type: str = "rule_based",
    render: bool = False,
) -> None:
    """
    Evaluate a trained agent.

    Args:
        model_path: Path to trained model
        n_episodes: Number of episodes to evaluate
        opponent_type: Type of opponent bots
        render: Whether to render games
    """
    print("=" * 60)
    print("Evaluating PPO Agent")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Episodes: {n_episodes}")
    print(f"Opponent type: {opponent_type}")
    print("=" * 60 + "\n")

    # Load model
    model = PPO.load(model_path)

    # Create environment
    env = create_env(opponent_type)
    if render:
        env.render_mode = "human"

    # Run episodes
    wins = 0
    total_reward = 0
    scores = []

    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated

            if render:
                env.render()

        total_reward += episode_reward
        final_score = info.get("agent_score", 0)
        scores.append(final_score)

        # Check if won (simplified - positive score)
        if final_score > 0:
            wins += 1

        if (episode + 1) % 10 == 0:
            print(
                f"Episode {episode + 1}/{n_episodes}: Avg Reward = {total_reward / (episode + 1):.2f}, Win Rate = {wins / (episode + 1) * 100:.1f}%"
            )

    env.close()

    # Print statistics
    import numpy as np

    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Average reward: {total_reward / n_episodes:.2f}")
    print(f"Win rate: {wins / n_episodes * 100:.1f}%")
    print(f"Average score: {np.mean(scores):.2f} ± {np.std(scores):.2f}")
    print(f"Best score: {max(scores)}")
    print(f"Worst score: {min(scores)}")
    print("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train Skull King RL Agent")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a new agent")
    train_parser.add_argument(
        "--timesteps",
        type=int,
        default=1_000_000,
        help="Total training timesteps",
    )
    train_parser.add_argument(
        "--envs",
        type=int,
        default=4,
        help="Number of parallel environments",
    )
    train_parser.add_argument(
        "--opponent",
        choices=["random", "rule_based"],
        default="rule_based",
        help="Type of opponent bots",
    )
    train_parser.add_argument(
        "--save-dir",
        type=str,
        default="./models",
        help="Directory to save models",
    )

    # Evaluate command
    eval_parser = subparsers.add_parser("eval", help="Evaluate a trained agent")
    eval_parser.add_argument(
        "model_path",
        type=str,
        help="Path to trained model",
    )
    eval_parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes",
    )
    eval_parser.add_argument(
        "--opponent",
        choices=["random", "rule_based"],
        default="rule_based",
        help="Type of opponent bots",
    )
    eval_parser.add_argument(
        "--render",
        action="store_true",
        help="Render games during evaluation",
    )

    args = parser.parse_args()

    if args.command == "train":
        train_ppo_agent(
            total_timesteps=args.timesteps,
            n_envs=args.envs,
            opponent_type=args.opponent,
            save_dir=args.save_dir,
        )
    elif args.command == "eval":
        evaluate_agent(
            model_path=args.model_path,
            n_episodes=args.episodes,
            opponent_type=args.opponent,
            render=args.render,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
