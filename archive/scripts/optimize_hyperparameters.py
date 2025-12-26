#!/usr/bin/env python3
"""Hyperparameter optimization for PPO training.

Uses a grid search or random search to find optimal hyperparameters.
Tests different combinations of:
- Learning rate
- Entropy coefficient
- Batch size
- Number of epochs
- GAE lambda
"""

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.gym_env.skullking_env_enhanced import SkullKingEnvEnhanced
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env


def create_env(opponent_type="random", difficulty="medium"):
    """Create environment for training."""
    return SkullKingEnvEnhanced(
        num_opponents=3,
        opponent_bot_type=opponent_type,
        opponent_difficulty=difficulty,
    )


def quick_train_and_evaluate(
    hyperparams: dict,
    train_steps: int = 50_000,
    eval_games: int = 20,
) -> float:
    """Quickly train a model with given hyperparameters and evaluate it.

    Returns average reward as the optimization metric.
    """
    print(f"\nTesting hyperparameters: {hyperparams}")

    # Create training environment
    env = make_vec_env(lambda: create_env("random", "medium"), n_envs=2)

    # Create model with test hyperparameters
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=hyperparams["learning_rate"],
        n_steps=hyperparams["n_steps"],
        batch_size=hyperparams["batch_size"],
        n_epochs=hyperparams["n_epochs"],
        gamma=hyperparams["gamma"],
        gae_lambda=hyperparams["gae_lambda"],
        ent_coef=hyperparams["ent_coef"],
        verbose=0,
    )

    # Train
    model.learn(total_timesteps=train_steps, progress_bar=False)
    env.close()

    # Evaluate
    eval_env = create_env("rule_based", "medium")
    total_reward = 0
    wins = 0

    for _ in range(eval_games):
        obs, _ = eval_env.reset()
        done = False
        episode_reward = 0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            episode_reward += reward
            done = terminated or truncated

        total_reward += episode_reward

        # Check if won
        if eval_env.game and eval_env.game.players:
            agent_score = eval_env.game.players[0].score
            opponent_scores = [p.score for p in eval_env.game.players[1:]]
            if agent_score == max([agent_score, *opponent_scores]):
                wins += 1

    eval_env.close()

    avg_reward = total_reward / eval_games
    win_rate = wins / eval_games

    print(f"  Result: Avg reward={avg_reward:.2f}, Win rate={100 * win_rate:.0f}%")

    return avg_reward


def grid_search_hyperparameters(
    param_grid: dict[str, list],
    train_steps: int = 50_000,
    eval_games: int = 20,
):
    """Perform grid search over hyperparameter space.

    Args:
        param_grid: Dictionary mapping parameter names to lists of values
        train_steps: Training steps per configuration
        eval_games: Evaluation games per configuration

    """
    print("=" * 70)
    print("HYPERPARAMETER GRID SEARCH")
    print("=" * 70)

    # Generate all combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(itertools.product(*param_values))

    print(f"\nTotal combinations to test: {len(combinations)}")
    print(f"Estimated time: ~{len(combinations) * train_steps / 50000 * 2:.0f} minutes")

    results = []

    for i, values in enumerate(combinations, 1):
        hyperparams = dict(zip(param_names, values, strict=False))

        print(f"\n[{i}/{len(combinations)}]", end=" ")
        avg_reward = quick_train_and_evaluate(hyperparams, train_steps, eval_games)

        results.append(
            {
                "hyperparams": hyperparams,
                "avg_reward": avg_reward,
            }
        )

    # Sort by performance
    results.sort(key=lambda x: x["avg_reward"], reverse=True)

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS (sorted by performance)")
    print("=" * 70)

    for i, result in enumerate(results[:5], 1):
        print(f"\n{i}. Avg Reward: {result['avg_reward']:.2f}")
        print("   Hyperparameters:")
        for key, value in result["hyperparams"].items():
            print(f"     {key}: {value}")

    # Save all results
    output_path = "./models/hyperparameter_search.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Full results saved to: {output_path}")

    return results


def random_search_hyperparameters(
    n_trials: int = 20,
    train_steps: int = 50_000,
    eval_games: int = 20,
):
    """Perform random search over hyperparameter space.

    Samples random combinations from continuous ranges.
    """
    print("=" * 70)
    print(f"HYPERPARAMETER RANDOM SEARCH ({n_trials} trials)")
    print("=" * 70)

    results = []

    for trial in range(1, n_trials + 1):
        # Sample random hyperparameters
        hyperparams = {
            "learning_rate": 10 ** np.random.uniform(-5, -3),  # 1e-5 to 1e-3
            "n_steps": np.random.choice([512, 1024, 2048, 4096]),
            "batch_size": np.random.choice([32, 64, 128, 256]),
            "n_epochs": np.random.choice([5, 10, 15, 20]),
            "gamma": np.random.uniform(0.95, 0.995),
            "gae_lambda": np.random.uniform(0.9, 0.99),
            "ent_coef": 10 ** np.random.uniform(-3, -1),  # 0.001 to 0.1
        }

        print(f"\n[Trial {trial}/{n_trials}]")
        avg_reward = quick_train_and_evaluate(hyperparams, train_steps, eval_games)

        results.append(
            {
                "hyperparams": hyperparams,
                "avg_reward": avg_reward,
            }
        )

    # Sort by performance
    results.sort(key=lambda x: x["avg_reward"], reverse=True)

    # Print top results
    print("\n" + "=" * 70)
    print("TOP 5 CONFIGURATIONS")
    print("=" * 70)

    for i, result in enumerate(results[:5], 1):
        print(f"\n{i}. Avg Reward: {result['avg_reward']:.2f}")
        print("   Hyperparameters:")
        for key, value in result["hyperparams"].items():
            if isinstance(value, float):
                print(f"     {key}: {value:.6f}")
            else:
                print(f"     {key}: {value}")

    # Save results
    output_path = "./models/hyperparameter_random_search.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✅ Results saved to: {output_path}")

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Hyperparameter Optimization")

    subparsers = parser.add_subparsers(dest="command")

    # Grid search
    grid_parser = subparsers.add_parser("grid", help="Grid search")
    grid_parser.add_argument("--steps", type=int, default=50_000, help="Training steps")
    grid_parser.add_argument("--eval-games", type=int, default=20, help="Eval games")

    # Random search
    random_parser = subparsers.add_parser("random", help="Random search")
    random_parser.add_argument("--trials", type=int, default=20, help="Number of trials")
    random_parser.add_argument("--steps", type=int, default=50_000, help="Training steps")
    random_parser.add_argument("--eval-games", type=int, default=20, help="Eval games")

    args = parser.parse_args()

    if args.command == "grid":
        # Define parameter grid
        param_grid = {
            "learning_rate": [1e-4, 3e-4, 1e-3],
            "n_steps": [1024, 2048],
            "batch_size": [64, 128],
            "n_epochs": [10, 15],
            "gamma": [0.99],
            "gae_lambda": [0.95],
            "ent_coef": [0.01, 0.02],
        }

        grid_search_hyperparameters(param_grid, args.steps, args.eval_games)

    elif args.command == "random":
        random_search_hyperparameters(args.trials, args.steps, args.eval_games)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
