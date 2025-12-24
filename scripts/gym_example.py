#!/usr/bin/env python3
"""Example script showing how to use the Skull King Gymnasium environment.

This demonstrates:
1. Creating the environment
2. Running random actions
3. Training a simple agent (placeholder)
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from app.gym_env import SkullKingEnv


def random_agent_example():
    """Run a game with random actions."""
    print("\n" + "=" * 60)
    print("Skull King Gymnasium Environment - Random Agent Example")
    print("=" * 60 + "\n")

    # Create environment
    env = SkullKingEnv(
        num_opponents=3,
        opponent_bot_type="rule_based",
        render_mode="human",
    )

    # Reset environment
    observation, info = env.reset(seed=42)
    print(f"Initial observation shape: {observation.shape}")
    print(f"Initial info: {info}\n")

    total_reward = 0
    step_count = 0

    # Run episode
    while True:
        # Random action
        action = env.action_space.sample()

        # Take step
        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        step_count += 1

        # Render
        env.render()

        # Check if episode is done
        if terminated or truncated:
            print("\nEpisode finished!")
            print(f"  Total steps: {step_count}")
            print(f"  Total reward: {total_reward:.2f}")
            print(f"  Final agent score: {info.get('agent_score', 0)}")
            break

    env.close()


def multiple_episodes_example():
    """Run multiple episodes to collect statistics."""
    print("\n" + "=" * 60)
    print("Running 5 episodes with random actions")
    print("=" * 60 + "\n")

    env = SkullKingEnv(
        num_opponents=3,
        opponent_bot_type="rule_based",
        render_mode=None,  # No rendering for speed
    )

    rewards = []
    scores = []
    wins = 0

    for episode in range(5):
        observation, info = env.reset(seed=episode)
        total_reward = 0

        while True:
            action = env.action_space.sample()
            _observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if terminated or truncated:
                final_score = info.get("agent_score", 0)
                rewards.append(total_reward)
                scores.append(final_score)

                # Check if agent won (simple heuristic)
                if final_score > 0:  # Simplified winning condition
                    wins += 1

                print(f"Episode {episode + 1}: Reward = {total_reward:.2f}, Score = {final_score}")
                break

    env.close()

    print("\nStatistics over 5 episodes:")
    print(f"  Average reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    print(f"  Average score: {np.mean(scores):.2f} ± {np.std(scores):.2f}")
    print(f"  Win rate: {wins}/5 ({wins / 5 * 100:.0f}%)")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Skull King Gym Environment Examples")
    parser.add_argument(
        "--mode",
        choices=["single", "multiple"],
        default="single",
        help="Run mode: single episode or multiple episodes",
    )

    args = parser.parse_args()

    if args.mode == "single":
        random_agent_example()
    else:
        multiple_episodes_example()


if __name__ == "__main__":
    main()
