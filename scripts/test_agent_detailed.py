"""Detailed agent testing with gameplay analysis."""

import sys
from collections import defaultdict

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from app.gym_env.skullking_env_masked import SkullKingEnvMasked


def mask_fn(env):
    return env.action_masks()


# Load best model
model_path = "./models/masked_ppo/best_model/best_model.zip"
try:
    model = MaskablePPO.load(model_path)
    print(f"✓ Loaded model from {model_path}\n")
except Exception as e:
    print(f"⚠ Could not load {model_path}: {e}\n")
    model = None

if not model:
    print("No model to test. Exiting.")
    sys.exit(1)

# Test against different opponents
opponent_configs = [
    ("random", "medium", "Random Medium"),
    ("random", "hard", "Random Hard"),
    ("rule_based", "easy", "Rule-based Easy"),
    ("rule_based", "medium", "Rule-based Medium"),
]

print("=" * 70)
print("DETAILED AGENT PERFORMANCE TESTING")
print("=" * 70)

overall_stats = defaultdict(list)

for opp_type, opp_diff, label in opponent_configs:
    print(f"\n{'=' * 70}")
    print(f"Testing vs {label}")
    print(f"{'=' * 70}")

    env = SkullKingEnvMasked(
        num_opponents=3, opponent_bot_type=opp_type, opponent_difficulty=opp_diff
    )
    env = ActionMasker(env, mask_fn)

    rewards = []
    ranks = []
    bid_accuracies = []
    episodes = 20

    for _episode in range(episodes):
        obs, info = env.reset()
        total_reward = 0
        done = False

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated

        rewards.append(total_reward)

        # Extract final info
        if "ranking" in info:
            ranks.append(info["ranking"])

        # Try to get bid accuracy
        if hasattr(env, "game") and env.game:
            agent_player = env.game.get_player(env.agent_player_id)
            if agent_player:
                perfect_bids = 0
                total_rounds = 0
                for round_obj in env.game.rounds:
                    if agent_player.id in round_obj.scores:
                        bid = agent_player.bid if hasattr(agent_player, "bid") else 0
                        tricks_won = round_obj.get_tricks_won(agent_player.id)
                        if abs(bid - tricks_won) == 0:
                            perfect_bids += 1
                        total_rounds += 1

                if total_rounds > 0:
                    bid_accuracies.append(perfect_bids / total_rounds)

    # Statistics
    avg_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    avg_rank = np.mean(ranks) if ranks else 0
    win_rate = (sum(1 for r in ranks if r == 1) / len(ranks) * 100) if ranks else 0
    avg_bid_acc = np.mean(bid_accuracies) if bid_accuracies else 0

    print(f"\nResults ({episodes} episodes):")
    print(f"  Average reward: {avg_reward:.1f} ± {std_reward:.1f}")
    print(f"  Average rank: {avg_rank:.2f} / 4")
    print(f"  Win rate: {win_rate:.1f}%")
    print(f"  Perfect bid rate: {avg_bid_acc * 100:.1f}%")

    overall_stats["rewards"].append(avg_reward)
    overall_stats["ranks"].append(avg_rank)
    overall_stats["win_rates"].append(win_rate)
    overall_stats["labels"].append(label)

print(f"\n{'=' * 70}")
print("OVERALL SUMMARY")
print(f"{'=' * 70}")

for i, label in enumerate(overall_stats["labels"]):
    print(f"\n{label}:")
    print(f"  Avg Reward: {overall_stats['rewards'][i]:.1f}")
    print(f"  Avg Rank: {overall_stats['ranks'][i]:.2f}")
    print(f"  Win Rate: {overall_stats['win_rates'][i]:.1f}%")

print(f"\n{'=' * 70}")
print("PERFORMANCE ANALYSIS")
print(f"{'=' * 70}")

# Analyze performance
all_win_rates = overall_stats["win_rates"]
avg_win_rate = np.mean(all_win_rates)
baseline_random = 25.0  # Expected for random 1/4 players

print(f"\nAverage win rate: {avg_win_rate:.1f}%")
print(f"Baseline (random): {baseline_random:.1f}%")
print(
    f"Improvement: {avg_win_rate - baseline_random:+.1f}% ({(avg_win_rate / baseline_random - 1) * 100:+.1f}%)"
)

if avg_win_rate > 35:
    print("\n✅ STRONG PERFORMANCE - Agent is learning well!")
elif avg_win_rate > 28:
    print("\n✓ GOOD PERFORMANCE - Agent above baseline")
else:
    print("\n⚠ MODERATE PERFORMANCE - Room for improvement")

print(f"\n{'=' * 70}")
