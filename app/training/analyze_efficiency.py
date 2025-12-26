#!/usr/bin/env python3
"""Analyze training efficiency from log file.

Usage:
    uv run python -m app.training.analyze_efficiency training_v8.log
"""

import re
import sys
from pathlib import Path


def parse_log(log_path: str) -> list[dict]:
    """Extract eval results and timing from training log."""
    results = []

    with open(log_path) as f:
        content = f.read()

    # Find all eval blocks with rewards
    eval_pattern = r"Eval num_timesteps=(\d+), episode_reward=([\d.]+) \+/- ([\d.]+)"
    evals = re.findall(eval_pattern, content)

    # Find all iteration blocks with timing
    # Pattern: total_timesteps and time_elapsed from the same block
    iter_pattern = r"time_elapsed\s*\|\s*(\d+)\s*\|.*?total_timesteps\s*\|\s*(\d+)"
    iterations = re.findall(iter_pattern, content, re.DOTALL)

    # Build timestep -> elapsed_time mapping
    time_map = {}
    for elapsed, timesteps in iterations:
        time_map[int(timesteps)] = int(elapsed)

    # Also try alternative pattern order
    iter_pattern2 = r"total_timesteps\s*\|\s*(\d+).*?time_elapsed\s*\|\s*(\d+)"
    for timesteps, elapsed in re.findall(iter_pattern2, content, re.DOTALL):
        time_map[int(timesteps)] = int(elapsed)

    # For each eval, find the closest timestep in time_map
    sorted_times = sorted(time_map.keys())

    for timesteps, reward, std in evals:
        timesteps = int(timesteps)
        reward = float(reward)
        std = float(std)

        # Find closest iteration timestep
        elapsed = 0
        for ts in sorted_times:
            if ts >= timesteps:
                elapsed = time_map[ts]
                break
            elapsed = time_map[ts]  # Use last known time if eval is after all iterations

        # Interpolate if we have the data
        if elapsed == 0 and sorted_times:
            # Estimate based on FPS from first iteration
            first_ts = sorted_times[0]
            first_time = time_map[first_ts]
            if first_time > 0:
                fps = first_ts / first_time
                elapsed = int(timesteps / fps)

        results.append(
            {
                "timesteps": timesteps,
                "elapsed_seconds": elapsed,
                "elapsed_minutes": elapsed / 60,
                "mean_reward": reward,
                "std_reward": std,
            }
        )

    return results


def analyze(results: list[dict]) -> None:
    """Print efficiency analysis."""
    if not results:
        print("No eval results found")
        return

    print("=" * 70)
    print("TRAINING EFFICIENCY ANALYSIS")
    print("=" * 70)
    print()

    # Table header
    print(f"{'Time':>8} {'Steps':>12} {'Reward':>10} {'FPS':>8} {'Δ/hour':>10}")
    print("-" * 70)

    first = results[0]
    prev = first

    for r in results:
        elapsed_min = r["elapsed_minutes"]
        timesteps = r["timesteps"]
        reward = r["mean_reward"]

        # FPS
        fps = timesteps / r["elapsed_seconds"] if r["elapsed_seconds"] > 0 else 0

        # Reward change per hour since first eval
        if r["elapsed_seconds"] > first["elapsed_seconds"]:
            time_delta_hours = (r["elapsed_seconds"] - first["elapsed_seconds"]) / 3600
            reward_delta = reward - first["mean_reward"]
            reward_per_hour = reward_delta / time_delta_hours if time_delta_hours > 0 else 0
        else:
            reward_per_hour = 0

        print(
            f"{elapsed_min:>7.1f}m "
            f"{timesteps:>11,} "
            f"{reward:>9.1f} "
            f"{fps:>7,.0f} "
            f"{reward_per_hour:>+9.1f}"
        )

        prev = r

    print("-" * 70)

    # Summary
    last = results[-1]
    total_seconds = last["elapsed_seconds"] if last["elapsed_seconds"] > 0 else 1
    total_hours = total_seconds / 3600
    total_reward_gain = last["mean_reward"] - first["mean_reward"]

    print()
    print(f"Total time:        {total_hours:.2f} hours ({last['elapsed_minutes']:.1f} min)")
    print(f"Total timesteps:   {last['timesteps']:,}")
    print(f"Average FPS:       {last['timesteps'] / total_seconds:,.0f}")
    print(f"First eval reward: {first['mean_reward']:.2f}")
    print(f"Final eval reward: {last['mean_reward']:.2f}")
    print(f"Total reward gain: {total_reward_gain:+.2f}")
    if total_hours > 0:
        print(f"Reward gain/hour:  {total_reward_gain / total_hours:+.2f}")
    print("=" * 70)


def main():
    if len(sys.argv) < 2:
        log_path = "training_v8.log"
    else:
        log_path = sys.argv[1]

    if not Path(log_path).exists():
        print(f"Log file not found: {log_path}")
        sys.exit(1)

    results = parse_log(log_path)
    analyze(results)


if __name__ == "__main__":
    main()
