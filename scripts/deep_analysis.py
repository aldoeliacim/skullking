"""Deep analysis of Skull King RL training - Ultra-thinking mode."""

import re

import matplotlib
import numpy as np

matplotlib.use("Agg")
from collections import defaultdict


def parse_training_log(log_file):
    """Parse training log and extract all metrics."""
    with open(log_file) as f:
        content = f.read()

    metrics = defaultdict(list)
    timesteps = []

    # Extract all timestep blocks
    blocks = re.findall(
        r"total_timesteps\s+\|\s+(\d+).*?(?=total_timesteps|\Z)", content, re.DOTALL
    )

    for match in re.finditer(r"total_timesteps\s+\|\s+(\d+)", content):
        ts = int(match.group(1))
        timesteps.append(ts)

        # Get block around this timestep
        start = max(0, match.start() - 800)
        end = min(len(content), match.end() + 800)
        block = content[start:end]

        # Extract all metrics
        metric_patterns = {
            "ep_rew_mean": r"ep_rew_mean\s+\|\s+([\d.]+)",
            "ep_len_mean": r"ep_len_mean\s+\|\s+([\d.]+)",
            "value_loss": r"value_loss\s+\|\s+([\d.e+]+)",
            "explained_variance": r"explained_variance\s+\|\s+([-\d.]+)",
            "entropy_loss": r"entropy_loss\s+\|\s+([-\d.]+)",
            "approx_kl": r"approx_kl\s+\|\s+([\d.e+]+)",
            "clip_fraction": r"clip_fraction\s+\|\s+([\d.]+)",
            "policy_gradient_loss": r"policy_gradient_loss\s+\|\s+([-\d.e+]+)",
        }

        for metric_name, pattern in metric_patterns.items():
            match_metric = re.search(pattern, block)
            if match_metric:
                metrics[metric_name].append(float(match_metric.group(1)))
            else:
                metrics[metric_name].append(None)

    return timesteps, metrics


def calculate_trends(timesteps, values):
    """Calculate trend statistics."""
    valid_data = [(t, v) for t, v in zip(timesteps, values, strict=False) if v is not None]
    if len(valid_data) < 2:
        return None

    ts, vs = zip(*valid_data, strict=False)
    ts = np.array(ts)
    vs = np.array(vs)

    # Linear regression
    A = np.vstack([ts, np.ones(len(ts))]).T
    slope, intercept = np.linalg.lstsq(A, vs, rcond=None)[0]

    return {
        "slope": slope,
        "start": vs[0],
        "end": vs[-1],
        "mean": np.mean(vs),
        "std": np.std(vs),
        "min": np.min(vs),
        "max": np.max(vs),
        "trend": "improving" if slope > 0 else "declining",
    }


def analyze_learning_rate(timesteps, metrics):
    """Analyze learning rate and convergence."""
    rewards = [v for v in metrics["ep_rew_mean"] if v is not None]
    if len(rewards) < 10:
        return "Insufficient data"

    # Compare first 25% vs last 25%
    split = len(rewards) // 4
    early = rewards[:split]
    late = rewards[-split:]

    improvement = np.mean(late) - np.mean(early)
    improvement_pct = (improvement / abs(np.mean(early))) * 100 if np.mean(early) != 0 else 0

    # Check if learning is slowing
    recent_variance = np.std(rewards[-5:])
    overall_variance = np.std(rewards)

    return {
        "improvement": improvement,
        "improvement_pct": improvement_pct,
        "recent_variance": recent_variance,
        "slowing": recent_variance < overall_variance * 0.5,
    }


def identify_issues(timesteps, metrics):
    """Identify potential issues in training."""
    issues = []
    recommendations = []

    # Check explained variance
    ev_trend = calculate_trends(timesteps, metrics["explained_variance"])
    if ev_trend and ev_trend["end"] < 0.5:
        issues.append("Low explained variance (<0.5)")
        recommendations.append("• Increase vf_coef or n_epochs")
    elif ev_trend and ev_trend["end"] > 0.85:
        issues.append("Very high explained variance (>0.85) - possible overfitting to value")
        recommendations.append("• Consider reducing vf_coef to 0.8")

    # Check entropy
    entropy_trend = calculate_trends(timesteps, metrics["entropy_loss"])
    if entropy_trend and abs(entropy_trend["end"]) < 0.8:
        issues.append("Low entropy - agent may be converging prematurely")
        recommendations.append("• Increase ent_coef to encourage exploration")

    # Check value loss trend
    vl_trend = calculate_trends(timesteps, metrics["value_loss"])
    if vl_trend and vl_trend["slope"] > 0:
        issues.append("Value loss increasing - value function diverging")
        recommendations.append("• Reduce learning rate or increase n_epochs")

    # Check KL divergence
    kl_values = [v for v in metrics["approx_kl"] if v is not None]
    if kl_values and np.mean(kl_values[-5:]) > 0.01:
        issues.append("High KL divergence - policy updates too large")
        recommendations.append("• Reduce learning rate or clip_range")

    return issues, recommendations


def main():
    print("=" * 70)
    print("SKULL KING RL - ULTRA-DEEP ANALYSIS")
    print("=" * 70)

    timesteps, metrics = parse_training_log("training_v2.log")

    print(
        f"\nTraining Progress: {timesteps[-1]:,} / 1,500,000 steps ({100 * timesteps[-1] / 1500000:.1f}%)"
    )
    print(f"Data points: {len(timesteps)}")

    print("\n" + "=" * 70)
    print("METRIC TRENDS")
    print("=" * 70)

    key_metrics = ["ep_rew_mean", "explained_variance", "value_loss", "entropy_loss"]

    for metric in key_metrics:
        trend = calculate_trends(timesteps, metrics[metric])
        if trend:
            print(f"\n{metric}:")
            print(f"  Start: {trend['start']:.3f}")
            print(f"  End: {trend['end']:.3f}")
            print(
                f"  Change: {trend['end'] - trend['start']:+.3f} ({((trend['end'] / trend['start'] - 1) * 100) if trend['start'] != 0 else 0:+.1f}%)"
            )
            print(
                f"  Slope: {trend['slope']:.6f} per step ({'↗ improving' if trend['slope'] > 0 else '↘ declining'})"
            )
            print(f"  Mean ± Std: {trend['mean']:.3f} ± {trend['std']:.3f}")

    print("\n" + "=" * 70)
    print("LEARNING ANALYSIS")
    print("=" * 70)

    learning = analyze_learning_rate(timesteps, metrics)
    if isinstance(learning, dict):
        print(
            f"\nImprovement (early vs late): {learning['improvement']:+.1f} ({learning['improvement_pct']:+.1f}%)"
        )
        print(f"Recent variance: {learning['recent_variance']:.1f}")
        print(f"Learning status: {'Slowing down' if learning['slowing'] else 'Still improving'}")

    print("\n" + "=" * 70)
    print("ISSUE DETECTION")
    print("=" * 70)

    issues, recommendations = identify_issues(timesteps, metrics)

    if issues:
        print("\n⚠️  Issues Found:")
        for issue in issues:
            print(f"  • {issue}")
        print("\n💡 Recommendations:")
        for rec in recommendations:
            print(f"  {rec}")
    else:
        print("\n✅ No major issues detected - training progressing well!")

    print("\n" + "=" * 70)
    print("ADVANCED INSIGHTS")
    print("=" * 70)

    # Variance analysis
    ev_values = [v for v in metrics["explained_variance"] if v is not None]
    reward_values = [v for v in metrics["ep_rew_mean"] if v is not None]

    if len(ev_values) >= 5:
        recent_ev = np.mean(ev_values[-5:])
        ev_growth_rate = (ev_values[-1] - ev_values[0]) / len(ev_values)
        print("\nExplained Variance Analysis:")
        print(f"  Current: {recent_ev:.3f}")
        print(f"  Growth rate: {ev_growth_rate:.6f} per iteration")
        print(
            f"  Status: {'Excellent' if recent_ev > 0.7 else 'Good' if recent_ev > 0.5 else 'Needs improvement'}"
        )

        if recent_ev > 0.8:
            print("  ⚡ High EV achieved! Value function is learning very well.")

    if len(reward_values) >= 10:
        recent_rewards = reward_values[-10:]
        reward_stability = np.std(recent_rewards) / (np.mean(recent_rewards) + 1e-8)
        print("\nReward Stability Analysis:")
        print(f"  Recent 10 avg: {np.mean(recent_rewards):.1f} ± {np.std(recent_rewards):.1f}")
        print(f"  Coefficient of variation: {reward_stability:.3f}")
        print(
            f"  Status: {'Very stable' if reward_stability < 0.1 else 'Stable' if reward_stability < 0.2 else 'Unstable'}"
        )

    # Entropy analysis
    entropy_values = [abs(v) for v in metrics["entropy_loss"] if v is not None]
    if len(entropy_values) >= 5:
        recent_entropy = np.mean(entropy_values[-5:])
        print("\nExploration Analysis:")
        print(f"  Current entropy: {recent_entropy:.3f}")
        print(
            f"  Status: {'Good exploration' if recent_entropy > 1.0 else 'Moderate' if recent_entropy > 0.8 else 'Low - may need more exploration'}"
        )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
