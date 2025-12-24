#!/usr/bin/env python3
"""
Analyze and visualize training results.

Features:
- Compare multiple trained models
- Generate performance reports
- Plot learning curves
- Analyze bidding accuracy over time
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from stable_baselines3 import PPO

from app.gym_env.skullking_env_enhanced import SkullKingEnvEnhanced


def evaluate_model_comprehensive(
    model_path: str,
    n_games: int = 100,
) -> dict[str, any]:
    """
    Comprehensively evaluate a trained model.

    Returns detailed statistics including:
    - Win rates against each opponent type
    - Bidding accuracy statistics
    - Average scores and rankings
    - Strategic play metrics
    """
    print(f"\n{'=' * 70}")
    print(f"COMPREHENSIVE EVALUATION: {model_path}")
    print(f"{'=' * 70}\n")

    model = PPO.load(model_path)
    results = {}

    opponent_configs = [
        ("random", "medium"),
        ("rule_based", "easy"),
        ("rule_based", "medium"),
        ("rule_based", "hard"),
    ]

    for opp_type, difficulty in opponent_configs:
        key = f"{opp_type}_{difficulty}"
        print(f"\n📊 Testing vs {opp_type} ({difficulty})...")
        print("-" * 70)

        env = SkullKingEnvEnhanced(
            num_opponents=3,
            opponent_bot_type=opp_type,
            opponent_difficulty=difficulty,
        )

        game_data = []
        bidding_errors = []

        for game_num in range(n_games):
            obs, _ = env.reset()
            done = False
            total_reward = 0
            round_bids = []
            round_actuals = []

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                done = terminated or truncated

                # Track bidding accuracy
                if env.game and env.game.get_current_round():
                    current_round = env.game.get_current_round()
                    agent_player = env.game.get_player(env.agent_player_id)

                    if agent_player and agent_player.bid is not None:
                        if current_round.is_complete():
                            tricks_won = current_round.get_tricks_won(env.agent_player_id)
                            round_bids.append(agent_player.bid)
                            round_actuals.append(tricks_won)
                            bidding_errors.append(abs(agent_player.bid - tricks_won))

            # Get final stats
            if env.game and env.game.players:
                agent_score = env.game.players[0].score
                opponent_scores = [p.score for p in env.game.players[1:]]
                ranking = 1 + sum(1 for s in opponent_scores if s > agent_score)

                game_data.append(
                    {
                        "reward": total_reward,
                        "score": agent_score,
                        "ranking": ranking,
                        "won": ranking == 1,
                        "top2": ranking <= 2,
                    }
                )

                if game_num < 5:  # Print first 5 games
                    print(
                        f"  Game {game_num + 1:3d}: Score={agent_score:4d} | "
                        f"Opp={opponent_scores} | Rank={ranking}/4"
                    )

        env.close()

        # Calculate statistics
        rewards = [g["reward"] for g in game_data]
        scores = [g["score"] for g in game_data]
        rankings = [g["ranking"] for g in game_data]
        wins = sum(1 for g in game_data if g["won"])
        top2 = sum(1 for g in game_data if g["top2"])

        results[key] = {
            "games_played": n_games,
            "win_rate": wins / n_games,
            "top2_rate": top2 / n_games,
            "avg_reward": np.mean(rewards),
            "std_reward": np.std(rewards),
            "avg_score": np.mean(scores),
            "std_score": np.std(scores),
            "avg_ranking": np.mean(rankings),
            "ranking_distribution": {
                "1st": sum(1 for r in rankings if r == 1),
                "2nd": sum(1 for r in rankings if r == 2),
                "3rd": sum(1 for r in rankings if r == 3),
                "4th": sum(1 for r in rankings if r == 4),
            },
            "bidding_accuracy": {
                "mean_error": np.mean(bidding_errors) if bidding_errors else 0,
                "perfect_bids": sum(1 for e in bidding_errors if e == 0),
                "within_1": sum(1 for e in bidding_errors if e <= 1),
                "within_2": sum(1 for e in bidding_errors if e <= 2),
                "total_bids": len(bidding_errors),
            },
        }

        # Print summary
        print(f"\n  Summary ({n_games} games):")
        print(f"    Win rate: {100 * wins / n_games:.1f}%")
        print(f"    Top-2 rate: {100 * top2 / n_games:.1f}%")
        print(f"    Avg score: {np.mean(scores):.1f} ± {np.std(scores):.1f}")
        print(f"    Avg ranking: {np.mean(rankings):.2f}")
        print(f"    Avg reward: {np.mean(rewards):.2f}")

        if bidding_errors:
            perfect_pct = (
                100 * results[key]["bidding_accuracy"]["perfect_bids"] / len(bidding_errors)
            )
            within1_pct = 100 * results[key]["bidding_accuracy"]["within_1"] / len(bidding_errors)
            print(f"    Bidding: {perfect_pct:.1f}% perfect, {within1_pct:.1f}% within ±1")

    return results


def compare_models(model_paths: list[str], n_games: int = 50):
    """Compare multiple trained models."""
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    all_results = {}
    for model_path in model_paths:
        model_name = Path(model_path).stem
        all_results[model_name] = evaluate_model_comprehensive(model_path, n_games)

    # Generate comparison table
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)

    opponent_types = list(all_results[list(all_results.keys())[0]].keys())

    for opp_type in opponent_types:
        print(f"\n{opp_type.upper().replace('_', ' ')}:")
        print(f"{'Model':<30} {'Win%':<10} {'Top2%':<10} {'Avg Rank':<12} {'Bid Acc':<10}")
        print("-" * 70)

        for model_name, results in all_results.items():
            stats = results[opp_type]
            bid_acc = stats["bidding_accuracy"]
            perfect_pct = (
                100 * bid_acc["perfect_bids"] / bid_acc["total_bids"]
                if bid_acc["total_bids"] > 0
                else 0
            )

            print(
                f"{model_name:<30} "
                f"{100 * stats['win_rate']:<10.1f} "
                f"{100 * stats['top2_rate']:<10.1f} "
                f"{stats['avg_ranking']:<12.2f} "
                f"{perfect_pct:<10.1f}"
            )

    # Save results to JSON
    output_path = "./models/comparison_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n✅ Results saved to: {output_path}")

    return all_results


def analyze_checkpoints(checkpoint_dir: str):
    """
    Analyze training checkpoints to see progression over time.

    Plots win rate and bidding accuracy as training progresses.
    """
    print(f"\n{'=' * 70}")
    print(f"CHECKPOINT ANALYSIS: {checkpoint_dir}")
    print(f"{'=' * 70}\n")

    # Find all checkpoint files
    checkpoint_files = sorted(Path(checkpoint_dir).glob("*.zip"))

    if not checkpoint_files:
        print(f"❌ No checkpoints found in {checkpoint_dir}")
        return None

    print(f"Found {len(checkpoint_files)} checkpoints")

    progression = []

    for ckpt_path in checkpoint_files:
        # Extract step number from filename
        step_str = ckpt_path.stem.split("_")[-2]
        steps = int(step_str)

        print(f"\nEvaluating checkpoint at {steps:,} steps...")

        # Quick evaluation (10 games per opponent type)
        model = PPO.load(str(ckpt_path))

        # Test against rule-based medium (primary benchmark)
        env = SkullKingEnvEnhanced(
            num_opponents=3,
            opponent_bot_type="rule_based",
            opponent_difficulty="medium",
        )

        wins = 0
        bidding_errors = []

        for _ in range(10):
            obs, _ = env.reset()
            done = False

            round_bids = []
            round_actuals = []

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                if env.game and env.game.get_current_round():
                    current_round = env.game.get_current_round()
                    agent_player = env.game.get_player(env.agent_player_id)

                    if agent_player and agent_player.bid is not None:
                        if current_round.is_complete():
                            tricks_won = current_round.get_tricks_won(env.agent_player_id)
                            bidding_errors.append(abs(agent_player.bid - tricks_won))

            # Check if won
            if env.game and env.game.players:
                agent_score = env.game.players[0].score
                opponent_scores = [p.score for p in env.game.players[1:]]
                if agent_score == max([agent_score] + opponent_scores):
                    wins += 1

        env.close()

        win_rate = wins / 10
        avg_bid_error = np.mean(bidding_errors) if bidding_errors else 0

        progression.append(
            {
                "steps": steps,
                "win_rate": win_rate,
                "avg_bid_error": avg_bid_error,
            }
        )

        print(f"  Win rate: {100 * win_rate:.0f}%, Avg bid error: {avg_bid_error:.2f}")

    # Print progression summary
    print("\n" + "=" * 70)
    print("TRAINING PROGRESSION")
    print("=" * 70)
    print(f"{'Steps':<15} {'Win Rate':<15} {'Bid Error':<15} {'Improvement':<15}")
    print("-" * 70)

    for i, data in enumerate(progression):
        improvement = ""
        if i > 0:
            prev_win = progression[i - 1]["win_rate"]
            delta = data["win_rate"] - prev_win
            if delta > 0:
                improvement = f"+{100 * delta:.1f}%"
            elif delta < 0:
                improvement = f"{100 * delta:.1f}%"
            else:
                improvement = "="

        print(
            f"{data['steps']:<15,} "
            f"{100 * data['win_rate']:<15.1f} "
            f"{data['avg_bid_error']:<15.2f} "
            f"{improvement:<15}"
        )

    return progression


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Analyze Training Results")

    subparsers = parser.add_subparsers(dest="command")

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a single model")
    eval_parser.add_argument("--model-path", required=True, help="Path to model")
    eval_parser.add_argument("--games", type=int, default=100, help="Number of games")

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare multiple models")
    compare_parser.add_argument("--models", nargs="+", required=True, help="Model paths")
    compare_parser.add_argument("--games", type=int, default=50, help="Games per model")

    # Checkpoints command
    ckpt_parser = subparsers.add_parser("checkpoints", help="Analyze training checkpoints")
    ckpt_parser.add_argument("--dir", required=True, help="Checkpoint directory")

    args = parser.parse_args()

    if args.command == "evaluate":
        evaluate_model_comprehensive(args.model_path, args.games)
    elif args.command == "compare":
        compare_models(args.models, args.games)
    elif args.command == "checkpoints":
        analyze_checkpoints(args.dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
