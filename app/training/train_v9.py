#!/usr/bin/env python3
"""Train Hierarchical RL agents for Skull King (V9).

V9 Training Features:
1. Hierarchical RL: Manager (bidding) + Worker (card play) policies
2. Numba-accelerated observation encoding
3. Larger network architecture (use GPU headroom)
4. Action masking for valid moves only
5. Mixed opponent evaluation
6. Early stopping with plateau detection

Manager Environment:
- Observes: Hand strength, position, opponent patterns
- Action: Bid 0-10
- Horizon: 10 decisions per game (one per round)
- Reward: Round-end score based on bid accuracy

Worker Environment:
- Observes: Current trick, bid goal, cards remaining
- Action: Card to play from hand
- Horizon: 1-10 cards per round
- Reward: Trick-level shaping toward bid goal

Usage:
    # Train Manager policy
    uv run python -m app.training.train_v9 train-manager --timesteps 5000000

    # Train Worker policy
    uv run python -m app.training.train_v9 train-worker --timesteps 5000000

    # Train both sequentially
    uv run python -m app.training.train_v9 train-both

See TRAINING_LOG.md and V9_OPTIMIZATION_PLAN.md for details.
"""

import argparse
from pathlib import Path

import torch
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from app.gym_env.skullking_env_hierarchical import (
    ManagerEnv,
    WorkerEnv,
)
from app.training.callbacks import (
    MixedOpponentEvalCallback,
    PhaseSchedulerCallback,
    RoundStatsCallback,
)

# V9 Configuration (benchmark-optimized for RTX 4080 SUPER + Ryzen 9 7900X)
#
# HARDWARE BENCHMARKS:
# - Hierarchical envs are 2.8x faster than flat masked env (51μs vs 145μs per step)
# - DummyVecEnv outperforms SubprocVecEnv (env stepping too fast for subprocess overhead)
# - Large network [2048,2048,1024] achieves 79% GPU vs 30-46% for standard network
# - VRAM usage: ~3.7GB per model (17GB available)
#
# GAME FLOW ANALYSIS (see EPISODE_DESIGN.md):
# - Game has 10 rounds: 65 total decisions (10 bids + 55 plays)
# - Early rounds (1-3): 14% of decisions, simple (special cards dominate)
# - Late rounds (7-10): 58% of decisions, complex (multi-trick planning)
# - Bidding has delayed reward: bid at round start, feedback at round end
# - Card play has dense reward: trick-level shaping
#
# EPOCH STRATEGY:
# - Manager (bidding): MORE epochs (sparse reward, need signal extraction)
# - Worker (card play): FEWER epochs (dense reward, avoid overfitting)

DEFAULT_TIMESTEPS = 5_000_000
DEFAULT_N_ENVS = 256  # Optimal for hierarchical envs with DummyVecEnv
DEFAULT_BATCH_SIZE = 16384
DEFAULT_N_STEPS = 1024  # More frequent updates for fresh data

# Phase-specific epoch configuration
MANAGER_N_EPOCHS = 25  # Higher: sparse reward (bid → round-end feedback)
WORKER_N_EPOCHS = 12  # Lower: dense reward (trick-level shaping)
DEFAULT_N_EPOCHS = 15  # Fallback

DEFAULT_SAVE_DIR = "./models/hierarchical_v9"

# Phase curriculum schedule (phases: 0=early, 1=mid, 2=late)
# Start with late rounds (complex), then progressively add earlier rounds
PHASE_SCHEDULE = [
    (0, (2,)),  # 0 steps: Late only (rounds 7-10)
    (1_000_000, (1, 2)),  # 1M steps: Mid + Late (rounds 4-10)
    (2_000_000, (0, 1, 2)),  # 2M steps: All rounds
]

# V9 Network Architecture (large network to maximize GPU utilization)
# Benchmark: 79% GPU utilization with this config vs 30-46% with [512,512,256]
POLICY_KWARGS = {
    "net_arch": {
        "pi": [2048, 2048, 1024],  # Large policy network
        "vf": [2048, 2048, 1024],  # Large value network
    },
    "activation_fn": torch.nn.ReLU,
}


def manager_mask_fn(env: ManagerEnv):
    """Extract action masks from ManagerEnv."""
    return env.action_masks()


def worker_mask_fn(env: WorkerEnv):
    """Extract action masks from WorkerEnv."""
    return env.action_masks()


def create_manager_env(
    opponent_type: str = "rule_based",
    difficulty: str = "medium",
    use_weighted_sampling: bool = True,
    allowed_phases: tuple[int, ...] | None = None,
):
    """Create Manager environment with action masking.

    Args:
        opponent_type: Type of opponent bot
        difficulty: Opponent difficulty level
        use_weighted_sampling: Use round-weighted sampling (favor late rounds)
        allowed_phases: Tuple of allowed phase indices for curriculum
                       (0=early, 1=mid, 2=late). None means all phases.
    """
    env = ManagerEnv(
        num_opponents=3,
        opponent_bot_type=opponent_type,
        opponent_difficulty=difficulty,
        use_weighted_sampling=use_weighted_sampling,
        allowed_phases=allowed_phases,
    )
    return ActionMasker(env, manager_mask_fn)


def create_worker_env(
    opponent_type: str = "rule_based",
    difficulty: str = "medium",
    fixed_goal: int | None = None,
    use_weighted_sampling: bool = True,
    allowed_phases: tuple[int, ...] | None = None,
):
    """Create Worker environment with action masking.

    Args:
        opponent_type: Type of opponent bot
        difficulty: Opponent difficulty level
        fixed_goal: If set, use this bid goal. None for random goals.
        use_weighted_sampling: Use round-weighted sampling (favor late rounds)
        allowed_phases: Tuple of allowed phase indices for curriculum
                       (0=early, 1=mid, 2=late). None means all phases.
    """
    env = WorkerEnv(
        num_opponents=3,
        opponent_bot_type=opponent_type,
        opponent_difficulty=difficulty,
        fixed_goal=fixed_goal,
        use_weighted_sampling=use_weighted_sampling,
        allowed_phases=allowed_phases,
    )
    return ActionMasker(env, worker_mask_fn)


def train_manager(
    total_timesteps: int = DEFAULT_TIMESTEPS,
    n_envs: int = DEFAULT_N_ENVS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    n_steps: int = DEFAULT_N_STEPS,
    n_epochs: int = MANAGER_N_EPOCHS,  # Higher epochs: sparse reward (bid → round-end)
    use_subproc: bool = False,  # DummyVecEnv is faster for hierarchical envs
    save_dir: str = DEFAULT_SAVE_DIR,
    load_path: str | None = None,
    use_phase_curriculum: bool = True,
) -> str:
    """Train Manager (bidding) policy.

    Bidding has SPARSE reward (feedback only at round end), so we use more
    epochs to extract maximum learning signal from each batch of experiences.

    Features:
    - Round-weighted sampling: Late rounds sampled 4x more than early rounds
    - Phase curriculum: Start with late rounds (complex), progressively add earlier
    - Phase embedding: Explicit early/mid/late encoding in observations

    Returns path to trained model.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vec_env_cls = SubprocVecEnv if use_subproc else DummyVecEnv
    vec_env_name = "SubprocVecEnv" if use_subproc else "DummyVecEnv"

    # Initial phases for curriculum
    initial_phases = PHASE_SCHEDULE[0][1] if use_phase_curriculum else None

    print("=" * 60)
    print("SKULL KING V9 - Manager (Bidding) Policy Training")
    print("=" * 60)
    print(f"Device: {device.upper()}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Parallel envs: {n_envs} ({vec_env_name})")
    print(f"Batch size: {batch_size}, n_steps: {n_steps}, n_epochs: {n_epochs}")
    print(f"Network: {POLICY_KWARGS['net_arch']['pi']}")
    print(f"Phase curriculum: {use_phase_curriculum} (starting with phases {initial_phases})")
    print("Round-weighted sampling: Enabled")
    print("=" * 60 + "\n")

    # Create directories
    save_path = Path(save_dir) / "manager"
    save_path.mkdir(parents=True, exist_ok=True)
    (save_path / "checkpoints").mkdir(exist_ok=True)
    (save_path / "best_model").mkdir(exist_ok=True)

    # Create environments with weighted sampling and phase curriculum
    print(f"Creating {n_envs} manager environments...")
    vec_env = make_vec_env(
        lambda: create_manager_env(
            "rule_based",
            "medium",
            use_weighted_sampling=True,
            allowed_phases=initial_phases,
        ),
        n_envs=n_envs,
        vec_env_cls=vec_env_cls,
    )

    eval_env = make_vec_env(
        lambda: create_manager_env(
            "rule_based",
            "hard",
            use_weighted_sampling=False,  # Uniform for eval
            allowed_phases=None,  # All phases for eval
        ),
        n_envs=1,
        vec_env_cls=DummyVecEnv,
    )

    # Create or load model
    if load_path:
        print(f"Loading model from {load_path}...")
        model = MaskablePPO.load(load_path, env=vec_env, device=device)
    else:
        print("Creating new MaskablePPO model...")
        model = MaskablePPO(
            "MlpPolicy",
            vec_env,
            learning_rate=3e-4,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.02,  # Higher entropy for bid exploration
            vf_coef=0.5,
            max_grad_norm=0.5,
            clip_range=0.2,
            policy_kwargs=POLICY_KWARGS,
            tensorboard_log=str(save_path / "tensorboard"),
            verbose=1,
            device=device,
        )

    # Setup callbacks
    callbacks = []

    checkpoint_cb = CheckpointCallback(
        save_freq=50_000 // n_envs,
        save_path=str(save_path / "checkpoints"),
        name_prefix="manager",
    )
    callbacks.append(checkpoint_cb)

    eval_cb = MixedOpponentEvalCallback(
        eval_env,
        opponent_configs=[("rule_based", "hard")],
        n_eval_episodes=10,
        eval_freq=100_000,
        best_model_save_path=str(save_path / "best_model"),
        log_path=str(save_path / "eval_logs"),
        deterministic=True,
        early_stopping=False,  # Disabled: curriculum learning causes temporary performance drops
        plateau_window=5,
        plateau_threshold=5.0,
        min_evals_before_stopping=10,
    )
    callbacks.append(eval_cb)

    # Phase curriculum: progressively unlock rounds
    if use_phase_curriculum:
        phase_cb = PhaseSchedulerCallback(schedule=PHASE_SCHEDULE, verbose=1)
        callbacks.append(phase_cb)

    # Round statistics tracking
    round_stats_cb = RoundStatsCallback(log_interval=50000, verbose=1)
    callbacks.append(round_stats_cb)

    # Train
    print("\nStarting training...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
        reset_num_timesteps=load_path is None,
    )

    # Save final model
    final_path = str(save_path / "final_model.zip")
    model.save(final_path)
    print(f"\nManager training complete. Model saved to {final_path}")

    vec_env.close()
    eval_env.close()

    return final_path


def train_worker(
    total_timesteps: int = DEFAULT_TIMESTEPS,
    n_envs: int = DEFAULT_N_ENVS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    n_steps: int = DEFAULT_N_STEPS,
    n_epochs: int = WORKER_N_EPOCHS,  # Lower epochs: dense reward (trick-level shaping)
    use_subproc: bool = False,  # DummyVecEnv is faster for hierarchical envs
    save_dir: str = DEFAULT_SAVE_DIR,
    load_path: str | None = None,
    fixed_goal: int | None = None,
    use_phase_curriculum: bool = True,
) -> str:
    """Train Worker (card-playing) policy.

    Card play has DENSE reward (trick-level shaping), so we use fewer epochs
    to avoid overfitting to the current batch and collect more fresh data.

    Features:
    - Round-weighted sampling: Late rounds sampled 4x more than early rounds
    - Phase curriculum: Start with late rounds (complex), progressively add earlier
    - Phase embedding: Explicit early/mid/late encoding in observations

    Args:
        fixed_goal: If set, train worker for specific bid goal.
                   If None, randomly sample goals for generalization.
        use_phase_curriculum: Enable progressive phase unlocking

    Returns path to trained model.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vec_env_cls = SubprocVecEnv if use_subproc else DummyVecEnv
    vec_env_name = "SubprocVecEnv" if use_subproc else "DummyVecEnv"

    # Initial phases for curriculum
    initial_phases = PHASE_SCHEDULE[0][1] if use_phase_curriculum else None

    print("=" * 60)
    print("SKULL KING V9 - Worker (Card Play) Policy Training")
    print("=" * 60)
    print(f"Device: {device.upper()}")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Parallel envs: {n_envs} ({vec_env_name})")
    print(f"Batch size: {batch_size}, n_steps: {n_steps}, n_epochs: {n_epochs}")
    print(f"Network: {POLICY_KWARGS['net_arch']['pi']}")
    print(f"Fixed goal: {fixed_goal if fixed_goal is not None else 'random'}")
    print(f"Phase curriculum: {use_phase_curriculum} (starting with phases {initial_phases})")
    print("Round-weighted sampling: Enabled")
    print("=" * 60 + "\n")

    # Create directories
    save_path = Path(save_dir) / "worker"
    save_path.mkdir(parents=True, exist_ok=True)
    (save_path / "checkpoints").mkdir(exist_ok=True)
    (save_path / "best_model").mkdir(exist_ok=True)

    # Create environments with weighted sampling and phase curriculum
    print(f"Creating {n_envs} worker environments...")
    vec_env = make_vec_env(
        lambda: create_worker_env(
            "rule_based",
            "medium",
            fixed_goal,
            use_weighted_sampling=True,
            allowed_phases=initial_phases,
        ),
        n_envs=n_envs,
        vec_env_cls=vec_env_cls,
    )

    eval_env = make_vec_env(
        lambda: create_worker_env(
            "rule_based",
            "hard",
            fixed_goal,
            use_weighted_sampling=False,  # Uniform for eval
            allowed_phases=None,  # All phases for eval
        ),
        n_envs=1,
        vec_env_cls=DummyVecEnv,
    )

    # Create or load model
    if load_path:
        print(f"Loading model from {load_path}...")
        model = MaskablePPO.load(load_path, env=vec_env, device=device)
    else:
        print("Creating new MaskablePPO model...")
        model = MaskablePPO(
            "MlpPolicy",
            vec_env,
            learning_rate=3e-4,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,  # Lower entropy for card play
            vf_coef=0.5,
            max_grad_norm=0.5,
            clip_range=0.2,
            policy_kwargs=POLICY_KWARGS,
            tensorboard_log=str(save_path / "tensorboard"),
            verbose=1,
            device=device,
        )

    # Setup callbacks
    callbacks = []

    checkpoint_cb = CheckpointCallback(
        save_freq=50_000 // n_envs,
        save_path=str(save_path / "checkpoints"),
        name_prefix="worker",
    )
    callbacks.append(checkpoint_cb)

    eval_cb = MixedOpponentEvalCallback(
        eval_env,
        opponent_configs=[("rule_based", "hard")],
        n_eval_episodes=10,
        eval_freq=100_000,
        best_model_save_path=str(save_path / "best_model"),
        log_path=str(save_path / "eval_logs"),
        deterministic=True,
        early_stopping=False,  # Disabled: curriculum learning causes temporary performance drops
        plateau_window=5,
        plateau_threshold=3.0,
        min_evals_before_stopping=10,
    )
    callbacks.append(eval_cb)

    # Phase curriculum: progressively unlock rounds
    if use_phase_curriculum:
        phase_cb = PhaseSchedulerCallback(schedule=PHASE_SCHEDULE, verbose=1)
        callbacks.append(phase_cb)

    # Round statistics tracking
    round_stats_cb = RoundStatsCallback(log_interval=50000, verbose=1)
    callbacks.append(round_stats_cb)

    # Train
    print("\nStarting training...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True,
        reset_num_timesteps=load_path is None,
    )

    # Save final model
    final_path = str(save_path / "final_model.zip")
    model.save(final_path)
    print(f"\nWorker training complete. Model saved to {final_path}")

    vec_env.close()
    eval_env.close()

    return final_path


def train_both(
    manager_timesteps: int = 3_000_000,
    worker_timesteps: int = 5_000_000,
    **kwargs,
) -> tuple[str, str]:
    """Train both Manager and Worker policies sequentially.

    Manager is trained first, then Worker.
    """
    print("=" * 60)
    print("SKULL KING V9 - Full Hierarchical Training")
    print("=" * 60)
    print(f"Manager timesteps: {manager_timesteps:,}")
    print(f"Worker timesteps: {worker_timesteps:,}")
    print("=" * 60 + "\n")

    # Train Manager
    manager_path = train_manager(total_timesteps=manager_timesteps, **kwargs)

    print("\n" + "=" * 60)
    print("Manager training complete. Starting Worker training...")
    print("=" * 60 + "\n")

    # Train Worker
    worker_path = train_worker(total_timesteps=worker_timesteps, **kwargs)

    print("\n" + "=" * 60)
    print("Full hierarchical training complete!")
    print(f"Manager model: {manager_path}")
    print(f"Worker model: {worker_path}")
    print("=" * 60)

    return manager_path, worker_path


def main():
    """Parse arguments and run training."""
    parser = argparse.ArgumentParser(description="Train Skull King V9 Hierarchical RL")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Common arguments
    common_parser = argparse.ArgumentParser(add_help=False)
    common_parser.add_argument(
        "--timesteps", type=int, default=DEFAULT_TIMESTEPS, help="Total timesteps"
    )
    common_parser.add_argument(
        "--envs", type=int, default=DEFAULT_N_ENVS, help="Number of parallel envs"
    )
    common_parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size"
    )
    common_parser.add_argument(
        "--n-steps", type=int, default=DEFAULT_N_STEPS, help="Steps per env before update"
    )
    common_parser.add_argument(
        "--n-epochs", type=int, default=DEFAULT_N_EPOCHS, help="PPO epochs per rollout"
    )
    common_parser.add_argument(
        "--subproc", action="store_true", help="Use SubprocVecEnv (default: DummyVecEnv)"
    )
    common_parser.add_argument(
        "--save-dir", type=str, default=DEFAULT_SAVE_DIR, help="Save directory"
    )
    common_parser.add_argument("--load", type=str, default=None, help="Load model path")

    # train-manager command (no extra args beyond common)
    subparsers.add_parser("train-manager", parents=[common_parser], help="Train Manager policy")

    # train-worker command
    worker_parser = subparsers.add_parser(
        "train-worker", parents=[common_parser], help="Train Worker policy"
    )
    worker_parser.add_argument(
        "--fixed-goal", type=int, default=None, help="Fixed bid goal for training"
    )

    # train-both command
    both_parser = subparsers.add_parser(
        "train-both", parents=[common_parser], help="Train both Manager and Worker"
    )
    both_parser.add_argument(
        "--manager-timesteps",
        type=int,
        default=3_000_000,
        help="Manager timesteps",
    )
    both_parser.add_argument(
        "--worker-timesteps",
        type=int,
        default=5_000_000,
        help="Worker timesteps",
    )

    args = parser.parse_args()

    kwargs = {
        "n_envs": args.envs,
        "batch_size": args.batch_size,
        "n_steps": args.n_steps,
        "n_epochs": args.n_epochs,
        "use_subproc": args.subproc,  # Default: False (DummyVecEnv is faster)
        "save_dir": args.save_dir,
        "load_path": args.load,
    }

    if args.command == "train-manager":
        train_manager(total_timesteps=args.timesteps, **kwargs)
    elif args.command == "train-worker":
        train_worker(total_timesteps=args.timesteps, fixed_goal=args.fixed_goal, **kwargs)
    elif args.command == "train-both":
        train_both(
            manager_timesteps=args.manager_timesteps,
            worker_timesteps=args.worker_timesteps,
            **kwargs,
        )


if __name__ == "__main__":
    main()
