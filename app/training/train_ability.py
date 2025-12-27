"""Training script for ability-aware RL agent.

Trains an agent that makes strategic decisions about pirate abilities,
not just bidding and card play.

Usage:
    uv run python -m app.training.train_ability train
    uv run python -m app.training.train_ability train --timesteps 5000000
    uv run python -m app.training.train_ability resume --checkpoint models/ability_v1/checkpoint.zip
"""

import argparse
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import torch
from gymnasium import Env
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from app.gym_env.skullking_env_ability import AbilityAwareEnv, DecisionPhase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Training configuration
TRAINING_CONFIG = {
    "total_timesteps": 5_000_000,
    "n_envs": 256,
    "n_steps": 1024,
    "batch_size": 16384,
    "n_epochs": 15,
    "learning_rate": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "ent_coef": 0.015,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "clip_range": 0.2,
    "eval_freq": 100_000,
    "n_eval_episodes": 20,
}

# Policy network architecture
POLICY_KWARGS = {
    "net_arch": {
        "pi": [512, 512, 256],
        "vf": [512, 512, 256],
    },
    "activation_fn": torch.nn.ReLU,
}


class AbilityMetricsCallback(BaseCallback):
    """Tracks ability-specific metrics during training."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.ability_counts: dict[str, int] = {
            "rosie": 0,
            "bendt": 0,
            "roatan": 0,
            "harry": 0,
        }
        self.ability_rewards: dict[str, list[float]] = {
            "rosie": [],
            "bendt": [],
            "roatan": [],
            "harry": [],
        }

    def _on_step(self) -> bool:
        # Extract ability decisions from info
        for info in self.locals.get("infos", []):
            if "ability_decisions" in info:
                decisions = info["ability_decisions"]
                for ability, actions in decisions.items():
                    if actions:
                        self.ability_counts[ability] += len(actions)

        return True

    def _on_rollout_end(self) -> None:
        # Log ability metrics
        total = sum(self.ability_counts.values())
        if total > 0:
            self.logger.record("ability/total_decisions", total)
            for ability, count in self.ability_counts.items():
                self.logger.record(f"ability/{ability}_count", count)
                self.logger.record(f"ability/{ability}_ratio", count / total)

        # Reset for next rollout
        for key in self.ability_counts:
            self.ability_counts[key] = 0


class AbilityCurriculumCallback(BaseCallback):
    """Curriculum learning for ability decisions.

    Stages:
    1. 0-500k: Abilities disabled (learn basic card play)
    2. 500k-1M: Harry only (simplest - end of round decision)
    3. 1M-2M: Harry + Roatán (both are post-trick)
    4. 2M-3M: All abilities enabled
    5. 3M+: Full training with harder opponents
    """

    CURRICULUM: ClassVar[list[tuple[int, dict[str, Any]]]] = [
        (0, {"enable_abilities": False, "opponent": "rule_based", "difficulty": "easy"}),
        (500_000, {"enable_abilities": True, "abilities": ["harry"]}),
        (1_000_000, {"enable_abilities": True, "abilities": ["harry", "roatan"]}),
        (2_000_000, {"enable_abilities": True, "abilities": "all"}),
        (3_000_000, {"opponent": "rule_based", "difficulty": "medium"}),
        (4_000_000, {"opponent": "rule_based", "difficulty": "hard"}),
    ]

    def __init__(self, envs: DummyVecEnv | SubprocVecEnv, verbose: int = 0):
        super().__init__(verbose)
        self.envs = envs
        self.current_stage = 0

    def _on_step(self) -> bool:
        # Check if we should advance curriculum
        for i, (timestep, config) in enumerate(self.CURRICULUM):
            if self.num_timesteps >= timestep and i > self.current_stage:
                self.current_stage = i
                logger.info("Curriculum stage %d at %d steps: %s", i, self.num_timesteps, config)
                self._apply_config(config)
                break

        return True

    def _apply_config(self, config: dict[str, Any]) -> None:
        """Apply curriculum config to environments."""
        # This would update env settings via env methods
        # For now, log the change
        logger.info("Applying curriculum config: %s", config)


class PhaseDistributionCallback(BaseCallback):
    """Tracks distribution of decision phases encountered."""

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.phase_counts: dict[str, int] = {p.name: 0 for p in DecisionPhase}

    def _on_step(self) -> bool:
        # Would need to extract phase from observations
        return True

    def _on_rollout_end(self) -> None:
        total = sum(self.phase_counts.values())
        if total > 0:
            for phase, count in self.phase_counts.items():
                self.logger.record(f"phase/{phase}", count / total)

        # Reset
        for key in self.phase_counts:
            self.phase_counts[key] = 0


def make_env(
    opponent_type: str = "rule_based",
    difficulty: str = "medium",
    enable_abilities: bool = True,
    seed: int | None = None,
) -> Env:
    """Create a single ability-aware environment."""

    def _init() -> ActionMasker:
        env = AbilityAwareEnv(
            opponent_bot_type=opponent_type,
            opponent_difficulty=difficulty,
            enable_abilities=enable_abilities,
        )
        if seed is not None:
            env.reset(seed=seed)

        def mask_fn(e: AbilityAwareEnv) -> np.ndarray:
            return e.action_masks()

        return ActionMasker(env, mask_fn)

    return _init


def create_vectorized_envs(
    n_envs: int,
    opponent_type: str = "rule_based",
    difficulty: str = "medium",
    enable_abilities: bool = True,
    use_subproc: bool = False,
) -> DummyVecEnv | SubprocVecEnv:
    """Create vectorized environments for training."""
    env_fns = [make_env(opponent_type, difficulty, enable_abilities, seed=i) for i in range(n_envs)]

    if use_subproc and n_envs > 1:
        return SubprocVecEnv(env_fns)
    return DummyVecEnv(env_fns)


def train(
    timesteps: int = TRAINING_CONFIG["total_timesteps"],
    n_envs: int = TRAINING_CONFIG["n_envs"],
    output_dir: str = "models/ability_v1",
    tensorboard_log: str = "runs/ability_v1",
    enable_abilities: bool = True,
    use_subproc: bool = False,
) -> None:
    """Train ability-aware agent from scratch."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Creating %d training environments...", n_envs)
    train_envs = create_vectorized_envs(
        n_envs=n_envs,
        opponent_type="rule_based",
        difficulty="easy",
        enable_abilities=enable_abilities,
        use_subproc=use_subproc,
    )

    logger.info("Creating evaluation environment...")
    eval_env = create_vectorized_envs(
        n_envs=1,
        opponent_type="rule_based",
        difficulty="medium",
        enable_abilities=enable_abilities,
    )

    logger.info("Initializing MaskablePPO...")
    model = MaskablePPO(
        "MlpPolicy",
        train_envs,
        learning_rate=TRAINING_CONFIG["learning_rate"],
        n_steps=TRAINING_CONFIG["n_steps"],
        batch_size=TRAINING_CONFIG["batch_size"],
        n_epochs=TRAINING_CONFIG["n_epochs"],
        gamma=TRAINING_CONFIG["gamma"],
        gae_lambda=TRAINING_CONFIG["gae_lambda"],
        ent_coef=TRAINING_CONFIG["ent_coef"],
        vf_coef=TRAINING_CONFIG["vf_coef"],
        max_grad_norm=TRAINING_CONFIG["max_grad_norm"],
        clip_range=TRAINING_CONFIG["clip_range"],
        policy_kwargs=POLICY_KWARGS,
        tensorboard_log=tensorboard_log,
        verbose=1,
        device="auto",
    )

    # Callbacks
    callbacks = CallbackList(
        [
            AbilityMetricsCallback(verbose=1),
            AbilityCurriculumCallback(train_envs, verbose=1),
            MaskableEvalCallback(
                eval_env,
                best_model_save_path=str(output_path / "best_model"),
                log_path=str(output_path / "eval_logs"),
                eval_freq=TRAINING_CONFIG["eval_freq"] // n_envs,
                n_eval_episodes=TRAINING_CONFIG["n_eval_episodes"],
                deterministic=True,
            ),
        ]
    )

    logger.info("Starting training for %d timesteps...", timesteps)
    start_time = datetime.now(tz=UTC)

    try:
        model.learn(
            total_timesteps=timesteps,
            callback=callbacks,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")

    # Save final model
    final_path = output_path / "final_model.zip"
    model.save(final_path)
    logger.info("Saved final model to %s", final_path)

    elapsed = datetime.now(tz=UTC) - start_time
    logger.info("Training completed in %s", elapsed)

    train_envs.close()
    eval_env.close()


def resume(
    checkpoint: str,
    timesteps: int = 2_000_000,
    output_dir: str | None = None,
) -> None:
    """Resume training from a checkpoint."""
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        logger.error("Checkpoint not found: %s", checkpoint)
        sys.exit(1)

    if output_dir is None:
        output_dir = str(checkpoint_path.parent)

    output_path = Path(output_dir)

    logger.info("Loading model from %s...", checkpoint)
    model = MaskablePPO.load(checkpoint)

    logger.info("Creating environments...")
    train_envs = create_vectorized_envs(
        n_envs=TRAINING_CONFIG["n_envs"],
        opponent_type="rule_based",
        difficulty="medium",
        enable_abilities=True,
    )

    model.set_env(train_envs)

    callbacks = CallbackList(
        [
            AbilityMetricsCallback(verbose=1),
        ]
    )

    logger.info("Resuming training for %d additional timesteps...", timesteps)

    try:
        model.learn(
            total_timesteps=timesteps,
            callback=callbacks,
            reset_num_timesteps=False,
            progress_bar=True,
        )
    except KeyboardInterrupt:
        logger.info("Training interrupted")

    final_path = output_path / "resumed_model.zip"
    model.save(final_path)
    logger.info("Saved model to %s", final_path)

    train_envs.close()


def evaluate(
    model_path: str,
    n_episodes: int = 100,
    opponent_type: str = "rule_based",
    difficulty: str = "hard",
) -> None:
    """Evaluate a trained model."""
    logger.info("Loading model from %s...", model_path)
    model = MaskablePPO.load(model_path)

    env = AbilityAwareEnv(
        opponent_bot_type=opponent_type,
        opponent_difficulty=difficulty,
        enable_abilities=True,
    )

    def mask_fn(e: AbilityAwareEnv) -> np.ndarray:
        return e.action_masks()

    masked_env = ActionMasker(env, mask_fn)

    # Track metrics
    episode_rewards: list[float] = []
    goal_achieved: list[bool] = []
    ability_usage: dict[str, int] = {"rosie": 0, "bendt": 0, "roatan": 0, "harry": 0}

    for _ in range(n_episodes):
        obs, info = masked_env.reset()
        done = False
        total_reward = 0.0

        while not done:
            masks = masked_env.action_masks()
            action, _ = model.predict(obs, action_masks=masks, deterministic=True)
            obs, reward, terminated, truncated, info = masked_env.step(action)
            total_reward += reward
            done = terminated or truncated

        episode_rewards.append(total_reward)
        if info.get("goal_achieved"):
            goal_achieved.append(True)
        else:
            goal_achieved.append(False)

        # Track ability usage
        if "ability_decisions" in info:
            for ability, decisions in info["ability_decisions"].items():
                ability_usage[ability] += len(decisions)

    # Report results
    logger.info("=" * 50)
    logger.info("Evaluation Results (%d episodes)", n_episodes)
    logger.info("=" * 50)
    logger.info("Mean reward: %.2f ± %.2f", np.mean(episode_rewards), np.std(episode_rewards))
    logger.info("Goal achievement rate: %.1f%%", 100 * np.mean(goal_achieved))
    logger.info("Ability usage:")
    total_abilities = sum(ability_usage.values())
    for ability, count in ability_usage.items():
        pct = 100 * count / total_abilities if total_abilities > 0 else 0
        logger.info("  %s: %d (%.1f%%)", ability, count, pct)


def benchmark(
    n_steps: int = 10000,
    n_envs: int = 64,
) -> None:
    """Benchmark environment performance."""
    logger.info("Creating %d environments...", n_envs)
    envs = create_vectorized_envs(
        n_envs=n_envs,
        opponent_type="rule_based",
        difficulty="medium",
        enable_abilities=True,
    )

    envs.reset()
    start = time.time()

    for _ in range(n_steps):
        # Random actions
        actions = np.array([envs.action_space.sample() for _ in range(n_envs)])
        envs.step(actions)  # Ignore return values for benchmarking

    elapsed = time.time() - start
    total_steps = n_steps * n_envs
    fps = total_steps / elapsed

    logger.info("=" * 50)
    logger.info("Benchmark Results")
    logger.info("=" * 50)
    logger.info("Envs: %d", n_envs)
    logger.info("Steps per env: %d", n_steps)
    logger.info("Total steps: %d", total_steps)
    logger.info("Time: %.2f seconds", elapsed)
    logger.info("FPS: %.0f", fps)
    logger.info("μs per step: %.1f", 1_000_000 / fps)

    envs.close()


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Train ability-aware RL agent")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train from scratch")
    train_parser.add_argument("--timesteps", type=int, default=TRAINING_CONFIG["total_timesteps"])
    train_parser.add_argument("--n-envs", type=int, default=TRAINING_CONFIG["n_envs"])
    train_parser.add_argument("--output-dir", type=str, default="models/ability_v1")
    train_parser.add_argument("--tensorboard-log", type=str, default="runs/ability_v1")
    train_parser.add_argument("--enable-abilities", action="store_true", default=True)
    train_parser.add_argument("--no-abilities", action="store_false", dest="enable_abilities")
    train_parser.add_argument("--use-subproc", action="store_true", default=False)

    # Resume command
    resume_parser = subparsers.add_parser("resume", help="Resume from checkpoint")
    resume_parser.add_argument("checkpoint", type=str)
    resume_parser.add_argument("--timesteps", type=int, default=2_000_000)
    resume_parser.add_argument("--output-dir", type=str, default=None)

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate trained model")
    eval_parser.add_argument("model_path", type=str)
    eval_parser.add_argument("--n-episodes", type=int, default=100)
    eval_parser.add_argument("--opponent-type", type=str, default="rule_based")
    eval_parser.add_argument("--difficulty", type=str, default="hard")

    # Benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Benchmark environment performance")
    bench_parser.add_argument("--n-steps", type=int, default=10000)
    bench_parser.add_argument("--n-envs", type=int, default=64)

    args = parser.parse_args()

    if args.command == "train":
        train(
            timesteps=args.timesteps,
            n_envs=args.n_envs,
            output_dir=args.output_dir,
            tensorboard_log=args.tensorboard_log,
            enable_abilities=args.enable_abilities,
            use_subproc=args.use_subproc,
        )
    elif args.command == "resume":
        resume(
            checkpoint=args.checkpoint,
            timesteps=args.timesteps,
            output_dir=args.output_dir,
        )
    elif args.command == "evaluate":
        evaluate(
            model_path=args.model_path,
            n_episodes=args.n_episodes,
            opponent_type=args.opponent_type,
            difficulty=args.difficulty,
        )
    elif args.command == "benchmark":
        benchmark(
            n_steps=args.n_steps,
            n_envs=args.n_envs,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
