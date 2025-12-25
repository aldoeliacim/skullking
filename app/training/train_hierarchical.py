"""Hierarchical RL Training for Skull King.

V7+V8: Combines performance optimizations with hierarchical policy architecture.

Training Approach:
1. Pre-train Worker policy with fixed goals (curriculum over goal values)
2. Pre-train Manager policy with rule-based Worker
3. Joint fine-tuning with Manager using trained Worker

This separates bid decisions (Manager) from card-play decisions (Worker),
providing clearer credit assignment and faster convergence.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from app.gym_env.skullking_env_hierarchical import (
    ManagerEnv,
    WorkerEnv,
    create_manager_env,
    create_worker_env,
    mask_fn_manager,
    mask_fn_worker,
)

console = Console()
logger = logging.getLogger(__name__)


# Training configurations
DEFAULT_CONFIG = {
    # V7 Performance settings
    "n_envs": 64,  # Parallel environments (reduced from 128 for hierarchical)
    "use_subproc": True,  # SubprocVecEnv for multi-core
    "batch_size": 2048,  # Larger batches for GPU
    "use_torch_compile": True,  # torch.compile optimization

    # Training steps
    "worker_pretrain_steps": 2_000_000,
    "manager_pretrain_steps": 1_000_000,
    "joint_finetune_steps": 3_000_000,

    # PPO hyperparameters
    "learning_rate": 3e-4,
    "n_epochs": 15,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "clip_range": 0.2,

    # Network architecture
    "policy_kwargs": {
        "net_arch": {"pi": [256, 256], "vf": [256, 256]},
        "activation_fn": torch.nn.ReLU,
    },

    # Paths
    "model_dir": "models/hierarchical",
    "tensorboard_dir": "models/hierarchical/tensorboard",
}


class WorkerCurriculumCallback(BaseCallback):
    """Curriculum over goal values for Worker pre-training.

    Starts with easier goals (0, high values) and gradually introduces harder goals.
    """

    def __init__(
        self,
        goal_schedule: list[tuple[int, list[int]]],
        verbose: int = 0,
    ):
        """Initialize curriculum.

        Args:
            goal_schedule: List of (timestep, allowed_goals) tuples
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.goal_schedule = sorted(goal_schedule, key=lambda x: x[0])
        self.current_goals = list(range(11))  # All goals initially

    def _on_step(self) -> bool:
        # Update goals based on schedule
        for timestep, goals in self.goal_schedule:
            if self.num_timesteps >= timestep:
                self.current_goals = goals

        # Update environment goal sampling
        # This would require environment modification to support
        return True


class HierarchicalProgressCallback(BaseCallback):
    """Progress callback with Rich console output."""

    def __init__(
        self,
        total_timesteps: int,
        phase_name: str = "Training",
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.phase_name = phase_name
        self.progress = None
        self.task = None

    def _on_training_start(self) -> None:
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn(f"[bold blue]{self.phase_name}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        )
        self.progress.start()
        self.task = self.progress.add_task(self.phase_name, total=self.total_timesteps)

    def _on_step(self) -> bool:
        if self.progress and self.task is not None:
            self.progress.update(self.task, completed=self.num_timesteps)
        return True

    def _on_training_end(self) -> None:
        if self.progress:
            self.progress.stop()


def make_worker_env(
    rank: int,
    opponent_type: str = "rule_based",
    difficulty: str = "medium",
    fixed_goal: int | None = None,
) -> callable:
    """Create a Worker environment factory."""
    def _init() -> ActionMasker:
        env = WorkerEnv(
            opponent_bot_type=opponent_type,
            opponent_difficulty=difficulty,
            fixed_goal=fixed_goal,
        )
        return ActionMasker(env, mask_fn_worker)
    return _init


def make_manager_env(
    rank: int,
    opponent_type: str = "rule_based",
    difficulty: str = "medium",
    worker_policy: Any = None,
) -> callable:
    """Create a Manager environment factory."""
    def _init() -> ActionMasker:
        env = ManagerEnv(
            worker_policy=worker_policy,
            opponent_bot_type=opponent_type,
            opponent_difficulty=difficulty,
        )
        return ActionMasker(env, mask_fn_manager)
    return _init


def create_vec_env(
    make_env_fn: callable,
    n_envs: int,
    use_subproc: bool = True,
    **env_kwargs,
) -> SubprocVecEnv | DummyVecEnv:
    """Create vectorized environment."""
    env_fns = [make_env_fn(i, **env_kwargs) for i in range(n_envs)]

    if use_subproc:
        return SubprocVecEnv(env_fns)
    return DummyVecEnv(env_fns)


def train_worker(config: dict) -> MaskablePPO:
    """Pre-train Worker policy with goal curriculum."""
    console.print("\n[bold green]Phase 1: Pre-training Worker Policy[/bold green]")
    console.print(f"Steps: {config['worker_pretrain_steps']:,}")

    # Create environments with curriculum
    # Start with fixed goals for stability
    vec_env = create_vec_env(
        make_worker_env,
        n_envs=config["n_envs"],
        use_subproc=config["use_subproc"],
        opponent_type="rule_based",
        difficulty="medium",
        fixed_goal=None,  # Random goals
    )

    # Create model
    model = MaskablePPO(
        "MlpPolicy",
        vec_env,
        learning_rate=config["learning_rate"],
        n_steps=2048,
        batch_size=config["batch_size"],
        n_epochs=config["n_epochs"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        ent_coef=config["ent_coef"],
        vf_coef=config["vf_coef"],
        max_grad_norm=config["max_grad_norm"],
        clip_range=config["clip_range"],
        policy_kwargs=config["policy_kwargs"],
        tensorboard_log=config["tensorboard_dir"],
        verbose=0,
    )

    # Compile for speed
    if config["use_torch_compile"] and hasattr(torch, "compile"):
        try:
            model.policy = torch.compile(model.policy, mode="reduce-overhead")
            console.print("[dim]torch.compile enabled[/dim]")
        except Exception as e:
            console.print(f"[yellow]torch.compile failed: {e}[/yellow]")

    # Callbacks
    callbacks = CallbackList([
        HierarchicalProgressCallback(
            config["worker_pretrain_steps"],
            phase_name="Worker Pre-training",
        ),
        CheckpointCallback(
            save_freq=500_000,
            save_path=config["model_dir"],
            name_prefix="worker_checkpoint",
        ),
    ])

    # Train
    model.learn(
        total_timesteps=config["worker_pretrain_steps"],
        callback=callbacks,
        tb_log_name="worker",
    )

    # Save
    worker_path = Path(config["model_dir"]) / "worker_pretrained.zip"
    model.save(worker_path)
    console.print(f"[green]Worker saved: {worker_path}[/green]")

    vec_env.close()
    return model


def train_manager(config: dict, worker_policy: MaskablePPO | None = None) -> MaskablePPO:
    """Pre-train Manager policy."""
    console.print("\n[bold green]Phase 2: Pre-training Manager Policy[/bold green]")
    console.print(f"Steps: {config['manager_pretrain_steps']:,}")

    # Create environments
    vec_env = create_vec_env(
        make_manager_env,
        n_envs=config["n_envs"],
        use_subproc=config["use_subproc"],
        opponent_type="rule_based",
        difficulty="medium",
        worker_policy=worker_policy,
    )

    # Create model
    model = MaskablePPO(
        "MlpPolicy",
        vec_env,
        learning_rate=config["learning_rate"],
        n_steps=512,  # Shorter episodes for Manager
        batch_size=min(config["batch_size"], 512),
        n_epochs=config["n_epochs"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        ent_coef=config["ent_coef"],
        vf_coef=config["vf_coef"],
        max_grad_norm=config["max_grad_norm"],
        clip_range=config["clip_range"],
        policy_kwargs=config["policy_kwargs"],
        tensorboard_log=config["tensorboard_dir"],
        verbose=0,
    )

    # Compile
    if config["use_torch_compile"] and hasattr(torch, "compile"):
        try:
            model.policy = torch.compile(model.policy, mode="reduce-overhead")
        except Exception:
            pass

    # Callbacks
    callbacks = CallbackList([
        HierarchicalProgressCallback(
            config["manager_pretrain_steps"],
            phase_name="Manager Pre-training",
        ),
        CheckpointCallback(
            save_freq=200_000,
            save_path=config["model_dir"],
            name_prefix="manager_checkpoint",
        ),
    ])

    # Train
    model.learn(
        total_timesteps=config["manager_pretrain_steps"],
        callback=callbacks,
        tb_log_name="manager",
    )

    # Save
    manager_path = Path(config["model_dir"]) / "manager_pretrained.zip"
    model.save(manager_path)
    console.print(f"[green]Manager saved: {manager_path}[/green]")

    vec_env.close()
    return model


def joint_finetune(
    config: dict,
    worker_model: MaskablePPO,
    manager_model: MaskablePPO,
) -> tuple[MaskablePPO, MaskablePPO]:
    """Joint fine-tuning of both policies."""
    console.print("\n[bold green]Phase 3: Joint Fine-tuning[/bold green]")
    console.print(f"Steps: {config['joint_finetune_steps']:,}")

    # Fine-tune Manager with trained Worker
    vec_env = create_vec_env(
        make_manager_env,
        n_envs=config["n_envs"],
        use_subproc=config["use_subproc"],
        opponent_type="rule_based",
        difficulty="hard",
        worker_policy=worker_model,
    )

    manager_model.set_env(vec_env)

    callbacks = CallbackList([
        HierarchicalProgressCallback(
            config["joint_finetune_steps"],
            phase_name="Joint Fine-tuning",
        ),
        CheckpointCallback(
            save_freq=500_000,
            save_path=config["model_dir"],
            name_prefix="joint_checkpoint",
        ),
    ])

    manager_model.learn(
        total_timesteps=config["joint_finetune_steps"],
        callback=callbacks,
        tb_log_name="joint",
        reset_num_timesteps=False,
    )

    # Save final models
    worker_path = Path(config["model_dir"]) / "worker_final.zip"
    manager_path = Path(config["model_dir"]) / "manager_final.zip"
    worker_model.save(worker_path)
    manager_model.save(manager_path)

    console.print(f"[green]Final Worker: {worker_path}[/green]")
    console.print(f"[green]Final Manager: {manager_path}[/green]")

    vec_env.close()
    return worker_model, manager_model


def train(config: dict | None = None) -> None:
    """Run full hierarchical training pipeline."""
    config = config or DEFAULT_CONFIG.copy()

    # Setup directories
    Path(config["model_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["tensorboard_dir"]).mkdir(parents=True, exist_ok=True)

    console.print("[bold]Skull King Hierarchical RL Training (V7+V8)[/bold]")
    console.print(f"Environments: {config['n_envs']}")
    console.print(f"Total steps: {config['worker_pretrain_steps'] + config['manager_pretrain_steps'] + config['joint_finetune_steps']:,}")

    # Phase 1: Pre-train Worker
    worker_model = train_worker(config)

    # Phase 2: Pre-train Manager (with trained Worker)
    manager_model = train_manager(config, worker_policy=worker_model)

    # Phase 3: Joint fine-tuning
    worker_final, manager_final = joint_finetune(config, worker_model, manager_model)

    console.print("\n[bold green]Training Complete![/bold green]")
    console.print(f"Worker: {config['model_dir']}/worker_final.zip")
    console.print(f"Manager: {config['model_dir']}/manager_final.zip")
    console.print(f"\nTensorboard: tensorboard --logdir {config['tensorboard_dir']}")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Hierarchical RL Training for Skull King")
    parser.add_argument(
        "--worker-steps",
        type=int,
        default=DEFAULT_CONFIG["worker_pretrain_steps"],
        help="Worker pre-training steps",
    )
    parser.add_argument(
        "--manager-steps",
        type=int,
        default=DEFAULT_CONFIG["manager_pretrain_steps"],
        help="Manager pre-training steps",
    )
    parser.add_argument(
        "--joint-steps",
        type=int,
        default=DEFAULT_CONFIG["joint_finetune_steps"],
        help="Joint fine-tuning steps",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=DEFAULT_CONFIG["n_envs"],
        help="Number of parallel environments",
    )
    parser.add_argument(
        "--no-subproc",
        action="store_true",
        help="Use DummyVecEnv instead of SubprocVecEnv",
    )

    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    config["worker_pretrain_steps"] = args.worker_steps
    config["manager_pretrain_steps"] = args.manager_steps
    config["joint_finetune_steps"] = args.joint_steps
    config["n_envs"] = args.n_envs
    config["use_subproc"] = not args.no_subproc

    train(config)


if __name__ == "__main__":
    main()
