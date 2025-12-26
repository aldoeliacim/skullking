"""Benchmark script for RL training throughput optimization.

Tests different configurations (n_envs, batch_size, vec_env type)
to find optimal FPS for the current hardware.

Usage:
    uv run python -m app.training.benchmark_training
"""

import gc
import time
from dataclasses import dataclass

import numpy as np
import torch
from rich.console import Console
from rich.table import Table
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from app.gym_env.skullking_env_masked import SkullKingEnvMasked


def mask_fn(env: SkullKingEnvMasked) -> np.ndarray:
    """Get action mask from environment."""
    return env.action_masks()


console = Console()


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""

    n_envs: int
    batch_size: int
    vec_env_type: str
    n_steps: int
    fps: float
    gpu_util: float
    gpu_mem_used: float
    cpu_util: float
    total_time: float
    steps_completed: int


def get_gpu_stats() -> tuple[float, float]:
    """Get GPU utilization and memory usage."""
    try:
        import subprocess

        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        util, mem = result.stdout.strip().split(", ")
        return float(util), float(mem)
    except Exception:
        return 0.0, 0.0


def make_env(rank: int) -> callable:
    """Create SkullKingEnvMasked environment factory."""

    def _init() -> ActionMasker:
        env = SkullKingEnvMasked(
            opponent_bot_type="rule_based",
            opponent_difficulty="medium",
        )
        return ActionMasker(env, mask_fn)

    return _init


def run_benchmark(
    n_envs: int,
    batch_size: int,
    use_subproc: bool,
    n_steps: int = 2048,
    total_steps: int = 100_000,  # More steps for stable measurements
    use_compile: bool = False,
) -> BenchmarkResult:
    """Run a single benchmark configuration."""

    # Create vectorized environment
    env_fns = [make_env(i) for i in range(n_envs)]

    if use_subproc:
        vec_env = SubprocVecEnv(env_fns)
        vec_env_type = "SubprocVecEnv"
    else:
        vec_env = DummyVecEnv(env_fns)
        vec_env_type = "DummyVecEnv"

    # Create model - larger network for 16GB VRAM
    model = MaskablePPO(
        "MlpPolicy",
        vec_env,
        learning_rate=3e-4,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        clip_range=0.2,
        policy_kwargs={
            # Larger network for more capacity
            "net_arch": {"pi": [512, 512, 256], "vf": [512, 512, 256]},
            "activation_fn": torch.nn.ReLU,
        },
        verbose=0,
        device="cuda",
    )

    # Optional torch.compile
    if use_compile and hasattr(torch, "compile"):
        try:
            model.policy = torch.compile(model.policy, mode="reduce-overhead")
        except Exception:
            pass

    # Warmup
    model.learn(total_timesteps=n_envs * n_steps, progress_bar=False)

    # Benchmark
    gpu_utils = []
    gpu_mems = []

    start_time = time.perf_counter()
    start_steps = model.num_timesteps

    # Sample GPU stats during training
    def sample_gpu():
        util, mem = get_gpu_stats()
        gpu_utils.append(util)
        gpu_mems.append(mem)

    # Run training with periodic GPU sampling
    remaining = total_steps
    while remaining > 0:
        chunk = min(remaining, n_envs * n_steps)
        model.learn(total_timesteps=chunk, progress_bar=False, reset_num_timesteps=False)
        remaining -= chunk
        sample_gpu()

    end_time = time.perf_counter()
    end_steps = model.num_timesteps

    elapsed = end_time - start_time
    steps_done = end_steps - start_steps
    fps = steps_done / elapsed

    avg_gpu_util = np.mean(gpu_utils) if gpu_utils else 0
    avg_gpu_mem = np.mean(gpu_mems) if gpu_mems else 0

    # Cleanup
    vec_env.close()
    del model
    gc.collect()
    torch.cuda.empty_cache()

    return BenchmarkResult(
        n_envs=n_envs,
        batch_size=batch_size,
        vec_env_type=vec_env_type,
        n_steps=n_steps,
        fps=fps,
        gpu_util=avg_gpu_util,
        gpu_mem_used=avg_gpu_mem,
        cpu_util=0,  # Would need psutil
        total_time=elapsed,
        steps_completed=steps_done,
    )


def main():
    """Run benchmark suite."""
    console.print("[bold]Hierarchical RL Training Benchmark[/bold]")
    console.print(f"GPU: {torch.cuda.get_device_name()}")
    console.print(f"CUDA: {torch.version.cuda}")
    console.print()

    # Configurations to test
    # Best from previous run: 256 envs, batch 32768, SubprocVecEnv = 6,009 FPS
    # Try pushing higher with more envs
    configs = [
        # (n_envs, batch_size, use_subproc, n_steps)
        # Previous best baseline
        (256, 32768, True, 2048),
        # Push more envs
        (384, 32768, True, 2048),
        (512, 32768, True, 2048),
        (512, 65536, True, 2048),
        (768, 32768, True, 2048),
        (768, 65536, True, 2048),
    ]

    results = []

    for n_envs, batch_size, use_subproc, n_steps in configs:
        config_str = f"envs={n_envs}, batch={batch_size}, {'Subproc' if use_subproc else 'Dummy'}, steps={n_steps}"
        console.print(f"[dim]Testing: {config_str}[/dim]")

        try:
            result = run_benchmark(
                n_envs=n_envs,
                batch_size=batch_size,
                use_subproc=use_subproc,
                n_steps=n_steps,
                total_steps=50_000,  # Quick measurement
            )
            results.append(result)
            console.print(f"  [green]FPS: {result.fps:,.0f}, GPU: {result.gpu_util:.0f}%[/green]")
        except Exception as e:
            console.print(f"  [red]Failed: {e}[/red]")

        # Brief pause between tests
        time.sleep(1)

    # Results table
    console.print()
    table = Table(title="Benchmark Results")
    table.add_column("Config", style="cyan")
    table.add_column("FPS", justify="right", style="green")
    table.add_column("GPU %", justify="right")
    table.add_column("GPU MB", justify="right")
    table.add_column("Time (s)", justify="right")

    # Sort by FPS
    results.sort(key=lambda r: r.fps, reverse=True)

    for r in results:
        config = f"{r.n_envs} envs, {r.batch_size} batch, {r.vec_env_type[:6]}, {r.n_steps} steps"
        table.add_row(
            config,
            f"{r.fps:,.0f}",
            f"{r.gpu_util:.0f}%",
            f"{r.gpu_mem_used:,.0f}",
            f"{r.total_time:.1f}",
        )

    console.print(table)

    # Best config
    if results:
        best = results[0]
        console.print()
        console.print("[bold green]Best Configuration:[/bold green]")
        console.print(f"  n_envs: {best.n_envs}")
        console.print(f"  batch_size: {best.batch_size}")
        console.print(f"  vec_env: {best.vec_env_type}")
        console.print(f"  n_steps: {best.n_steps}")
        console.print(f"  FPS: {best.fps:,.0f}")
        console.print(f"  GPU Utilization: {best.gpu_util:.0f}%")


if __name__ == "__main__":
    main()
