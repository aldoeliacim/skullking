# Skull King RL Training Log

Training history and results for the MaskablePPO agent.

**Hardware:** AMD Ryzen 9 7900X, 64GB DDR5, NVIDIA RTX 4080 SUPER (16GB)

---

## V6 (December 25, 2024) - Loot Alliances & Enhanced Observations

**Status:** Completed ✅

### Motivation

V5 model doesn't understand loot alliance mechanics (+20 bonus when both allied players make their bids). Added alliance observations so the agent can recognize and leverage loot card alliances.

### Changes from V5

**Observation Space: 182 → 190 dims (+8 dims)**

| New Observation | Dims | Description |
|-----------------|------|-------------|
| Has loot card | 1 | Binary: agent has loot in hand |
| Loot card count | 1 | Normalized count (0, 0.5, 1.0) |
| Alliance status | 4 | Binary mask: allied with players (multi-alliance) |
| Ally bid accuracy | 1 | Average (tricks_won - bid) / round_num across allies |
| Alliance potential | 1 | Sum of potential bonuses (0.2 per ally on track) |

**Reward Shaping Enhancements:**

- Alliance bonus: +2.0 per successful alliance at round end
- Multi-alliance support (could be +4.0 with 2 allies)

### Configuration

| Parameter | Value |
|-----------|-------|
| Timesteps | 10,092,544 |
| Training time | 2h 20m |
| Throughput | ~1,188 fps |
| Parallel envs | 32 (DummyVecEnv) |
| Learning rate | 3e-4 |
| Batch size | 1024 |
| Network | [256, 256] |

### Results

| Metric | Value |
|--------|-------|
| Final reward (mean) | 79.4 |
| Explained variance | 0.901 |
| Value loss | 23.9 |
| Entropy loss | -0.314 |
| Clip fraction | 0.0915 |
| KL divergence | 0.0113 |

**Evaluation (21 episodes, mixed opponents):**

| Timestep | Avg Reward | Std Dev |
|----------|------------|---------|
| 10.06M | 78.73 | ±8.22 |
| 10.07M | 80.92 | ±7.48 |
| 10.08M | 82.27 | ±12.47 |
| 10.09M | 84.81 | ±10.23 |
| Final | 81.35 | ±9.49 |

### Analysis

**Strengths:**
- Stable training (KL ~0.011, explained variance >0.9)
- Consistent with V5 performance (79-85 range)
- Alliance observations integrated successfully

**Observations:**
- Performance similar to V5 despite new observations
- Alliance situations may be too rare to significantly impact training
- Or +2.0 reward signal needs to be stronger

**Bottleneck Identified:**
- GPU utilization only 31% (starved for data)
- CPU-bound environment stepping (single-threaded DummyVecEnv)
- FPS ~1,188 could be 3-4x higher with SubprocVecEnv

### Model Files

- `models/masked_ppo/masked_ppo_final.zip` (2.8 MB)
- `models/masked_ppo/best_model/best_model.zip`

---

## V7 (December 25, 2024) - Performance Benchmarking

**Status:** Completed ✅

### Motivation

V6 training bottleneck analysis showed 31% GPU utilization with ~1,188 FPS. V7 systematically benchmarked configurations to find optimal hyperparameters for the RTX 4080 SUPER + Ryzen 9 7900X hardware.

### Methodology

Ran `app/training/benchmark_hierarchical.py` with various configurations:
- Tested n_envs: 256, 384, 512, 768
- Tested batch_size: 32768, 65536
- Fixed n_steps: 2048
- All tests used SubprocVecEnv for multi-core parallelism
- Each config ran 50,000 steps for stable FPS measurement

### Benchmark Results

| n_envs | batch_size | FPS | GPU % | GPU MB | Time (s) |
|--------|------------|-----|-------|--------|----------|
| **768** | **32768** | **6,836** | **50%** | 2,532 | 230.1 |
| 768 | 65536 | 6,614 | 36% | 3,082 | 237.8 |
| 512 | 32768 | 6,573 | 52% | 2,527 | 159.5 |
| 384 | 32768 | 6,478 | 49% | 2,529 | 121.4 |
| 512 | 65536 | 6,470 | 37% | 3,082 | 162.1 |
| 256 | 32768 | 6,299 | 43% | 2,521 | 83.2 |

### Optimal Configuration Found

| Parameter | V6 | V7 Optimal | Improvement |
|-----------|-----|------------|-------------|
| Vec env | DummyVecEnv | SubprocVecEnv | Multi-core |
| Parallel envs | 32 | **768** | 24x |
| Batch size | 1024 | **32768** | 32x |
| n_steps | 4096 | **2048** | Optimized |
| FPS | 1,188 | **6,836** | **5.8x faster** |
| GPU util | 31% | 50% | CPU-bound |
| GPU memory | ~1.5 GB | 2.5 GB | 16 GB available |

### Analysis

**Key Findings:**
- Training is **CPU-bound**, not GPU-bound (GPU only at 50%)
- 768 envs saturates the 24-thread Ryzen 9 7900X
- Larger batch (65536) doesn't help - adds overhead without benefit
- n_steps=2048 optimal for rollout/update balance
- SubprocVecEnv essential for multi-core parallelism (DummyVecEnv is single-threaded)

**Bottleneck Analysis:**
- Python GIL limits DummyVecEnv to single-core
- SubprocVecEnv spawns separate processes, bypassing GIL
- GPU waits for CPU to collect rollouts
- Further gains would require faster environment stepping (C++/Rust)

### Outcome

Optimal configuration applied to V8 training script (`app/training/train_ppo.py`).

---

## V8 (December 25, 2024) - Optimized Training at Scale

**Status:** In Progress 🔄

### Motivation

Apply V7 benchmark findings to train at maximum throughput. Extended to 50M timesteps (5x previous) to fully leverage the performance gains.

### Changes from V6

**Training Infrastructure (from V7 benchmarks):**

| Feature | V6 | V8 |
|---------|----|----|
| Vec env | DummyVecEnv | SubprocVecEnv |
| Parallel envs | 32 | 768 |
| Batch size | 1024 | 32768 |
| n_steps | 4096 | 2048 |
| Network | [256, 256] | [512, 512, 256] |
| torch.compile | ❌ | ✅ |
| Total timesteps | 10M | 50M |

**Network Architecture Upgrade:**
- Larger network [512, 512, 256] for increased capacity
- Separate pi/vf architectures for policy and value heads
- ReLU activation throughout

### Configuration

```bash
uv run python -m app.training.train_ppo train --timesteps 50000000
# Uses: 768 envs, batch 32768, n_steps 2048, SubprocVecEnv
```

### Final Results

**Status:** Completed (stopped at plateau) ✅

**Training duration:** 79.1 minutes (1.32 hours)
**Total timesteps:** 31,045,632
**Average FPS:** 6,539

| Metric | V6 Final | V8 Final | Comparison |
|--------|----------|----------|------------|
| ep_rew_mean | 79.4 | 80.6 | +1.5% |
| Best eval reward | 81.35 | **85.0** | **+4.5%** |
| Final eval reward | 81.35 | 80.6 | Similar |
| Explained variance | 0.901 | 0.900 | Same |
| Training time | 2h 20m | **1h 19m** | **44% faster** |
| FPS | 1,188 | **6,539** | **5.5x faster** |

**Evaluation Trajectory (21 episodes, mixed opponents):**

| Time | Timesteps | Reward | Δ/hour | Analysis |
|------|-----------|--------|--------|----------|
| 4m | 500K | 37.2 | - | Early learning |
| 12m | 2M | 68.1 | +450 | Rapid gains |
| 20m | 5M | 77.2 | +200 | Core strategy |
| 40m | 13M | 84.3 | +90 | Refinement |
| 67m | 26M | 81.4 | +42 | **Plateau** |
| 79m | 31M | 80.6 | +33 | Diminishing returns |

### Early Stopping Analysis

Training was stopped manually after detecting plateau:
- Reward range over last 10 evals: 79.8 - 85.0 (variance ~5)
- Δreward/hour dropped from +450 to +33
- No improvement from best (85.0) for 15+ evals

**Conclusion:** Further training yields diminishing returns. Need algorithmic improvements (Hierarchical RL) to break plateau.

### Self-Play Activity

Self-play activated at 2M steps, rotating through checkpoints:
- Prevents overfitting to fixed opponent strategies
- Loaded checkpoints at 5M, 8M, 12M, 16M, 20M, 22M steps

### Model Files

- `models/masked_ppo/checkpoints/` (every 100K steps)
- `models/masked_ppo/best_model/best_model.zip` (reward: 85.0)
- Training log: `training_v8.log`

---

## V9 (Planned) - Hierarchical RL + Performance Optimizations

**Status:** Implementation Phase 🔧

### Motivation

V8 plateaued at reward ~80-85 despite 5.5x faster training. Need algorithmic improvements to break through:

1. **Hierarchical RL**: Separate bidding and card-play policies
2. **Performance**: Numba/Cython for faster env stepping
3. **GPU utilization**: Larger networks to use more VRAM

### Architecture

```
Manager Policy (Bidding)          Worker Policy (Card Play)
├── Obs: Hand strength, position  ├── Obs: Current trick, bid goal
├── Action: Bid 0-10              ├── Action: Card to play
├── Horizon: 10 decisions/game    ├── Horizon: 1-10 per round
└── Reward: Round-end score       └── Reward: Trick-level shaping
```

### Expected Benefits

| Metric | V8 | V9 (Expected) |
|--------|-----|---------------|
| Max reward | 85 | 90+ |
| Sample efficiency | 1x | 2-3x |
| FPS | 6,500 | 15,000+ (with Numba) |
| GPU utilization | 50% | 70-80% |

### Implementation Blockers (Must Fix)

The hierarchical environment needs API refactoring:

```python
# Current issues in skullking_env_hierarchical.py:

# 1. Round.start_trick() doesn't exist
trick = current_round.start_trick()  # ❌ No such method

# Fix: Create Trick manually (like skullking_env_masked.py)
trick = Trick(number=trick_number, starter_player_index=starter_index)
current_round.tricks.append(trick)  # ✅
```

**Files to fix:**
- `app/gym_env/skullking_env_hierarchical.py` - 4 occurrences of `start_trick()`

### Performance Optimizations

See `V9_OPTIMIZATION_PLAN.md` for detailed plan:
- Numba JIT for observation encoding
- Cython for game logic hot paths
- Larger network [1024, 512, 256]
- EnvPool integration (long-term)

### References

- `ADVANCED_RL_TECHNIQUES.md` Section 2: Hierarchical RL design
- `V9_OPTIMIZATION_PLAN.md`: Performance optimization details
- `archive/training_v8_experimental/train_hierarchical.py`: Training script template

---

## V5 (December 25, 2024) - Completed

**Status:** Completed ✅

### Changes from V4

- Extended training: 5M → 10M timesteps
- Mixed opponent evaluation: 21 episodes across easy/medium/hard
- Self-play callback: activates at 2M steps, updates every 200k
- Observation space: 171 → 182 dims (round one-hot, bid goal)
- Action mask verification logging

### Configuration

| Parameter | Value |
|-----------|-------|
| Timesteps | 10,000,000 |
| Training time | 2h 24m |
| Throughput | ~1,155 fps |
| Parallel envs | 32 |
| Learning rate | 3e-4 |
| Batch size | 1024 |
| Network | [256, 256] |

### Results

| Metric | Value |
|--------|-------|
| Final reward (mean) | 79.9 - 81.8 |
| Explained variance | 0.902 - 0.904 |
| Value loss | 22.6 - 23.3 |
| Entropy loss | -0.316 to -0.32 |
| Clip fraction | 0.089 - 0.091 |

**Evaluation (21 episodes, mixed opponents):**

| Phase | Avg Reward | Std Dev |
|-------|------------|---------|
| Pre-selfplay (9.9M) | 78-83 | ±8-12 |
| Post-selfplay (10M+) | 77-85 | ±7-13 |

### Analysis

**Strengths:**
- Excellent value learning (explained variance >0.9)
- Stable training (low KL divergence ~0.011)
- Consistent performance across opponent types

**Observations:**
- Self-play activated at 10M with 4.6M checkpoint
- Reward variance ±8-13 is acceptable
- No overfitting to specific opponent types

**Limitations:**
- No awareness of loot alliance mechanics
- Cannot optimize for +20 alliance bonus

### Model Files

- `models/masked_ppo/masked_ppo_final.zip`
- `models/masked_ppo/best_model/best_model.zip`

---

## V4 (December 24, 2024)

**Status:** Completed

### Changes from V3

- Extended training: 1.5M → 5M timesteps
- Larger network: [256, 256]
- More parallel envs: 32
- Optimized batch size: 1024 for GPU utilization
- 8-phase curriculum schedule

### Configuration

| Parameter | Value |
|-----------|-------|
| Timesteps | 5,000,000 |
| Training time | ~45 min |
| Throughput | ~1,870 fps |
| Explained variance | 0.906 |

### Curriculum

| Steps | Opponent | Difficulty |
|-------|----------|------------|
| 0 | random | easy |
| 50k | random | medium |
| 150k | random | hard |
| 250k | rule_based | easy |
| 400k | rule_based | medium |
| 600k | rule_based | hard |
| 850k | rule_based | medium |
| 1.1M | rule_based | hard |

### Results

| Opponent | Avg Reward | Result |
|----------|------------|--------|
| Random (easy) | 65.8 ± 10.1 | WIN |
| Random (medium) | 61.2 ± 7.6 | WIN |
| Random (hard) | 67.7 ± 7.9 | WIN |
| Rule-based (easy) | 67.1 ± 9.8 | WIN |
| Rule-based (medium) | 60.2 ± 2.6 | WIN |
| Rule-based (hard) | 63.7 ± 2.8 | WIN |

### Model Files

- `models/masked_ppo/masked_ppo_final.zip` (2.7 MB)
- `models/masked_ppo/best_model/best_model.zip`

---

## V3 (December 2024)

**Status:** Completed

### Changes from V2

- Extended training to 1.5M steps
- Continued with normalized rewards

### Results

| Metric | Value |
|--------|-------|
| Training time | 38 min |
| Explained variance | 0.89 |
| Value loss | 27.1 |
| Eval reward | 226 |
| vs Rule-based MEDIUM | 100% win rate (20 games) |
| vs Rule-based HARD | 100% win rate (50 games) |

---

## V2 (December 2024)

**Status:** Completed

### Problem Solved

V1 had extreme reward variance (±792) causing unstable training.

**Root cause:** Reward scale mismatch

- Dense rewards: -0.5 to +3 per step
- Round penalties: -80 for bad bids (40x larger!)
- One bad round undid 40 good trick rewards

### Changes from V1

1. **Normalized rewards**
   - Round completion: -80/+20 → -5/+5
   - Game completion: -35/+80 → -5/+10

2. **Hyperparameter tuning**
   - n_epochs: 15 → 20
   - vf_coef: 0.5 → 1.0
   - gae_lambda: 0.98 → 0.99

### Results @ 50k steps

| Metric | V1 @ 150k | V2 @ 50k | Change |
|--------|-----------|----------|--------|
| Explained Variance | 0.158 | 0.443 | +180% |
| Value Loss | 973 | 43.7 | -94% |
| Eval Variance | ±391 | ±109 | -72% |
| Reward Variance | ±792 | ±13 | -98% |

---

## V1 (December 2024)

**Status:** Deprecated

### Initial Implementation

- MaskablePPO with action masking
- Dense reward shaping (bid quality, trick outcomes)
- 171-dim observation space
- Basic curriculum (random → rule-based)

### Problems Identified

- Extreme reward variance: ±792
- Poor value learning: explained variance 0.158
- Unstable training due to reward scale mismatch

### Results @ 150k steps

| Metric | Value |
|--------|-------|
| Training reward | 190 |
| Eval reward | 358 ± 391 |
| Explained variance | 0.158 |
| Value loss | 973 |

---

## Observations & Learnings

### Reward Shaping

- Dense rewards must be comparable in scale to sparse rewards
- Per-step rewards should be ~1/episode_length of final outcome
- Normalizing to [-5, +5] range works well

### Hyperparameters

- Higher vf_coef helps when value function struggles
- Larger batch sizes maximize GPU utilization (32768 optimal for RTX 4080 SUPER)
- Network capacity matters: [512, 512, 256] enables learning at 50M+ steps
- n_epochs=15 sufficient; more doesn't help significantly

### Performance Optimization (V7/V8 Lessons)

- **SubprocVecEnv is essential** - DummyVecEnv is single-threaded due to Python GIL
- **CPU is the bottleneck** - GPU sits at 50% waiting for rollout collection
- **Optimal env count = CPU threads** - 768 envs for 24-thread Ryzen 9 7900X
- **Batch size sweet spot exists** - 32768 optimal; 65536 adds overhead without benefit
- **torch.compile helps** - Reduces forward pass overhead
- **n_steps affects throughput** - 2048 better than 4096 for update frequency

### Scaling Insights

| Scale Factor | V6 | V8 | Impact |
|--------------|----|----|--------|
| Parallel envs | 32 | 768 | 24x more experience/step |
| Batch size | 1024 | 32768 | 32x larger gradient estimates |
| Total steps | 10M | 50M | 5x more training |
| FPS | 1,188 | 6,643 | 5.6x faster |
| Time for 10M | 2h20m | 25min | 5.6x faster |

### Curriculum

- Start with random opponents for basic mechanics
- Transition to rule-based for strategic play
- Self-play prevents overfitting to fixed strategies
- 8-phase curriculum covers full difficulty range

### Evaluation

- 5 episodes too few for stable metrics
- 21+ episodes recommended
- Test against multiple opponent types (easy/medium/hard)
- Eval every 500K steps is sufficient (more frequent slows training)

### Network Architecture

- Larger networks need more warmup time (explained variance starts low)
- Separate pi/vf heads allow independent capacity tuning
- [512, 512, 256] with ReLU works well for 50M+ step training
