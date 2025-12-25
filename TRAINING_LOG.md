# Skull King RL Training Log

Training history and results for the MaskablePPO agent.

**Hardware:** AMD Ryzen 9 7900X, 64GB DDR5, NVIDIA RTX 4080 SUPER (16GB)

---

## V5 (December 25, 2024) - In Progress

**Status:** Training (10M timesteps)

### Changes from V4

- Extended training: 5M → 10M timesteps
- Mixed opponent evaluation: 21 episodes across easy/medium/hard
- Self-play callback: activates at 2M steps, updates every 200k
- Action mask verification logging

### Configuration

| Parameter | Value |
|-----------|-------|
| Timesteps | 10,000,000 |
| Parallel envs | 32 |
| Learning rate | 3e-4 |
| Batch size | 1024 |
| Network | [256, 256] |

### Results

*Training in progress...*

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
- Larger batch sizes (1024) maximize GPU utilization
- [256, 256] network sufficient for this complexity

### Curriculum

- Start with random opponents for basic mechanics
- Transition to rule-based for strategic play
- Self-play prevents overfitting to fixed strategies

### Evaluation

- 5 episodes too few for stable metrics
- 21+ episodes recommended
- Test against multiple opponent types
