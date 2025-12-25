# Skull King RL Training Log

Training history and results for the MaskablePPO agent.

**Hardware:** AMD Ryzen 9 7900X, 64GB DDR5, NVIDIA RTX 4080 SUPER (16GB)

---

## V6 (Planned) - Loot Alliances & Enhanced Observations

**Status:** Ready to Train

### Motivation

V5 model doesn't understand loot alliance mechanics (new feature: +20 bonus when both allied players make their bids). Need to add alliance observations so the agent can:
1. Recognize when it holds loot cards
2. Track active alliances (supports multiple alliances)
3. Consider alliance bonus in bidding/play strategy

### Changes from V5

**Observation Space: 182 → 190 dims (+8 dims)**

| New Observation | Dims | Description |
|-----------------|------|-------------|
| Has loot card | 1 | Binary: agent has loot in hand |
| Loot card count | 1 | Normalized count (0, 0.5, 1.0) |
| Alliance status | 4 | Binary mask: allied with players (multi-alliance support) |
| Ally bid accuracy | 1 | Average (tricks_won - bid) / round_num across all allies |
| Alliance potential | 1 | Sum of potential bonuses (0.2 per ally on track) |

**Reward Shaping Enhancements:**

1. **Alliance bonus in round reward** (CRITICAL FIX):
   - +2.0 per successful alliance at round end (normalized from +20)
   - Only awarded when BOTH agent and ally make their bids
   - Supports multiple alliances (could be +4.0 with 2 allies)

2. Round reward now: base bid accuracy (-5 to +5) + alliance bonus (0 to +4)

**Multi-Alliance Support:**

- A player can have multiple allies (e.g., winning tricks with loot cards from multiple players)
- Alliance status uses binary mask (not one-hot) to track all allies
- Ally accuracy averaged across all allies
- Alliance potential sums bonuses from all on-track allies

### Configuration

| Parameter | Value |
|-----------|-------|
| Timesteps | 10,000,000 |
| Observation dims | 190 |
| Parallel envs | 32 |
| Learning rate | 3e-4 |
| Network | [256, 256] |

### Expected Improvements

- Model understands loot card strategic value
- Learns alliance formation is valuable (+20 each when both win)
- Better card play to win tricks containing loot cards
- Supports multiple simultaneous alliances

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
