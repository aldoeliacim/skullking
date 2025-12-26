# V9 Actual Benchmarks (December 26, 2024)

## Previous Benchmarks Were Wrong

The "2.8x faster" hierarchical env claim was based on a **broken Worker environment**:
- Worker env terminated immediately (episode length = 1)
- Cards were not being dealt (`deal_cards()` never called)
- Agent hand was empty, so episodes returned 0 reward instantly

## Corrected Benchmark Results

### Environment Stepping (raw, no training)

| Environment | Before Fix | After Fix | After Optimization |
|-------------|------------|-----------|-------------------|
| ManagerEnv | ~11,000 FPS | ~11,000 FPS | ~10,500 FPS |
| WorkerEnv | ~5,500 FPS* | ~5,500 FPS | **~17,500 FPS** |

*Was incorrectly measured due to broken episodes

### Training FPS (with gradient updates)

| Config | Before Optimization | After Optimization |
|--------|--------------------|--------------------|
| 256 envs, [1024,1024,512] | ~3,000 FPS | **~9,600 FPS** |
| 128 envs, [512,512,256] | ~2,600 FPS | ~7,000 FPS |

### GPU Utilization

| Metric | Before | After |
|--------|--------|-------|
| GPU Compute | 58% | **82%** |
| GPU Memory | 2.7 GB | 3.3 GB |
| Available | 16 GB | 12.7 GB unused |

---

## Critical Bottleneck Found

### The Problem: `reset()` Simulated Past Rounds

```python
# OLD: For round 10, simulate rounds 1-9 (45 card plays!)
for r in range(1, self.current_round_num):
    self._simulate_full_round(r)  # Full bot decisions + trick resolution
```

**Profile results:**
- `reset()`: 77% of total time (0.938s / 1.217s)
- `_simulate_full_round()`: 0.865s per 2000 steps

### The Fix: Skip Past Round Simulation

```python
# NEW: Jump directly to target round
self.game.current_round_number = self.current_round_num - 1
self.game.start_new_round()
self.game.deal_cards()
```

**Why this is safe:**
1. Worker observations don't include game history
2. Single-round training doesn't need cumulative scores
3. Bot decisions don't depend on past round outcomes

**Impact:**
- `reset()`: 0.938s → 0.073s (**12.8x faster**)
- Overall: 1.217s → 0.389s (**3.1x faster**)

---

## Current Resource Utilization

| Resource | Capacity | Current Usage | Bottleneck? |
|----------|----------|---------------|-------------|
| **CPU** | 24 threads | ~60% (256 envs) | Partially |
| **GPU Compute** | RTX 4080 SUPER | 82% | **Almost optimal** |
| **GPU VRAM** | 16 GB | 3.3 GB (21%) | No |
| **RAM** | 64 GB | ~8 GB | No |

### Remaining Bottlenecks (in order)

1. **Python GIL** - DummyVecEnv runs sequentially
2. **Observation encoding** - `_get_worker_obs()` 27% of step time
3. **Bot decision logic** - `RuleBasedBot.pick_card()` 20% of step time
4. **Trick determination** - `determine_winner()` 15% of step time

---

## Recommendations for Maximum Throughput

### Immediate (No Code Changes)

| Change | Expected Impact | Effort |
|--------|-----------------|--------|
| Increase n_envs to 512+ | +20% FPS | Config |
| Increase batch_size to 65536 | +10% GPU util | Config |
| Increase n_epochs to 25 | Better learning/step | Config |
| Use SubprocVecEnv | +50% FPS (multicore) | Config |

### Short-term (Python Optimizations)

| Change | Expected Impact | Effort |
|--------|-----------------|--------|
| Numba-compile observation encoding | 2-3x for that function | Low |
| Simplify bot logic for training | 1.5x for pick_card | Low |
| Pre-allocate numpy arrays | 1.2x for obs | Low |

### Long-term (Architecture Changes)

| Change | Expected Impact | Effort |
|--------|-----------------|--------|
| Cython game logic | 5-10x env stepping | Medium |
| Rust/PyO3 environment | 20-50x env stepping | High |
| EnvPool C++ implementation | 50-100x stepping | High |

---

## Optimal Configuration (Current)

```python
# Recommended V9 Worker training config
n_envs = 512          # Increase from 256
batch_size = 65536    # Increase from 32768
n_steps = 512         # Keep
n_epochs = 20         # Increase from 15
net_arch = [1024, 1024, 512]  # Good GPU utilization

# Expected: ~10,000-12,000 FPS with 85%+ GPU
```

---

## Episode Design Audit

### Current Design

**Manager (Bidding):**
- Episode = Full game (10 rounds)
- 10 decisions per episode
- Sparse reward (round-end only)

**Worker (Card Play):**
- Episode = Single round
- 1-10 decisions per episode (varies)
- Dense reward (trick-level)

### Problems Identified

1. **Episode length variance** (Worker):
   - Round 1: 1 step
   - Round 10: 10 steps
   - Short episodes dominate due to faster resets

2. **Phase curriculum ineffective**:
   - RoundStats shows all 0 counts for phases
   - Phase embedding not being utilized

3. **Credit assignment still hard** (Manager):
   - 10 card plays between bid and reward
   - Hindsight relabeling not implemented

### Recommendations

1. **Use uniform episode lengths**:
   ```python
   # Instead of random rounds, weight by episode length
   # This ensures equal training signal per step
   ```

2. **Implement hindsight relabeling**:
   ```python
   # After each round, create synthetic "correct bid" experience
   # 3-4x more training signal per episode
   ```

3. **Consider trick-level episodes** for Worker:
   ```python
   # Episode = 1 trick (1 decision)
   # Pro: Consistent length, immediate credit
   # Con: Loses multi-trick context
   ```

---

## Deep Episode Design Analysis

### Game State Flow

```
FULL GAME (55-65 agent decisions):
┌─────────────────────────────────────────────────────────────────┐
│ Round 1                                                         │
│ ┌─────┐ ┌─────────────────────────────────────────────────────┐ │
│ │ BID │→│ Trick 1 (1 card each)                               │ │
│ └─────┘ └─────────────────────────────────────────────────────┘ │
│                                                                 │
│ Round 2                                                         │
│ ┌─────┐ ┌─────────────────────────────────────────────────────┐ │
│ │ BID │→│ Trick 1 │ Trick 2                                   │ │
│ └─────┘ └─────────────────────────────────────────────────────┘ │
│                                                                 │
│ ...                                                             │
│                                                                 │
│ Round 10                                                        │
│ ┌─────┐ ┌─────────────────────────────────────────────────────┐ │
│ │ BID │→│ T1 │ T2 │ T3 │ T4 │ T5 │ T6 │ T7 │ T8 │ T9 │ T10   │ │
│ └─────┘ └─────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

### Current Episode Boundaries

**Manager (ManagerEnv):**
```
Episode = Full 10-round game
├── Decision 1: Bid for Round 1
├── (Worker plays Round 1)
├── Decision 2: Bid for Round 2
├── (Worker plays Round 2)
├── ...
└── Decision 10: Bid for Round 10

Reward: Cumulative score at game end
Problem: 10 bids → 1 reward (sparse, hard to credit assign)
```

**Worker (WorkerEnv):**
```
Episode = Single round (1-10 tricks)
├── Start with hand + bid goal
├── Decision 1: Play card for Trick 1
├── Decision 2: Play card for Trick 2
├── ...
└── Decision N: Play card for Trick N

Reward: Trick-level shaping + round-end bonus
Problem: Variable length (1-10 decisions per episode)
```

### Learning Signal Analysis

| Episode Type | Decisions | Reward Sparsity | Credit Assignment |
|--------------|-----------|-----------------|-------------------|
| Full Game | 55-65 | Very sparse (1 reward) | Very hard |
| Per-Round | 1-10 | Medium (N+1 rewards) | Medium |
| Per-Trick | 1 | Dense (1 per decision) | Easy |

### The Core Problem: Episode Length Variance

With round-weighted sampling favoring late rounds (7-10):

```
Round  | Decisions | Weight | Effective Training Steps
-------|-----------|--------|-------------------------
   1   |     1     |  0.5   |  0.5 (underweighted)
   2   |     2     |  0.6   |  1.2
   3   |     3     |  0.7   |  2.1
   7   |     7     |  1.2   |  8.4
   8   |     8     |  1.4   | 11.2
   9   |     9     |  1.6   | 14.4
  10   |    10     |  2.0   | 20.0 (heavily weighted)
```

**This seems good, but...**

The PPO buffer collects N steps regardless of episode boundaries:
- If buffer = 512 steps, round 10 gives ~50 complete episodes
- Round 1 gives ~500 complete episodes
- Short episodes = more episode boundaries = more variance in returns

### Proposed Solutions

#### Option A: Step-Weighted Sampling (Not Episode-Weighted)

```python
# Weight by decision count, not episode count
# Each step should have equal probability of being from any round
STEP_WEIGHTS = {r: r for r in range(1, 11)}  # Linear with round
# Round 10 = 10x more likely than Round 1
```

#### Option B: Fixed-Step Episodes

```python
# Always collect exactly K decisions per episode
# Pad short rounds, truncate long rounds
class FixedStepWorkerEnv:
    def __init__(self, steps_per_episode=8):
        self.target_steps = steps_per_episode

    def step(self, action):
        # ... play trick ...
        if self.step_count >= self.target_steps:
            truncated = True  # Force end
```

#### Option C: Trick-Level Episodes (Maximum Credit Assignment)

```python
class TrickEnv:
    """One decision per episode = perfect credit assignment."""

    def step(self, action):
        result = self._play_trick(action)
        reward = self._compute_trick_reward(result)
        done = True  # Always end after 1 decision
        return obs, reward, done, False, info
```

**Pros:**
- Perfect credit assignment (action → immediate reward)
- Consistent episode length (always 1)
- No variance from episode boundaries

**Cons:**
- Loses multi-trick planning context
- May learn myopic strategies (win this trick vs. achieve bid)

#### Option D: Hierarchical with Trick-Level (Recommended)

```
┌─────────────────────────────────────────┐
│ Meta-Controller (Round-Level)           │
│ - Observes: Hand, bid, tricks_won       │
│ - Decides: Strategy for next N tricks   │
│   (aggressive/defensive/balanced)       │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│ Trick-Controller (Trick-Level)          │
│ - Observes: Current trick + strategy    │
│ - Decides: Which card to play           │
│ - Reward: Immediate (trick outcome)     │
└─────────────────────────────────────────┘
```

### Recommendation

**For V9 immediate improvement:**
1. Keep current round-level Worker episodes
2. Implement step-weighted sampling (weight = round number)
3. This naturally balances training signal per step

**For V10+ if plateau persists:**
1. Implement TrickEnv for maximum credit assignment
2. Use attention mechanism to maintain multi-trick context
3. Potentially add meta-controller for strategy selection
