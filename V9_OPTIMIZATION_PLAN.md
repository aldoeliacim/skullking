# V9 Optimization Plan: Maximum Resource Utilization

## V8 Final Results

```
Total time:        79.1 minutes (1.32 hours)
Total timesteps:   31,045,632
Average FPS:       6,539
First eval reward: 37.18
Final eval reward: 80.62
Total reward gain: +43.44
Reward gain/hour:  +32.94 (down from +450 at start)
```

**Conclusion:** Training plateaued at reward ~80. Diminishing returns after 20 minutes.

---

## Current Resource Utilization

| Resource | Capacity | V8 Usage | Bottleneck? |
|----------|----------|----------|-------------|
| **CPU** | 24 threads (7900X) | 100% (768 envs) | **YES** |
| GPU Compute | RTX 4080 SUPER | 50% | Waiting for CPU |
| GPU VRAM | 16 GB | 2.5 GB (16%) | No |
| RAM | 64 GB | ~30 GB | No |

**Primary bottleneck: Python environment stepping speed**

---

## Optimization Categories

### 1. Environment Stepping (CRITICAL - 10-100x potential)

#### Option A: EnvPool (Recommended)

EnvPool is a C++ vectorized environment library achieving 1M+ FPS on Atari.

**Integration approach:**

```python
# envpool uses C++ backend with Python bindings
import envpool

# Option 1: Wrap existing env (limited speedup)
# Option 2: Native C++ implementation (maximum speedup)
```

**Implementation plan:**
1. Create `SkullKingEnvPool` C++ class
2. Implement game logic in C++ (cards, tricks, scoring)
3. Use EnvPool's batch API
4. Expected: 50,000+ FPS (10x current)

**Complexity:** High (C++ rewrite)
**Impact:** 10-100x faster stepping

#### Option B: Cython Compilation

Compile hot paths to C while keeping Python structure.

```python
# skullking_env_fast.pyx
cimport numpy as np

cdef class FastGameState:
    cdef int[:] hands
    cdef int current_player

    cpdef np.ndarray get_observation(self):
        # Compiled C code
        ...
```

**Hot paths to compile:**
- `_get_obs()` - observation encoding (called every step)
- `_calculate_reward()` - reward computation
- `Game.play_card()` - game state updates
- `RuleBasedBot.choose_action()` - opponent decisions

**Complexity:** Medium
**Impact:** 5-10x faster stepping

#### Option C: Numba JIT

JIT compile numpy operations without code changes.

```python
from numba import jit, njit

@njit
def encode_hand(cards: np.ndarray) -> np.ndarray:
    # Compiled at first call
    result = np.zeros(74, dtype=np.float32)
    for card_id in cards:
        result[card_id] = 1.0
    return result
```

**Best for:**
- Observation encoding (numpy heavy)
- Reward shaping calculations
- Action mask computation

**Complexity:** Low
**Impact:** 2-5x for compiled functions

#### Option D: Rust Environment (via PyO3)

```rust
// Fast game logic in Rust
use pyo3::prelude::*;

#[pyclass]
struct SkullKingEnv {
    game_state: GameState,
}

#[pymethods]
impl SkullKingEnv {
    fn step(&mut self, action: i32) -> (Vec<f32>, f32, bool) {
        // ~100x faster than Python
    }
}
```

**Complexity:** High
**Impact:** 50-100x faster stepping

---

### 2. GPU Utilization (Currently 50%)

#### A. Larger Network Architecture

```python
# Current (V8): 2.5 GB VRAM, 50% GPU
policy_kwargs = {
    "net_arch": {"pi": [512, 512, 256], "vf": [512, 512, 256]},
}

# V9 Option 1: Wider layers (4 GB VRAM)
policy_kwargs = {
    "net_arch": {"pi": [1024, 1024, 512], "vf": [1024, 1024, 512]},
}

# V9 Option 2: Deeper network (5 GB VRAM)
policy_kwargs = {
    "net_arch": {"pi": [512, 512, 512, 256, 256], "vf": [512, 512, 512, 256, 256]},
}

# V9 Option 3: Transformer-style attention (8+ GB VRAM)
# Custom policy class with self-attention over cards
```

#### B. Mixed Precision Training (FP16)

```python
import torch

# Enable automatic mixed precision
scaler = torch.cuda.amp.GradScaler()

with torch.cuda.amp.autocast():
    # Forward pass in FP16
    action, value = policy(obs)
    loss = compute_loss(...)

# Backward pass with scaling
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**Benefits:**
- 2x memory efficiency
- Faster on Tensor Cores
- Can double batch size

**Note:** SB3 doesn't natively support; requires custom policy.

#### C. More Training Epochs

```python
# Current
n_epochs = 15

# V9: More gradient updates per rollout
n_epochs = 30  # or even 50

# Trade-off: More GPU compute, same data collection speed
```

#### D. Parallel Model Training

Train multiple models simultaneously on one GPU:

```python
# Train 4 models with different seeds
models = [
    MaskablePPO(..., seed=i, device="cuda")
    for i in range(4)
]

# Each uses 4GB VRAM (16GB total)
# Different hyperparameters or architectures
```

---

### 3. Memory Optimization

#### A. More Parallel Environments

```python
# Current
n_envs = 768  # Saturates 24 threads

# With faster env stepping (EnvPool/Cython):
n_envs = 2048  # or more
# RAM: ~80MB per env = 160GB needed (exceeds 64GB)
# Need to optimize env memory footprint
```

#### B. Pinned Memory for Faster Transfers

```python
# Pin tensors for faster CPU→GPU transfer
obs_buffer = torch.zeros(batch_size, obs_dim, pin_memory=True)

# Non-blocking transfer
obs_gpu = obs_buffer.to("cuda", non_blocking=True)
```

#### C. Memory-Mapped Checkpoints

```python
import torch

# Async checkpoint saving (don't block training)
def save_async(model, path):
    state_dict = model.state_dict()
    torch.save(state_dict, path)  # In background thread
```

---

### 4. Algorithm Efficiency (V9 Hierarchical RL)

#### A. Manager/Worker Architecture

```
Manager Policy (Bidding)          Worker Policy (Card Play)
├── Obs: Hand strength, position  ├── Obs: Current trick, bid goal
├── Action: Bid 0-10              ├── Action: Card to play
├── Horizon: 10 decisions/game    ├── Horizon: 1-10 per round
└── Reward: Round-end score       └── Reward: Trick-level shaping
```

**Expected benefits:**
- 2-3x sample efficiency (shorter horizons)
- Clearer credit assignment
- Interpretable bid/play strategies

#### B. Hindsight Goal Relabeling

```python
# After a round, relabel with actual outcome
def relabel_experience(trajectory, actual_tricks_won):
    for transition in trajectory:
        # Replace bid goal with what actually happened
        transition.obs["bid_goal"] = actual_tricks_won
        # Recompute reward as if that was the plan
        transition.reward = compute_reward(actual_tricks_won, actual_tricks_won)
```

**Benefit:** Learn from "failures" as if they were planned.

---

### 5. Evaluation Efficiency

#### A. Reduce Eval Frequency

```python
# Current: Every 500K steps
eval_freq = 500_000

# V9: Every 1M steps (since we have early stopping)
eval_freq = 1_000_000
```

#### B. Parallel Evaluation

```python
# Eval in separate process while training continues
from multiprocessing import Process

def eval_worker(model_path, result_queue):
    model = load(model_path)
    reward = evaluate(model)
    result_queue.put(reward)

# Non-blocking eval
Process(target=eval_worker, args=(path, queue)).start()
```

---

## V9 Implementation Priority

| Priority | Optimization | Impact | Effort | Dependencies |
|----------|--------------|--------|--------|--------------|
| **1** | Hierarchical RL | Break plateau | Medium | Fix env API |
| **2** | Cython hot paths | 5-10x FPS | Medium | None |
| **3** | Larger network | Use GPU | Low | None |
| **4** | Mixed precision | 2x memory | Medium | Custom policy |
| **5** | EnvPool native | 50x+ FPS | High | C++ rewrite |
| **6** | Numba JIT | 2-5x FPS | Low | None |

---

## Quick Wins for V9

### 1. Numba-accelerated observation encoding

```python
from numba import njit
import numpy as np

@njit(cache=True)
def encode_hand_fast(card_ids: np.ndarray, output: np.ndarray) -> None:
    """Encode hand as one-hot vector (compiled)."""
    output[:] = 0
    for i in range(len(card_ids)):
        if card_ids[i] >= 0:
            output[card_ids[i]] = 1.0

@njit(cache=True)
def encode_played_cards_fast(played: np.ndarray, output: np.ndarray) -> None:
    """Encode played cards (compiled)."""
    output[:] = 0
    for i in range(len(played)):
        if played[i] >= 0:
            output[played[i]] = 1.0
```

### 2. Cython game logic skeleton

```python
# setup.py
from Cython.Build import cythonize

setup(
    ext_modules=cythonize([
        "app/game_fast.pyx",
        "app/observation_fast.pyx",
    ])
)
```

### 3. Immediate config changes

```python
# V9 training config
DEFAULT_N_ENVS = 768
DEFAULT_BATCH_SIZE = 32768
DEFAULT_N_STEPS = 2048
DEFAULT_N_EPOCHS = 30  # Increased from 15

# Larger network
policy_kwargs = {
    "net_arch": {"pi": [1024, 512, 256], "vf": [1024, 512, 256]},
}
```

---

## Measurement Plan

### Metrics to Track

1. **FPS** - Environment steps per second
2. **GPU Utilization %** - Target: 80%+
3. **Δreward/hour** - Training efficiency
4. **Time to reward=X** - How fast to reach milestones
5. **Sample efficiency** - Reward per 1M steps

### Benchmark Protocol

```python
def benchmark_v9():
    configs = [
        ("baseline", {}),
        ("numba", {"use_numba": True}),
        ("cython", {"use_cython": True}),
        ("large_net", {"net_arch": [1024, 512, 256]}),
        ("hierarchical", {"use_hierarchical": True}),
    ]

    for name, config in configs:
        fps, gpu_util, reward_rate = run_benchmark(config)
        log(f"{name}: FPS={fps}, GPU={gpu_util}%, Δ/hour={reward_rate}")
```

---

## Files to Create/Modify

1. `app/gym_env/skullking_env_hierarchical.py` - Fix API issues
2. `app/gym_env/observation_fast.pyx` - Cython observation encoding
3. `app/training/train_hierarchical.py` - V9 training script
4. `app/training/benchmark_v9.py` - V9 benchmarking
5. `requirements-dev.txt` - Add Cython, Numba

---

## Expected V9 Results

| Metric | V8 | V9 (Projected) |
|--------|-----|----------------|
| FPS | 6,500 | 15,000-30,000 |
| GPU Util | 50% | 70-80% |
| Time to reward=80 | 20 min | 5-10 min |
| Max reward | 80-85 | 90+ (hierarchical) |
| Sample efficiency | 1x | 2-3x |

---

## Implementation Status

### Completed ✅

1. [x] Fix hierarchical env API (Game, Player constructors)
2. [x] Implement Numba-accelerated observation encoding
3. [x] Create V9 training script with larger network
4. [x] Benchmark hierarchical env speedup (2.8x faster)
5. [x] Implement Manager/Worker hierarchical training
6. [x] **Episode Design Optimizations:**
   - [x] Round-weighted sampling (late rounds 4x more likely)
   - [x] Phase curriculum callback (late → mid → all)
   - [x] Phase embedding in observations (+3 dims)
   - [x] Phase-specific epochs (Manager: 25, Worker: 12)
   - [x] Round stats tracking callback

### Pending

7. [ ] If significant improvement: proceed with Cython
8. [ ] Long-term: Evaluate EnvPool C++ rewrite ROI

---

## Episode Design Optimizations (New)

Based on `EPISODE_DESIGN.md` analysis:

### Game Flow Analysis

| Phase | Rounds | Decisions | Complexity |
|-------|--------|-----------|------------|
| Early | 1-3 | 14% | Low - special cards dominate |
| Mid | 4-6 | 28% | Medium - suit management |
| Late | 7-10 | 58% | High - multi-trick planning |

### Implemented Solutions

**1. Round-Weighted Sampling**
```python
ROUND_WEIGHTS = {
    1: 0.5, 2: 0.6, 3: 0.7,     # Early: less weight
    4: 0.8, 5: 0.9, 6: 1.0,     # Mid: medium weight
    7: 1.2, 8: 1.4, 9: 1.6, 10: 2.0  # Late: 4x early
}
```

**2. Phase Curriculum**
```python
PHASE_SCHEDULE = [
    (0, (2,)),           # Start: Late only
    (1_000_000, (1, 2)), # 1M: Mid + Late
    (2_000_000, (0, 1, 2)), # 2M: All rounds
]
```

**3. Phase Embedding**
- 3-dim one-hot: (early, mid, late)
- ManagerEnv: 168 → 171 dims
- WorkerEnv: 200 → 203 dims

**4. Phase-Specific Epochs**
```python
MANAGER_N_EPOCHS = 25  # Sparse reward (bid → round-end)
WORKER_N_EPOCHS = 12   # Dense reward (trick-level shaping)
```

**5. New Callbacks**
- `PhaseSchedulerCallback`: Progressive phase unlocking
- `RoundStatsCallback`: Per-phase performance tracking
