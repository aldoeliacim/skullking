# Skull King RL Episode Design

## Game Phase Analysis

### Phase Characteristics

| Phase | Rounds | Cards | Complexity | Strategy Focus |
|-------|--------|-------|------------|----------------|
| **Early** | 1-3 | 1-3 | Low | Special cards dominate, bids often 0-1 |
| **Mid** | 4-6 | 4-6 | Medium | Suit management, trump timing |
| **Late** | 7-10 | 7-10 | High | Multi-trick planning, position play |

### Decision Density

```
Game Total: 65 decisions (10 bids + 55 plays)

Round Distribution:
  Early (1-3):  3 bids +  6 plays = 9 decisions (14%)
  Mid   (4-6):  3 bids + 15 plays = 18 decisions (28%)
  Late  (7-10): 4 bids + 34 plays = 38 decisions (58%)
```

**Key insight: Late rounds contain 58% of all decisions but have the most complexity.**

---

## Current Problems

### 1. Credit Assignment Gap

Bidding suffers from **delayed reward**:

```
Round 10 timeline:
┌─────────┐                                    ┌─────────┐
│   BID   │ ──── 10 card plays ───────────────>│ REWARD  │
└─────────┘                                    └─────────┘
     │                                              │
     └── Up to 10 actions of delay ────────────────┘
```

Was the bad score due to:
- Wrong bid? (Hand evaluation error)
- Wrong play? (Execution error)
- Bad luck? (Opponent had counter)

### 2. PPO Epochs on Stale Data

```python
# Current approach
collect(256 * 1024 = 262K experiences)  # Fast (env stepping)
for epoch in range(20):                  # 20 passes
    gradient_update(same_262K_data)      # No new information!
```

For fast-stepping environments (20K steps/sec), this is wasteful:
- More epochs = more GPU work on **same data**
- Better: fewer epochs, more fresh data collection

### 3. Variable Episode Length

Worker episodes range from 1-10 steps:
- Round 1: 1 step (high variance)
- Round 10: 10 steps (low variance)

Short episodes dominate training → biased toward simple scenarios.

---

## Proposed Solutions

### Solution 1: Phase-Specific Training

Train separate or phase-conditioned policies:

```python
# Option A: Separate models per phase
EarlyRoundPolicy:  rounds 1-3  (simple decisions)
MidRoundPolicy:    rounds 4-6  (medium complexity)
LateRoundPolicy:   rounds 7-10 (full complexity)

# Option B: Single model with phase embedding
observation = [
    hand_encoding,
    round_onehot,      # Already have this
    phase_onehot,      # NEW: [early, mid, late]
    phase_complexity,  # NEW: normalized 0-1
]
```

### Solution 2: Round-Weighted Sampling

Weight training by round importance:

```python
# Round weights based on:
# 1. Decision count (late rounds have more)
# 2. Point impact (bid 0 in round 10 = ±100 pts)
# 3. Complexity (late rounds need more training)

ROUND_WEIGHTS = {
    1: 0.5, 2: 0.6, 3: 0.7,     # Early: less weight
    4: 0.8, 5: 0.9, 6: 1.0,     # Mid: medium weight
    7: 1.2, 8: 1.4, 9: 1.6, 10: 2.0  # Late: high weight
}

# Sample episodes proportionally
def sample_round():
    return random.choices(range(1, 11), weights=ROUND_WEIGHTS.values())[0]
```

### Solution 3: Hindsight Bid Relabeling

After each round, create synthetic "what-if" experiences:

```python
def hindsight_relabel(round_experience):
    """Create synthetic bid experiences from actual outcome."""
    actual_bid = round_experience.bid
    actual_tricks_won = round_experience.tricks_won

    synthetic_experiences = []

    # What if we had bid the correct amount?
    if actual_tricks_won != actual_bid:
        synthetic = copy(round_experience)
        synthetic.bid = actual_tricks_won  # "Correct" bid
        synthetic.reward = compute_reward(actual_tricks_won, actual_tricks_won)  # Success!
        synthetic_experiences.append(synthetic)

    # What if we had bid ±1?
    for delta in [-1, +1]:
        alt_bid = actual_bid + delta
        if 0 <= alt_bid <= round_experience.round_num:
            synthetic = copy(round_experience)
            synthetic.bid = alt_bid
            synthetic.reward = compute_reward(alt_bid, actual_tricks_won)
            synthetic_experiences.append(synthetic)

    return synthetic_experiences
```

**Benefits:**
- 3-4x more training signal per round
- Learns from "failures" as if they were successes
- Reduces bid exploration needed

### Solution 4: Adaptive Epochs by Phase

Different phases need different learning intensity:

```python
# Bidding (sparse reward)
MANAGER_EPOCHS = 25  # More epochs to extract signal

# Card Play (dense reward)
WORKER_EPOCHS = {
    'early': 8,   # Simple, fast learning
    'mid':   12,  # Medium complexity
    'late':  18,  # Complex, needs more updates
}

# Or: Adaptive based on reward variance
def adaptive_epochs(reward_variance):
    # High variance → more epochs to stabilize
    # Low variance → fewer epochs (already learned)
    base = 10
    return base + int(reward_variance * 20)
```

### Solution 5: Trick-Level Episodes (Alternative)

Instead of round = episode, use trick = episode:

```python
class TrickEnv(gym.Env):
    """Episode = one trick decision."""

    def __init__(self):
        # Observation: current trick + context
        self.observation_space = spaces.Box(...)
        # Action: card to play
        self.action_space = spaces.Discrete(11)

    def step(self, action):
        # Play card
        result = self.play_card(action)

        # Immediate reward based on trick outcome + goal progress
        reward = self._trick_reward(result)

        # Episode done after ONE action
        done = True

        return obs, reward, done, truncated, info
```

**Pros:**
- Fixed episode length (1 step)
- Immediate credit assignment
- No variance from round length

**Cons:**
- Loses multi-trick planning context
- May learn myopic strategy (win this trick vs. win the game)

---

## Recommended Architecture

### For V9 Training:

1. **Manager (Bidding)**
   - Episode: Full game (10 rounds)
   - See cumulative context (score, opponent bids)
   - Use hindsight relabeling
   - Epochs: 25 (sparse signal)

2. **Worker (Card Play)**
   - Episode: Single round
   - Fixed goal from Manager
   - Round-weighted sampling (favor late rounds)
   - Epochs: 10-15 (dense signal)

3. **Round Curriculum**
   ```python
   PHASE_CURRICULUM = [
       (0, 'late'),      # Start with complex (7-10)
       (1_000_000, 'mid'),    # Add medium (4-6)
       (2_000_000, 'all'),    # Full game
   ]
   ```

4. **Epoch Strategy**
   ```python
   # Manager: high epochs (sparse reward)
   manager_config = {'n_epochs': 25}

   # Worker: adaptive by round
   worker_config = {
       'n_epochs': 12,  # Base
       'late_round_multiplier': 1.5,  # More for rounds 7-10
   }
   ```

---

## Implementation Priority

| Priority | Change | Effort | Impact | Status |
|----------|--------|--------|--------|--------|
| 1 | Round-weighted sampling | Low | Medium | ✅ Done |
| 2 | Phase-specific epochs | Low | Medium | ✅ Done |
| 3 | Phase embedding (early/mid/late) | Low | Medium | ✅ Done |
| 4 | Late-round curriculum | Low | Medium | ✅ Done |
| 5 | Round stats tracking | Low | Medium | ✅ Done |
| 6 | Hindsight bid relabeling | Medium | High | 🔄 Callback ready |
| 7 | Trick-level episodes | High | Unknown | ❌ Future |

**Implemented in V9:**
- `ROUND_WEIGHTS`: Late rounds (7-10) sampled 4x more than early rounds
- `get_phase()`: Maps rounds to phases (0=early, 1=mid, 2=late)
- `PhaseSchedulerCallback`: Progressive phase unlocking during training
- `RoundStatsCallback`: Per-phase performance tracking
- Phase embedding: 3-dim one-hot in observations (Manager: 171 dims, Worker: 203 dims)

---

## Metrics to Track

1. **Per-phase win rate**: Do we perform differently in early/mid/late?
2. **Bid accuracy by round**: Is round 10 bidding worse than round 3?
3. **Credit assignment**: Does hindsight relabeling improve bid learning?
4. **Epoch efficiency**: Are more epochs actually helping or overfitting?
