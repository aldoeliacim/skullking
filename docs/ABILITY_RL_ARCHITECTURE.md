# Ability-Aware RL Architecture for Skull King

## Executive Summary

This document outlines a comprehensive architecture for training an RL agent that can make strategic decisions about pirate abilities in Skull King. The key insight is that abilities are **not just modifiers to game state** — they are **strategic decision points** that significantly impact game outcomes.

---

## 1. The Challenge

### Current System Limitations

The current RL environment (`SkullKingEnvMasked`) only handles two decision types:
- **Bidding**: Actions 0-10 (predict tricks won)
- **Card Play**: Actions 0-9 (select card from hand)

Pirate abilities are resolved with hardcoded heuristics in `game_handler.py`:

| Pirate | Ability | Current Bot Logic | Quality |
|--------|---------|------------------|---------|
| Rosie | Choose starter | Always self | ❌ No strategy |
| Bendt | Draw 2, discard 2 | Discard first 2 | ❌ Random |
| Roatán | Bet 0/10/20 | Always bet 10 | ⚠️ Fixed |
| Jade | View deck | Ignore info | ❌ Wasted |
| Harry | Modify bid ±1 | Match tricks | ✅ Smart |

### Why Abilities Matter

1. **Rosie**: Choosing who leads can determine trick outcomes
2. **Bendt**: Optimal discard improves hand quality for remaining tricks
3. **Roatán**: Risk-reward tradeoff based on bid confidence
4. **Jade**: Information about undealt cards affects probability estimates
5. **Harry**: Last chance to hit bid = critical decision

---

## 2. Proposed Architecture

### 2.1 Multi-Phase State Machine

Instead of a flat action space, we model the game as a **state machine with decision phases**:

```
┌────────────┐     ┌────────────┐     ┌────────────────┐
│  BIDDING   │────▶│  PLAYING   │────▶│  ABILITY_*     │
└────────────┘     └────────────┘     └────────────────┘
                         │                    │
                         │◀───────────────────┘
                         │
                         ▼
                   ┌────────────┐
                   │ ROUND_END  │ (Harry check)
                   └────────────┘
```

**Decision Phases:**
```python
class DecisionPhase(Enum):
    BIDDING = 0
    PLAYING = 1
    ABILITY_ROSIE = 2      # Choose starter
    ABILITY_BENDT_1 = 3    # Select first discard
    ABILITY_BENDT_2 = 4    # Select second discard
    ABILITY_ROATAN = 5     # Extra bet
    ABILITY_JADE = 6       # Acknowledge (no real decision)
    ABILITY_HARRY = 7      # Modify bid
```

### 2.2 Observation Space Design

**Base Observations (existing, ~190 dims):**
- Hand encoding (one-hot per card type)
- Trick state (cards played, lead suit)
- Player state (bids, tricks won, scores)
- Round/game progress

**New: Phase Context (~40 additional dims):**

```python
phase_observations = {
    # Phase indicator (8 dims, one-hot)
    "decision_phase": one_hot(current_phase, 8),

    # Ability-specific context
    "ability_type": one_hot(current_ability, 6),  # none, rosie, bendt, roatan, jade, harry
    "ability_player_id": one_hot(triggering_player, 6),

    # For Rosie: player selection context
    "player_lead_strength": [strength_estimate(p) for p in players],  # 6 floats

    # For Bendt: discard context
    "drawn_cards": card_encoding(drawn_cards),  # 2 * card_dims
    "bendt_phase": 0 or 1,  # first or second discard
    "first_discard": card_encoding(selected_card) if phase_2 else zeros,

    # For Roatán: risk assessment
    "bid_confidence": (tricks_won - bid) / round_number,  # -1 to 1
    "tricks_remaining": remaining / round_number,

    # For Harry: bid modification context
    "current_bid": bid / 10,
    "tricks_won": tricks_won / 10,
    "bid_difference": (tricks_won - bid) / 10,  # +1 = over, -1 = under
}
```

**Total Observation Space: ~230 dimensions**

### 2.3 Multi-Head Action Architecture

Instead of one large action space, use **multiple specialized output heads**:

```
                    ┌─────────────────────────────────┐
                    │         Shared Encoder          │
                    │   (MLP: 230 → 256 → 256)       │
                    └─────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │               │           │           │               │
        ▼               ▼           ▼           ▼               ▼
   ┌─────────┐    ┌─────────┐ ┌─────────┐ ┌─────────┐    ┌─────────┐
   │head_bid │    │head_card│ │head_rosi│ │head_bend│    │head_harr│
   │ (11)    │    │  (13)   │ │  (6)    │ │  (13)   │    │  (3)    │
   └─────────┘    └─────────┘ └─────────┘ └─────────┘    └─────────┘
```

**PyTorch Implementation:**

```python
class MultiHeadPolicy(nn.Module):
    def __init__(self, obs_dim=230, hidden_dim=256):
        super().__init__()

        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Decision heads
        self.heads = nn.ModuleDict({
            'bid': nn.Linear(hidden_dim, 11),      # Bid 0-10
            'card': nn.Linear(hidden_dim, 13),     # Hand indices
            'rosie': nn.Linear(hidden_dim, 6),     # Player indices
            'bendt': nn.Linear(hidden_dim, 13),    # Card indices (discard)
            'roatan': nn.Linear(hidden_dim, 3),    # 0/10/20
            'harry': nn.Linear(hidden_dim, 3),     # -1/0/+1
        })

        # Value head
        self.value = nn.Linear(hidden_dim, 1)

    def forward(self, obs, phase):
        latent = self.encoder(obs)

        # Select active head based on phase
        head_map = {
            DecisionPhase.BIDDING: 'bid',
            DecisionPhase.PLAYING: 'card',
            DecisionPhase.ABILITY_ROSIE: 'rosie',
            DecisionPhase.ABILITY_BENDT_1: 'bendt',
            DecisionPhase.ABILITY_BENDT_2: 'bendt',
            DecisionPhase.ABILITY_ROATAN: 'roatan',
            DecisionPhase.ABILITY_HARRY: 'harry',
        }

        active_head = head_map[phase]
        logits = self.heads[active_head](latent)
        value = self.value(latent)

        return logits, value
```

### 2.4 Action Masking Per Phase

Each phase has its own masking logic:

```python
def get_action_mask(self, phase: DecisionPhase) -> np.ndarray:
    if phase == DecisionPhase.BIDDING:
        # Mask: valid bids 0 to round_number
        mask = np.zeros(11)
        mask[:round_number + 1] = 1

    elif phase == DecisionPhase.PLAYING:
        # Mask: valid cards based on suit-following rules
        mask = np.zeros(13)
        valid_indices = get_valid_card_indices(hand, trick)
        mask[valid_indices] = 1

    elif phase == DecisionPhase.ABILITY_ROSIE:
        # Mask: all players are valid choices
        mask = np.zeros(6)
        mask[:num_players] = 1

    elif phase == DecisionPhase.ABILITY_BENDT_1:
        # Mask: all cards in expanded hand
        mask = np.zeros(13)
        mask[:len(hand)] = 1

    elif phase == DecisionPhase.ABILITY_BENDT_2:
        # Mask: all cards except first selection
        mask = np.zeros(13)
        mask[:len(hand)] = 1
        mask[first_discard_index] = 0  # Exclude already selected

    elif phase == DecisionPhase.ABILITY_ROATAN:
        # Mask: all options valid
        mask = np.ones(3)

    elif phase == DecisionPhase.ABILITY_HARRY:
        # Mask: all options valid
        mask = np.ones(3)

    return mask
```

---

## 3. Reward Design

### 3.1 Base Rewards (existing)
- Trick win: +0.1 if progressing toward bid
- Bid accuracy: +1.0 for exact bid, penalty for over/under
- Game win: +10.0

### 3.2 Ability-Specific Rewards

**Rosie (choose_starter):**
```python
def rosie_reward(chosen_player, outcome):
    # Reward if chosen player leads to favorable outcome
    if chosen_player == self and won_next_trick:
        return +0.2  # Strategic self-selection
    elif chosen_player != self and helped_opponent:
        return -0.1  # Gave away advantage
    return 0.0
```

**Bendt (draw_discard):**
```python
def bendt_reward(discarded_cards, bid, remaining_tricks):
    # Reward for strategic discards
    reward = 0.0
    for card in discarded_cards:
        if is_escape(card) and bid > 0:
            reward += 0.1  # Good: discard escape when bidding high
        elif is_high_card(card) and bid == 0:
            reward += 0.1  # Good: discard winner when bidding zero
        elif is_high_card(card) and bid > 0:
            reward -= 0.2  # Bad: discarded a winner when need tricks
    return reward
```

**Roatán (extra_bet):**
```python
def roatan_reward(bet_amount, hit_bid):
    if hit_bid:
        return bet_amount / 100  # +0.0, +0.1, or +0.2
    else:
        return -bet_amount / 100  # -0.0, -0.1, or -0.2
```

**Jade (view_deck):**
```python
# No direct reward - but information should improve future decisions
# Intrinsic reward for information gain could be added
def jade_reward():
    return 0.0  # Implicit value through better play
```

**Harry (modify_bid):**
```python
def harry_reward(modifier, final_bid, tricks_won):
    original_bid = final_bid - modifier

    # Reward for successful adjustment
    if tricks_won == final_bid:
        if original_bid != tricks_won:
            return +0.3  # Harry saved the bid
        else:
            return +0.1  # Correct to not change
    else:
        if modifier != 0:
            return -0.2  # Wasted the adjustment
    return 0.0
```

---

## 4. Training Strategy

### 4.1 Curriculum Learning

**Stage 1: Base Training (no abilities)**
- Train with abilities disabled
- Learn bidding and card play fundamentals
- Target: 60%+ win rate vs rule-based bots

**Stage 2: Single Ability Introduction**
```python
ability_curriculum = [
    (1_000_000, {'harry': True}),      # Harry first (simplest)
    (2_000_000, {'roatan': True}),     # Then Roatán
    (3_000_000, {'rosie': True}),      # Then Rosie
    (4_000_000, {'bendt': True}),      # Then Bendt (most complex)
    (5_000_000, {'jade': True}),       # Finally Jade
]
```

**Stage 3: Full Ability Training**
- All abilities enabled
- Self-play against ability-aware opponents
- Mixed opponents for robustness

### 4.2 Experience Replay Prioritization

Abilities are rare events. Use prioritized experience replay:

```python
class AbilityPrioritizedBuffer:
    def __init__(self, capacity, ability_boost=5.0):
        self.buffer = []
        self.priorities = []
        self.ability_boost = ability_boost

    def add(self, experience, has_ability):
        priority = self.ability_boost if has_ability else 1.0
        self.buffer.append(experience)
        self.priorities.append(priority)

    def sample(self, batch_size):
        probs = np.array(self.priorities) / sum(self.priorities)
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        return [self.buffer[i] for i in indices]
```

### 4.3 Auxiliary Tasks

Train auxiliary prediction heads to improve representation:

1. **Ability Prediction**: Given hand/position, predict which pirate might win
2. **Bid Confidence**: Predict final bid accuracy given current state
3. **Card Counting**: Predict remaining card distribution

---

## 5. Environment Implementation

### 5.1 Extended Step Function

```python
def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
    reward = 0.0
    done = False

    if self.decision_phase == DecisionPhase.BIDDING:
        self._apply_bid(action)
        self._advance_to_next_bidder_or_start_play()

    elif self.decision_phase == DecisionPhase.PLAYING:
        card_played = self._apply_card_play(action)
        reward += self._get_card_reward(card_played)

        # Check for ability trigger
        ability = self._check_ability_trigger()
        if ability:
            self.decision_phase = self._get_ability_phase(ability)
            self.pending_ability = ability
        else:
            self._advance_to_next_player_or_end_trick()

    elif self.decision_phase in ABILITY_PHASES:
        reward += self._resolve_ability(action)

        # Bendt needs second phase
        if self.decision_phase == DecisionPhase.ABILITY_BENDT_1:
            self.decision_phase = DecisionPhase.ABILITY_BENDT_2
            self.first_bendt_discard = action
        else:
            self.decision_phase = DecisionPhase.PLAYING
            self._advance_to_next_player_or_end_trick()

    obs = self._get_observation()
    done = self._check_game_end()

    return obs, reward, done, False, {}
```

### 5.2 Modified Observation Builder

```python
def _get_observation(self) -> np.ndarray:
    # Base observations (existing)
    base_obs = self._build_base_observation()  # 190 dims

    # Phase indicator
    phase_one_hot = np.zeros(8)
    phase_one_hot[self.decision_phase.value] = 1

    # Ability context
    ability_context = self._build_ability_context()  # ~32 dims

    return np.concatenate([base_obs, phase_one_hot, ability_context])

def _build_ability_context(self) -> np.ndarray:
    context = np.zeros(32)

    if self.pending_ability is None:
        return context

    ability = self.pending_ability

    if ability.type == AbilityType.CHOOSE_STARTER:
        # Rosie context: player strengths
        for i, player in enumerate(self.game.players):
            context[i] = self._estimate_lead_strength(player)

    elif ability.type == AbilityType.DRAW_DISCARD:
        # Bendt context: drawn cards + phase
        context[:14] = self._encode_cards(ability.drawn_cards)
        context[14] = 1 if self.decision_phase == DecisionPhase.ABILITY_BENDT_2 else 0
        if hasattr(self, 'first_bendt_discard'):
            context[15 + self.first_bendt_discard] = 1

    elif ability.type == AbilityType.EXTRA_BET:
        # Roatán context: confidence
        context[0] = self._calculate_bid_confidence()

    elif ability.type == AbilityType.MODIFY_BID:
        # Harry context: bid/tricks
        player = self.game.get_player(self.agent_player_id)
        context[0] = player.bid / 10
        context[1] = self.round.get_tricks_won(player.id) / 10
        context[2] = (context[1] - context[0])  # Difference

    return context
```

---

## 6. Evaluation Metrics

### 6.1 Ability-Specific Metrics

```python
class AbilityMetrics:
    def __init__(self):
        self.rosie_strategic_rate = 0  # % of Rosie choices that were strategic
        self.bendt_discard_quality = 0  # Average quality of discards
        self.roatan_bet_accuracy = 0   # % of bets won
        self.harry_save_rate = 0       # % of Harry uses that saved bids

    def evaluate(self, episode_data):
        # Rosie: Did choosing self lead to winning the trick?
        # Bendt: Did discards improve hand quality?
        # Roatán: Win rate on bets
        # Harry: How often did ±1 result in hitting bid?
```

### 6.2 Comparison Baselines

1. **Random ability decisions**: Baseline
2. **Rule-based heuristics**: Current implementation
3. **Learned policy**: Proposed architecture

---

## 7. Implementation Roadmap

### Phase 1: Environment Extension (2 weeks)
- [ ] Add DecisionPhase enum and state tracking
- [ ] Extend observation space with ability context
- [ ] Implement multi-phase step function
- [ ] Add ability-specific masking

### Phase 2: Policy Architecture (1 week)
- [ ] Implement MultiHeadPolicy network
- [ ] Integrate with MaskablePPO
- [ ] Add phase-conditional forward pass

### Phase 3: Training Pipeline (2 weeks)
- [ ] Implement curriculum learning
- [ ] Add ability-specific rewards
- [ ] Set up prioritized experience buffer
- [ ] Create evaluation suite

### Phase 4: Iteration (ongoing)
- [ ] Train and evaluate
- [ ] Tune hyperparameters
- [ ] Add auxiliary tasks if needed
- [ ] Self-play refinement

---

## 8. Alternative Approaches Considered

### 8.1 Flat Action Space
- **Pros**: Simpler implementation
- **Cons**: 50+ actions, sparse signal for rare abilities
- **Verdict**: Rejected due to masking complexity

### 8.2 Separate Models Per Ability
- **Pros**: Specialized learning
- **Cons**: No transfer learning, complex orchestration
- **Verdict**: Rejected in favor of shared encoder

### 8.3 LLM-Based Reasoning
- **Pros**: Zero-shot ability decisions
- **Cons**: Latency, cost, not suitable for training loop
- **Verdict**: Could be used for data augmentation/imitation learning

---

## 9. Conclusion

This architecture addresses the key challenge of learning strategic ability decisions by:

1. **Modeling abilities as decision phases** with phase-specific observations
2. **Using multi-head outputs** for efficient action selection
3. **Applying curriculum learning** to handle ability rarity
4. **Designing ability-specific rewards** for dense signal

The result should be an agent that not only plays cards well but also makes intelligent strategic decisions about pirate abilities, significantly improving overall performance.
