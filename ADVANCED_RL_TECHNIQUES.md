# Advanced RL/ML Techniques for Skull King

## Table of Contents

1. [Current Approach Analysis](#1-current-approach-analysis)
2. [Hierarchical RL (Options Framework)](#2-hierarchical-rl-options-framework)
3. [Round-as-Episode with Meta-Learning](#3-round-as-episode-with-meta-learning)
4. [Transformer-Based Architecture](#4-transformer-based-architecture)
5. [Counterfactual Regret Minimization (CFR)](#5-counterfactual-regret-minimization-cfr)
6. [Intrinsic Motivation & Curiosity-Driven Learning](#6-intrinsic-motivation--curiosity-driven-learning)
7. [Population-Based Training with League](#7-population-based-training-with-league)
8. [Monte Carlo Tree Search + RL (AlphaZero-style)](#8-monte-carlo-tree-search--rl-alphazero-style)
9. [Opponent Modeling](#9-opponent-modeling)
10. [Implementation Roadmap](#10-implementation-roadmap)

---

## 1. Current Approach Analysis

### 1.1 Episode Structure

Our current implementation treats a **full game as one episode**:

```
Episode (1 full game ≈ 55 steps):
├── Round 1:  1 bid  +  1 trick  =  2 agent actions
├── Round 2:  1 bid  +  2 tricks =  3 agent actions
├── Round 3:  1 bid  +  3 tricks =  4 agent actions
├── ...
└── Round 10: 1 bid  + 10 tricks = 11 agent actions
                                   ─────────────────
                        Total:     10 bids + 55 tricks = 65 actions
                        Agent:     10 bids + ~14 tricks ≈ 24 actions (in 4-player)
```

**Note:** In a 4-player game, the agent plays approximately 1/4 of the tricks, so ~55/4 ≈ 14 trick actions plus 10 bids ≈ 24 actions per episode. However, the environment steps through all players, so from the environment's perspective it's ~55 steps.

### 1.2 Current Observation Space (171 dimensions)

```python
observation = [
    game_phase,           # 4 dims (one-hot: PENDING/BIDDING/PICKING/ENDED)
    hand_encoding,        # 90 dims (10 cards × 9 features)
    trick_state,          # 36 dims (4 players × 9 features)
    bidding_context,      # 8 dims (round, hand size, bid, tricks won/needed)
    opponent_state,       # 9 dims (3 opponents × 3 features)
    hand_strength,        # 4 dims (pirates, kings, mermaids, high cards)
    trick_position,       # 4 dims (position in trick order)
    opponent_patterns,    # 6 dims (bidding patterns)
    cards_played,         # 5 dims (card types seen)
    round_progression,    # 1 dim
    bid_pressure,         # 1 dim
    position_advantage,   # 1 dim
    trump_strength,       # 2 dims
]
```

### 1.3 Current Action Space

```python
action_space = Discrete(11)
# During bidding: action 0-10 = bid value (capped at round number)
# During picking: action 0-10 = card index in hand
```

### 1.4 Problems with Current Approach

| Problem | Description | Impact |
|---------|-------------|--------|
| **Credit Assignment** | Hard to attribute final reward to specific decisions | Agent struggles to learn which bids/plays were good |
| **Long Horizon** | 55 steps with γ=0.995 means γ^55 ≈ 0.76 | Late-game rewards heavily discounted |
| **Non-Stationarity** | Round 1 (1 card) vs Round 10 (10 cards) are fundamentally different | Single policy must handle 10 different "games" |
| **Sparse Terminal Reward** | Final ranking only known after 55 decisions | Delayed feedback hampers learning |
| **Mixed Objectives** | Bidding and playing require different skills | Same network must master two distinct tasks |

### 1.5 Current Reward Shaping

We mitigate some issues with dense rewards:

```python
# Per-action rewards
bid_quality_reward    = 0 to +2    # How well bid matches hand strength
card_play_reward      = -0.5 to +1 # Strategic card selection
valid_action_bonus    = +0.1       # Encourages valid moves

# Per-trick rewards
trick_outcome_reward  = -2 to +3   # Win/lose relative to need

# Per-round rewards (normalized)
round_accuracy_reward = -5 to +5   # Bid accuracy

# Per-game rewards (normalized)
final_ranking_reward  = -5 to +10  # Placement bonus
```

---

## 2. Hierarchical RL (Options Framework)

### 2.1 Concept

Hierarchical RL decomposes complex tasks into a hierarchy of simpler sub-tasks. For Skull King, this maps naturally to the game structure:

```
┌─────────────────────────────────────────────────────────────┐
│                    MANAGER POLICY (πₘ)                       │
│  ────────────────────────────────────────────────────────── │
│  Activated: Once per round (during bidding phase)           │
│  Input: Full hand, game state, opponent history             │
│  Output: Bid value (0 to round_number)                      │
│  Reward: Round completion accuracy (-5 to +5)               │
│  Horizon: 10 decisions per game                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ Bid = "Goal" for worker
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    WORKER POLICY (πw)                        │
│  ────────────────────────────────────────────────────────── │
│  Activated: Every trick (during picking phase)              │
│  Input: Hand, trick state, TARGET BID, tricks won so far    │
│  Output: Card index to play                                 │
│  Reward: Trick outcome relative to goal (-2 to +3)          │
│  Horizon: 1-10 decisions per round                          │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Theoretical Foundation

Based on the **Options Framework** (Sutton, Precup & Singh, 1999):

- **Option** ω = (I, π, β) where:
  - I = initiation set (states where option can start)
  - π = intra-option policy (how to act within option)
  - β = termination condition (when option ends)

For Skull King:
- **Bidding Option**: I = {BIDDING phase}, π = bid policy, β = bid placed
- **Playing Option**: I = {PICKING phase}, π = card policy, β = round complete

### 2.3 Implementation Architecture

```python
class HierarchicalSkullKingAgent:
    """Two-level hierarchical agent for Skull King."""

    def __init__(self):
        # Manager: Selects bid (goal) at start of each round
        self.manager = MaskablePPO(
            "MlpPolicy",
            manager_env,
            policy_kwargs={"net_arch": [128, 128]},
        )

        # Worker: Plays cards to achieve bid goal
        self.worker = MaskablePPO(
            "MlpPolicy",
            worker_env,
            policy_kwargs={"net_arch": [256, 256]},
        )

    def act(self, obs, phase):
        if phase == "BIDDING":
            bid = self.manager.predict(obs["manager_obs"])
            self.current_goal = bid
            return bid
        else:  # PICKING
            # Augment worker obs with goal
            worker_obs = np.concatenate([
                obs["worker_obs"],
                [self.current_goal / 10.0]  # Normalized goal
            ])
            return self.worker.predict(worker_obs)
```

### 2.4 Manager Environment

```python
class ManagerEnv(gym.Env):
    """Environment for the bidding (manager) policy."""

    observation_space = spaces.Box(
        low=-1, high=1,
        shape=(100,),  # Hand + game context
        dtype=np.float32
    )
    action_space = spaces.Discrete(11)  # Bid 0-10

    def step(self, bid):
        # Execute full round with worker policy
        self.inner_env.set_bid(bid)
        round_reward = self.inner_env.play_round(self.worker_policy)

        # Reward based on bid accuracy
        tricks_won = self.inner_env.get_tricks_won()
        accuracy = abs(bid - tricks_won)

        if accuracy == 0:
            reward = 5.0  # Perfect bid
        elif accuracy == 1:
            reward = 2.0
        else:
            reward = -accuracy

        return obs, reward, done, truncated, info
```

### 2.5 Worker Environment

```python
class WorkerEnv(gym.Env):
    """Environment for the card-playing (worker) policy."""

    def __init__(self):
        # Observation includes goal (target bid)
        self.observation_space = spaces.Box(
            low=-1, high=1,
            shape=(172,),  # 171 base + 1 goal
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(11)  # Card index

    def step(self, card_idx):
        # Play card
        result = self.game.play_card(card_idx)

        # Reward relative to goal progress
        tricks_won = self.get_tricks_won()
        tricks_needed = self.goal - tricks_won
        tricks_remaining = self.round_number - self.trick_number

        if result.won_trick:
            if tricks_needed > 0:
                reward = 3.0  # Needed win
            else:
                reward = -2.0  # Unwanted win
        else:
            if tricks_needed <= tricks_remaining:
                reward = 0.5  # Still achievable
            else:
                reward = -1.0  # Goal now impossible

        return obs, reward, done, truncated, info
```

### 2.6 Training Procedure

```python
def train_hierarchical():
    """Train hierarchical agent in stages."""

    # Stage 1: Pre-train worker with fixed bids
    print("Stage 1: Training worker policy...")
    for bid_value in range(11):
        worker_env = WorkerEnv(fixed_goal=bid_value)
        worker.learn(total_timesteps=100_000)

    # Stage 2: Train manager with frozen worker
    print("Stage 2: Training manager policy...")
    manager_env = ManagerEnv(worker_policy=worker)
    manager.learn(total_timesteps=500_000)

    # Stage 3: Joint fine-tuning
    print("Stage 3: Joint fine-tuning...")
    for iteration in range(10):
        # Fine-tune worker with manager's bids
        worker.learn(total_timesteps=50_000)
        # Fine-tune manager with updated worker
        manager.learn(total_timesteps=50_000)
```

### 2.7 Hindsight Goal Relabeling

Inspired by **Hindsight Experience Replay (HER)**, we can relabel failed episodes:

```python
def hindsight_relabel(trajectory, actual_tricks_won):
    """Relabel trajectory as if we bid correctly."""

    original_goal = trajectory.goal

    # Create "successful" version with actual outcome as goal
    relabeled = trajectory.copy()
    relabeled.goal = actual_tricks_won

    # Recompute rewards with new goal
    for t, transition in enumerate(relabeled):
        transition.reward = compute_reward(
            transition.state,
            transition.action,
            goal=actual_tricks_won  # Pretend this was our goal
        )

    # Add both to replay buffer
    replay_buffer.add(trajectory)        # Original (may have failed)
    replay_buffer.add(relabeled)         # "Successful" relabeled version
```

### 2.8 Expected Benefits

| Metric | Current | Hierarchical (Expected) |
|--------|---------|------------------------|
| Credit assignment | Difficult | Clear separation |
| Sample efficiency | Baseline | 2-3x improvement |
| Bid accuracy | ~60% | ~80% |
| Learning stability | High variance | More stable |
| Interpretability | Black box | Inspect each policy |

### 2.9 References

- Sutton, Precup & Singh (1999). "Between MDPs and semi-MDPs: A Framework for Temporal Abstraction in Reinforcement Learning"
- Bacon, Harb & Precup (2017). "The Option-Critic Architecture"
- Nachum et al. (2018). "Data-Efficient Hierarchical Reinforcement Learning"
- Andrychowicz et al. (2017). "Hindsight Experience Replay"

---

## 3. Round-as-Episode with Meta-Learning

### 3.1 Concept

Instead of treating a full game as one episode, treat each **round as a separate episode**. Use meta-learning to transfer knowledge across rounds with different card counts.

```
Game reframed as 10 related but distinct MDPs:

┌─────────────┐  ┌─────────────┐       ┌──────────────┐
│  Round 1    │  │  Round 2    │  ...  │  Round 10    │
│  MDP M₁     │  │  MDP M₂     │       │  MDP M₁₀     │
│  1 card     │  │  2 cards    │       │  10 cards    │
│  1 trick    │  │  2 tricks   │       │  10 tricks   │
└─────────────┘  └─────────────┘       └──────────────┘
       │                │                     │
       └────────────────┼─────────────────────┘
                        ▼
              ┌─────────────────┐
              │  Meta-Learner   │
              │  Finds shared   │
              │  structure      │
              └─────────────────┘
```

### 3.2 Why This Helps

1. **Shorter episodes**: Round 10 has only 11 steps vs 55 for full game
2. **Denser feedback**: Round reward after each round, not just game end
3. **Structured curriculum**: Can train on easier rounds (1-3) first
4. **Transfer learning**: Skills learned in round 5 apply to round 6

### 3.3 MAML (Model-Agnostic Meta-Learning)

Learn an initialization θ* that can quickly adapt to any round:

```python
class MAMLAgent:
    """Meta-learning agent that adapts to each round."""

    def __init__(self):
        self.meta_policy = PolicyNetwork()
        self.meta_optimizer = Adam(lr=0.001)
        self.inner_lr = 0.01
        self.inner_steps = 5

    def adapt_to_round(self, round_number, experiences):
        """Fast adaptation to specific round."""
        # Clone meta-parameters
        adapted_params = self.meta_policy.parameters().clone()

        # Inner loop: adapt to this round
        for _ in range(self.inner_steps):
            loss = self.compute_policy_loss(experiences, adapted_params)
            grads = torch.autograd.grad(loss, adapted_params)
            adapted_params = [p - self.inner_lr * g
                              for p, g in zip(adapted_params, grads)]

        return adapted_params

    def meta_train(self, task_batch):
        """Outer loop: optimize for fast adaptation."""
        meta_loss = 0

        for round_number, experiences in task_batch:
            # Adapt to round
            adapted = self.adapt_to_round(round_number, experiences[:10])

            # Evaluate adapted policy on held-out experiences
            eval_loss = self.compute_policy_loss(
                experiences[10:],
                adapted
            )
            meta_loss += eval_loss

        # Update meta-parameters
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()
```

### 3.4 RL² (Learning to Reinforcement Learn)

Use an RNN that learns to learn within an episode:

```python
class RL2Agent(nn.Module):
    """RNN-based agent that adapts within episode."""

    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()

        # Input: observation + previous action + previous reward
        input_dim = obs_dim + act_dim + 1

        self.rnn = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=2,
            batch_first=True
        )

        self.policy_head = nn.Linear(hidden_dim, act_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, obs, prev_action, prev_reward, hidden):
        # Concatenate inputs
        x = torch.cat([obs, prev_action, prev_reward], dim=-1)

        # RNN maintains adaptation state
        rnn_out, new_hidden = self.rnn(x.unsqueeze(1), hidden)

        # Policy and value outputs
        policy = F.softmax(self.policy_head(rnn_out), dim=-1)
        value = self.value_head(rnn_out)

        return policy, value, new_hidden
```

**Training procedure for RL²:**

```python
def train_rl2():
    """Train RL² agent across rounds."""

    for epoch in range(1000):
        # Sample batch of rounds
        rounds = sample_rounds(batch_size=32)

        for round_data in rounds:
            # Reset hidden state at round start
            hidden = agent.init_hidden()

            prev_action = torch.zeros(act_dim)
            prev_reward = torch.zeros(1)

            trajectory = []

            for obs in round_data.observations:
                # Agent adapts through hidden state
                policy, value, hidden = agent(
                    obs, prev_action, prev_reward, hidden
                )

                action = sample(policy)
                reward = env.step(action)

                trajectory.append((obs, action, reward, value))

                prev_action = one_hot(action)
                prev_reward = reward

            # Update with PPO on trajectory
            loss = ppo_loss(trajectory)
            loss.backward()
            optimizer.step()
```

### 3.5 Contextual Policies

Simpler approach: condition policy on round number explicitly:

```python
class ContextualPolicy(nn.Module):
    """Policy conditioned on round context."""

    def __init__(self, obs_dim, act_dim, num_rounds=10):
        super().__init__()

        # Round embedding
        self.round_embedding = nn.Embedding(num_rounds, 32)

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(obs_dim + 32, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        # Round-specific heads (optional)
        self.policy_heads = nn.ModuleList([
            nn.Linear(256, act_dim) for _ in range(num_rounds)
        ])

    def forward(self, obs, round_number):
        # Embed round context
        round_ctx = self.round_embedding(round_number)

        # Concatenate and process
        x = torch.cat([obs, round_ctx], dim=-1)
        features = self.shared(x)

        # Use round-specific head
        logits = self.policy_heads[round_number](features)

        return F.softmax(logits, dim=-1)
```

### 3.6 Progressive Training Curriculum

```python
def progressive_round_training():
    """Train on progressively harder rounds."""

    curriculum = [
        (1, 100_000),    # Round 1: simple, fast learning
        (2, 100_000),    # Round 2: slightly more complex
        (3, 150_000),    # Round 3: medium
        (1, 50_000),     # Revisit round 1 (prevent forgetting)
        (4, 150_000),
        (5, 200_000),
        (2, 50_000),     # Revisit
        (6, 200_000),
        (7, 250_000),
        (8, 300_000),
        (9, 350_000),
        (10, 400_000),
        ("all", 500_000), # Mixed training on all rounds
    ]

    for round_or_all, timesteps in curriculum:
        if round_or_all == "all":
            env = MixedRoundEnv(rounds=range(1, 11))
        else:
            env = SingleRoundEnv(round_number=round_or_all)

        agent.learn(env, total_timesteps=timesteps)
```

### 3.7 References

- Finn, Abbeel & Levine (2017). "Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks"
- Duan et al. (2016). "RL²: Fast Reinforcement Learning via Slow Reinforcement Learning"
- Rakelly et al. (2019). "Efficient Off-Policy Meta-Reinforcement Learning via Probabilistic Context Variables"

---

## 4. Transformer-Based Architecture

### 4.1 Concept

Replace the MLP policy network with a **Transformer** architecture that can:
1. Handle variable-length inputs (hands of 1-10 cards)
2. Capture card-to-card relationships via attention
3. Maintain history of the game through memory

### 4.2 Why Transformers for Card Games

```
Traditional MLP approach:
┌─────────────────────────────────────────────┐
│  Fixed-size input: [card1, card2, ..., pad] │
│  Each card encoded independently            │
│  No explicit modeling of card relationships │
└─────────────────────────────────────────────┘

Transformer approach:
┌─────────────────────────────────────────────┐
│  Variable-length: [card1, card2, ..., cardN]│
│  Self-attention captures relationships      │
│  "This mermaid is valuable because they     │
│   have the skull king"                      │
└─────────────────────────────────────────────┘
```

### 4.3 Card Transformer Architecture

```python
class CardTransformer(nn.Module):
    """Transformer for processing card game states."""

    def __init__(
        self,
        card_dim: int = 16,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        max_cards: int = 10,
    ):
        super().__init__()

        # Card embedding (card type + suit + number + special flags)
        self.card_embedding = nn.Sequential(
            nn.Linear(9, card_dim),  # 9 features per card
            nn.ReLU(),
            nn.Linear(card_dim, hidden_dim),
        )

        # Positional encoding for card order in hand
        self.pos_encoding = nn.Embedding(max_cards, hidden_dim)

        # Special tokens
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.bid_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # Context encoder (game state, opponent info)
        self.context_encoder = nn.Sequential(
            nn.Linear(50, hidden_dim),
            nn.ReLU(),
        )

        # Output heads
        self.bid_head = nn.Linear(hidden_dim, 11)  # Bid 0-10
        self.play_head = nn.Linear(hidden_dim, 1)  # Score per card
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, cards, context, mask=None):
        """
        Args:
            cards: [batch, num_cards, 9] card features
            context: [batch, 50] game context
            mask: [batch, num_cards] valid card mask
        """
        batch_size, num_cards, _ = cards.shape

        # Embed cards
        card_embeds = self.card_embedding(cards)

        # Add positional encoding
        positions = torch.arange(num_cards, device=cards.device)
        card_embeds = card_embeds + self.pos_encoding(positions)

        # Embed context and add as first token
        ctx_embed = self.context_encoder(context).unsqueeze(1)

        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        # Full sequence: [CLS, context, card1, card2, ...]
        sequence = torch.cat([cls_tokens, ctx_embed, card_embeds], dim=1)

        # Create attention mask (don't attend to padding)
        if mask is not None:
            # Extend mask for CLS and context tokens
            full_mask = torch.cat([
                torch.ones(batch_size, 2, device=mask.device),
                mask
            ], dim=1)
            attn_mask = (full_mask == 0)
        else:
            attn_mask = None

        # Apply transformer
        output = self.transformer(
            sequence,
            src_key_padding_mask=attn_mask
        )

        # CLS token output for global decisions
        cls_output = output[:, 0, :]

        # Card outputs for card selection
        card_outputs = output[:, 2:, :]  # Skip CLS and context

        # Compute outputs
        bid_logits = self.bid_head(cls_output)
        card_scores = self.play_head(card_outputs).squeeze(-1)
        value = self.value_head(cls_output)

        return {
            "bid_logits": bid_logits,
            "card_scores": card_scores,
            "value": value,
            "attention": self.get_attention_weights(),
        }

    def get_attention_weights(self):
        """Extract attention weights for interpretability."""
        # Implementation depends on transformer version
        pass
```

### 4.4 Decision Transformer (Offline RL)

Reframe RL as sequence modeling - predict actions given desired returns:

```python
class DecisionTransformer(nn.Module):
    """
    Decision Transformer for Skull King.

    Input sequence: [R̂₁, s₁, a₁, R̂₂, s₂, a₂, ..., R̂ₜ, sₜ, ?]

    Where R̂ₜ is the return-to-go (desired future reward).
    """

    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        hidden_dim: int = 128,
        max_length: int = 60,  # Max game length
    ):
        super().__init__()

        # Embeddings for each modality
        self.state_embed = nn.Linear(state_dim, hidden_dim)
        self.action_embed = nn.Embedding(act_dim, hidden_dim)
        self.return_embed = nn.Linear(1, hidden_dim)

        # Timestep embedding
        self.timestep_embed = nn.Embedding(max_length, hidden_dim)

        # GPT-style transformer
        self.transformer = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(
                d_model=hidden_dim,
                nhead=4,
                dim_feedforward=hidden_dim * 4,
            ),
            num_layers=4,
        )

        # Prediction heads
        self.predict_action = nn.Linear(hidden_dim, act_dim)
        self.predict_state = nn.Linear(hidden_dim, state_dim)
        self.predict_return = nn.Linear(hidden_dim, 1)

    def forward(self, returns_to_go, states, actions, timesteps):
        """
        Args:
            returns_to_go: [batch, seq_len, 1] desired future returns
            states: [batch, seq_len, state_dim]
            actions: [batch, seq_len] action indices
            timesteps: [batch, seq_len] time indices
        """
        batch_size, seq_len = states.shape[:2]

        # Embed all modalities
        return_embeds = self.return_embed(returns_to_go)
        state_embeds = self.state_embed(states)
        action_embeds = self.action_embed(actions)
        time_embeds = self.timestep_embed(timesteps)

        # Add timestep embeddings
        return_embeds = return_embeds + time_embeds
        state_embeds = state_embeds + time_embeds
        action_embeds = action_embeds + time_embeds

        # Interleave: [R₁, s₁, a₁, R₂, s₂, a₂, ...]
        # Shape: [batch, seq_len * 3, hidden_dim]
        stacked = torch.stack([
            return_embeds, state_embeds, action_embeds
        ], dim=2).reshape(batch_size, seq_len * 3, -1)

        # Causal attention mask
        mask = self.generate_causal_mask(seq_len * 3)

        # Transform
        output = self.transformer(stacked, stacked, tgt_mask=mask)

        # Predict next action from state positions
        # State positions are at indices 1, 4, 7, ... (3k+1)
        state_outputs = output[:, 1::3, :]
        action_preds = self.predict_action(state_outputs)

        return action_preds

    def generate_causal_mask(self, size):
        """Generate causal attention mask."""
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        return mask.bool()
```

**Training Decision Transformer:**

```python
def train_decision_transformer(dataset, model, epochs=100):
    """
    Train Decision Transformer on offline dataset.

    Dataset contains trajectories: [(s₁, a₁, r₁), (s₂, a₂, r₂), ...]
    """
    optimizer = Adam(model.parameters(), lr=1e-4)

    for epoch in range(epochs):
        for batch in dataset.iterate_batches():
            states, actions, rewards = batch

            # Compute returns-to-go
            returns_to_go = compute_returns_to_go(rewards)

            # Create timesteps
            timesteps = torch.arange(states.shape[1])

            # Forward pass
            action_preds = model(returns_to_go, states, actions, timesteps)

            # Cross-entropy loss on action predictions
            loss = F.cross_entropy(
                action_preds[:, :-1].reshape(-1, act_dim),
                actions[:, 1:].reshape(-1)
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def evaluate_decision_transformer(model, env, target_return):
    """
    Evaluate by conditioning on desired return.
    """
    state = env.reset()

    states = [state]
    actions = []
    returns_to_go = [target_return]
    timesteps = [0]

    done = False
    while not done:
        # Predict action
        action_pred = model(
            torch.tensor(returns_to_go).unsqueeze(0),
            torch.tensor(states).unsqueeze(0),
            torch.tensor(actions + [0]).unsqueeze(0),  # Dummy action
            torch.tensor(timesteps).unsqueeze(0),
        )

        action = action_pred[0, -1].argmax().item()

        # Step environment
        next_state, reward, done, _ = env.step(action)

        # Update sequences
        states.append(next_state)
        actions.append(action)
        returns_to_go.append(returns_to_go[-1] - reward)
        timesteps.append(len(timesteps))

    return sum(rewards)
```

### 4.5 Attention Visualization for Interpretability

```python
def visualize_card_attention(model, hand, trick):
    """Visualize which cards the model attends to."""

    with torch.no_grad():
        output = model(hand, trick)
        attention_weights = output["attention"]  # [heads, cards, cards]

    # Average across heads
    avg_attention = attention_weights.mean(dim=0)

    # Plot heatmap
    import matplotlib.pyplot as plt
    import seaborn as sns

    card_names = [card_to_string(c) for c in hand]

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        avg_attention.numpy(),
        xticklabels=card_names,
        yticklabels=card_names,
        cmap="Blues",
    )
    plt.title("Card-to-Card Attention")
    plt.savefig("attention_viz.png")
```

### 4.6 References

- Vaswani et al. (2017). "Attention Is All You Need"
- Chen et al. (2021). "Decision Transformer: Reinforcement Learning via Sequence Modeling"
- Janner et al. (2021). "Offline Reinforcement Learning as One Big Sequence Modeling Problem"
- Li et al. (2023). "Transformer-based Policy for Card Games"

---

## 5. Counterfactual Regret Minimization (CFR)

### 5.1 Concept

CFR is a game-theoretic algorithm that finds Nash equilibrium strategies for imperfect-information games. It works by:

1. Tracking **regret** for not taking each action
2. Playing proportionally to positive regrets
3. Averaging strategies over iterations

### 5.2 Why CFR for Skull King

Skull King has **imperfect information**:
- Players don't see each other's hands
- Future cards are unknown
- Bidding reveals partial information about hands

CFR excels at games where:
- Information asymmetry is important
- Bluffing/deception can be valuable
- Opponent modeling matters

### 5.3 Basic CFR Algorithm

```python
class CFRAgent:
    """Counterfactual Regret Minimization for Skull King."""

    def __init__(self):
        # Cumulative regrets for each information set
        self.regret_sum = defaultdict(lambda: np.zeros(11))

        # Cumulative strategy for averaging
        self.strategy_sum = defaultdict(lambda: np.zeros(11))

    def get_info_set(self, state):
        """
        Convert state to information set (what player knows).

        In Skull King, this includes:
        - Our hand
        - Cards played in current trick
        - All bids (visible)
        - Tricks won by each player
        - But NOT other players' hands
        """
        return (
            tuple(sorted(state.hand)),
            tuple(state.trick_cards),
            tuple(state.bids),
            tuple(state.tricks_won),
            state.round_number,
        )

    def get_strategy(self, info_set, valid_actions):
        """Get current strategy from regrets."""
        regrets = self.regret_sum[info_set]

        # Use regret matching: strategy proportional to positive regrets
        positive_regrets = np.maximum(regrets, 0)

        if positive_regrets.sum() > 0:
            strategy = positive_regrets / positive_regrets.sum()
        else:
            # Uniform over valid actions
            strategy = np.zeros(11)
            strategy[valid_actions] = 1.0 / len(valid_actions)

        return strategy

    def cfr(self, state, player, reach_probs):
        """
        Run CFR traversal.

        Args:
            state: Current game state
            player: Player we're computing strategy for
            reach_probs: Probability of reaching this state for each player

        Returns:
            Expected utility for the player
        """
        if state.is_terminal():
            return state.get_utility(player)

        info_set = self.get_info_set(state)
        valid_actions = state.get_valid_actions()
        strategy = self.get_strategy(info_set, valid_actions)

        # Compute counterfactual values for each action
        action_utilities = np.zeros(11)

        for action in valid_actions:
            next_state = state.apply_action(action)

            # Update reach probability
            new_reach = reach_probs.copy()
            new_reach[state.current_player] *= strategy[action]

            action_utilities[action] = self.cfr(
                next_state, player, new_reach
            )

        # Expected utility under current strategy
        utility = np.dot(strategy, action_utilities)

        if state.current_player == player:
            # Compute regrets
            for action in valid_actions:
                regret = action_utilities[action] - utility
                self.regret_sum[info_set][action] += (
                    reach_probs[1 - player] * regret  # Opponent reach
                )

            # Accumulate strategy for averaging
            self.strategy_sum[info_set] += (
                reach_probs[player] * strategy
            )

        return utility

    def train(self, iterations=100000):
        """Train CFR agent."""
        for i in range(iterations):
            # Sample initial state
            state = SkullKingState.new_game()

            # Run CFR for each player
            for player in range(4):
                reach_probs = np.ones(4)
                self.cfr(state, player, reach_probs)

            if i % 10000 == 0:
                print(f"Iteration {i}, exploitability: {self.exploitability()}")

    def get_average_strategy(self, info_set):
        """Get the averaged strategy (converges to Nash equilibrium)."""
        strategy_sum = self.strategy_sum[info_set]

        if strategy_sum.sum() > 0:
            return strategy_sum / strategy_sum.sum()
        else:
            return np.ones(11) / 11  # Uniform
```

### 5.4 Deep CFR

For large state spaces, use neural networks to generalize:

```python
class DeepCFR:
    """Deep Counterfactual Regret Minimization."""

    def __init__(self, state_dim, action_dim):
        # Value network predicts regrets
        self.value_net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

        # Strategy network for final policy
        self.strategy_net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Softmax(dim=-1),
        )

        # Memory buffers
        self.advantage_memory = ReplayBuffer(max_size=1_000_000)
        self.strategy_memory = ReplayBuffer(max_size=1_000_000)

    def traverse(self, state, player, t):
        """Single CFR traversal with sampling."""
        if state.is_terminal():
            return state.get_utility(player)

        info_set_features = self.encode_info_set(state)
        valid_actions = state.get_valid_actions()

        if state.current_player == player:
            # Regret matching using value network
            advantages = self.value_net(info_set_features)
            strategy = self.regret_matching(advantages, valid_actions)

            # Sample action
            action = np.random.choice(11, p=strategy.numpy())

            # Traverse
            next_state = state.apply_action(action)
            utility = self.traverse(next_state, player, t)

            # Compute advantages for all actions
            action_values = np.zeros(11)
            for a in valid_actions:
                if a == action:
                    action_values[a] = utility
                else:
                    alt_state = state.apply_action(a)
                    action_values[a] = self.traverse(alt_state, player, t)

            # Store in memory
            advantages = action_values - utility
            self.advantage_memory.add({
                "info_set": info_set_features,
                "advantages": advantages,
                "iteration": t,
            })

            return utility
        else:
            # Opponent: use average strategy
            strategy = self.strategy_net(info_set_features)
            action = np.random.choice(11, p=strategy.numpy())

            next_state = state.apply_action(action)
            return self.traverse(next_state, player, t)

    def train(self, iterations=100000):
        """Train Deep CFR."""
        for t in range(iterations):
            # CFR traversal
            state = SkullKingState.new_game()
            for player in range(4):
                self.traverse(state, player, t)

            # Train value network on advantages
            if len(self.advantage_memory) > 1000:
                batch = self.advantage_memory.sample(256)
                self.train_value_network(batch)

            # Train strategy network periodically
            if t % 100 == 0:
                self.train_strategy_network()
```

### 5.5 CFR+ Improvements

```python
class CFRPlus(CFRAgent):
    """CFR+ with faster convergence."""

    def get_strategy(self, info_set, valid_actions):
        """Use regret matching+ (floor regrets at 0)."""
        regrets = self.regret_sum[info_set]

        # CFR+ floors cumulative regrets at 0 each iteration
        self.regret_sum[info_set] = np.maximum(regrets, 0)

        regrets = self.regret_sum[info_set]

        if regrets.sum() > 0:
            strategy = regrets / regrets.sum()
        else:
            strategy = np.zeros(11)
            strategy[valid_actions] = 1.0 / len(valid_actions)

        return strategy

    def update_regrets(self, info_set, regrets, iteration):
        """Weighted averaging for CFR+."""
        # Weight by iteration (recent iterations matter more)
        weight = iteration

        self.regret_sum[info_set] = np.maximum(
            self.regret_sum[info_set] + weight * regrets,
            0  # Floor at 0
        )
```

### 5.6 Handling 4-Player Games

Standard CFR is for 2-player zero-sum games. For 4-player Skull King:

```python
class MultiPlayerCFR(CFRAgent):
    """CFR adapted for 4-player games."""

    def cfr(self, state, reach_probs):
        """Compute strategy for all players simultaneously."""
        if state.is_terminal():
            # Return utilities for all players
            return [state.get_utility(p) for p in range(4)]

        current_player = state.current_player
        info_set = self.get_info_set(state)
        valid_actions = state.get_valid_actions()
        strategy = self.get_strategy(info_set, valid_actions)

        action_utilities = [np.zeros(11) for _ in range(4)]

        for action in valid_actions:
            next_state = state.apply_action(action)

            new_reach = reach_probs.copy()
            new_reach[current_player] *= strategy[action]

            utils = self.cfr(next_state, new_reach)

            for p in range(4):
                action_utilities[p][action] = utils[p]

        # Compute expected utilities
        utilities = [np.dot(strategy, au) for au in action_utilities]

        # Update regrets for current player
        opponent_reach = np.prod([
            reach_probs[p] for p in range(4) if p != current_player
        ])

        for action in valid_actions:
            regret = action_utilities[current_player][action] - utilities[current_player]
            self.regret_sum[info_set][action] += opponent_reach * regret

        self.strategy_sum[info_set] += reach_probs[current_player] * strategy

        return utilities
```

### 5.7 References

- Zinkevich et al. (2007). "Regret Minimization in Games with Incomplete Information"
- Lanctot et al. (2009). "Monte Carlo Sampling for Regret Minimization in Extensive Games"
- Brown & Sandholm (2019). "Superhuman AI for multiplayer poker"
- Steinberger (2019). "Single Deep Counterfactual Regret Minimization"

---

## 6. Intrinsic Motivation & Curiosity-Driven Learning

### 6.1 Concept

Add **intrinsic rewards** that encourage exploration beyond extrinsic game rewards:

```
Total Reward = Extrinsic (game score) + β × Intrinsic (curiosity/novelty)
```

### 6.2 Why Intrinsic Motivation for Skull King

- **Sparse rewards**: Game outcome only known after many decisions
- **Strategy discovery**: Novel strategies (e.g., bidding 0) may be unexplored
- **Opponent adaptation**: Curiosity can drive learning about opponent behaviors

### 6.3 Intrinsic Curiosity Module (ICM)

Reward agent for visiting states where its forward model is surprised:

```python
class ICM(nn.Module):
    """Intrinsic Curiosity Module."""

    def __init__(self, state_dim, action_dim, feature_dim=128):
        super().__init__()

        # Feature encoder (shared)
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim),
        )

        # Forward model: predict next state features from (state, action)
        self.forward_model = nn.Sequential(
            nn.Linear(feature_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim),
        )

        # Inverse model: predict action from (state, next_state)
        self.inverse_model = nn.Sequential(
            nn.Linear(feature_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, state, action, next_state):
        # Encode states
        phi_s = self.encoder(state)
        phi_ns = self.encoder(next_state)

        # Forward prediction
        action_onehot = F.one_hot(action, num_classes=11).float()
        pred_phi_ns = self.forward_model(torch.cat([phi_s, action_onehot], dim=-1))

        # Inverse prediction
        pred_action = self.inverse_model(torch.cat([phi_s, phi_ns], dim=-1))

        # Intrinsic reward = forward prediction error
        intrinsic_reward = F.mse_loss(pred_phi_ns, phi_ns, reduction='none').mean(dim=-1)

        # Inverse loss for training encoder
        inverse_loss = F.cross_entropy(pred_action, action)

        # Forward loss for training forward model
        forward_loss = F.mse_loss(pred_phi_ns, phi_ns.detach())

        return intrinsic_reward, inverse_loss, forward_loss
```

**Integration with PPO:**

```python
class CuriousPPO:
    """PPO with Intrinsic Curiosity Module."""

    def __init__(self, env):
        self.policy = MaskablePPO("MlpPolicy", env)
        self.icm = ICM(state_dim=171, action_dim=11)
        self.beta = 0.1  # Intrinsic reward weight

    def compute_rewards(self, states, actions, next_states, extrinsic_rewards):
        """Combine extrinsic and intrinsic rewards."""
        intrinsic_rewards, inv_loss, fwd_loss = self.icm(
            states, actions, next_states
        )

        # Normalize intrinsic rewards
        intrinsic_rewards = (intrinsic_rewards - intrinsic_rewards.mean()) / (
            intrinsic_rewards.std() + 1e-8
        )

        total_rewards = extrinsic_rewards + self.beta * intrinsic_rewards

        return total_rewards, inv_loss, fwd_loss

    def train_step(self, batch):
        # Compute combined rewards
        rewards, inv_loss, fwd_loss = self.compute_rewards(
            batch.states, batch.actions, batch.next_states, batch.rewards
        )

        # Update ICM
        icm_loss = inv_loss + fwd_loss
        self.icm_optimizer.zero_grad()
        icm_loss.backward()
        self.icm_optimizer.step()

        # Update policy with combined rewards
        batch.rewards = rewards
        self.policy.train_on_batch(batch)
```

### 6.4 Random Network Distillation (RND)

Simpler approach: reward novelty based on prediction error of a random target:

```python
class RND(nn.Module):
    """Random Network Distillation for exploration."""

    def __init__(self, state_dim, feature_dim=128):
        super().__init__()

        # Target network (random, frozen)
        self.target = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim),
        )

        # Freeze target
        for param in self.target.parameters():
            param.requires_grad = False

        # Predictor network (trained to match target)
        self.predictor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim),
        )

    def forward(self, state):
        # Target output (fixed random features)
        target_features = self.target(state)

        # Predictor tries to match
        pred_features = self.predictor(state)

        # Intrinsic reward = prediction error
        # Novel states have high error (predictor hasn't seen them)
        intrinsic_reward = F.mse_loss(pred_features, target_features, reduction='none')
        intrinsic_reward = intrinsic_reward.mean(dim=-1)

        return intrinsic_reward

    def train_predictor(self, states):
        """Train predictor to reduce novelty of visited states."""
        target_features = self.target(states).detach()
        pred_features = self.predictor(states)

        loss = F.mse_loss(pred_features, target_features)
        return loss
```

### 6.5 Empowerment (Information-Theoretic Exploration)

Maximize agent's control over future states:

```python
class Empowerment(nn.Module):
    """
    Empowerment = I(a; s' | s) = H(s' | s) - H(s' | s, a)

    Agent seeks states where its actions have maximum influence.
    """

    def __init__(self, state_dim, action_dim):
        super().__init__()

        # Forward dynamics model P(s' | s, a)
        self.dynamics = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, state_dim * 2),  # Mean and log_std
        )

        # Source distribution q(a | s) for variational bound
        self.source = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def compute_empowerment(self, state, num_samples=16):
        """Estimate empowerment via variational bound."""
        batch_size = state.shape[0]

        # Sample actions from source
        action_logits = self.source(state)
        action_dist = Categorical(logits=action_logits)
        actions = action_dist.sample((num_samples,))  # [samples, batch]

        # Predict next states
        actions_onehot = F.one_hot(actions, num_classes=11).float()
        state_expanded = state.unsqueeze(0).expand(num_samples, -1, -1)

        dynamics_input = torch.cat([state_expanded, actions_onehot], dim=-1)
        dynamics_out = self.dynamics(dynamics_input)

        mean, log_std = dynamics_out.chunk(2, dim=-1)
        next_state_dist = Normal(mean, log_std.exp())

        # Estimate mutual information
        # I(a; s' | s) ≈ log q(a | s) - log p(a | s, s')

        log_q_a = action_dist.log_prob(actions)

        # Approximate p(a | s, s') with learned inverse model
        # (simplified: use uniform prior)
        log_p_a = -np.log(11)  # Uniform

        empowerment = (log_q_a - log_p_a).mean(dim=0)

        return empowerment
```

### 6.6 Scheduled Intrinsic Motivation

Decay curiosity as agent masters the game:

```python
class ScheduledCuriosity:
    """Decay intrinsic motivation over training."""

    def __init__(self, initial_beta=0.5, final_beta=0.01, decay_steps=1_000_000):
        self.initial_beta = initial_beta
        self.final_beta = final_beta
        self.decay_steps = decay_steps
        self.current_step = 0

    def get_beta(self):
        """Get current intrinsic reward weight."""
        progress = min(self.current_step / self.decay_steps, 1.0)

        # Exponential decay
        beta = self.initial_beta * (self.final_beta / self.initial_beta) ** progress

        return beta

    def step(self):
        self.current_step += 1
```

### 6.7 References

- Pathak et al. (2017). "Curiosity-driven Exploration by Self-Supervised Prediction"
- Burda et al. (2018). "Exploration by Random Network Distillation"
- Salge, Glackin & Polani (2014). "Empowerment – An Introduction"

---

## 7. Population-Based Training with League

### 7.1 Concept

Train a population of diverse agents that compete against each other, similar to AlphaStar:

```
┌─────────────────────────────────────────────────────────────┐
│                      AGENT POPULATION                        │
│                                                              │
│  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐      │
│  │ A₁  │  │ A₂  │  │ A₃  │  │ A₄  │  │ A₅  │  │ ...│      │
│  │Main │  │Main │  │Expl.│  │Expl.│  │Past │  │     │      │
│  └──┬──┘  └──┬──┘  └──┬──┘  └──┬──┘  └──┬──┘  └─────┘      │
│     │        │        │        │        │                   │
│     └────────┴────────┴────────┴────────┘                   │
│                       │                                      │
│                       ▼                                      │
│              ┌─────────────────┐                             │
│              │   MATCHMAKING   │                             │
│              │  (prioritized)  │                             │
│              └─────────────────┘                             │
│                       │                                      │
│                       ▼                                      │
│              ┌─────────────────┐                             │
│              │   PLAY GAMES    │                             │
│              │   & LEARN       │                             │
│              └─────────────────┘                             │
└─────────────────────────────────────────────────────────────┘
```

### 7.2 Agent Types

1. **Main Agents**: Primary agents being trained
2. **Exploiters**: Agents specifically trained to beat main agents
3. **League Exploiters**: Agents trained to beat entire league
4. **Past Checkpoints**: Frozen snapshots of main agents

### 7.3 Implementation

```python
class LeagueTraining:
    """Population-based training with prioritized matchmaking."""

    def __init__(self, num_main=4, num_exploiters=4, checkpoint_interval=100_000):
        self.population = {
            "main": [self.create_agent(f"main_{i}") for i in range(num_main)],
            "exploiter": [self.create_agent(f"exploiter_{i}") for i in range(num_exploiters)],
            "past": [],  # Frozen checkpoints
        }

        # Win rates for prioritized matchmaking
        self.matchup_stats = defaultdict(lambda: {"wins": 0, "games": 0})

        self.checkpoint_interval = checkpoint_interval
        self.total_games = 0

    def create_agent(self, name):
        """Create a new agent."""
        return {
            "name": name,
            "model": MaskablePPO("MlpPolicy", env),
            "games_played": 0,
            "elo": 1000,
        }

    def select_opponent(self, agent):
        """Prioritized fictitious self-play matchmaking."""
        agent_type = agent["name"].split("_")[0]

        if agent_type == "main":
            # Main agents play diverse opponents
            opponent_pool = (
                self.population["main"] +
                self.population["exploiter"] +
                self.population["past"][-10:]  # Recent checkpoints
            )
            # Prioritize opponents that beat us
            weights = []
            for opp in opponent_pool:
                key = (agent["name"], opp["name"])
                stats = self.matchup_stats[key]
                if stats["games"] > 0:
                    # Higher weight for opponents we lose to
                    win_rate = stats["wins"] / stats["games"]
                    weights.append(1.0 - win_rate + 0.1)
                else:
                    weights.append(1.0)  # Unexplored matchup

        elif agent_type == "exploiter":
            # Exploiters focus on main agents
            opponent_pool = self.population["main"]
            weights = [1.0] * len(opponent_pool)

        # Remove self from pool
        valid_idx = [i for i, o in enumerate(opponent_pool) if o["name"] != agent["name"]]
        opponent_pool = [opponent_pool[i] for i in valid_idx]
        weights = [weights[i] for i in valid_idx]

        # Sample opponent
        weights = np.array(weights) / sum(weights)
        opponent = np.random.choice(opponent_pool, p=weights)

        return opponent

    def play_match(self, agents):
        """Play a 4-player game and return results."""
        env = SkullKingEnv()
        obs = env.reset()

        done = False
        while not done:
            for i, agent in enumerate(agents):
                if env.current_player == i:
                    action, _ = agent["model"].predict(obs)
                    obs, reward, done, info = env.step(action)
                    if done:
                        break

        # Get final rankings
        rankings = env.get_rankings()  # [(player_idx, score), ...]

        return rankings

    def update_elo(self, agents, rankings):
        """Update ELO ratings based on game result."""
        K = 32  # ELO K-factor

        # Compute expected scores
        for i, agent_i in enumerate(agents):
            for j, agent_j in enumerate(agents):
                if i >= j:
                    continue

                # Expected score
                expected_i = 1 / (1 + 10 ** ((agent_j["elo"] - agent_i["elo"]) / 400))

                # Actual score (1 if i beat j, 0.5 if tie, 0 if lost)
                rank_i = next(r for r, (idx, _) in enumerate(rankings) if idx == i)
                rank_j = next(r for r, (idx, _) in enumerate(rankings) if idx == j)

                if rank_i < rank_j:
                    actual_i = 1.0
                elif rank_i > rank_j:
                    actual_i = 0.0
                else:
                    actual_i = 0.5

                # Update ELO
                agent_i["elo"] += K * (actual_i - expected_i)
                agent_j["elo"] += K * ((1 - actual_i) - (1 - expected_i))

    def train_step(self):
        """One training step for all agents."""
        for agent in self.population["main"] + self.population["exploiter"]:
            # Select opponents
            opponents = [self.select_opponent(agent) for _ in range(3)]
            all_players = [agent] + opponents
            np.random.shuffle(all_players)  # Randomize seating

            # Play game
            rankings = self.play_match(all_players)

            # Update stats
            self.update_elo(all_players, rankings)
            self.update_matchup_stats(all_players, rankings)

            # Train on experience
            agent["model"].learn(total_timesteps=1)
            agent["games_played"] += 1

        self.total_games += 1

        # Checkpoint main agents
        if self.total_games % self.checkpoint_interval == 0:
            for agent in self.population["main"]:
                checkpoint = {
                    "name": f"{agent['name']}_t{self.total_games}",
                    "model": agent["model"].copy(),  # Freeze copy
                    "elo": agent["elo"],
                    "games_played": agent["games_played"],
                }
                self.population["past"].append(checkpoint)

    def train(self, total_steps=10_000_000):
        """Full training loop."""
        steps = 0
        while steps < total_steps:
            self.train_step()
            steps += len(self.population["main"]) + len(self.population["exploiter"])

            if steps % 100_000 == 0:
                self.log_progress()

    def log_progress(self):
        """Log training progress."""
        print(f"\n{'='*60}")
        print(f"Games played: {self.total_games:,}")
        print(f"\nELO Rankings:")

        all_agents = (
            self.population["main"] +
            self.population["exploiter"] +
            self.population["past"][-5:]
        )

        for agent in sorted(all_agents, key=lambda a: -a["elo"]):
            print(f"  {agent['name']:20s}: {agent['elo']:.0f}")
```

### 7.4 Hyperparameter Mutation

Allow population to explore hyperparameter space:

```python
class PBTTraining(LeagueTraining):
    """Population-Based Training with hyperparameter evolution."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Hyperparameters to evolve
        self.hp_ranges = {
            "learning_rate": (1e-5, 1e-3, "log"),
            "ent_coef": (0.001, 0.1, "log"),
            "gamma": (0.95, 0.999, "linear"),
            "gae_lambda": (0.9, 0.99, "linear"),
        }

    def mutate_hyperparameters(self, agent):
        """Mutate agent's hyperparameters."""
        for hp_name, (low, high, scale) in self.hp_ranges.items():
            if np.random.random() < 0.2:  # 20% mutation chance
                current = getattr(agent["model"], hp_name)

                if scale == "log":
                    # Perturb in log space
                    log_current = np.log(current)
                    log_new = log_current + np.random.normal(0, 0.2)
                    new_value = np.clip(np.exp(log_new), low, high)
                else:
                    # Linear perturbation
                    new_value = current + np.random.normal(0, (high - low) * 0.1)
                    new_value = np.clip(new_value, low, high)

                setattr(agent["model"], hp_name, new_value)

    def exploit_and_explore(self):
        """Replace worst agents with mutated copies of best."""
        # Sort by ELO
        main_agents = sorted(
            self.population["main"],
            key=lambda a: a["elo"],
            reverse=True
        )

        # Bottom 20% copies from top 20%
        num_replace = max(1, len(main_agents) // 5)

        for i in range(num_replace):
            worst = main_agents[-(i+1)]
            best = main_agents[i]

            # Copy weights
            worst["model"].set_parameters(best["model"].get_parameters())
            worst["elo"] = best["elo"] - 50  # Small ELO penalty

            # Mutate hyperparameters
            self.mutate_hyperparameters(worst)
```

### 7.5 References

- Jaderberg et al. (2017). "Population Based Training of Neural Networks"
- Vinyals et al. (2019). "Grandmaster level in StarCraft II using multi-agent reinforcement learning"
- Lanctot et al. (2017). "A Unified Game-Theoretic Approach to Multiagent Reinforcement Learning"

---

## 8. Monte Carlo Tree Search + RL (AlphaZero-style)

### 8.1 Concept

Combine neural network evaluation with tree search planning:

```
┌─────────────────────────────────────────────────────────────┐
│                     MCTS + NEURAL NET                        │
│                                                              │
│     Neural Network (Policy + Value)                          │
│     ┌─────────────────────────────────┐                      │
│     │  Input: Game State               │                      │
│     │  Output: π(a|s), v(s)           │                      │
│     └─────────────────────────────────┘                      │
│                     │                                        │
│                     ▼                                        │
│     ┌─────────────────────────────────┐                      │
│     │         MCTS Planning            │                      │
│     │  1. SELECT (UCB)                │                      │
│     │  2. EXPAND (use π for priors)   │                      │
│     │  3. EVALUATE (use v for value)  │                      │
│     │  4. BACKUP                      │                      │
│     └─────────────────────────────────┘                      │
│                     │                                        │
│                     ▼                                        │
│              Best Action                                     │
└─────────────────────────────────────────────────────────────┘
```

### 8.2 MCTS for Skull King

```python
class MCTSNode:
    """Node in Monte Carlo search tree."""

    def __init__(self, state, parent=None, prior=1.0):
        self.state = state
        self.parent = parent
        self.prior = prior  # P(a|s) from neural network

        self.children = {}  # action -> MCTSNode
        self.visit_count = 0
        self.value_sum = 0.0

    @property
    def value(self):
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ucb_score(self, c_puct=1.0):
        """Upper Confidence Bound for trees."""
        if self.visit_count == 0:
            return float('inf')

        exploration = c_puct * self.prior * np.sqrt(self.parent.visit_count) / (1 + self.visit_count)

        return self.value + exploration

    def select_child(self):
        """Select child with highest UCB score."""
        return max(self.children.items(), key=lambda x: x[1].ucb_score())[1]

    def expand(self, action_priors):
        """Expand node with all legal actions."""
        for action, prior in action_priors.items():
            if action not in self.children:
                next_state = self.state.apply_action(action)
                self.children[action] = MCTSNode(next_state, parent=self, prior=prior)

    def backup(self, value):
        """Propagate value up the tree."""
        node = self
        while node is not None:
            node.visit_count += 1
            node.value_sum += value
            value = -value  # Flip for opponent (simplified for 2-player)
            node = node.parent


class AlphaZeroAgent:
    """AlphaZero-style agent for Skull King."""

    def __init__(self, state_dim, action_dim):
        # Neural network for policy and value
        self.network = DualHeadNetwork(state_dim, action_dim)

        # MCTS parameters
        self.num_simulations = 800
        self.c_puct = 1.0
        self.temperature = 1.0

    def search(self, root_state):
        """Run MCTS from root state."""
        root = MCTSNode(root_state)

        # Initial expansion
        policy, value = self.network(root_state.to_tensor())
        root.expand(self.mask_policy(policy, root_state.valid_actions()))

        for _ in range(self.num_simulations):
            node = root

            # SELECT: traverse to leaf
            while node.children:
                node = node.select_child()

            # EVALUATE with neural network
            if not node.state.is_terminal():
                policy, value = self.network(node.state.to_tensor())

                # EXPAND
                node.expand(self.mask_policy(policy, node.state.valid_actions()))
            else:
                value = node.state.get_utility(root_state.current_player)

            # BACKUP
            node.backup(value)

        return root

    def mask_policy(self, policy, valid_actions):
        """Mask invalid actions and renormalize."""
        masked = {a: policy[a] for a in valid_actions}
        total = sum(masked.values())
        return {a: p / total for a, p in masked.items()}

    def get_action(self, state, temperature=1.0):
        """Select action using MCTS."""
        root = self.search(state)

        # Get visit counts
        visits = {a: child.visit_count for a, child in root.children.items()}

        if temperature == 0:
            # Greedy selection
            return max(visits, key=visits.get)
        else:
            # Sample proportional to visit count
            actions = list(visits.keys())
            counts = np.array([visits[a] ** (1/temperature) for a in actions])
            probs = counts / counts.sum()
            return np.random.choice(actions, p=probs)

    def get_training_targets(self, root):
        """Get policy targets from MCTS visit counts."""
        total_visits = sum(child.visit_count for child in root.children.values())

        policy_target = np.zeros(11)
        for action, child in root.children.items():
            policy_target[action] = child.visit_count / total_visits

        value_target = root.value

        return policy_target, value_target


class DualHeadNetwork(nn.Module):
    """Neural network with policy and value heads."""

    def __init__(self, state_dim, action_dim):
        super().__init__()

        # Shared trunk
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )

        # Policy head
        self.policy_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1),
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh(),  # Value in [-1, 1]
        )

    def forward(self, state):
        features = self.trunk(state)
        policy = self.policy_head(features)
        value = self.value_head(features)
        return policy, value
```

### 8.3 Training Loop

```python
def train_alphazero(agent, num_iterations=1000, games_per_iteration=100):
    """AlphaZero training loop."""

    replay_buffer = ReplayBuffer(max_size=500_000)

    for iteration in range(num_iterations):
        # Self-play to generate training data
        for _ in range(games_per_iteration):
            game_data = self_play_game(agent)
            replay_buffer.extend(game_data)

        # Train network on replay buffer
        for _ in range(1000):
            batch = replay_buffer.sample(256)
            loss = train_network(agent.network, batch)

        # Evaluate against previous version
        if iteration % 10 == 0:
            win_rate = evaluate(agent, agent_previous)
            if win_rate > 0.55:
                agent_previous = agent.copy()
                print(f"New best agent at iteration {iteration}")

def self_play_game(agent):
    """Generate training data from self-play."""
    state = SkullKingState.new_game()
    game_data = []

    while not state.is_terminal():
        # Run MCTS
        root = agent.search(state)

        # Get training targets
        policy_target, _ = agent.get_training_targets(root)

        # Store state and policy target
        game_data.append({
            "state": state.to_tensor(),
            "policy_target": policy_target,
        })

        # Select action with temperature
        action = agent.get_action(state, temperature=1.0)
        state = state.apply_action(action)

    # Add value targets based on game outcome
    outcome = state.get_outcome()
    for i, data in enumerate(game_data):
        player = i % 4  # Determine which player's turn
        data["value_target"] = outcome[player]

    return game_data
```

### 8.4 Handling Imperfect Information

Skull King has hidden information. Adaptations:

```python
class InformationSetMCTS(AlphaZeroAgent):
    """MCTS for imperfect information games."""

    def search(self, info_set):
        """
        Search from information set (what player knows).
        Sample possible opponent hands for each simulation.
        """
        root = MCTSNode(info_set)

        for _ in range(self.num_simulations):
            # Sample a determinization (possible world)
            world = self.sample_world(info_set)

            node = root

            # SELECT
            while node.children and not world.is_terminal():
                node = node.select_child()
                world = world.apply_action(node.action)

            # EXPAND & EVALUATE
            if not world.is_terminal():
                policy, value = self.network(world.to_tensor())
                node.expand(self.mask_policy(policy, world.valid_actions()))
            else:
                value = world.get_utility(info_set.current_player)

            # BACKUP
            node.backup(value)

        return root

    def sample_world(self, info_set):
        """
        Sample a consistent game state given information set.

        Must respect:
        - Cards in our hand
        - Cards already played
        - Bids made by all players
        """
        # Cards we know about
        known_cards = set(info_set.our_hand) | set(info_set.played_cards)

        # Remaining cards
        remaining = [c for c in ALL_CARDS if c not in known_cards]
        np.random.shuffle(remaining)

        # Deal to opponents
        opponent_hands = []
        idx = 0
        for opp in range(3):
            hand_size = info_set.opponent_hand_sizes[opp]
            opponent_hands.append(remaining[idx:idx + hand_size])
            idx += hand_size

        return FullGameState(
            our_hand=info_set.our_hand,
            opponent_hands=opponent_hands,
            played_cards=info_set.played_cards,
            bids=info_set.bids,
        )
```

### 8.5 References

- Silver et al. (2017). "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm"
- Silver et al. (2018). "A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play"
- Cowling, Powley & Whitehouse (2012). "Information Set Monte Carlo Tree Search"

---

## 9. Opponent Modeling

### 9.1 Concept

Explicitly model opponent strategies to adapt play:

```
┌─────────────────────────────────────────────────────────────┐
│                   OPPONENT MODELING                          │
│                                                              │
│   Observations of opponent actions                           │
│   ┌─────────────────────────────────┐                        │
│   │  Opp1: bid 3 with hand strength X│                        │
│   │  Opp2: plays high cards early    │                        │
│   │  Opp3: always bids conservatively│                        │
│   └─────────────────────────────────┘                        │
│                     │                                        │
│                     ▼                                        │
│   ┌─────────────────────────────────┐                        │
│   │     OPPONENT MODEL NETWORK       │                        │
│   │  Predicts opponent's next action │                        │
│   └─────────────────────────────────┘                        │
│                     │                                        │
│                     ▼                                        │
│   ┌─────────────────────────────────┐                        │
│   │       ADAPTIVE POLICY            │                        │
│   │  Conditions on opponent model    │                        │
│   └─────────────────────────────────┘                        │
└─────────────────────────────────────────────────────────────┘
```

### 9.2 Opponent Embedding Network

```python
class OpponentModeler(nn.Module):
    """Learn embeddings of opponent strategies."""

    def __init__(self, action_dim=11, embed_dim=32):
        super().__init__()

        # Encode opponent action history
        self.action_encoder = nn.Embedding(action_dim + 1, 16)  # +1 for padding

        # RNN to process history
        self.history_rnn = nn.GRU(
            input_size=16 + 32,  # action + context
            hidden_size=64,
            num_layers=2,
            batch_first=True,
        )

        # Output opponent embedding
        self.embed_head = nn.Linear(64, embed_dim)

        # Predict opponent's next action
        self.predict_head = nn.Linear(64, action_dim)

    def forward(self, opponent_history, context):
        """
        Args:
            opponent_history: [batch, seq_len] past actions
            context: [batch, seq_len, 32] game context per action

        Returns:
            opponent_embedding: [batch, embed_dim]
            action_prediction: [batch, action_dim]
        """
        # Encode actions
        action_embeds = self.action_encoder(opponent_history)

        # Combine with context
        rnn_input = torch.cat([action_embeds, context], dim=-1)

        # Process history
        _, hidden = self.history_rnn(rnn_input)
        hidden = hidden[-1]  # Last layer

        # Outputs
        embedding = self.embed_head(hidden)
        prediction = F.softmax(self.predict_head(hidden), dim=-1)

        return embedding, prediction


class AdaptivePolicy(nn.Module):
    """Policy that conditions on opponent models."""

    def __init__(self, state_dim, action_dim, num_opponents=3):
        super().__init__()

        self.opponent_modeler = OpponentModeler()

        # Main policy with opponent conditioning
        self.policy = nn.Sequential(
            nn.Linear(state_dim + 32 * num_opponents, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, state, opponent_histories, contexts):
        # Get opponent embeddings
        opponent_embeds = []
        for i in range(3):
            embed, _ = self.opponent_modeler(
                opponent_histories[i],
                contexts[i]
            )
            opponent_embeds.append(embed)

        # Concatenate with state
        full_input = torch.cat([state] + opponent_embeds, dim=-1)

        # Output policy
        return F.softmax(self.policy(full_input), dim=-1)
```

### 9.3 Theory of Mind

Model what opponents think about us:

```python
class TheoryOfMind(nn.Module):
    """
    Recursive opponent modeling:
    - Level 0: Opponent plays randomly
    - Level 1: Opponent models us as Level 0
    - Level 2: Opponent models us as Level 1
    - ...
    """

    def __init__(self, state_dim, action_dim, max_level=2):
        super().__init__()

        self.max_level = max_level

        # Policy for each level of reasoning
        self.level_policies = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim, 128),
                nn.ReLU(),
                nn.Linear(128, action_dim),
            )
            for _ in range(max_level + 1)
        ])

        # Belief network: estimate opponent's reasoning level
        self.belief_net = nn.Sequential(
            nn.Linear(state_dim + 64, 128),  # state + opponent history
            nn.ReLU(),
            nn.Linear(128, max_level + 1),
            nn.Softmax(dim=-1),
        )

    def get_action(self, state, opponent_history, level=None):
        """Get action at specified reasoning level."""
        if level is None:
            # Estimate opponent's level
            beliefs = self.belief_net(torch.cat([state, opponent_history], dim=-1))
            level = beliefs.argmax().item()

        # Best respond to opponent at level-1
        if level > 0:
            # Assume opponent uses level-1 policy
            opp_policy = self.level_policies[level - 1](state)
            # Adjust our state based on opponent prediction
            adjusted_state = self.anticipate_opponent(state, opp_policy)
        else:
            adjusted_state = state

        return self.level_policies[level](adjusted_state)
```

### 9.4 References

- He et al. (2016). "Opponent Modeling in Deep Reinforcement Learning"
- Rabinowitz et al. (2018). "Machine Theory of Mind"
- Foerster et al. (2018). "Learning with Opponent-Learning Awareness"

---

## 10. Implementation Roadmap

### 10.1 Priority Order

Based on expected impact vs implementation complexity:

| Priority | Technique | Impact | Complexity | Time Est. |
|----------|-----------|--------|------------|-----------|
| 1 | **Hierarchical RL** | High | Medium | 1-2 days |
| 2 | **Transformer Architecture** | High | Medium | 1-2 days |
| 3 | **Round-as-Episode** | Medium | Low | 0.5 day |
| 4 | **Intrinsic Motivation** | Medium | Low | 0.5 day |
| 5 | **Population Training** | High | High | 2-3 days |
| 6 | **MCTS + RL** | Very High | High | 3-5 days |
| 7 | **Deep CFR** | High | Very High | 5-7 days |
| 8 | **Opponent Modeling** | Medium | Medium | 1-2 days |

### 10.2 Recommended Implementation Order

```
Phase 1: Quick Wins (1-2 days)
├── Add round number to observations
├── Implement round-as-episode option
├── Add RND curiosity bonus
└── Measure baseline improvements

Phase 2: Hierarchical (2-3 days)
├── Implement ManagerEnv and WorkerEnv
├── Train policies separately
├── Joint fine-tuning
└── Compare to baseline

Phase 3: Architecture Upgrade (2-3 days)
├── Implement CardTransformer
├── Integrate with MaskablePPO
├── Train and compare
└── Attention visualization

Phase 4: Advanced Training (1 week)
├── Population-based training
├── League matchmaking
├── Self-play with past checkpoints
└── Final evaluation
```

### 10.3 Evaluation Metrics

```python
def comprehensive_evaluation(agent):
    """Evaluate agent on multiple dimensions."""

    metrics = {}

    # Win rate vs different opponents
    for opponent in ["random", "rule_easy", "rule_medium", "rule_hard"]:
        win_rate = evaluate_vs_opponent(agent, opponent, games=100)
        metrics[f"win_rate_{opponent}"] = win_rate

    # Bid accuracy
    bid_accuracy = evaluate_bid_accuracy(agent, games=100)
    metrics["bid_accuracy"] = bid_accuracy

    # Strategy diversity
    action_entropy = measure_action_entropy(agent, games=100)
    metrics["action_entropy"] = action_entropy

    # Head-to-head vs previous versions
    if hasattr(agent, "previous_version"):
        h2h = head_to_head(agent, agent.previous_version, games=100)
        metrics["h2h_improvement"] = h2h

    return metrics
```

### 10.4 File Structure

```
skullking/
├── app/
│   └── gym_env/
│       ├── skullking_env_masked.py      # Current env
│       ├── skullking_env_hierarchical.py # Hierarchical env
│       └── skullking_env_round.py        # Round-as-episode env
├── scripts/
│   ├── train_masked_ppo.py              # Current training
│   ├── train_hierarchical.py            # Hierarchical RL
│   ├── train_transformer.py             # Transformer policy
│   ├── train_population.py              # Population-based
│   └── train_alphazero.py               # MCTS + RL
├── models/
│   ├── transformer.py                   # CardTransformer
│   ├── hierarchical.py                  # Manager/Worker
│   ├── curiosity.py                     # ICM/RND
│   └── opponent_model.py                # Opponent modeling
└── ADVANCED_RL_TECHNIQUES.md            # This document
```

---

## Summary

This document covers 8 advanced RL/ML techniques applicable to Skull King:

1. **Hierarchical RL**: Natural decomposition into bidding and card-playing
2. **Meta-Learning**: Transfer across rounds with different card counts
3. **Transformers**: Attention for card relationships and variable hands
4. **CFR**: Game-theoretic approach for imperfect information
5. **Curiosity**: Intrinsic motivation for exploration
6. **Population Training**: Diverse agents prevent overfitting
7. **MCTS + RL**: Planning with neural evaluation
8. **Opponent Modeling**: Adaptive play against different strategies

The recommended starting point is **Hierarchical RL** combined with **Transformer architecture**, as these provide high impact with moderate implementation complexity and directly address the structural challenges of Skull King.
