# Skull King - Modern Python Implementation

A complete rewrite of the Skull King card game in modern Python with bot AI players and Gymnasium environment for reinforcement learning.

## 🎮 Features

### Core Game Engine
- ✅ Complete Skull King game rules implementation
- ✅ Support for 2-7 players
- ✅ 10 rounds with escalating complexity
- ✅ All special cards and rules (Skull King, Mermaids, Pirates, Kraken, White Whale)
- ✅ Comprehensive scoring system with bonus points

### AI Bot Players
- ✅ **RandomBot**: Baseline bot making random valid moves
- ✅ **RuleBasedBot**: Intelligent heuristic-based strategy
  - Smart bidding based on hand strength
  - Strategic card play (win/lose tactics)
  - Special card handling
  - Difficulty levels (Easy, Medium, Hard)
- ✅ **RLBot**: Interface for reinforcement learning agents

### Reinforcement Learning
- ✅ **Gymnasium Environment**: Full OpenAI Gym-compatible environment
  - Observation space: Game state encoding (hand, bids, scores, tricks)
  - Action space: Bidding and card selection
  - Reward shaping: Score-based with bonuses
  - Multi-agent support: Train against bot opponents
- ✅ Ready for RL algorithm integration (PPO, DQN, A2C, etc.)

### Modern Python Stack
- ✅ Python 3.11+ with type hints throughout
- ✅ Poetry for dependency management
- ✅ Pydantic V2 for data validation
- ✅ FastAPI ready (async web framework)
- ✅ Comprehensive test suite with pytest
- ✅ Code quality tools (Black, Ruff, MyPy)

## 📦 Installation

### Prerequisites
- Python 3.11 or higher
- Poetry (for dependency management)

### Setup

```bash
# Install Poetry if you haven't
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
cd skullking
poetry install

# Activate virtual environment
poetry shell

# Or run commands with poetry run
poetry run python scripts/play_bot_game.py
```

## 🤖 Playing with Bots

### Watch Bots Play

Watch AI bots play a complete game:

```bash
# 4 rule-based bots (default)
python scripts/play_bot_game.py

# 6 players with custom mix
python scripts/play_bot_game.py --players 6 --random 2

# This creates 2 random bots and 4 rule-based bots
```

Example output:
```
============================================================
Setting up Skull King game with 4 bots
============================================================

  Player 1: Bot1 (RuleBasedBot (medium))
  Player 2: Bot2 (RuleBasedBot (medium))
  Player 3: Bot3 (RuleBasedBot (medium))
  Player 4: Bot4 (RuleBasedBot (medium))

============================================================
ROUND 1
============================================================

BIDDING PHASE
----------------------------------------
Bot1 bids: 1
Bot2 bids: 0
Bot3 bids: 1
Bot4 bids: 0

  Trick 1:
  ------------------------------------
    Bot1: Parrot5
    Bot2: Escape1
    Bot3: Roger7
    Bot4: Chest3
  → Winner: Bot3 with Roger7

ROUND RESULTS
----------------------------------------
Bot1: Bid 1, Won 0 ✗ | Score: -10 (Total: -10)
Bot2: Bid 0, Won 0 ✓ | Score: +10 (Total: +10)
Bot3: Bid 1, Won 1 ✓ | Score: +20 (Total: +20)
Bot4: Bid 0, Won 0 ✓ | Score: +10 (Total: +10)
```

## 🏋️ Training Reinforcement Learning Agents

### Using the Gymnasium Environment

```python
from app.gym_env import SkullKingEnv

# Create environment
env = SkullKingEnv(
    num_opponents=3,
    opponent_bot_type="rule_based",  # or "random"
    render_mode="human"  # or "ansi" or None
)

# Reset environment
observation, info = env.reset(seed=42)

# Take actions
while True:
    action = env.action_space.sample()  # Your RL agent here
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        break

env.close()
```

### Run Example

```bash
# Single episode with rendering
python scripts/gym_example.py --mode single

# Multiple episodes for statistics
python scripts/gym_example.py --mode multiple
```

### Observation Space

The observation is a flattened vector containing:
- **Player's hand**: One-hot encoding of cards (10 slots × 71 card types)
- **Current trick cards**: Cards played so far (7 slots × 71 card types)
- **Bids**: All player bids (7 players × 11 possible bids)
- **Scores**: Current scores for all players (normalized)
- **Tricks won**: Tricks won this round per player
- **Metadata**: Round number, game phase, etc.

### Action Space

- **During bidding**: Discrete(11) - Bid 0 to 10
- **During picking**: Discrete(10) - Pick card at index 0-9

### Reward Function

- **Round end**: +/- score delta
- **Invalid moves**: -0.5 to -1.0 penalty
- **Game end**: Bonus based on final ranking (+50 for 1st, -25 for last)

### Training with Stable-Baselines3

```python
from stable_baselines3 import PPO
from app.gym_env import SkullKingEnv

# Create environment
env = SkullKingEnv(num_opponents=3, opponent_bot_type="rule_based")

# Train PPO agent
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

# Save model
model.save("skullking_ppo")

# Load and use
model = PPO.load("skullking_ppo")
obs, info = env.reset()
action, _states = model.predict(obs)
```

## 🧪 Running Tests

```bash
# Run all tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=app --cov-report=html

# Run specific test file
poetry run pytest tests/test_models/test_card.py

# Run with verbose output
poetry run pytest -v
```

## 🏗️ Project Structure

```
skullking/
├── app/
│   ├── models/              # Game domain models
│   │   ├── card.py          # Card definitions and winner logic
│   │   ├── deck.py          # Deck shuffling and dealing
│   │   ├── player.py        # Player state
│   │   ├── round.py         # Round logic
│   │   ├── trick.py         # Trick resolution
│   │   ├── game.py          # Game orchestration
│   │   └── enums.py         # Enums and constants
│   ├── bots/                # AI bot players
│   │   ├── base_bot.py      # Abstract bot interface
│   │   ├── random_bot.py    # Random strategy
│   │   ├── rule_based_bot.py # Heuristic strategy
│   │   └── rl_bot.py        # RL agent interface
│   ├── gym_env/             # Gymnasium environment
│   │   └── skullking_env.py # RL environment
│   ├── api/                 # FastAPI handlers (TODO)
│   ├── repositories/        # Data access (TODO)
│   └── config.py            # Configuration
├── tests/                   # Test suite
│   ├── test_models/
│   └── test_bots/
├── scripts/
│   ├── play_bot_game.py     # Watch bots play
│   └── gym_example.py       # Gym environment demo
├── pyproject.toml           # Poetry configuration
└── README_PYTHON.md         # This file
```

## 🎯 Bot Strategies

### RandomBot
- **Bidding**: Random number between 0 and round number
- **Playing**: Random valid card
- **Use case**: Baseline for evaluation

### RuleBasedBot
- **Bidding Strategy**:
  - Evaluates hand strength (Skull King = 0.9, Pirates = 0.6, etc.)
  - Sums expected trick wins
  - Adds randomness based on difficulty level
- **Playing Strategy**:
  - Calculates if it needs to win current trick
  - Plays strong cards when trying to win
  - Plays weak cards when trying to lose
  - Adjusts based on difficulty (easy bots make mistakes)

### Difficulty Levels
- **Easy**: More random decisions, less accurate bidding
- **Medium**: Balanced strategy
- **Hard**: Accurate bidding, optimal play

## 📊 Game Rules

### Deck Composition (63 cards)
- 1 Skull King
- 1 White Whale, 1 Kraken
- 2 Mermaids
- 5 Pirates
- 14 Jolly Rogers (trump suit)
- 14 Parrots, 14 Maps, 14 Chests (standard suits)
- 5 Escapes

### Card Hierarchy
1. **Skull King** beats Pirates
2. **Pirates** beat Mermaids
3. **Mermaids** beat Skull King
4. **Special**: Mermaid wins if Skull King + Pirate + Mermaid present
5. **Jolly Roger** (trump) beats standard suits
6. **Highest card of same type** wins
7. **Kraken**: No one wins the trick
8. **White Whale**: Highest suit card wins
9. **Escape**: Always loses

### Scoring
- **Bid correct (non-zero)**: 20 × bid + bonus points
- **Bid correct (zero)**: 10 × round number
- **Bid wrong (non-zero)**: -10 × difference
- **Bid wrong (zero)**: -10 × round number

### Bonus Points
- Capturing 14 of standard suits: +10 each
- Capturing 14 of Jolly Roger: +20
- Pirate capturing Mermaid: +20
- Skull King capturing Pirate: +30
- Mermaid capturing Skull King: +40

## 🔬 Next Steps for RL Training

1. **Observation Engineering**: Refine observation space encoding
2. **Reward Shaping**: Tune rewards for better learning
3. **Algorithm Selection**: Try PPO, DQN, A2C
4. **Hyperparameter Tuning**: Optimize learning rate, batch size, etc.
5. **Self-Play**: Train agents against each other
6. **Curriculum Learning**: Start with easier opponents
7. **Evaluation**: Compare against rule-based bots

## 🚀 Future Enhancements

- [ ] FastAPI WebSocket server for multiplayer
- [ ] MongoDB repository layer
- [ ] Redis pub/sub for game events
- [ ] Web UI integration
- [ ] Advanced RL agents (multi-agent RL)
- [ ] Tournament system for bot evaluation
- [ ] ELO ratings for bots
- [ ] More bot strategies (MCTS, neural networks)

## 📝 Development

### Code Quality

```bash
# Format code
poetry run black app/ tests/

# Lint code
poetry run ruff check app/ tests/

# Type checking
poetry run mypy app/
```

### Environment Variables

Copy `.env.example` to `.env` and configure:

```bash
# Bot Configuration
ENABLE_BOTS=true
BOT_THINK_TIME_MIN=0.5
BOT_THINK_TIME_MAX=2.0
DEFAULT_BOT_STRATEGY=rule_based

# Game Configuration
MAX_PLAYERS=7
ROUNDS_COUNT=10
WAIT_TIME_SECONDS=15
```

## 📚 Additional Resources

- [Skull King Official Rules](https://www.grandpabecksgames.com/products/skull-king)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

## 🤝 Contributing

1. Write tests for new features
2. Follow Python type hints
3. Run code quality tools before committing
4. Document new bot strategies

## 📄 License

See LICENSE file for details.

---

**Ready to train your Skull King champion? Start with the Gymnasium environment and compete against the bots!** 🏴‍☠️👑
