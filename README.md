# Skull King

A modern Python implementation of the [Skull King](https://www.grandpabecksgames.com/pages/skull-king) card game with multiplayer WebSocket support, AI bots, and a Gymnasium environment for reinforcement learning.

![Screenshot of Skull King online board game.](./screenshot.png)

## Tech Stack

| Layer | Technology |
|-------|------------|
| **Backend** | Python 3.11+, FastAPI, WebSockets |
| **Frontend** | React Native (Expo), TypeScript, Zustand |
| **Database** | MongoDB, Redis (pub/sub) |
| **AI/ML** | Gymnasium, Stable-Baselines3, MaskablePPO |
| **DevOps** | Docker, Docker Compose, uv, Bun |
| **Quality** | Ruff, Oxlint, Biome, Pytest, Pre-commit |

## Features

- **Complete Rules**: All 74 cards including Skull King, Pirates with abilities, Mermaids, Kraken, White Whale, Loot, and Escape cards
- **Pirate Abilities**: Rosie (pick next leader), Bendt (draw 2/discard 2), Harry (adjust bid ±1), Jade (see undealt cards), Roatan (bonus bet)
- **Loot Alliances**: Form alliances with trick winners for +20 bonus when both make their bids
- **Multiplayer**: Real-time WebSocket gameplay for 2-8 players with spectator mode
- **AI Bots**: Easy/Medium/Hard rule-based bots and trained neural network (MaskablePPO)
- **Reinforcement Learning**: Gymnasium environment with action masking for training agents
- **Internationalization**: English and Spanish support
- **Cross-Platform**: Web, iOS, and Android via React Native (Expo)

## Quick Start

### Local Development

```bash
# Install uv (if needed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies and run
uv sync
uv run uvicorn app.main:app --reload --port 8000

# Open http://localhost:8000
```

### Docker

```bash
# Development (hot reload)
docker compose -f docker-compose.dev.yml up -d

# Production
docker compose up -d --build

# Access at http://localhost:8000
```

### Run Tests

```bash
uv run pytest tests/ -v
```

## Game Rules

### Card Hierarchy

1. **Skull King** beats Pirates (+30 bonus each)
2. **Pirates** (5 unique) beat Mermaids (+20 bonus)
3. **Mermaids** (2) beat Skull King (+40 bonus)
4. **Special**: Mermaid wins if Skull King + Pirate + Mermaid all present
5. **Jolly Roger** (black trump) beats standard suits
6. **Kraken**: No one wins the trick
7. **White Whale**: Highest standard suit card wins, destroys specials
8. **Loot** (2): Acts as Escape, forms alliance with trick winner
9. **Escape** (5): Always loses
10. **Scary Mary**: Choose Pirate or Escape when played

### Scoring

| Bid | Result | Points |
|-----|--------|--------|
| 1+ tricks | Correct | +20 per trick + bonuses |
| 1+ tricks | Wrong | -10 per trick off |
| Zero | Correct | +10 x round number |
| Zero | Wrong | -10 x round number |

**Bonuses**: Skull King captures Pirate (+30), Pirate captures Mermaid (+20), Mermaid captures Skull King (+40), Loot alliance (+20 each if both make bids), 14 of trump suit (+10 per 14)

## Project Structure

```
skullking/
├── app/
│   ├── api/          # FastAPI routes, WebSocket, game handler
│   ├── models/       # Game domain models (Card, Deck, Game, etc.)
│   ├── bots/         # AI bots (RandomBot, RuleBasedBot, RLBot)
│   ├── gym_env/      # Gymnasium RL environment (SkullKingEnvMasked)
│   ├── training/     # MaskablePPO training with curriculum learning
│   ├── repositories/ # MongoDB data access layer
│   └── services/     # Game serialization, persistence
├── frontend/         # React Native (Expo) app - see frontend/README.md
├── models/           # Trained RL model checkpoints
├── scripts/          # Utility scripts (see scripts/README.md)
├── static/           # Standalone web client for local play
├── archive/          # Legacy code preserved for reference
└── tests/            # Pytest test suite (319 tests)
```

## Training RL Agents

### Flat Environment (V8)

```python
from sb3_contrib import MaskablePPO
from app.gym_env import SkullKingEnvMasked

env = SkullKingEnvMasked(num_opponents=3)
model = MaskablePPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1_000_000)
model.save("models/skull_king_ppo")
```

```bash
# Train new model (10M steps, ~2.5 hours on RTX 4080)
uv run python -m app.training.train_ppo train --timesteps 10000000

# Resume from checkpoint
uv run python -m app.training.train_ppo resume --load models/masked_ppo/best_model/best_model.zip
```

### Hierarchical RL (V9)

Separate Manager (bidding) and Worker (card play) policies for better credit assignment:

```bash
# Train with V9 hierarchical approach
uv run python -m app.training.train_v9 --manager-timesteps 5000000 --worker-timesteps 10000000

# Features: phase curriculum, round-weighted sampling, phase embeddings
```

See [TRAINING_LOG.md](./TRAINING_LOG.md) for training history and [V9_OPTIMIZATION_PLAN.md](./V9_OPTIMIZATION_PLAN.md) for optimization details.

## Docker Services

| Service | Port | Description |
|---------|------|-------------|
| app | 8000 | FastAPI application |
| mongodb | 27017 | Game persistence |
| redis | 6379 | Pub/Sub messaging |

### Useful Commands

```bash
# View logs
docker compose logs -f app

# Run bot game in container
docker compose exec app python scripts/play_bot_game.py

# Access MongoDB
docker compose exec mongodb mongosh skullking

# Stop everything
docker compose down -v
```

## Development

```bash
# Lint and format
uv run ruff check . --fix
uv run ruff format .

# Run pre-commit
pre-commit run --all-files

# Type check
uv run mypy app/
```

## Environment Variables

```bash
MONGODB_HOST=localhost
MONGODB_PORT=27017
MONGODB_DATABASE=skullking
BROKER_REDIS_HOST=localhost
BROKER_REDIS_PORT=6379
JWT_SECRET=your-secret-key
ENABLE_BOTS=true
FRONTEND_URL=http://localhost:5173
RL_MODEL_PATH=models/masked_ppo/masked_ppo_final.zip
```

## Resources

- [Official Skull King Rules](https://www.grandpabecksgames.com/products/skull-king)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- [FastAPI](https://fastapi.tiangolo.com/)
