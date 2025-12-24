# Skull King

A modern Python implementation of the [Skull King](https://www.grandpabecksgames.com/pages/skull-king) card game with multiplayer WebSocket support, AI bots, and a Gymnasium environment for reinforcement learning.

![Screenshot of Skull King online board game.](./screenshot.png)

## Tech Stack

| Layer | Technology |
|-------|------------|
| **Backend** | Python 3.11+, FastAPI, WebSockets |
| **Frontend** | Vanilla JS, CSS3, i18n |
| **Database** | MongoDB (optional), Redis |
| **AI/ML** | Gymnasium, Stable-Baselines3, sb3-contrib |
| **DevOps** | Docker, Docker Compose, uv |
| **Quality** | Ruff, Pre-commit, Pytest |

## Features

- **Complete Rules**: All Skull King cards including pirates with abilities, Kraken, White Whale
- **Multiplayer**: Real-time WebSocket gameplay for 2-8 players
- **AI Bots**: RandomBot, RuleBasedBot, and trained RL agents
- **Reinforcement Learning**: Gymnasium environment for training agents
- **Internationalization**: English and Spanish support

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
2. **Pirates** beat Mermaids (+20 bonus)
3. **Mermaids** beat Skull King (+50 bonus)
4. **Special**: Mermaid wins if Skull King + Pirate + Mermaid all present
5. **Jolly Roger** (black trump) beats standard suits
6. **Kraken**: No one wins the trick
7. **White Whale**: Highest standard suit card wins
8. **Escape**: Always loses

### Scoring

| Bid | Result | Points |
|-----|--------|--------|
| 1+ tricks | Correct | +20 per trick + bonuses |
| 1+ tricks | Wrong | -10 per trick off |
| Zero | Correct | +10 x round number |
| Zero | Wrong | -10 x round number |

## Project Structure

```
skullking/
├── app/
│   ├── api/          # FastAPI routes, WebSocket, game handler
│   ├── models/       # Game domain models
│   ├── bots/         # AI bot implementations
│   ├── gym_env/      # Gymnasium RL environment
│   └── services/     # Business logic
├── static/           # Frontend (JS, CSS, i18n)
├── tests/            # Test suite
└── scripts/          # Training and utility scripts
```

## Training RL Agents

```python
from sb3_contrib import MaskablePPO
from app.gym_env import SkullKingEnvMasked

env = SkullKingEnvMasked(num_opponents=3)
model = MaskablePPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1_000_000)
model.save("models/skull_king_ppo")
```

Or use the training script:

```bash
uv run python scripts/train_masked_ppo.py
```

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
MONGODB_URL=mongodb://localhost:27017
REDIS_URL=redis://localhost:6379
JWT_SECRET=your-secret-key
ENABLE_BOTS=true
FRONTEND_URL=http://localhost:8000
```

## Resources

- [Official Skull King Rules](https://www.grandpabecksgames.com/products/skull-king)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- [FastAPI](https://fastapi.tiangolo.com/)
