# Skull King

A modern Python implementation of the [Skull King](https://www.grandpabecksgames.com/pages/skull-king) card game with bot AI players and Gymnasium environment for reinforcement learning.

![Screenshot of Skull King online board game.](./screenshot.png)

Skull King is a trick-taking game where you bid the exact number of tricks you predict you'll win each round. Battle your rivals while keeping your bid afloat and seizing opportunities to sink your opponents! The pirate with the highest score earns the title of Captain of the Seven Seas!

## Features

### Core Game Engine
- Complete Skull King rules implementation
- Support for 2-7 players
- 10 rounds with escalating complexity
- All special cards (Skull King, Mermaids, Pirates, Kraken, White Whale)
- Comprehensive scoring system with bonus points

### AI Bot Players
- **RandomBot**: Baseline bot making random valid moves
- **RuleBasedBot**: Intelligent heuristic-based strategy with difficulty levels
- **RLBot**: Interface for reinforcement learning agents

### Reinforcement Learning
- **Gymnasium Environment**: OpenAI Gym-compatible environment
- Observation/action space encoding for RL algorithms
- Ready for PPO, DQN, A2C, etc.

### Modern Python Stack
- Python 3.11+ with type hints
- FastAPI WebSocket server for multiplayer
- Poetry for dependency management
- Pre-commit hooks with ruff linting
- Comprehensive test suite (46 tests)

## Installation

```bash
# Install Poetry if needed
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Activate virtual environment
poetry shell
```

## Quick Start

### Watch Bots Play

```bash
# 4 rule-based bots (default)
python scripts/play_bot_game.py

# 6 players with custom mix
python scripts/play_bot_game.py --players 6 --random 2
```

### Train RL Agents

```python
from stable_baselines3 import PPO
from app.gym_env import SkullKingEnv

env = SkullKingEnv(num_opponents=3, opponent_bot_type="rule_based")
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
model.save("skullking_ppo")
```

### Run Tests

```bash
pytest tests/
```

## Cards

### Suit Cards (14 each)
- Parrot (yellow)
- Pirate Map (green)
- Treasure Chest (purple)
- Jolly Roger (black - trump)

### Special Cards
- Skull King (1)
- Mermaid (2)
- Pirate (5)
- Escape (5)
- Kraken (1)
- White Whale (1)

> **Loot** and **Tigress** cards are not implemented due to complexity.

## Rules

### Card Hierarchy
1. **Skull King** beats Pirates
2. **Pirates** beat Mermaids
3. **Mermaids** beat Skull King
4. **Special**: Mermaid wins if Skull King + Pirate + Mermaid all present
5. **Jolly Roger** (trump) beats standard suits
6. **Kraken**: No one wins the trick
7. **White Whale**: Highest suit card wins
8. **Escape**: Always loses

### Following Suit
- If a suit card leads, you must follow that suit if able
- Special cards (Mermaid, Pirate, Skull King, etc.) have no suit to follow
- Escape defers suit-setting to next player

### Scoring

**Bidding One or More:**
- Correct bid: 20 points per trick won + bonus points
- Wrong bid: -10 points per trick off

**Bidding Zero:**
- Correct: +10 points × round number
- Wrong: -10 points × round number

### Bonus Points
- Standard 14s captured: +10 each
- Jolly Roger 14 captured: +20
- Pirate captures Mermaid: +20
- Skull King captures Pirate: +30
- Mermaid captures Skull King: +40

## Project Structure

```
skullking/
├── app/
│   ├── api/                 # FastAPI routes & WebSocket
│   │   ├── routes.py
│   │   ├── websocket.py
│   │   └── game_handler.py  # Game logic handler
│   ├── models/              # Game domain models
│   ├── bots/                # AI bot players
│   ├── gym_env/             # Gymnasium environment
│   ├── repositories/        # Data access
│   └── services/            # Business logic
├── tests/                   # Test suite
├── scripts/                 # Utility scripts
└── pyproject.toml
```

## Development

### Code Quality

```bash
# Run pre-commit hooks
pre-commit run --all-files

# Run tests
pytest tests/ -v

# Type checking
mypy app/
```

### Environment Variables

Copy `.env.example` to `.env`:

```bash
ENABLE_BOTS=true
BOT_THINK_TIME_MIN=0.5
BOT_THINK_TIME_MAX=2.0
DEFAULT_BOT_STRATEGY=rule_based
MAX_PLAYERS=7
ROUNDS_COUNT=10
```

## Production Deployment

### Docker

```bash
docker compose -f docker-compose-production.yml up -d --build
```

### Nginx Configuration

```nginx
server {
  listen 80;
  server_name api.skullking.ir;
  return 301 https://$host$request_uri;
}

server {
  listen 443 ssl;
  ssl_certificate /etc/letsencrypt/live/skullking.ir/fullchain.pem;
  ssl_certificate_key /etc/letsencrypt/live/skullking.ir/privkey.pem;
  server_name api.skullking.ir;

  location / {
    proxy_pass http://127.0.0.1:3002/;
    proxy_set_header Host $host;
    proxy_set_header X-Forwarded-Host $http_host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
  }

  location /games/join {
    proxy_pass http://127.0.0.1:3002$request_uri;
    proxy_set_header Host $host;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_read_timeout 300;
    proxy_connect_timeout 300;
    proxy_send_timeout 300;
  }
}
```

## Resources

- [Skull King Official Rules](https://www.grandpabecksgames.com/products/skull-king)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

## License

See LICENSE file for details.
