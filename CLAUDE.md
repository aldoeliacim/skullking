# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Development Commands

### Backend (Python)
```bash
uv sync                                    # Install dependencies
uv run uvicorn app.main:app --reload       # Run dev server (port 8000)
uv run pytest tests/ -v                    # Run all tests
uv run pytest tests/test_models/ -v        # Run specific test directory
uv run pytest tests/test_X.py::test_name   # Run single test
uv run ruff check . --fix                  # Lint and auto-fix
uv run ruff format .                       # Format code
uv run mypy app/                           # Type check
pre-commit run --all-files                 # Run all pre-commit hooks
```

### Frontend (React Native/Expo)
```bash
cd frontend
bun install                                # Install dependencies
bun run start                              # Start Expo dev server
bun run build:web                          # Build for web (outputs to dist/)
bun run lint:check                         # Lint with oxlint + biome
bun run typecheck                          # TypeScript check
```

### Docker
```bash
docker compose up -d --build               # Build and run production
docker compose -f docker-compose.dev.yml up -d  # Development with hot reload
./scripts/deploy.sh all                    # Deploy frontend to PVE, backend to Docker
```

### RL Training
```bash
uv run python -m app.training.train_ppo train --timesteps 10000000
uv run python -m app.training.train_v9 --manager-timesteps 5000000
```

## Architecture

### Backend Layer Separation
The codebase enforces strict import boundaries via import-linter:
- `app/models/` → Pure domain logic, no API/DB imports
- `app/bots/` → AI implementations, no API imports
- `app/gym_env/` → Gymnasium environments, no training imports
- `app/api/` → FastAPI routes, WebSocket handlers
- `app/repositories/` → MongoDB data access
- `app/services/` → Cross-cutting concerns (serialization, pub/sub)

### Gymnasium Environments (app/gym_env/)
Three environment tiers with increasing complexity:
1. **SkullKingEnvMasked** - Flat MaskablePPO env (190-dim obs, single policy)
2. **WorkerEnv** (in skullking_env_hierarchical.py) - Card-play only (203-dim obs)
3. **AbilityAwareEnv** - Extends WorkerEnv with pirate ability decisions (243-dim obs)

All envs use action masking via `action_masks()` method for invalid action handling.

### Frontend State Management
- **Zustand stores** in `frontend/src/stores/`: `gameStore.ts`, `wsStore.ts`
- **WebSocket connection** in `frontend/src/services/websocket.ts`
- **Message handlers** in `frontend/src/stores/messageHandlers.ts`
- Routes: `/` (home), `/lobby/[id]`, `/game/[id]`

### Game Flow
1. WebSocket connects → `ws://.../ws/{game_id}/{player_id}`
2. All players bid → triggers `phase: "playing"`
3. Cards played in trick order → `GameHandler._complete_trick()`
4. Pirate abilities resolve (Rosie, Bendt, Harry, Roatan)
5. Round scoring in `Round.calculate_scores()`
6. After 10 rounds → final scores, game ends

### Key Domain Models (app/models/)
- **CardId enum** - 74 unique cards with IDs like `SKULL_KING`, `PIRATE_ROSIE`, `PARROT1`-`PARROT14`
- **Trick.get_valid_cards()** - Suit-following rules
- **Trick.determine_winner()** - Complex hierarchy (SK > Pirates > Mermaids, with exceptions)
- **Round.calculate_scores()** - Bid accuracy scoring with bonuses

## Code Quality Tools

Pre-commit runs: ruff, mypy, pyright, vulture, bandit, xenon, import-linter, gitleaks

**Pyright** catches class attribute shadowing bugs (e.g., child class overriding parent's `PHASE_DIM`).

**Beartype** provides runtime type checking on critical gym env methods.

**Hypothesis** generates property-based tests in `tests/test_gym_env/test_ability_env_hypothesis.py`.

## Error Handling Pattern

Backend uses `ErrorCode` enum (`app/api/responses.py`) for i18n-ready error messages:
- Error codes map to translation keys (e.g., `ErrorCode.NOT_YOUR_TURN` → `"error.notYourTurn"`)
- Frontend looks up localized strings in `src/i18n/{en,es}.json`
- All game handler errors use `await self._send_error(game_id, player_id, ErrorCode.ENUM_VALUE)`

## Testing Patterns

Tests use NumPy boolean compatibility: use `assert mask[0]` not `assert mask[0] is True`.

CardId format: `CardId.PARROT1` not `CardId.YELLOW_1` (suits are PARROT, PIRATE_MAP, TREASURE_CHEST, JOLLY_ROGER).
