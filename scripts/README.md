# Scripts

Utility scripts for development and testing.

## Available Scripts

### check_i18n.py
Check i18n translation files for missing keys between languages.
```bash
uv run python scripts/check_i18n.py
```

### gym_example.py
Example of using the Gymnasium environment for testing.
```bash
uv run python scripts/gym_example.py
```

### play_bot_game.py
Run a complete game with bots for testing and visualization.
```bash
uv run python scripts/play_bot_game.py
```

## Training

For RL training, use the training module:
```bash
# Train new model
uv run python -m app.training.train_ppo train --timesteps 10000000

# Resume training
uv run python -m app.training.train_ppo resume --load models/masked_ppo/best_model/best_model.zip
```

## Archived Scripts

Legacy and experimental scripts are in `archive/scripts/`.
