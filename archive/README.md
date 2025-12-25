# Archive

This folder contains legacy code that is no longer actively used but preserved for reference.

## Contents

### scripts/
Legacy training and analysis scripts replaced by `app/training/`:
- `train_rl_agent.py` - Original PPO training (no action masking)
- `train_advanced_rl.py` - Curriculum learning experiments
- `train_selfplay.py` - Self-play concepts (now in train_ppo.py)
- `train_ultra_ppo.py` - Experimental training
- `train_masked_ppo_v2.py` - Intermediate version
- `optimize_hyperparameters.py` - Optuna hyperparameter search
- `analyze_training.py` - Training log analysis
- `deep_analysis.py` - Detailed model analysis

### gym_env/
Legacy Gymnasium environments replaced by `SkullKingEnvMasked`:
- `skullking_env.py` - Original env (no masking, basic observations)
- `skullking_env_enhanced.py` - Enhanced observations (no masking)

### bots/
Legacy bot implementations:
- `selfplay_bot.py` - PPO-based self-play (replaced by MaskablePPO version in gym_env)

### logs/
Old training logs (current logs go to models/*/tensorboard)

## Current Active Code

- Training: `uv run python -m app.training.train_ppo`
- Gym Environment: `app.gym_env.SkullKingEnvMasked`
- Models: `models/masked_ppo/` (V5+)
