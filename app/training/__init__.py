"""Training module for Skull King RL agents.

Usage:
    # Train new model
    uv run python -m app.training.train_ppo train --timesteps 10000000

    # Resume training
    uv run python -m app.training.train_ppo resume --load models/masked_ppo/best_model/best_model.zip

Components:
    - train_ppo: Main MaskablePPO training with curriculum learning
    - callbacks: Training callbacks (Curriculum, MixedOpponentEval, SelfPlay)

See TRAINING_LOG.md for training history and hyperparameters.
"""

from app.training.callbacks import (
    CurriculumCallback,
    MixedOpponentEvalCallback,
    SelfPlayCallback,
)

__all__ = [
    "CurriculumCallback",
    "MixedOpponentEvalCallback",
    "SelfPlayCallback",
]
