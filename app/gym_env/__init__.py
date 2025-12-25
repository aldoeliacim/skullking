"""Gymnasium environment for Skull King.

Active environment:
- SkullKingEnvMasked: MaskablePPO-compatible env with action masking,
  dense rewards, and 190-dim observations (V6)

Legacy environments archived in archive/gym_env/:
- SkullKingEnv: Original env (no masking)
- SkullKingEnvEnhanced: Enhanced observations (no masking)
"""

from app.gym_env.skullking_env_masked import SkullKingEnvMasked

__all__ = ["SkullKingEnvMasked"]
