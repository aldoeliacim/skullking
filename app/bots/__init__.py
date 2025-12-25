"""Bot AI players for Skull King.

Available bots:
- RandomBot: Plays random valid cards
- RuleBasedBot: Strategic play with difficulty levels (easy/medium/hard)

For RL self-play, see SelfPlayBot in app.gym_env.skullking_env_masked
(legacy PPO version archived in archive/bots/)
"""

from app.bots.base_bot import BaseBot, BotDifficulty
from app.bots.random_bot import RandomBot
from app.bots.rule_based_bot import RuleBasedBot

__all__ = ["BaseBot", "BotDifficulty", "RandomBot", "RuleBasedBot"]
