"""Game constants for Skull King."""

# Game limits
MAX_PLAYERS = 8
MIN_PLAYERS = 2
MAX_ROUNDS = 10

# Card mechanics
BENDT_DRAW_DISCARD_COUNT = 2  # Bendt draws 2 cards, then discards 2
HIGH_CARD_THRESHOLD = 10  # Cards with number >= this are "high"

# Publisher service
REDIS_PUBLISH_TIMEOUT = 5

# Scoring (for reference - actual logic in models/round.py)
BONUS_POINTS_14 = 10
BONUS_POINTS_CAPTURE = 20
BONUS_MERMAID_CAPTURES_SKULL_KING = 40
CORRECT_BID_MULTIPLIER = 20
ZERO_BID_MULTIPLIER = 10
