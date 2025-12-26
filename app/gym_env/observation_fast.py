"""Numba-accelerated observation encoding for Skull King RL environments.

Pre-computes card properties into numpy arrays for fast JIT-compiled encoding.
Provides 2-5x speedup for observation generation.
"""

import numpy as np
from numba import njit

from app.models.card import CardId, CardType, get_card

# Pre-compute card properties for all 74 cards
# These arrays are indexed by card_id (1-74, 0 is unused)
NUM_CARDS = 75  # 0-74 inclusive

# Card property arrays (computed once at module load)
CARD_NUMBERS = np.zeros(NUM_CARDS, dtype=np.float32)
CARD_IS_STANDARD_SUIT = np.zeros(NUM_CARDS, dtype=np.bool_)
CARD_IS_ROGER = np.zeros(NUM_CARDS, dtype=np.bool_)
CARD_IS_PIRATE = np.zeros(NUM_CARDS, dtype=np.bool_)
CARD_IS_TIGRESS = np.zeros(NUM_CARDS, dtype=np.bool_)
CARD_IS_KING = np.zeros(NUM_CARDS, dtype=np.bool_)
CARD_IS_MERMAID = np.zeros(NUM_CARDS, dtype=np.bool_)
CARD_IS_ESCAPE = np.zeros(NUM_CARDS, dtype=np.bool_)
CARD_IS_LOOT = np.zeros(NUM_CARDS, dtype=np.bool_)
CARD_IS_WHALE = np.zeros(NUM_CARDS, dtype=np.bool_)
CARD_IS_KRAKEN = np.zeros(NUM_CARDS, dtype=np.bool_)
CARD_IS_BEAST = np.zeros(NUM_CARDS, dtype=np.bool_)


def _init_card_arrays() -> None:
    """Initialize card property arrays from Card objects."""
    for card_id in CardId:
        i = int(card_id)
        card = get_card(card_id)
        CARD_NUMBERS[i] = card.number if card.number else 0
        CARD_IS_STANDARD_SUIT[i] = card.is_standard_suit()
        CARD_IS_ROGER[i] = card.is_roger()
        CARD_IS_PIRATE[i] = card.is_pirate()
        CARD_IS_TIGRESS[i] = card.is_tigress()
        CARD_IS_KING[i] = card.is_king()
        CARD_IS_MERMAID[i] = card.is_mermaid()
        CARD_IS_ESCAPE[i] = card.is_escape()
        CARD_IS_LOOT[i] = card.is_loot()
        CARD_IS_WHALE[i] = card.is_whale()
        CARD_IS_KRAKEN[i] = card.is_kraken()
        CARD_IS_BEAST[i] = card.is_beast()


# Initialize arrays on module load
_init_card_arrays()


@njit(cache=True)
def encode_card_fast(
    card_id: int,
    card_numbers: np.ndarray,
    is_standard_suit: np.ndarray,
    is_roger: np.ndarray,
    is_pirate: np.ndarray,
    is_tigress: np.ndarray,
    is_king: np.ndarray,
    is_mermaid: np.ndarray,
    is_escape: np.ndarray,
    is_loot: np.ndarray,
    is_whale: np.ndarray,
    is_kraken: np.ndarray,
    is_beast: np.ndarray,
    output: np.ndarray,
) -> None:
    """Encode a single card into 9 features (JIT-compiled).

    Features:
    - [0-4]: Card type one-hot (standard, pirate, king, mermaid, escape)
    - [5]: Normalized card number
    - [6-8]: Special flags (high-power, king/beast, mermaid)
    """
    # Card type one-hot (5 dims)
    output[0:5] = 0.0

    if is_standard_suit[card_id] or is_roger[card_id]:
        output[0] = 1.0  # Standard suit
    elif is_pirate[card_id]:
        output[1] = 1.0  # Pirate
    elif is_tigress[card_id]:
        output[1] = 0.5  # Partial pirate
        output[4] = 0.5  # Partial escape
    elif is_king[card_id]:
        output[2] = 1.0  # King
    elif is_mermaid[card_id]:
        output[3] = 1.0  # Mermaid
    elif is_escape[card_id] or is_loot[card_id]:
        output[4] = 1.0  # Escape
    elif is_whale[card_id] or is_kraken[card_id]:
        output[2] = 0.5  # Beast power

    # Card number (1 dim)
    output[5] = card_numbers[card_id] / 14.0

    # Special flags (3 dims)
    output[6] = 1.0 if (is_pirate[card_id] or is_tigress[card_id] or is_beast[card_id]) else 0.0
    output[7] = 1.0 if (is_king[card_id] or is_beast[card_id]) else 0.0
    output[8] = 1.0 if is_mermaid[card_id] else 0.0


@njit(cache=True)
def encode_hand_fast(
    hand: np.ndarray,
    hand_size: int,
    card_numbers: np.ndarray,
    is_standard_suit: np.ndarray,
    is_roger: np.ndarray,
    is_pirate: np.ndarray,
    is_tigress: np.ndarray,
    is_king: np.ndarray,
    is_mermaid: np.ndarray,
    is_escape: np.ndarray,
    is_loot: np.ndarray,
    is_whale: np.ndarray,
    is_kraken: np.ndarray,
    is_beast: np.ndarray,
    output: np.ndarray,
) -> None:
    """Encode up to 10 cards in hand (90 dims = 10 cards x 9 features).

    JIT-compiled for maximum performance.
    """
    output[:] = 0.0
    card_encoding = np.zeros(9, dtype=np.float32)

    for i in range(10):
        if i < hand_size:
            card_id = hand[i]
            encode_card_fast(
                card_id,
                card_numbers,
                is_standard_suit,
                is_roger,
                is_pirate,
                is_tigress,
                is_king,
                is_mermaid,
                is_escape,
                is_loot,
                is_whale,
                is_kraken,
                is_beast,
                card_encoding,
            )
            output[i * 9 : (i + 1) * 9] = card_encoding


@njit(cache=True)
def encode_trick_fast(
    trick_cards: np.ndarray,
    num_cards: int,
    card_numbers: np.ndarray,
    is_standard_suit: np.ndarray,
    is_roger: np.ndarray,
    is_pirate: np.ndarray,
    is_tigress: np.ndarray,
    is_king: np.ndarray,
    is_mermaid: np.ndarray,
    is_escape: np.ndarray,
    is_loot: np.ndarray,
    is_whale: np.ndarray,
    is_kraken: np.ndarray,
    is_beast: np.ndarray,
    output: np.ndarray,
) -> None:
    """Encode up to 4 cards in trick (36 dims = 4 cards x 9 features).

    JIT-compiled for maximum performance.
    """
    output[:] = 0.0
    card_encoding = np.zeros(9, dtype=np.float32)

    for i in range(4):
        if i < num_cards:
            card_id = trick_cards[i]
            encode_card_fast(
                card_id,
                card_numbers,
                is_standard_suit,
                is_roger,
                is_pirate,
                is_tigress,
                is_king,
                is_mermaid,
                is_escape,
                is_loot,
                is_whale,
                is_kraken,
                is_beast,
                card_encoding,
            )
            output[i * 9 : (i + 1) * 9] = card_encoding


@njit(cache=True)
def encode_round_onehot_fast(round_number: int, output: np.ndarray) -> None:
    """Encode round number as one-hot (10 dims)."""
    output[:] = 0.0
    if 1 <= round_number <= 10:
        output[round_number - 1] = 1.0


@njit(cache=True)
def encode_game_phase_fast(phase_idx: int, output: np.ndarray) -> None:
    """Encode game phase as one-hot (4 dims)."""
    output[:] = 0.0
    if 0 <= phase_idx < 4:
        output[phase_idx] = 1.0


@njit(cache=True)
def count_card_type_fast(
    hand: np.ndarray,
    hand_size: int,
    type_mask: np.ndarray,
) -> int:
    """Count cards in hand matching a type mask."""
    count = 0
    for i in range(hand_size):
        if type_mask[hand[i]]:
            count += 1
    return count


class FastObservationEncoder:
    """Helper class providing fast observation encoding using pre-compiled functions."""

    def __init__(self) -> None:
        """Initialize encoder with pre-computed card arrays."""
        # Store references to module-level arrays for use in methods
        self.card_numbers = CARD_NUMBERS
        self.is_standard_suit = CARD_IS_STANDARD_SUIT
        self.is_roger = CARD_IS_ROGER
        self.is_pirate = CARD_IS_PIRATE
        self.is_tigress = CARD_IS_TIGRESS
        self.is_king = CARD_IS_KING
        self.is_mermaid = CARD_IS_MERMAID
        self.is_escape = CARD_IS_ESCAPE
        self.is_loot = CARD_IS_LOOT
        self.is_whale = CARD_IS_WHALE
        self.is_kraken = CARD_IS_KRAKEN
        self.is_beast = CARD_IS_BEAST

        # Pre-allocate output buffers
        self._hand_buffer = np.zeros(90, dtype=np.float32)
        self._trick_buffer = np.zeros(36, dtype=np.float32)
        self._card_buffer = np.zeros(9, dtype=np.float32)
        self._round_buffer = np.zeros(10, dtype=np.float32)
        self._phase_buffer = np.zeros(4, dtype=np.float32)

    def encode_hand(self, hand: list[int]) -> np.ndarray:
        """Encode hand using JIT-compiled function."""
        hand_arr = np.array(hand, dtype=np.int64)
        encode_hand_fast(
            hand_arr,
            len(hand),
            self.card_numbers,
            self.is_standard_suit,
            self.is_roger,
            self.is_pirate,
            self.is_tigress,
            self.is_king,
            self.is_mermaid,
            self.is_escape,
            self.is_loot,
            self.is_whale,
            self.is_kraken,
            self.is_beast,
            self._hand_buffer,
        )
        return self._hand_buffer.copy()

    def encode_trick(self, cards: list[int]) -> np.ndarray:
        """Encode trick cards using JIT-compiled function."""
        cards_arr = np.array(cards, dtype=np.int64)
        encode_trick_fast(
            cards_arr,
            len(cards),
            self.card_numbers,
            self.is_standard_suit,
            self.is_roger,
            self.is_pirate,
            self.is_tigress,
            self.is_king,
            self.is_mermaid,
            self.is_escape,
            self.is_loot,
            self.is_whale,
            self.is_kraken,
            self.is_beast,
            self._trick_buffer,
        )
        return self._trick_buffer.copy()

    def encode_card(self, card_id: int) -> np.ndarray:
        """Encode single card using JIT-compiled function."""
        encode_card_fast(
            card_id,
            self.card_numbers,
            self.is_standard_suit,
            self.is_roger,
            self.is_pirate,
            self.is_tigress,
            self.is_king,
            self.is_mermaid,
            self.is_escape,
            self.is_loot,
            self.is_whale,
            self.is_kraken,
            self.is_beast,
            self._card_buffer,
        )
        return self._card_buffer.copy()

    def encode_round_onehot(self, round_number: int) -> np.ndarray:
        """Encode round number as one-hot."""
        encode_round_onehot_fast(round_number, self._round_buffer)
        return self._round_buffer.copy()

    def encode_game_phase(self, phase_idx: int) -> np.ndarray:
        """Encode game phase as one-hot."""
        encode_game_phase_fast(phase_idx, self._phase_buffer)
        return self._phase_buffer.copy()

    def count_pirates(self, hand: list[int]) -> int:
        """Count pirate cards in hand."""
        hand_arr = np.array(hand, dtype=np.int64)
        return count_card_type_fast(hand_arr, len(hand), self.is_pirate)

    def count_kings(self, hand: list[int]) -> int:
        """Count king cards in hand."""
        hand_arr = np.array(hand, dtype=np.int64)
        return count_card_type_fast(hand_arr, len(hand), self.is_king)

    def count_mermaids(self, hand: list[int]) -> int:
        """Count mermaid cards in hand."""
        hand_arr = np.array(hand, dtype=np.int64)
        return count_card_type_fast(hand_arr, len(hand), self.is_mermaid)
