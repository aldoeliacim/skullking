/**
 * Development-mode state validation for catching frontend/backend mismatches.
 *
 * These invariants should always hold if frontend correctly mirrors backend state.
 * Violations indicate bugs in message handling logic.
 */

import type { GameState } from './gameStore';

interface ValidationError {
  rule: string;
  message: string;
  context: Record<string, unknown>;
}

/**
 * Validate game state invariants after processing a message.
 * Only runs in development mode.
 */
export function validateGameState(state: GameState, messageType: string): ValidationError[] {
  if (process.env.NODE_ENV === 'production') {
    return [];
  }

  const errors: ValidationError[] = [];

  // Helper to add error
  const addError = (rule: string, message: string, context: Record<string, unknown> = {}) => {
    errors.push({ rule, message, context: { ...context, messageType } });
  };

  // === Trick Card Invariants ===

  // Trick cards should never exceed player count
  if (state.trickCards.length > state.players.length) {
    addError(
      'TRICK_CARDS_OVERFLOW',
      `Trick has ${state.trickCards.length} cards but only ${state.players.length} players`,
      { trickCards: state.trickCards.length, players: state.players.length },
    );
  }

  // All trick cards should have unique player IDs
  const trickPlayerIds = state.trickCards.map((tc) => tc.player_id);
  const uniqueTrickPlayerIds = new Set(trickPlayerIds);
  if (trickPlayerIds.length !== uniqueTrickPlayerIds.size) {
    addError('DUPLICATE_TRICK_PLAYER', 'Same player has multiple cards in trick', {
      playerIds: trickPlayerIds,
    });
  }

  // === Round/Trick Number Invariants ===

  // Current trick should not exceed current round
  if (state.currentTrick > state.currentRound) {
    addError('TRICK_EXCEEDS_ROUND', `Trick ${state.currentTrick} > Round ${state.currentRound}`, {
      currentTrick: state.currentTrick,
      currentRound: state.currentRound,
    });
  }

  // Round should be between 0 and 10
  if (state.currentRound < 0 || state.currentRound > 10) {
    addError('INVALID_ROUND', `Round ${state.currentRound} out of valid range 0-10`, {
      currentRound: state.currentRound,
    });
  }

  // === Phase-Specific Invariants ===

  if (state.phase === 'PICKING') {
    // Must have a picking player during PICKING phase
    if (!state.pickingPlayerId) {
      addError('MISSING_PICKING_PLAYER', 'Phase is PICKING but no pickingPlayerId set', {
        phase: state.phase,
      });
    }

    // Picking player must exist in players list
    if (state.pickingPlayerId && !state.players.find((p) => p.id === state.pickingPlayerId)) {
      addError('INVALID_PICKING_PLAYER', 'pickingPlayerId not found in players list', {
        pickingPlayerId: state.pickingPlayerId,
        playerIds: state.players.map((p) => p.id),
      });
    }
  }

  if (state.phase === 'BIDDING') {
    // Trick cards should be empty during bidding
    if (state.trickCards.length > 0) {
      addError('CARDS_DURING_BIDDING', 'Trick cards present during BIDDING phase', {
        trickCards: state.trickCards.length,
      });
    }
  }

  // === Player Invariants ===

  // All players should have valid state
  state.players.forEach((player) => {
    if (player.tricks_won < 0) {
      addError('NEGATIVE_TRICKS', `Player ${player.username} has negative tricks_won`, {
        playerId: player.id,
        tricksWon: player.tricks_won,
      });
    }

    // Tricks won should not exceed current trick number
    if (player.tricks_won > state.currentTrick) {
      addError(
        'TRICKS_EXCEED_CURRENT',
        `Player ${player.username} won ${player.tricks_won} tricks but only ${state.currentTrick} played`,
        { playerId: player.id, tricksWon: player.tricks_won, currentTrick: state.currentTrick },
      );
    }
  });

  // === Hand Invariants ===

  // Hand size should not exceed round number (cards dealt = round number)
  if (state.hand.length > state.currentRound && state.currentRound > 0) {
    addError(
      'HAND_SIZE_OVERFLOW',
      `Hand has ${state.hand.length} cards but round is ${state.currentRound}`,
      {
        handSize: state.hand.length,
        currentRound: state.currentRound,
      },
    );
  }

  // All cards in hand should have valid IDs
  state.hand.forEach((card) => {
    if (!card.id) {
      addError('INVALID_CARD_ID', 'Card in hand has no ID', { card });
    }
  });

  return errors;
}

/**
 * Log validation errors to console in development mode.
 */
export function logValidationErrors(errors: ValidationError[]): void {
  if (errors.length === 0) {
    return;
  }

  console.group(`[StateValidator] ${errors.length} invariant violation(s) detected`);
  errors.forEach((error) => {
    console.warn(`[${error.rule}] ${error.message}`, error.context);
  });
  console.groupEnd();
}

/**
 * Message semantic documentation.
 * Maps each message type to what frontend state changes it implies.
 */
export const MESSAGE_SEMANTICS: Record<string, string> = {
  DEAL: 'New round starting. Hand updated, tricks/bids reset.',
  START_BIDDING: 'Bidding phase begins. Show bid UI.',
  BADE: 'A player placed a bid. Update their bid value.',
  END_BIDDING: 'All bids placed. Transition to PICKING phase.',
  START_PICKING: 'NEW TRICK starting. Clear trick cards. Set current picker.',
  PICKED: 'Card played. Add to trick area. Remove from hand if ours.',
  NEXT_TRICK: 'SAME TRICK, next player. Just update picker ID.',
  ANNOUNCE_TRICK_WINNER: 'Trick complete. Show winner. Update tricks_won.',
  ANNOUNCE_SCORES: 'Round complete. Update all scores.',
  END_GAME: 'Game over. Show final results.',
};
