/**
 * API types for backend communication.
 *
 * These types represent the exact format expected by the backend.
 * Use these when sending data to ensure type safety at API boundaries.
 */

// Backend expects card IDs as integers
export type CardIdNumeric = number;

// Backend command payloads
// Index signature allows compatibility with Record<string, unknown>
export interface PickPayload {
  card_id: CardIdNumeric;
  tigress_choice?: 'pirate' | 'escape';
  [key: string]: unknown;
}

export interface BidPayload {
  bid: number;
}

export interface AddBotPayload {
  bot_type: string;
  difficulty: string;
}

export interface RemoveBotPayload {
  bot_id: string;
}

// Helper to convert frontend card ID (string) to backend format (number)
export function toCardIdNumeric(cardId: string): CardIdNumeric {
  const num = parseInt(cardId, 10);
  if (Number.isNaN(num)) {
    throw new Error(`Invalid card ID: ${cardId}`);
  }
  return num;
}
