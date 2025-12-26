/**
 * Card utility functions for Skull King.
 *
 * Card ID ranges:
 * - 1: Skull King
 * - 2: Whale
 * - 3: Kraken
 * - 4-5: Mermaids
 * - 6-10: Pirates
 * - 11-24: Roger (trump suit)
 * - 25-38: Parrot
 * - 39-52: Map
 * - 53-66: Chest
 * - 67-71: Escapes
 * - 72: Tigress
 * - 73-74: Loot
 */

import type { Card, TrickCard } from '../stores/gameStore';

export type Suit = 'roger' | 'parrot' | 'map' | 'chest';
export type CardType =
  | 'skull_king'
  | 'white_whale'
  | 'kraken'
  | 'mermaid'
  | 'pirate'
  | 'suit'
  | 'escape'
  | 'tigress'
  | 'loot';

// Suit ranges for efficient lookup
const SUIT_RANGES: Array<{ min: number; max: number; suit: Suit; base: number }> = [
  { min: 11, max: 24, suit: 'roger', base: 10 },
  { min: 25, max: 38, suit: 'parrot', base: 24 },
  { min: 39, max: 52, suit: 'map', base: 38 },
  { min: 53, max: 66, suit: 'chest', base: 52 },
];

/**
 * Check if a card is a special card (can always be played regardless of suit).
 * Special cards: Skull King, Whale, Kraken, Mermaids, Pirates, Escapes, Tigress, Loot
 */
export function isSpecialCard(cardId: number): boolean {
  return cardId <= 10 || cardId >= 67;
}

/**
 * Check if a card is a trump card (Jolly Roger / Black suit).
 * Trump cards can always be played regardless of the led suit.
 */
export function isTrumpCard(cardId: number): boolean {
  return cardId >= 11 && cardId <= 24;
}

/**
 * Check if a card can always be played regardless of suit (special or trump).
 */
export function isAlwaysPlayable(cardId: number): boolean {
  return isSpecialCard(cardId) || isTrumpCard(cardId);
}

/**
 * Get the suit of a card. Returns null for special cards.
 */
export function getCardSuit(cardId: number): Suit | null {
  for (const range of SUIT_RANGES) {
    if (cardId >= range.min && cardId <= range.max) {
      return range.suit;
    }
  }
  return null;
}

/**
 * Get the number (1-14) for a suit card. Returns null for special cards.
 */
export function getCardNumber(cardId: number): number | null {
  for (const range of SUIT_RANGES) {
    if (cardId >= range.min && cardId <= range.max) {
      return cardId - range.base;
    }
  }
  return null;
}

/**
 * Get the card type from ID.
 */
export function getCardType(cardId: number): CardType | null {
  if (cardId === 1) return 'skull_king';
  if (cardId === 2) return 'white_whale';
  if (cardId === 3) return 'kraken';
  if (cardId >= 4 && cardId <= 5) return 'mermaid';
  if (cardId >= 6 && cardId <= 10) return 'pirate';
  if (cardId >= 11 && cardId <= 66) return 'suit';
  if (cardId >= 67 && cardId <= 71) return 'escape';
  if (cardId === 72) return 'tigress';
  if (cardId >= 73 && cardId <= 74) return 'loot';
  return null;
}

/**
 * Find the led suit from trick cards.
 * The led suit is determined by the first non-special card played.
 * Returns null if no suit has been led yet (all cards so far are special).
 */
export function getLedSuit(trickCards: TrickCard[]): Suit | null {
  for (const tc of trickCards) {
    const cardId = parseInt(tc.card_id, 10);
    const suit = getCardSuit(cardId);
    if (suit !== null) {
      return suit;
    }
  }
  return null;
}

/**
 * Calculate which cards in hand are valid to play.
 * Rules:
 * - Special cards can ALWAYS be played
 * - If no suit has been led, any card can be played
 * - If a suit has been led and player has cards of that suit, they must play that suit
 * - If player doesn't have the led suit, they can play any card
 */
export function getValidCardIds(hand: Card[], trickCards: TrickCard[]): string[] {
  // If no cards in trick yet, all cards are valid
  if (trickCards.length === 0) {
    return hand.map((c) => c.id);
  }

  // Find the led suit
  const ledSuit = getLedSuit(trickCards);

  // If no suit led yet (all special cards), all cards are valid
  if (ledSuit === null) {
    return hand.map((c) => c.id);
  }

  // Separate always-playable cards (special + trump) and cards of the led suit
  const alwaysPlayableCards: string[] = [];
  const suitCards: string[] = [];

  for (const card of hand) {
    const cardId = parseInt(card.id, 10);
    if (isAlwaysPlayable(cardId)) {
      // Special cards and trump cards can always be played
      alwaysPlayableCards.push(card.id);
    } else if (getCardSuit(cardId) === ledSuit) {
      suitCards.push(card.id);
    }
  }

  // If player has cards of led suit, must play those (or always-playable cards)
  if (suitCards.length > 0) {
    return [...suitCards, ...alwaysPlayableCards];
  }

  // Player doesn't have led suit, can play anything
  return hand.map((c) => c.id);
}
