import type { Card, Suit, TrickCard } from '../types/game';

// Card ID ranges
const SKULL_KING = 1;
const WHITE_WHALE = 2;
const KRAKEN = 3;
const MERMAIDS = [4, 5];
const PIRATES = [6, 7, 8, 9, 10];
const ROGER = { start: 11, end: 24 }; // Trump suit
const PARROT = { start: 25, end: 38 };
const MAP = { start: 39, end: 52 };
const CHEST = { start: 53, end: 66 };
const ESCAPES = [67, 68, 69, 70, 71];
const TIGRESS = 72;
const LOOTS = [73, 74];

// Pirate names for images
const PIRATE_NAMES = ['bendt', 'harry', 'juanita', 'rascal', 'rosie'];

export function parseCard(cardId: number): Card | null {
  if (cardId < 1 || cardId > 74) return null;

  // Skull King
  if (cardId === SKULL_KING) {
    return { id: cardId, name: 'Skull King', type: 'skull_king', image: 'skullking.png' };
  }

  // White Whale
  if (cardId === WHITE_WHALE) {
    return { id: cardId, name: 'White Whale', type: 'white_whale', image: 'whale.png' };
  }

  // Kraken
  if (cardId === KRAKEN) {
    return { id: cardId, name: 'Kraken', type: 'kraken', image: 'kraken.png' };
  }

  // Mermaids
  if (MERMAIDS.includes(cardId)) {
    return { id: cardId, name: 'Mermaid', type: 'mermaid', image: 'siren.png' };
  }

  // Pirates
  if (cardId >= PIRATES[0] && cardId <= PIRATES[4]) {
    const pirateIndex = cardId - PIRATES[0];
    const pirateName = PIRATE_NAMES[pirateIndex];
    return {
      id: cardId,
      name: pirateName.charAt(0).toUpperCase() + pirateName.slice(1),
      type: 'pirate',
      image: `${pirateName}.png`,
    };
  }

  // Roger (Trump)
  if (cardId >= ROGER.start && cardId <= ROGER.end) {
    const number = cardId - ROGER.start + 1;
    return {
      id: cardId,
      name: `Roger ${number}`,
      type: 'suit',
      suit: 'roger',
      number,
      image: 'black.png',
    };
  }

  // Parrot
  if (cardId >= PARROT.start && cardId <= PARROT.end) {
    const number = cardId - PARROT.start + 1;
    return {
      id: cardId,
      name: `Parrot ${number}`,
      type: 'suit',
      suit: 'parrot',
      number,
      image: 'green.png',
    };
  }

  // Map
  if (cardId >= MAP.start && cardId <= MAP.end) {
    const number = cardId - MAP.start + 1;
    return {
      id: cardId,
      name: `Map ${number}`,
      type: 'suit',
      suit: 'map',
      number,
      image: 'purple.png',
    };
  }

  // Chest
  if (cardId >= CHEST.start && cardId <= CHEST.end) {
    const number = cardId - CHEST.start + 1;
    return {
      id: cardId,
      name: `Chest ${number}`,
      type: 'suit',
      suit: 'chest',
      number,
      image: 'yellow.png',
    };
  }

  // Escapes
  if (ESCAPES.includes(cardId)) {
    return { id: cardId, name: 'Escape', type: 'escape', image: 'flee.png' };
  }

  // Tigress
  if (cardId === TIGRESS) {
    return { id: cardId, name: 'Scary Mary', type: 'tigress', image: 'tigress.png' };
  }

  // Loot
  if (LOOTS.includes(cardId)) {
    return { id: cardId, name: 'Loot', type: 'loot', image: 'loot.png' };
  }

  return null;
}

export function isSpecialCard(cardId: number): boolean {
  return (
    cardId === SKULL_KING ||
    cardId === WHITE_WHALE ||
    cardId === KRAKEN ||
    MERMAIDS.includes(cardId) ||
    (cardId >= PIRATES[0] && cardId <= PIRATES[4]) ||
    ESCAPES.includes(cardId) ||
    cardId === TIGRESS ||
    LOOTS.includes(cardId)
  );
}

export function isTrumpCard(cardId: number): boolean {
  return cardId >= ROGER.start && cardId <= ROGER.end;
}

export function getCardSuit(cardId: number): Suit | null {
  if (cardId >= ROGER.start && cardId <= ROGER.end) return 'roger';
  if (cardId >= PARROT.start && cardId <= PARROT.end) return 'parrot';
  if (cardId >= MAP.start && cardId <= MAP.end) return 'map';
  if (cardId >= CHEST.start && cardId <= CHEST.end) return 'chest';
  return null;
}

export function getCardNumber(cardId: number): number | null {
  if (cardId >= ROGER.start && cardId <= ROGER.end) return cardId - ROGER.start + 1;
  if (cardId >= PARROT.start && cardId <= PARROT.end) return cardId - PARROT.start + 1;
  if (cardId >= MAP.start && cardId <= MAP.end) return cardId - MAP.start + 1;
  if (cardId >= CHEST.start && cardId <= CHEST.end) return cardId - CHEST.start + 1;
  return null;
}

export function getLedSuit(trickCards: TrickCard[]): Suit | null {
  for (const tc of trickCards) {
    if (!isSpecialCard(tc.card_id)) {
      return getCardSuit(tc.card_id);
    }
  }
  return null;
}

export function getValidCardIds(hand: number[], trickCards: TrickCard[]): number[] {
  // If no cards played yet, all cards are valid
  if (trickCards.length === 0) {
    return hand;
  }

  // Find the led suit
  const ledSuit = getLedSuit(trickCards);

  // If no regular suit was led, all cards are valid
  if (!ledSuit) {
    return hand;
  }

  // Check if player has cards of the led suit
  const suitCards = hand.filter((cardId) => getCardSuit(cardId) === ledSuit);

  // If player has cards of the led suit, they must play one of those
  // (but special cards are always allowed)
  if (suitCards.length > 0) {
    return hand.filter((cardId) => getCardSuit(cardId) === ledSuit || isSpecialCard(cardId));
  }

  // Player doesn't have the led suit, can play anything
  return hand;
}

export function getCardColor(card: Card): string {
  switch (card.type) {
    case 'skull_king':
      return 'var(--color-skull-king)';
    case 'pirate':
      return 'var(--color-pirate)';
    case 'mermaid':
      return 'var(--color-mermaid)';
    case 'kraken':
      return 'var(--color-kraken)';
    case 'white_whale':
      return 'var(--color-whale)';
    case 'escape':
    case 'loot':
      return 'var(--color-escape)';
    case 'tigress':
      return 'var(--color-pirate)';
    case 'suit':
      switch (card.suit) {
        case 'roger':
          return 'var(--color-roger)';
        case 'parrot':
          return 'var(--color-parrot)';
        case 'map':
          return 'var(--color-map)';
        case 'chest':
          return 'var(--color-chest)';
        default:
          return 'var(--color-bg-card)';
      }
    default:
      return 'var(--color-bg-card)';
  }
}
