import { create } from 'zustand';
import { type ConnectionState, websocket } from '../services/websocket';
import { getCardNumber, getCardSuit, getCardType } from '../utils/cardUtils';
import { handleMessage } from './messageHandlers';

// Types
export interface Player {
  id: string;
  username: string;
  is_bot: boolean;
  bot_type?: string;
  bot_difficulty?: string;
  score: number;
  bid: number | null;
  tricks_won: number;
  is_host?: boolean;
}

export interface Card {
  id: string;
  suit?: string;
  number?: number;
  type?: string;
  name?: string;
  image?: string;
}

export interface TrickCard {
  player_id: string;
  card_id: string;
  tigress_choice?: 'pirate' | 'escape' | undefined;
}

export interface AbilityData {
  type: string;
  pirate?: string;
  data?: Record<string, unknown>;
}

export type GamePhase = 'PENDING' | 'BIDDING' | 'PICKING' | 'ENDED';

// Alliance between loot player and trick winner
export interface LootAlliance {
  lootPlayerId: string;
  allyPlayerId: string;
}

export interface GameState {
  // Connection
  connectionState: ConnectionState;
  gameId: string | null;
  playerId: string | null;
  playerName: string | null;
  isSpectator: boolean;

  // Game state
  phase: GamePhase;
  players: Player[];
  currentRound: number;
  currentTrick: number;
  hand: Card[];
  trickCards: TrickCard[];
  pickingPlayerId: string | null;
  lootAlliances: LootAlliance[];

  // UI state
  showBidding: boolean;
  showResults: boolean;
  showAbility: boolean;
  abilityData: AbilityData | null;
  trickWinner: { playerId: string; playerName: string } | null;
  logs: Array<{ message: string; type: string; timestamp: number }>;

  // Actions
  connect: (gameId: string, playerId: string, playerName: string, isSpectator?: boolean) => void;
  disconnect: () => void;
  placeBid: (bid: number) => void;
  playCard: (cardId: string, tigressChoice?: 'pirate' | 'escape') => void;
  addBot: (botType: string, difficulty: string) => void;
  removeBot: (botId: string) => void;
  startGame: () => void;
  continueReady: () => void;
  resolveAbility: (data: Record<string, unknown>) => void;
  addLog: (message: string, type?: string) => void;
  clearTrickWinner: () => void;
  reset: () => void;
}

const initialState = {
  connectionState: 'disconnected' as ConnectionState,
  gameId: null,
  playerId: null,
  playerName: null,
  isSpectator: false,
  phase: 'PENDING' as GamePhase,
  players: [],
  currentRound: 0,
  currentTrick: 0,
  hand: [],
  trickCards: [],
  pickingPlayerId: null,
  lootAlliances: [] as LootAlliance[],
  showBidding: false,
  showResults: false,
  showAbility: false,
  abilityData: null,
  trickWinner: null,
  logs: [],
};

// Module-level storage for cleanup function (avoids hacky state mutation)
let cleanupFn: (() => void) | null = null;

export const useGameStore = create<GameState>((set, get) => ({
  ...initialState,

  connect: (gameId, playerId, playerName, isSpectator = false) => {
    const currentState = get();

    // If already connected to the same game, don't reconnect
    if (
      currentState.gameId === gameId &&
      currentState.playerId === playerId &&
      currentState.connectionState === 'connected'
    ) {
      return;
    }

    // If connected to a different game, disconnect first
    if (currentState.connectionState === 'connected' && currentState.gameId !== gameId) {
      websocket.disconnect();
    }

    set({ gameId, playerId, playerName, isSpectator });

    // Setup connection state handler
    const unsubscribeConnection = websocket.addConnectionStateHandler((state) => {
      set({ connectionState: state });
    });

    // Setup message handler
    const unsubscribeMessage = websocket.addMessageHandler((message) => {
      handleMessage(message, set, get);
    });

    // Connect
    websocket.connect(gameId, playerId, isSpectator, playerName);

    // Store cleanup function at module level
    cleanupFn = () => {
      unsubscribeConnection();
      unsubscribeMessage();
    };
  },

  disconnect: () => {
    if (cleanupFn) {
      cleanupFn();
      cleanupFn = null;
    }
    websocket.disconnect();
    set(initialState);
  },

  placeBid: (bid) => {
    websocket.placeBid(bid);
    set({ showBidding: false });
  },

  playCard: (cardId, tigressChoice) => {
    const { hand } = get();
    websocket.playCard(cardId, tigressChoice);
    // Optimistically remove card from hand
    set({ hand: hand.filter((c) => c.id !== cardId) });
  },

  addBot: (botType, difficulty) => {
    websocket.addBot(botType, difficulty);
  },

  removeBot: (botId) => {
    websocket.removeBot(botId);
  },

  startGame: () => {
    websocket.startGame();
  },

  continueReady: () => {
    websocket.continueReady();
  },

  resolveAbility: (data) => {
    websocket.resolveAbility(data);
    set({ showAbility: false, abilityData: null });
  },

  addLog: (message, type = 'info') => {
    const { logs } = get();
    const newLog = { message, type, timestamp: Date.now() };
    set({ logs: [...logs.slice(-49), newLog] }); // Keep last 50 logs
  },

  clearTrickWinner: () => {
    set({ trickWinner: null });
  },

  reset: () => {
    set(initialState);
  },
}));

// Pirate image names matching backend PIRATE_IDENTITY order
const PIRATE_IMAGES = ['rosie', 'bendt', 'rascal', 'juanita', 'harry'];
const PIRATE_NAMES: Record<number, string> = {
  6: 'Harry the Giant',
  7: 'Tortuga Jack',
  8: 'Bendt the Bandit',
  9: 'Bahij the Bandit',
  10: "Rosie D'Laney",
};

// Suit to image mapping (indexed by Suit type from cardUtils)
const SUIT_IMAGES = {
  roger: 'black.png',
  parrot: 'green.png',
  map: 'purple.png',
  chest: 'yellow.png',
} as const;

// Parse card ID to card object using centralized card utilities
export function parseCard(cardIdInput: string | number): Card {
  const numId = typeof cardIdInput === 'number' ? cardIdInput : parseInt(cardIdInput, 10);
  const card: Card = { id: String(numId) };

  const cardType = getCardType(numId);
  const suit = getCardSuit(numId);
  const number = getCardNumber(numId);

  if (suit && number) {
    // Suit card
    card.type = 'suit';
    card.suit = suit;
    card.number = number;
    card.image = SUIT_IMAGES[suit];
  } else if (cardType) {
    // Special card
    card.type = cardType;
    switch (cardType) {
      case 'skull_king':
        card.name = 'Skull King';
        card.image = 'skullking.png';
        break;
      case 'white_whale':
        card.name = 'White Whale';
        card.image = 'whale.png';
        break;
      case 'kraken':
        card.name = 'Kraken';
        card.image = 'kraken.png';
        break;
      case 'mermaid':
        card.name = `Mermaid ${numId - 3}`;
        card.image = 'siren.png';
        break;
      case 'pirate':
        card.name = PIRATE_NAMES[numId] || 'Pirate';
        card.image = `${PIRATE_IMAGES[numId - 6]}.png`;
        break;
      case 'escape':
        card.name = 'Escape';
        card.image = 'flee.png';
        break;
      case 'tigress':
        card.name = 'Scary Mary';
        card.image = 'tigress.png';
        break;
      case 'loot':
        card.name = 'Loot';
        card.image = 'loot.png';
        break;
    }
  } else {
    card.image = 'back.png';
  }

  return card;
}

export default useGameStore;
