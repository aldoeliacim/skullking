import { create } from 'zustand';
import { sessionStorage } from '../services/sessionStorage';
import { websocket } from '../services/websocket';
import { handleMessage } from './messageHandlers';

// Re-export types from shared types file for backward compatibility
export type {
  Player,
  Card,
  TrickCard,
  AbilityData,
  GamePhase,
  LootAlliance,
  ConnectionState,
  GameState,
} from '../types/game';
import type { ConnectionState, GamePhase, GameState, LootAlliance } from '../types/game';

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

    // Save session for reconnection after app background
    sessionStorage.saveSession({ gameId, playerId, playerName, isSpectator });

    // Setup connection state handler
    const unsubscribeConnection = websocket.addConnectionStateHandler((state) => {
      set({ connectionState: state });

      // Request full game state on reconnection
      if (state === 'connected' && currentState.phase !== 'PENDING') {
        websocket.requestGameState();
      }
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
    sessionStorage.clearSession();
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

// Re-export parseCard from cardUtils for backward compatibility
export { parseCard } from '../utils/cardUtils';

export default useGameStore;
