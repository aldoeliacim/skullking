import { create } from 'zustand';
import { type ConnectionState, type WebSocketMessage, websocket } from '../services/websocket';
import { getCardNumber, getCardSuit, getCardType } from '../utils/cardUtils';
import { logValidationErrors, validateGameState } from './stateValidator';

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

// Message handler
function handleMessage(
  message: WebSocketMessage,
  set: (state: Partial<GameState>) => void,
  get: () => GameState,
): void {
  const { type, content } = message;

  switch (type) {
    case 'JOINED': {
      const isBot = (content.is_bot as boolean) || false;
      const newPlayer: Player = {
        id: content.player_id as string,
        username: content.username as string,
        is_bot: isBot,
        score: 0,
        bid: null,
        tricks_won: 0,
        is_host: (content.is_host as boolean) || false,
        ...(isBot && content.bot_type ? { bot_type: content.bot_type as string } : {}),
      };
      const players = [...get().players.filter((p) => p.id !== newPlayer.id), newPlayer];
      set({ players });
      get().addLog(`${content.username} joined the game`);
      break;
    }

    case 'LEFT': {
      const players = get().players.filter((p) => p.id !== content.player_id);
      set({ players });
      get().addLog(`${content.username} left the game`);
      break;
    }

    case 'INIT': {
      // Initial game state on connect - use all data from backend
      const game = content.game as {
        id: string;
        slug: string;
        state: string;
        players: Array<{
          id: string;
          username: string;
          score: number;
          index: number;
          is_bot: boolean;
          is_connected: boolean;
          bid?: number | null;
          tricks_won?: number;
        }>;
      };
      if (game) {
        const players: Player[] = game.players.map((p) => ({
          id: p.id,
          username: p.username,
          is_bot: p.is_bot,
          score: p.score,
          bid: p.bid ?? null,
          tricks_won: p.tricks_won ?? 0,
        }));
        set({ players });
      }
      break;
    }

    case 'GAME_STATE': {
      // Full game state update - use all data from backend for reconnection support
      const stateContent = content as {
        id?: string;
        slug?: string;
        state?: string;
        players?: Array<{
          id: string;
          username: string;
          score: number;
          index: number;
          is_bot: boolean;
          is_connected: boolean;
          bid?: number | null;
          tricks_won?: number;
        }>;
        current_round?: {
          loot_alliances?: Record<string, string>;
        };
      };
      if (stateContent.players) {
        const players: Player[] = stateContent.players.map((p) => ({
          id: p.id,
          username: p.username,
          is_bot: p.is_bot,
          score: p.score,
          bid: p.bid ?? null,
          tricks_won: p.tricks_won ?? 0,
        }));
        set({ players });
      }
      // Parse loot alliances from current round
      if (stateContent.current_round?.loot_alliances) {
        const alliances = Object.entries(stateContent.current_round.loot_alliances).map(
          ([lootPlayerId, allyPlayerId]) => ({ lootPlayerId, allyPlayerId }),
        );
        set({ lootAlliances: alliances });
      }
      break;
    }

    case 'STARTED': {
      set({ phase: 'BIDDING' });
      get().addLog(`Game started with ${get().players.length} players`);
      break;
    }

    case 'DEAL': {
      const cardIds = content.cards as (string | number)[];
      const cards = cardIds.map((id) => parseCard(id));
      // Reset player tricks for new round - get players BEFORE setting state
      const resetPlayers = get().players.map((p) => ({ ...p, tricks_won: 0, bid: null }));
      // Single atomic update for all state changes
      set({
        hand: cards,
        currentRound: content.round as number,
        currentTrick: 0,
        trickCards: [],
        players: resetPlayers,
        lootAlliances: [], // Reset alliances for new round
      });
      get().addLog(`Round ${content.round} - Cards dealt`);
      break;
    }

    case 'START_BIDDING': {
      set({
        phase: 'BIDDING',
        showBidding: !get().isSpectator,
        currentRound: content.round as number,
        currentTrick: 0,
        trickCards: [],
      });
      break;
    }

    case 'BADE': {
      // Update player's bid (no separate bids object needed - player.bid is the source of truth)
      const playerId = content.player_id as string;
      const bidValue = content.bid as number;
      const players = get().players.map((p) => (p.id === playerId ? { ...p, bid: bidValue } : p));
      set({ players });

      const player = players.find((p) => p.id === playerId);
      get().addLog(`${player?.username || 'Player'} bid ${bidValue}`);
      break;
    }

    case 'END_BIDDING': {
      // Ensure all player bids are synced from backend (in case any were missed)
      const bidsArray = content.bids as Array<{ player_id: string; bid: number }>;
      const currentPlayers = get().players;
      const players = currentPlayers.map((p) => {
        const bidInfo = bidsArray.find((b) => b.player_id === p.id);
        return bidInfo ? { ...p, bid: bidInfo.bid } : p;
      });
      set({ players, phase: 'PICKING', showBidding: false });
      get().addLog('All bids placed - playing begins');
      break;
    }

    case 'START_PICKING': {
      // Backend sends START_PICKING only at the start of a new trick
      // (NEXT_TRICK is sent for subsequent players in the same trick)
      // So we ALWAYS clear trick cards here - no conditional logic needed
      set({
        phase: 'PICKING',
        pickingPlayerId: content.picking_player_id as string,
        currentTrick: content.trick as number,
        trickCards: [],
        trickWinner: null,
      });
      break;
    }

    case 'PICKED': {
      // Convert card_id to string (backend sends integer)
      const cardIdStr = String(content.card_id);
      const trickCard: TrickCard = {
        player_id: content.player_id as string,
        card_id: cardIdStr,
        tigress_choice: content.tigress_choice as 'pirate' | 'escape' | undefined,
      };

      // Get current state BEFORE updating
      const currentTrickCards = get().trickCards;
      const currentHand = get().hand;

      // Single atomic update for all state changes
      set({
        trickCards: [...currentTrickCards, trickCard],
        hand: currentHand.filter((c) => c.id !== cardIdStr),
      });
      break;
    }

    case 'NEXT_TRICK': {
      // Update picking player when turn advances
      set({ pickingPlayerId: content.picking_player_id as string });
      break;
    }

    case 'ANNOUNCE_TRICK_WINNER': {
      const winnerId = content.winner_player_id as string | null;

      // Handle Kraken case - no winner
      if (!winnerId) {
        set({ trickWinner: null });
        get().addLog('Kraken destroyed all cards - no winner!');
        break;
      }

      const winner = get().players.find((p) => p.id === winnerId);
      const currentTrickCards = get().trickCards;
      const currentAlliances = get().lootAlliances;

      // Detect loot cards and form alliances (card IDs 73, 74 are loot)
      const LOOT_CARD_IDS = ['73', '74'];
      const newAlliances: LootAlliance[] = [];
      for (const tc of currentTrickCards) {
        if (LOOT_CARD_IDS.includes(tc.card_id) && tc.player_id !== winnerId) {
          // Loot player forms alliance with winner
          newAlliances.push({ lootPlayerId: tc.player_id, allyPlayerId: winnerId });
        }
      }

      // Update tricks won
      const players = get().players.map((p) =>
        p.id === winnerId ? { ...p, tricks_won: (p.tricks_won || 0) + 1 } : p,
      );
      set({
        players,
        trickWinner: {
          playerId: winnerId,
          playerName: winner?.username || (content.winner_name as string) || 'Unknown',
        },
        lootAlliances: [...currentAlliances, ...newAlliances],
      });

      // Log alliance formation
      for (const alliance of newAlliances) {
        const lootPlayer = get().players.find((p) => p.id === alliance.lootPlayerId);
        get().addLog(
          `${lootPlayer?.username || 'Player'} formed alliance with ${winner?.username || 'winner'}`,
        );
      }

      get().addLog(`${winner?.username || content.winner_name} won the trick`);
      break;
    }

    case 'ANNOUNCE_SCORES': {
      const scores = content.scores as Array<{
        player_id: string;
        total_score: number;
        score_delta: number;
      }>;
      const players = get().players.map((p) => {
        const scoreInfo = scores.find((s) => s.player_id === p.id);
        return scoreInfo ? { ...p, score: scoreInfo.total_score } : p;
      });
      set({ players });
      get().addLog(`Round ${content.round} complete`);
      break;
    }

    case 'END_GAME': {
      set({ phase: 'ENDED', showResults: true });
      const leaderboard = content.leaderboard as Array<{ player_id: string; username: string }>;
      const winner = leaderboard?.[0];
      get().addLog(`${winner?.username || 'Someone'} won the game!`);
      break;
    }

    case 'ABILITY_TRIGGERED': {
      set({
        showAbility: !get().isSpectator,
        abilityData: {
          type: content.ability_type as string,
          pirate: content.pirate as string,
          data: content as Record<string, unknown>,
        },
      });
      break;
    }

    case 'ABILITY_RESOLVED': {
      set({ showAbility: false, abilityData: null });
      break;
    }

    case 'REPORT_ERROR': {
      console.error('[Game] Error:', content.error);
      get().addLog(`Error: ${content.error}`, 'error');
      break;
    }

    // VALID_CARDS no longer needed - frontend calculates valid cards dynamically

    default:
      console.log('[Game] Unhandled message:', type, content);
  }

  // Validate state invariants after each message (dev mode only)
  const errors = validateGameState(get() as GameState, type);
  logValidationErrors(errors);
}

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
