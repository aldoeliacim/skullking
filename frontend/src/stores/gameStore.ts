import { create } from 'zustand';
import { type ConnectionState, type WebSocketMessage, websocket } from '../services/websocket';

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
  bids: Record<string, number>;
  leadSuit: string | null;

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
  bids: {},
  leadSuit: null,
  showBidding: false,
  showResults: false,
  showAbility: false,
  abilityData: null,
  trickWinner: null,
  logs: [],
};

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

    // Store unsubscribe functions for cleanup
    // @ts-expect-error - storing for cleanup
    get()._cleanup = () => {
      unsubscribeConnection();
      unsubscribeMessage();
    };
  },

  disconnect: () => {
    // @ts-expect-error - cleanup function
    const cleanup = get()._cleanup;
    if (cleanup) {
      cleanup();
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
      // Initial game state on connect
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
        }>;
      };
      if (game) {
        const players: Player[] = game.players.map((p) => ({
          id: p.id,
          username: p.username,
          is_bot: p.is_bot,
          score: p.score,
          bid: null,
          tricks_won: 0,
        }));
        set({ players });
      }
      break;
    }

    case 'GAME_STATE': {
      // Full game state update
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
        }>;
      };
      if (stateContent.players) {
        const players: Player[] = stateContent.players.map((p) => ({
          id: p.id,
          username: p.username,
          is_bot: p.is_bot,
          score: p.score,
          bid: null,
          tricks_won: 0,
        }));
        set({ players });
      }
      break;
    }

    case 'STARTED': {
      set({ phase: 'BIDDING' });
      get().addLog(`Game started with ${get().players.length} players`);
      break;
    }

    case 'DEAL': {
      const cards = (content.cards as string[]).map((id) => parseCard(id));
      set({
        hand: cards,
        currentRound: content.round as number,
        currentTrick: 0,
        trickCards: [],
        bids: {},
      });
      // Reset player tricks for new round
      const players = get().players.map((p) => ({ ...p, tricks_won: 0, bid: null }));
      set({ players });
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
      const bids = { ...get().bids, [content.player_id as string]: content.bid as number };
      const players = get().players.map((p) =>
        p.id === content.player_id ? { ...p, bid: content.bid as number } : p,
      );
      set({ bids, players });

      const player = get().players.find((p) => p.id === content.player_id);
      get().addLog(`${player?.username || 'Player'} bid ${content.bid}`);
      break;
    }

    case 'END_BIDDING': {
      const bidsArray = content.bids as Array<{ player_id: string; bid: number }>;
      const bids: Record<string, number> = {};
      bidsArray.forEach((b) => {
        bids[b.player_id] = b.bid;
      });
      set({ bids, phase: 'PICKING', showBidding: false });
      get().addLog('All bids placed - playing begins');
      break;
    }

    case 'START_PICKING': {
      set({
        phase: 'PICKING',
        pickingPlayerId: content.picking_player_id as string,
        currentTrick: content.trick as number,
        leadSuit: (content.lead_suit as string) || null,
      });

      // Clear trick cards if new trick
      const newTrick = content.trick as number;
      if (newTrick > get().currentTrick || newTrick === 1) {
        set({ trickCards: [], trickWinner: null });
      }
      break;
    }

    case 'PICKED': {
      const trickCard: TrickCard = {
        player_id: content.player_id as string,
        card_id: content.card_id as string,
        tigress_choice: content.tigress_choice as 'pirate' | 'escape' | undefined,
      };
      set({ trickCards: [...get().trickCards, trickCard] });

      // Update lead suit if first card
      if (get().trickCards.length === 1 && content.lead_suit) {
        set({ leadSuit: content.lead_suit as string });
      }
      break;
    }

    case 'NEXT_TRICK': {
      // Update picking player when turn advances
      set({ pickingPlayerId: content.picking_player_id as string });
      break;
    }

    case 'ANNOUNCE_TRICK_WINNER': {
      const winnerId = content.winner_player_id as string;
      const winner = get().players.find((p) => p.id === winnerId);

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
      });

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

    default:
      console.log('[Game] Unhandled message:', type, content);
  }
}

// Parse card ID to card object
function parseCard(cardId: string): Card {
  const card: Card = { id: cardId };

  // Parse card type from ID
  if (
    cardId.startsWith('blue_') ||
    cardId.startsWith('yellow_') ||
    cardId.startsWith('green_') ||
    cardId.startsWith('purple_')
  ) {
    const parts = cardId.split('_');
    card.suit = parts[0] ?? '';
    card.number = parseInt(parts[1] ?? '0', 10);
    card.type = 'standard';
  } else if (cardId.startsWith('black_')) {
    const parts = cardId.split('_');
    card.suit = 'black';
    card.number = parseInt(parts[1] ?? '0', 10);
    card.type = 'black';
  } else if (cardId.startsWith('escape')) {
    card.type = 'escape';
    card.name = 'Escape';
  } else if (cardId.startsWith('pirate')) {
    card.type = 'pirate';
    card.name = getPirateName(cardId);
  } else if (cardId === 'skull_king') {
    card.type = 'skull_king';
    card.name = 'Skull King';
  } else if (cardId.startsWith('mermaid')) {
    card.type = 'mermaid';
    card.name = 'Mermaid';
  } else if (cardId === 'tigress') {
    card.type = 'tigress';
    card.name = 'Scary Mary';
  } else if (cardId === 'kraken') {
    card.type = 'kraken';
    card.name = 'Kraken';
  } else if (cardId === 'white_whale') {
    card.type = 'white_whale';
    card.name = 'White Whale';
  } else if (cardId.startsWith('loot')) {
    card.type = 'loot';
    card.name = 'Loot';
  }

  return card;
}

function getPirateName(cardId: string): string {
  const names: Record<string, string> = {
    pirate_1: 'Harry the Giant',
    pirate_2: 'Tortuga Jack',
    pirate_3: 'Bendt the Bandit',
    pirate_4: 'Bahij the Bandit',
    pirate_5: "Rosie D'Laney",
    pirate_6: 'Juanita Jade',
    pirate_rascal_1: 'Rascal of Roatan',
  };
  return names[cardId] || 'Pirate';
}

export default useGameStore;
