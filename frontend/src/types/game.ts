// Shared game types to avoid circular dependencies

// Connection state (mirrored from websocket.ts to avoid import cycles)
export type ConnectionState = 'disconnected' | 'connecting' | 'connected' | 'reconnecting';

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

// Full game state interface (moved here to break circular dependency)
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
