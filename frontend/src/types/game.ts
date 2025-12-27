// Game types for Skull King

export type GamePhase = 'PENDING' | 'BIDDING' | 'PICKING' | 'ENDED';
export type ConnectionState = 'disconnected' | 'connecting' | 'connected' | 'reconnecting';
export type BotDifficulty = 'easy' | 'medium' | 'hard';
export type BotType = 'rl' | 'rule_based' | 'random';
export type TigressChoice = 'pirate' | 'escape';

export interface Player {
  id: string;
  username: string;
  is_bot: boolean;
  score: number;
  bid: number | null;
  tricks_won: number;
  is_ready?: boolean;
}

export interface Card {
  id: number;
  name: string;
  type: CardType;
  suit?: Suit;
  number?: number;
  image?: string;
}

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

export type Suit = 'roger' | 'parrot' | 'map' | 'chest';

export interface TrickCard {
  player_id: string;
  card_id: number;
  tigress_choice?: TigressChoice;
}

export interface GameState {
  id: string;
  slug: string;
  phase: GamePhase;
  players: Player[];
  current_round: number;
  current_trick: number;
  picking_player_id: string | null;
  hand: number[];
  trick_cards: TrickCard[];
  valid_cards: number[];
  loot_alliances: Record<string, string>;
}

export interface AbilityData {
  type: AbilityType;
  player_id: string;
  options?: string[];
  cards?: number[];
  drawn_cards?: number[];
  current_bid?: number;
  deck_cards?: number[];
}

export type AbilityType =
  | 'choose_starter'   // Rosie
  | 'draw_and_discard' // Bendt
  | 'extra_bet'        // Roatán
  | 'view_deck'        // Jade
  | 'modify_bid';      // Harry

export interface GameLogEntry {
  id: string;
  message: string;
  timestamp: number;
}

// WebSocket message types
export type MessageType =
  | 'INIT'
  | 'GAME_STATE'
  | 'JOINED'
  | 'LEFT'
  | 'SPECTATOR_JOINED'
  | 'SPECTATOR_LEFT'
  | 'STARTED'
  | 'DEAL'
  | 'START_BIDDING'
  | 'BADE'
  | 'END_BIDDING'
  | 'START_PICKING'
  | 'PICKED'
  | 'NEXT_TRICK'
  | 'VALID_CARDS'
  | 'ANNOUNCE_TRICK_WINNER'
  | 'ANNOUNCE_SCORES'
  | 'END_GAME'
  | 'ABILITY_TRIGGERED'
  | 'ABILITY_RESOLVED'
  | 'SHOW_DECK'
  | 'CONTINUE_PROMPT'
  | 'ALL_READY'
  | 'REPORT_ERROR';

export interface WebSocketMessage {
  type: MessageType;
  [key: string]: unknown;
}

// API types
export interface CreateGameResponse {
  game_id: string;
  slug: string;
}

export interface GameInfo {
  id: string;
  slug: string;
  state: GamePhase;
  player_count: number;
  spectator_count: number;
  created_at: string;
  players: Array<{
    id: string;
    username: string;
    is_bot: boolean;
  }>;
}
