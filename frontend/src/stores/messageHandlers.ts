/**
 * WebSocket message handlers for game state updates.
 *
 * This module processes incoming WebSocket messages and updates
 * the game store state accordingly.
 */

import type { WebSocketMessage } from '../services/websocket';
import type { GamePhase, GameState, LootAlliance, Player, TrickCard } from './gameStore';
import { parseCard } from './gameStore';
import { logValidationErrors, validateGameState } from './stateValidator';

// Type for the set/get functions from Zustand
type SetState = (state: Partial<GameState>) => void;
type GetState = () => GameState;

/**
 * Handle JOINED message - a player joined the game.
 */
function handleJoined(content: Record<string, unknown>, set: SetState, get: GetState): void {
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
}

/**
 * Handle LEFT message - a player left the game.
 */
function handleLeft(content: Record<string, unknown>, set: SetState, get: GetState): void {
  const players = get().players.filter((p) => p.id !== content.player_id);
  set({ players });
  get().addLog(`${content.username} left the game`);
}

/**
 * Handle INIT message - initial game state on connect.
 */
function handleInit(content: Record<string, unknown>, set: SetState): void {
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
}

/**
 * Handle GAME_STATE message - full game state for reconnection.
 */
function handleGameState(content: Record<string, unknown>, set: SetState): void {
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
}

/**
 * Handle STARTED message - game has started.
 */
function handleStarted(set: SetState, get: GetState): void {
  set({ phase: 'BIDDING' as GamePhase });
  get().addLog(`Game started with ${get().players.length} players`);
}

/**
 * Handle DEAL message - cards dealt for new round.
 */
function handleDeal(content: Record<string, unknown>, set: SetState, get: GetState): void {
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
}

/**
 * Handle START_BIDDING message - bidding phase begins.
 */
function handleStartBidding(content: Record<string, unknown>, set: SetState, get: GetState): void {
  set({
    phase: 'BIDDING' as GamePhase,
    showBidding: !get().isSpectator,
    currentRound: content.round as number,
    currentTrick: 0,
    trickCards: [],
  });
}

/**
 * Handle BADE message - a player placed a bid.
 */
function handleBade(content: Record<string, unknown>, set: SetState, get: GetState): void {
  const playerId = content.player_id as string;
  const bidValue = content.bid as number;
  const players = get().players.map((p) => (p.id === playerId ? { ...p, bid: bidValue } : p));
  set({ players });

  const player = players.find((p) => p.id === playerId);
  get().addLog(`${player?.username || 'Player'} bid ${bidValue}`);
}

/**
 * Handle END_BIDDING message - all bids placed.
 */
function handleEndBidding(content: Record<string, unknown>, set: SetState, get: GetState): void {
  const bidsArray = content.bids as Array<{ player_id: string; bid: number }>;
  const currentPlayers = get().players;
  const players = currentPlayers.map((p) => {
    const bidInfo = bidsArray.find((b) => b.player_id === p.id);
    return bidInfo ? { ...p, bid: bidInfo.bid } : p;
  });
  set({ players, phase: 'PICKING' as GamePhase, showBidding: false });
  get().addLog('All bids placed - playing begins');
}

/**
 * Handle START_PICKING message - new trick starts.
 */
function handleStartPicking(content: Record<string, unknown>, set: SetState): void {
  set({
    phase: 'PICKING' as GamePhase,
    pickingPlayerId: content.picking_player_id as string,
    currentTrick: content.trick as number,
    trickCards: [],
    trickWinner: null,
  });
}

/**
 * Handle PICKED message - a player played a card.
 */
function handlePicked(content: Record<string, unknown>, set: SetState, get: GetState): void {
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
}

/**
 * Handle NEXT_TRICK message - turn advances to next player.
 */
function handleNextTrick(content: Record<string, unknown>, set: SetState): void {
  set({ pickingPlayerId: content.picking_player_id as string });
}

// Loot card IDs (cards 73 and 74)
const LOOT_CARD_IDS = ['73', '74'];

/**
 * Handle ANNOUNCE_TRICK_WINNER message - trick complete, winner announced.
 */
function handleAnnounceTrickWinner(
  content: Record<string, unknown>,
  set: SetState,
  get: GetState,
): void {
  const winnerId = content.winner_player_id as string | null;

  // Handle Kraken case - no winner
  if (!winnerId) {
    set({ trickWinner: null });
    get().addLog('Kraken destroyed all cards - no winner!');
    return;
  }

  const winner = get().players.find((p) => p.id === winnerId);
  const currentTrickCards = get().trickCards;
  const currentAlliances = get().lootAlliances;

  // Detect loot cards and form alliances
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
}

/**
 * Handle ANNOUNCE_SCORES message - round complete, scores updated.
 */
function handleAnnounceScores(
  content: Record<string, unknown>,
  set: SetState,
  get: GetState,
): void {
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
}

/**
 * Handle END_GAME message - game over.
 */
function handleEndGame(content: Record<string, unknown>, set: SetState, get: GetState): void {
  set({ phase: 'ENDED' as GamePhase, showResults: true });
  const leaderboard = content.leaderboard as Array<{ player_id: string; username: string }>;
  const winner = leaderboard?.[0];
  get().addLog(`${winner?.username || 'Someone'} won the game!`);
}

/**
 * Handle ABILITY_TRIGGERED message - pirate ability activated.
 */
function handleAbilityTriggered(
  content: Record<string, unknown>,
  set: SetState,
  get: GetState,
): void {
  set({
    showAbility: !get().isSpectator,
    abilityData: {
      type: content.ability_type as string,
      pirate: content.pirate as string,
      data: content as Record<string, unknown>,
    },
  });
}

/**
 * Handle ABILITY_RESOLVED message - pirate ability completed.
 */
function handleAbilityResolved(set: SetState): void {
  set({ showAbility: false, abilityData: null });
}

/**
 * Handle REPORT_ERROR message - error from backend.
 */
function handleReportError(content: Record<string, unknown>, get: GetState): void {
  console.error('[Game] Error:', content.error);
  get().addLog(`Error: ${content.error}`, 'error');
}

/**
 * Main message handler - routes messages to specific handlers.
 */
export function handleMessage(message: WebSocketMessage, set: SetState, get: GetState): void {
  const { type, content } = message;

  switch (type) {
    case 'JOINED':
      handleJoined(content, set, get);
      break;

    case 'LEFT':
      handleLeft(content, set, get);
      break;

    case 'INIT':
      handleInit(content, set);
      break;

    case 'GAME_STATE':
      handleGameState(content, set);
      break;

    case 'STARTED':
      handleStarted(set, get);
      break;

    case 'DEAL':
      handleDeal(content, set, get);
      break;

    case 'START_BIDDING':
      handleStartBidding(content, set, get);
      break;

    case 'BADE':
      handleBade(content, set, get);
      break;

    case 'END_BIDDING':
      handleEndBidding(content, set, get);
      break;

    case 'START_PICKING':
      handleStartPicking(content, set);
      break;

    case 'PICKED':
      handlePicked(content, set, get);
      break;

    case 'NEXT_TRICK':
      handleNextTrick(content, set);
      break;

    case 'ANNOUNCE_TRICK_WINNER':
      handleAnnounceTrickWinner(content, set, get);
      break;

    case 'ANNOUNCE_SCORES':
      handleAnnounceScores(content, set, get);
      break;

    case 'END_GAME':
      handleEndGame(content, set, get);
      break;

    case 'ABILITY_TRIGGERED':
      handleAbilityTriggered(content, set, get);
      break;

    case 'ABILITY_RESOLVED':
      handleAbilityResolved(set);
      break;

    case 'REPORT_ERROR':
      handleReportError(content, get);
      break;

    default:
      console.log('[Game] Unhandled message:', type, content);
  }

  // Validate state invariants after each message (dev mode only)
  const errors = validateGameState(get() as GameState, type);
  logValidationErrors(errors);
}
