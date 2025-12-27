import { create } from 'zustand';
import type {
  GamePhase,
  ConnectionState,
  Player,
  TrickCard,
  AbilityData,
  GameLogEntry,
  WebSocketMessage,
  BotType,
  BotDifficulty,
  TigressChoice,
} from '../types/game';
import { wsClient } from '../services/websocket';
import { parseCard } from '../utils/cardUtils';

interface GameStore {
  // Connection state
  connectionState: ConnectionState;
  gameId: string | null;
  playerId: string | null;
  playerName: string | null;
  isSpectator: boolean;
  isHost: boolean;

  // Game state
  phase: GamePhase;
  players: Player[];
  currentRound: number;
  currentTrick: number;
  pickingPlayerId: string | null;
  hand: number[];
  trickCards: TrickCard[];
  validCards: number[];
  lootAlliances: Record<string, string>;

  // UI state
  showBidding: boolean;
  showTigress: boolean;
  pendingTigressCardId: number | null;
  showAbility: boolean;
  abilityData: AbilityData | null;
  showResults: boolean;
  trickWinner: { playerId: string; playerName: string } | null;
  logs: GameLogEntry[];

  // Actions
  connect: (gameId: string, playerId: string, playerName: string, spectator?: boolean) => void;
  disconnect: () => void;
  handleMessage: (message: WebSocketMessage) => void;
  placeBid: (bid: number) => void;
  playCard: (cardId: number) => void;
  confirmTigress: (choice: TigressChoice) => void;
  cancelTigress: () => void;
  addBot: (type?: BotType, difficulty?: BotDifficulty) => void;
  removeBot: (botId: string) => void;
  clearBots: () => void;
  startGame: () => void;
  continueReady: () => void;
  resolveAbility: (data: Record<string, unknown>) => void;
  addLog: (message: string) => void;
  clearTrickWinner: () => void;
  reset: () => void;
}

const TIGRESS_CARD_ID = 72;

const initialState = {
  connectionState: 'disconnected' as ConnectionState,
  gameId: null,
  playerId: null,
  playerName: null,
  isSpectator: false,
  isHost: false,
  phase: 'PENDING' as GamePhase,
  players: [],
  currentRound: 1,
  currentTrick: 1,
  pickingPlayerId: null,
  hand: [],
  trickCards: [],
  validCards: [],
  lootAlliances: {},
  showBidding: false,
  showTigress: false,
  pendingTigressCardId: null,
  showAbility: false,
  abilityData: null,
  showResults: false,
  trickWinner: null,
  logs: [],
};

export const useGameStore = create<GameStore>((set, get) => ({
  ...initialState,

  connect: (gameId, playerId, playerName, spectator = false) => {
    set({
      gameId,
      playerId,
      playerName,
      isSpectator: spectator,
      connectionState: 'connecting',
    });

    wsClient.onConnectionChange((state) => {
      set({
        connectionState: state === 'connected' ? 'connected' : state === 'reconnecting' ? 'reconnecting' : 'disconnected',
      });
    });

    wsClient.onMessage((message) => {
      get().handleMessage(message);
    });

    wsClient.connect(gameId, playerId, playerName, spectator);
  },

  disconnect: () => {
    wsClient.disconnect();
    set(initialState);
  },

  handleMessage: (message) => {
    const { playerId } = get();

    switch (message.type) {
      case 'INIT':
      case 'GAME_STATE': {
        const players = (message.players as Player[]) || [];
        const hostPlayer = players[0];
        set({
          players,
          phase: (message.state as GamePhase) || 'PENDING',
          currentRound: (message.current_round as number) || 1,
          currentTrick: (message.current_trick as number) || 1,
          pickingPlayerId: message.picking_player_id as string | null,
          hand: (message.hand as number[]) || [],
          trickCards: (message.trick_cards as TrickCard[]) || [],
          validCards: (message.valid_cards as number[]) || [],
          isHost: hostPlayer?.id === playerId,
          lootAlliances: (message.loot_alliances as Record<string, string>) || {},
        });
        break;
      }

      case 'JOINED': {
        const newPlayer = message.player as Player;
        set((state) => ({
          players: [...state.players, newPlayer],
        }));
        get().addLog(`${newPlayer.username} joined`);
        break;
      }

      case 'LEFT': {
        const leftPlayerId = message.player_id as string;
        set((state) => ({
          players: state.players.filter((p) => p.id !== leftPlayerId),
        }));
        break;
      }

      case 'STARTED':
        set({ phase: 'BIDDING' });
        get().addLog('Game started!');
        break;

      case 'DEAL': {
        const hand = message.cards as number[];
        const round = message.round as number;
        set((state) => ({
          hand,
          currentRound: round,
          currentTrick: 1,
          trickCards: [],
          trickWinner: null,
          lootAlliances: {},
          players: state.players.map((p) => ({
            ...p,
            bid: null,
            tricks_won: 0,
          })),
        }));
        get().addLog(`Round ${round} - Cards dealt`);
        break;
      }

      case 'START_BIDDING':
        set({ phase: 'BIDDING', showBidding: true });
        break;

      case 'BADE': {
        const bidPlayerId = message.player_id as string;
        const bidValue = message.bid as number;
        set((state) => ({
          players: state.players.map((p) =>
            p.id === bidPlayerId ? { ...p, bid: bidValue } : p
          ),
        }));
        const player = get().players.find((p) => p.id === bidPlayerId);
        if (player) {
          get().addLog(`${player.username} bid ${bidValue}`);
        }
        break;
      }

      case 'END_BIDDING':
        set({ phase: 'PICKING', showBidding: false });
        break;

      case 'START_PICKING': {
        const pickingId = message.picking_player_id as string;
        set({
          pickingPlayerId: pickingId,
          trickCards: [],
        });
        break;
      }

      case 'PICKED': {
        const pickedPlayerId = message.player_id as string;
        const cardId = message.card_id as number;
        const tigressChoice = message.tigress_choice as TigressChoice | undefined;

        set((state) => {
          // Remove card from hand if it's our card
          const newHand = pickedPlayerId === playerId
            ? state.hand.filter((id) => id !== cardId)
            : state.hand;

          return {
            hand: newHand,
            trickCards: [
              ...state.trickCards,
              { player_id: pickedPlayerId, card_id: cardId, tigress_choice: tigressChoice },
            ],
          };
        });

        const player = get().players.find((p) => p.id === pickedPlayerId);
        const card = parseCard(cardId);
        if (player && card) {
          get().addLog(`${player.username} played ${card.name}`);
        }
        break;
      }

      case 'NEXT_TRICK': {
        const nextPlayerId = message.picking_player_id as string;
        set({ pickingPlayerId: nextPlayerId });
        break;
      }

      case 'VALID_CARDS': {
        const validCards = message.valid_cards as number[];
        set({ validCards });
        break;
      }

      case 'ANNOUNCE_TRICK_WINNER': {
        const winnerId = message.winner_player_id as string;
        const winnerPlayer = get().players.find((p) => p.id === winnerId);

        set((state) => ({
          players: state.players.map((p) =>
            p.id === winnerId ? { ...p, tricks_won: p.tricks_won + 1 } : p
          ),
          trickWinner: winnerPlayer
            ? { playerId: winnerId, playerName: winnerPlayer.username }
            : null,
        }));

        // Check for loot alliances
        const trickCards = get().trickCards;
        const lootCards = [73, 74];
        trickCards.forEach((tc) => {
          if (lootCards.includes(tc.card_id) && tc.player_id !== winnerId) {
            set((state) => ({
              lootAlliances: {
                ...state.lootAlliances,
                [tc.player_id]: winnerId,
              },
            }));
          }
        });

        if (winnerPlayer) {
          get().addLog(`${winnerPlayer.username} won the trick!`);
        }

        // Clear trick after delay
        setTimeout(() => {
          set({ trickCards: [], trickWinner: null });
        }, 2500);
        break;
      }

      case 'ANNOUNCE_SCORES': {
        const scores = message.scores as Array<{ player_id: string; score: number }>;
        set((state) => ({
          players: state.players.map((p) => {
            const scoreData = scores.find((s) => s.player_id === p.id);
            return scoreData ? { ...p, score: scoreData.score } : p;
          }),
        }));
        break;
      }

      case 'END_GAME':
        set({ phase: 'ENDED', showResults: true });
        get().addLog('Game over!');
        break;

      case 'ABILITY_TRIGGERED': {
        const abilityData = message as unknown as AbilityData;
        set({
          showAbility: true,
          abilityData,
        });
        break;
      }

      case 'ABILITY_RESOLVED':
        set({ showAbility: false, abilityData: null });
        break;

      case 'REPORT_ERROR': {
        const errorCode = message.error_code as string;
        get().addLog(`Error: ${errorCode}`);
        break;
      }
    }
  },

  placeBid: (bid) => {
    wsClient.placeBid(bid);
    set({ showBidding: false });
  },

  playCard: (cardId) => {
    // Check if it's the Tigress card
    if (cardId === TIGRESS_CARD_ID) {
      set({ showTigress: true, pendingTigressCardId: cardId });
      return;
    }
    wsClient.playCard(cardId);
  },

  confirmTigress: (choice) => {
    const { pendingTigressCardId } = get();
    if (pendingTigressCardId) {
      wsClient.playCard(pendingTigressCardId, choice);
      set({ showTigress: false, pendingTigressCardId: null });
    }
  },

  cancelTigress: () => {
    set({ showTigress: false, pendingTigressCardId: null });
  },

  addBot: (type = 'rl', difficulty = 'hard') => {
    wsClient.addBot(type, difficulty);
  },

  removeBot: (botId) => {
    wsClient.removeBot(botId);
  },

  clearBots: () => {
    wsClient.clearBots();
  },

  startGame: () => {
    wsClient.startGame();
  },

  continueReady: () => {
    wsClient.continueReady();
  },

  resolveAbility: (data) => {
    const { abilityData } = get();
    if (abilityData) {
      wsClient.resolveAbility(abilityData.type, data);
    }
  },

  addLog: (message) => {
    set((state) => ({
      logs: [
        ...state.logs.slice(-49),
        { id: crypto.randomUUID(), message, timestamp: Date.now() },
      ],
    }));
  },

  clearTrickWinner: () => {
    set({ trickWinner: null });
  },

  reset: () => {
    set(initialState);
  },
}));

export default useGameStore;
