import type { WebSocketMessage, BotType, BotDifficulty, TigressChoice, AbilityType } from '../types/game';
import { WS_BASE_URL } from './api';

type MessageHandler = (message: WebSocketMessage) => void;
type ConnectionHandler = (state: 'connected' | 'disconnected' | 'reconnecting') => void;

class WebSocketClient {
  private ws: WebSocket | null = null;
  private messageHandlers = new Set<MessageHandler>();
  private connectionHandlers = new Set<ConnectionHandler>();
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 10;
  private reconnectTimeout: number | null = null;
  private heartbeatInterval: number | null = null;
  private gameId: string | null = null;
  private playerId: string | null = null;
  private username: string | null = null;
  private isSpectator = false;

  connect(gameId: string, playerId: string, username: string, spectator = false) {
    this.gameId = gameId;
    this.playerId = playerId;
    this.username = username;
    this.isSpectator = spectator;
    this.reconnectAttempts = 0;
    this.doConnect();
  }

  private doConnect() {
    if (this.ws?.readyState === WebSocket.OPEN) {
      return;
    }

    const endpoint = this.isSpectator ? 'spectate' : 'join';
    const params = new URLSearchParams({
      game_id: this.gameId!,
      player_id: this.playerId!,
      username: this.username!,
    });

    const url = `${WS_BASE_URL}/games/${endpoint}?${params}`;

    try {
      this.ws = new WebSocket(url);
      this.notifyConnectionState('reconnecting');

      this.ws.onopen = () => {
        this.reconnectAttempts = 0;
        this.notifyConnectionState('connected');
        this.startHeartbeat();
      };

      this.ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data) as WebSocketMessage;
          this.messageHandlers.forEach((handler) => {
            try {
              handler(message);
            } catch (err) {
              console.error('Message handler error:', err);
            }
          });
        } catch (err) {
          console.error('Failed to parse message:', err);
        }
      };

      this.ws.onclose = () => {
        this.stopHeartbeat();
        this.notifyConnectionState('disconnected');
        this.attemptReconnect();
      };

      this.ws.onerror = (error) => {
        console.error('WebSocket error:', error);
      };
    } catch (err) {
      console.error('Failed to connect:', err);
      this.attemptReconnect();
    }
  }

  private attemptReconnect() {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      return;
    }

    this.reconnectAttempts++;
    const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 30000);

    this.reconnectTimeout = window.setTimeout(() => {
      this.notifyConnectionState('reconnecting');
      this.doConnect();
    }, delay);
  }

  private startHeartbeat() {
    this.heartbeatInterval = window.setInterval(() => {
      this.send({ command: 'PING' });
    }, 30000);
  }

  private stopHeartbeat() {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }
  }

  private notifyConnectionState(state: 'connected' | 'disconnected' | 'reconnecting') {
    this.connectionHandlers.forEach((handler) => handler(state));
  }

  disconnect() {
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }
    this.stopHeartbeat();

    if (this.ws) {
      this.ws.onclose = null;
      this.ws.close();
      this.ws = null;
    }

    this.notifyConnectionState('disconnected');
  }

  send(data: Record<string, unknown>) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
    }
  }

  // Game commands
  placeBid(bid: number) {
    this.send({ command: 'BID', bid });
  }

  playCard(cardId: number, tigressChoice?: TigressChoice) {
    this.send({
      command: 'PICK',
      card_id: cardId,
      ...(tigressChoice && { tigress_choice: tigressChoice }),
    });
  }

  addBot(botType: BotType = 'rl', difficulty: BotDifficulty = 'hard') {
    this.send({ command: 'ADD_BOT', bot_type: botType, difficulty });
  }

  removeBot(botId: string) {
    this.send({ command: 'REMOVE_BOT', player_id: botId });
  }

  clearBots() {
    this.send({ command: 'CLEAR_BOTS' });
  }

  startGame() {
    this.send({ command: 'START_GAME' });
  }

  continueReady() {
    this.send({ command: 'CONTINUE_READY' });
  }

  resolveAbility(abilityType: AbilityType, data: Record<string, unknown>) {
    const commandMap: Record<AbilityType, string> = {
      choose_starter: 'RESOLVE_ROSIE',
      draw_and_discard: 'RESOLVE_BENDT',
      extra_bet: 'RESOLVE_ROATAN',
      view_deck: 'RESOLVE_JADE',
      modify_bid: 'RESOLVE_HARRY',
    };
    this.send({ command: commandMap[abilityType], ...data });
  }

  syncState() {
    this.send({ command: 'SYNC_STATE' });
  }

  // Handler management
  onMessage(handler: MessageHandler) {
    this.messageHandlers.add(handler);
    return () => this.messageHandlers.delete(handler);
  }

  onConnectionChange(handler: ConnectionHandler) {
    this.connectionHandlers.add(handler);
    return () => this.connectionHandlers.delete(handler);
  }

  get isConnected() {
    return this.ws?.readyState === WebSocket.OPEN;
  }
}

export const wsClient = new WebSocketClient();
export default wsClient;
