import { WS_BASE_URL } from './api';

// WebSocket message types (matching backend Command enum)
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
  | 'ANNOUNCE_TRICK_WINNER'
  | 'ANNOUNCE_SCORES'
  | 'END_GAME'
  | 'ABILITY_TRIGGERED'
  | 'ABILITY_RESOLVED'
  | 'SHOW_DECK'
  | 'CONTINUE_PROMPT'
  | 'ALL_READY'
  | 'REPORT_ERROR';

// Backend sends "command", we map it to "type" for consistency
interface RawWebSocketMessage {
  command: string;
  content: Record<string, unknown>;
}

export interface WebSocketMessage {
  type: MessageType;
  content: Record<string, unknown>;
}

export type MessageHandler = (message: WebSocketMessage) => void;

// WebSocket connection state
export type ConnectionState = 'disconnected' | 'connecting' | 'connected' | 'reconnecting';

// WebSocket client class
class WebSocketClient {
  private ws: WebSocket | null = null;
  private messageHandlers: Set<MessageHandler> = new Set();
  private connectionStateHandlers: Set<(state: ConnectionState) => void> = new Set();
  private connectionState: ConnectionState = 'disconnected';
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 10;
  private reconnectDelay = 1000;
  private reconnectTimer: NodeJS.Timeout | null = null;
  private currentUrl: string | null = null;
  private heartbeatTimer: NodeJS.Timeout | null = null;
  private heartbeatInterval = 30000;

  // Connect to WebSocket
  connect(gameId: string, playerId: string, isSpectator = false, username = 'Player'): void {
    const endpoint = isSpectator ? 'spectate' : 'join';
    const params = new URLSearchParams({
      game_id: gameId,
      player_id: playerId,
      username: username,
    });
    this.currentUrl = `${WS_BASE_URL}/games/${endpoint}?${params.toString()}`;
    this.doConnect();
  }

  private doConnect(): void {
    if (!this.currentUrl) {
      return;
    }

    this.setConnectionState('connecting');

    try {
      this.ws = new WebSocket(this.currentUrl);

      this.ws.onopen = () => {
        this.reconnectAttempts = 0;
        this.setConnectionState('connected');
        this.startHeartbeat();
      };

      this.ws.onmessage = (event) => {
        try {
          const raw: RawWebSocketMessage = JSON.parse(event.data);
          // Convert backend "command" to frontend "type"
          const message: WebSocketMessage = {
            type: raw.command as MessageType,
            content: raw.content,
          };
          this.notifyHandlers(message);
        } catch (error) {
          console.error('[WebSocket] Failed to parse message:', error);
        }
      };

      this.ws.onerror = (error) => {
        console.error('[WebSocket] Error:', error);
      };

      this.ws.onclose = (event) => {
        this.stopHeartbeat();

        if (event.code !== 1000 && event.code !== 1001) {
          // Abnormal closure, attempt reconnect
          this.attemptReconnect();
        } else {
          this.setConnectionState('disconnected');
        }
      };
    } catch (error) {
      console.error('[WebSocket] Connection error:', error);
      this.attemptReconnect();
    }
  }

  private attemptReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      this.setConnectionState('disconnected');
      return;
    }

    this.setConnectionState('reconnecting');
    this.reconnectAttempts++;

    // Exponential backoff with jitter
    const delay = Math.min(
      this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1) + Math.random() * 1000,
      30000,
    );

    this.reconnectTimer = setTimeout(() => {
      this.doConnect();
    }, delay);
  }

  private startHeartbeat(): void {
    this.heartbeatTimer = setInterval(() => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify({ command: 'PING' }));
      }
    }, this.heartbeatInterval);
  }

  private stopHeartbeat(): void {
    if (this.heartbeatTimer) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  // Disconnect
  disconnect(): void {
    if (this.reconnectTimer) {
      clearTimeout(this.reconnectTimer);
      this.reconnectTimer = null;
    }

    this.stopHeartbeat();

    if (this.ws) {
      this.ws.close(1000, 'Client disconnect');
      this.ws = null;
    }

    this.currentUrl = null;
    this.reconnectAttempts = 0;
    this.setConnectionState('disconnected');
  }

  // Send message
  send(command: string, content: Record<string, unknown> = {}): boolean {
    if (this.ws?.readyState !== WebSocket.OPEN) {
      console.warn('[WebSocket] Cannot send, not connected');
      return false;
    }

    try {
      this.ws.send(JSON.stringify({ command, content }));
      return true;
    } catch (error) {
      console.error('[WebSocket] Send error:', error);
      return false;
    }
  }

  // Game actions
  placeBid(bid: number): boolean {
    return this.send('BID', { bid });
  }

  playCard(cardId: string, tigressChoice?: 'pirate' | 'escape'): boolean {
    // Backend expects card_id as integer
    const content: Record<string, unknown> = { card_id: parseInt(cardId, 10) };
    if (tigressChoice) {
      content.tigress_choice = tigressChoice;
    }
    return this.send('PICK', content);
  }

  addBot(botType: string, difficulty: string): boolean {
    return this.send('ADD_BOT', { bot_type: botType, difficulty });
  }

  removeBot(botId: string): boolean {
    return this.send('REMOVE_BOT', { bot_id: botId });
  }

  startGame(): boolean {
    return this.send('START_GAME');
  }

  continueReady(): boolean {
    return this.send('CONTINUE_READY');
  }

  resolveAbility(data: Record<string, unknown>): boolean {
    return this.send('RESOLVE_ABILITY', data);
  }

  // Message handlers
  addMessageHandler(handler: MessageHandler): () => void {
    this.messageHandlers.add(handler);
    return () => this.messageHandlers.delete(handler);
  }

  private notifyHandlers(message: WebSocketMessage): void {
    this.messageHandlers.forEach((handler) => {
      try {
        handler(message);
      } catch (error) {
        console.error('[WebSocket] Handler error:', error);
      }
    });
  }

  // Connection state handlers
  addConnectionStateHandler(handler: (state: ConnectionState) => void): () => void {
    this.connectionStateHandlers.add(handler);
    handler(this.connectionState);
    return () => this.connectionStateHandlers.delete(handler);
  }

  private setConnectionState(state: ConnectionState): void {
    this.connectionState = state;
    this.connectionStateHandlers.forEach((handler) => handler(state));
  }

  // Getters
  getConnectionState(): ConnectionState {
    return this.connectionState;
  }

  isConnected(): boolean {
    return this.connectionState === 'connected';
  }
}

// Export singleton instance
export const websocket = new WebSocketClient();

export default websocket;
