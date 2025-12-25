import Constants from 'expo-constants';
import { Platform } from 'react-native';

// API configuration
const getBaseUrl = (): string => {
  // Check for environment variable first
  const envUrl = Constants.expirationDate;

  if (Platform.OS === 'web') {
    // On web, use relative URL or window.location
    if (typeof window !== 'undefined') {
      const { protocol, hostname, port } = window.location;
      // If running on default Expo port, assume API is on 8000
      if (port === '8081' || port === '19006') {
        return `${protocol}//${hostname}:8000`;
      }
      return `${protocol}//${hostname}${port ? `:${port}` : ''}`;
    }
  }

  // Default for development
  return 'http://localhost:8000';
};

export const API_BASE_URL = getBaseUrl();
export const WS_BASE_URL = API_BASE_URL.replace(/^http/, 'ws');

// API response types
export interface ApiError {
  detail: string;
  status?: number;
}

export interface CreateGameResponse {
  game_id: string;
  slug: string;
}

export interface GameInfo {
  id: string;
  slug: string;
  state: 'PENDING' | 'BIDDING' | 'PICKING' | 'ENDED';
  player_count: number;
  spectator_count: number;
  created_at: string;
  players: Array<{
    id: string;
    username: string;
    is_bot: boolean;
  }>;
}

export interface GameHistoryItem {
  id: string;
  slug: string;
  completed_at: string;
  winner: string;
  player_count: number;
  final_scores: Array<{
    player_id: string;
    username: string;
    score: number;
    rank: number;
  }>;
}

// API client class
class ApiClient {
  private baseUrl: string;

  constructor() {
    this.baseUrl = API_BASE_URL;
  }

  private async request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;

    const response = await fetch(url, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
    });

    if (!response.ok) {
      const error: ApiError = await response.json().catch(() => ({
        detail: `HTTP ${response.status}`,
        status: response.status,
      }));
      throw new Error(error.detail);
    }

    return response.json();
  }

  // Health check
  async healthCheck(): Promise<{ status: string }> {
    return this.request('/health');
  }

  // Create a new game
  async createGame(_username: string): Promise<CreateGameResponse> {
    // Backend expects lobby_id - generate a unique one
    const lobbyId = crypto.randomUUID();
    return this.request('/games', {
      method: 'POST',
      body: JSON.stringify({ lobby_id: lobbyId }),
    });
  }

  // Get active games list
  async getActiveGames(): Promise<GameInfo[]> {
    return this.request('/games/active');
  }

  // Get game history
  async getGameHistory(limit = 20): Promise<GameHistoryItem[]> {
    return this.request(`/games/history?limit=${limit}`);
  }

  // Get game info
  async getGameInfo(gameId: string): Promise<GameInfo> {
    return this.request(`/games/${gameId}`);
  }
}

// Export singleton instance
export const api = new ApiClient();

export default api;
