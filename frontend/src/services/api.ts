import type { CreateGameResponse, GameInfo } from '../types/game';

// API configuration - detect backend URL
const getBaseUrl = (): string => {
  const { protocol, hostname, port } = window.location;

  // Development: Vite runs on 5173, backend on 8000
  if (port === '5173' || port === '3000') {
    return `${protocol}//${hostname}:8000`;
  }

  // Production: same origin
  return `${protocol}//${hostname}${port ? `:${port}` : ''}`;
};

export const API_BASE_URL = getBaseUrl();
export const WS_BASE_URL = API_BASE_URL.replace(/^http/, 'ws');

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
      const error = await response.json().catch(() => ({
        detail: `HTTP ${response.status}`,
      }));
      throw new Error(error.detail || 'Request failed');
    }

    return response.json();
  }

  async healthCheck(): Promise<{ status: string }> {
    return this.request('/health');
  }

  async createGame(): Promise<CreateGameResponse> {
    const lobbyId = crypto.randomUUID();
    return this.request('/games', {
      method: 'POST',
      body: JSON.stringify({ lobby_id: lobbyId }),
    });
  }

  async getActiveGames(): Promise<GameInfo[]> {
    return this.request('/games/active');
  }

  async getGameInfo(gameId: string): Promise<GameInfo> {
    return this.request(`/games/${gameId}`);
  }
}

export const api = new ApiClient();
export default api;
