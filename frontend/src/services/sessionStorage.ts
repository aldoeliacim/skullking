/**
 * Session storage service for persisting game session across app backgrounding.
 *
 * Saves game connection info to AsyncStorage so users can reconnect
 * when they return to the app after locking screen, switching tabs, etc.
 */

import AsyncStorage from '@react-native-async-storage/async-storage';

const SESSION_KEY = '@skullking/activeSession';

export interface GameSession {
  gameId: string;
  playerId: string;
  playerName: string;
  isSpectator: boolean;
  savedAt: number;
}

// Session expires after 2 hours of inactivity
const SESSION_EXPIRY_MS = 2 * 60 * 60 * 1000;

class SessionStorage {
  /**
   * Save active game session.
   */
  async saveSession(session: Omit<GameSession, 'savedAt'>): Promise<void> {
    try {
      const data: GameSession = {
        ...session,
        savedAt: Date.now(),
      };
      await AsyncStorage.setItem(SESSION_KEY, JSON.stringify(data));
    } catch (error) {
      console.error('[SessionStorage] Failed to save session:', error);
    }
  }

  /**
   * Load active game session if it exists and hasn't expired.
   */
  async loadSession(): Promise<GameSession | null> {
    try {
      const data = await AsyncStorage.getItem(SESSION_KEY);
      if (!data) {
        return null;
      }

      const session: GameSession = JSON.parse(data);

      // Check if session has expired
      if (Date.now() - session.savedAt > SESSION_EXPIRY_MS) {
        await this.clearSession();
        return null;
      }

      return session;
    } catch (error) {
      console.error('[SessionStorage] Failed to load session:', error);
      return null;
    }
  }

  /**
   * Clear saved session (e.g., when game ends or user explicitly leaves).
   */
  async clearSession(): Promise<void> {
    try {
      await AsyncStorage.removeItem(SESSION_KEY);
    } catch (error) {
      console.error('[SessionStorage] Failed to clear session:', error);
    }
  }

  /**
   * Check if a valid session exists.
   */
  async hasValidSession(): Promise<boolean> {
    const session = await this.loadSession();
    return session !== null;
  }
}

export const sessionStorage = new SessionStorage();

export default sessionStorage;
