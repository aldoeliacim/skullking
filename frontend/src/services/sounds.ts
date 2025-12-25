import AsyncStorage from '@react-native-async-storage/async-storage';
import { Audio } from 'expo-av';

const SOUND_ENABLED_KEY = '@skullking/soundEnabled';

// Sound file mapping - would be loaded from assets
type SoundName =
  | 'cardPlay'
  | 'cardDeal'
  | 'bidPlace'
  | 'trickWin'
  | 'yourTurn'
  | 'gameStart'
  | 'gameEnd'
  | 'buttonClick'
  | 'error'
  | 'mermaidCapture';

interface SoundConfig {
  volume: number;
  loop?: boolean;
}

const soundConfigs: Record<SoundName, SoundConfig> = {
  cardPlay: { volume: 0.5 },
  cardDeal: { volume: 0.3 },
  bidPlace: { volume: 0.4 },
  trickWin: { volume: 0.6 },
  yourTurn: { volume: 0.7 },
  gameStart: { volume: 0.6 },
  gameEnd: { volume: 0.7 },
  buttonClick: { volume: 0.2 },
  error: { volume: 0.5 },
  mermaidCapture: { volume: 0.8 },
};

class SoundService {
  private sounds: Map<SoundName, Audio.Sound> = new Map();
  private enabled = true;
  private initialized = false;

  async initialize(): Promise<void> {
    if (this.initialized) {
      return;
    }

    try {
      // Configure audio mode
      await Audio.setAudioModeAsync({
        playsInSilentModeIOS: true,
        staysActiveInBackground: false,
        shouldDuckAndroid: true,
      });

      // Load sound enabled setting
      const savedEnabled = await AsyncStorage.getItem(SOUND_ENABLED_KEY);
      this.enabled = savedEnabled !== 'false';

      this.initialized = true;
    } catch (error) {
      console.error('[SoundService] Failed to initialize:', error);
    }
  }

  async loadSound(name: SoundName, uri: string): Promise<void> {
    try {
      const { sound } = await Audio.Sound.createAsync(
        { uri },
        { volume: soundConfigs[name].volume },
      );
      this.sounds.set(name, sound);
    } catch (error) {
      console.error(`[SoundService] Failed to load sound ${name}:`, error);
    }
  }

  async play(name: SoundName): Promise<void> {
    if (!this.enabled) {
      return;
    }

    const sound = this.sounds.get(name);
    if (!sound) {
      // Sound not loaded, skip
      return;
    }

    try {
      await sound.setPositionAsync(0);
      await sound.playAsync();
    } catch (error) {
      console.error(`[SoundService] Failed to play sound ${name}:`, error);
    }
  }

  async setEnabled(enabled: boolean): Promise<void> {
    this.enabled = enabled;
    await AsyncStorage.setItem(SOUND_ENABLED_KEY, enabled ? 'true' : 'false');
  }

  isEnabled(): boolean {
    return this.enabled;
  }

  async cleanup(): Promise<void> {
    for (const sound of this.sounds.values()) {
      try {
        await sound.unloadAsync();
      } catch {
        // Ignore cleanup errors
      }
    }
    this.sounds.clear();
    this.initialized = false;
  }
}

// Export singleton instance
export const soundService = new SoundService();

// Convenience function
export const playSound = (name: SoundName): void => {
  soundService.play(name);
};

export default soundService;
