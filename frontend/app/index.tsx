import AsyncStorage from '@react-native-async-storage/async-storage';
import { useRouter } from 'expo-router';
import React, { useState, useCallback, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import {
  ActivityIndicator,
  KeyboardAvoidingView,
  Modal,
  Platform,
  Pressable,
  RefreshControl,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';
import Animated, { SlideInRight } from 'react-native-reanimated';
import { SafeAreaView } from 'react-native-safe-area-context';
import { SettingsButton } from '../src/components';
import { GameInfo, api } from '../src/services/api';
import { colors } from '../src/styles/theme';

const PLAYER_NAME_KEY = '@skullking/playerName';

export default function HomeScreen(): React.JSX.Element {
  const { t } = useTranslation();
  const router = useRouter();

  const [playerName, setPlayerName] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showBrowse, setShowBrowse] = useState(false);

  useEffect(() => {
    AsyncStorage.getItem(PLAYER_NAME_KEY).then((saved) => {
      if (saved) setPlayerName(saved);
    });
  }, []);

  const handleNameChange = useCallback((name: string) => {
    setPlayerName(name);
    setError(null);
    AsyncStorage.setItem(PLAYER_NAME_KEY, name);
  }, []);

  const handleCreateGame = useCallback(async () => {
    if (!playerName.trim()) {
      setError(t('login.errorEnterName'));
      return;
    }
    setIsLoading(true);
    setError(null);
    try {
      const response = await api.createGame(playerName.trim());
      router.push({
        pathname: '/lobby/[id]',
        params: {
          id: response.game_id,
          slug: response.slug,
          playerId: 'host',
          playerName: playerName.trim(),
        },
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : t('login.errorCreateFailed'));
    } finally {
      setIsLoading(false);
    }
  }, [playerName, router, t]);

  const handleSelectGame = useCallback(
    (game: GameInfo, spectate: boolean) => {
      setShowBrowse(false);
      if (spectate) {
        router.push({ pathname: '/game/[id]', params: { id: game.slug, spectator: 'true' } });
      } else {
        if (!playerName.trim()) {
          setError(t('login.errorEnterName'));
          return;
        }
        router.push({
          pathname: '/lobby/[id]',
          params: { id: game.slug, playerName: playerName.trim() },
        });
      }
    },
    [router, playerName, t],
  );

  return (
    <SafeAreaView style={styles.container}>
      <SettingsButton style={styles.settingsButton} />

      <KeyboardAvoidingView
        style={styles.keyboardView}
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
      >
        <ScrollView
          contentContainerStyle={styles.scrollContent}
          keyboardShouldPersistTaps="handled"
        >
          {/* Header */}
          <Text style={styles.emoji}>🏴‍☠️</Text>
          <Text style={styles.title}>Skull King</Text>
          <Text style={styles.subtitle}>{t('app.subtitle')}</Text>

          {/* Form */}
          <View style={styles.form}>
            <Text style={styles.label}>{t('login.yourName')}</Text>
            <TextInput
              style={styles.input}
              placeholder={t('login.enterName')}
              placeholderTextColor="#666"
              value={playerName}
              onChangeText={handleNameChange}
              autoCapitalize="words"
              autoCorrect={false}
              maxLength={20}
            />

            <View style={styles.buttonRow}>
              <Pressable
                style={({ pressed }) => [styles.primaryButton, pressed && styles.buttonPressed]}
                onPress={handleCreateGame}
                disabled={isLoading}
              >
                {isLoading ? (
                  <ActivityIndicator color="#000" />
                ) : (
                  <Text style={styles.primaryButtonText}>{t('login.createGame')}</Text>
                )}
              </Pressable>
              <Pressable
                style={({ pressed }) => [styles.secondaryButton, pressed && styles.buttonPressed]}
                onPress={() => setShowBrowse(true)}
              >
                <Text style={styles.secondaryButtonText}>{t('login.browseGames')}</Text>
              </Pressable>
            </View>

            {error && <Text style={styles.error}>{error}</Text>}
          </View>
        </ScrollView>
      </KeyboardAvoidingView>

      {/* Browse Modal */}
      <BrowseModal
        visible={showBrowse}
        onClose={() => setShowBrowse(false)}
        onSelectGame={handleSelectGame}
      />
    </SafeAreaView>
  );
}

function BrowseModal({
  visible,
  onClose,
  onSelectGame,
}: {
  visible: boolean;
  onClose: () => void;
  onSelectGame: (game: GameInfo, spectate: boolean) => void;
}) {
  const { t } = useTranslation();
  const [games, setGames] = useState<GameInfo[]>([]);
  const [loading, setLoading] = useState(false);

  const loadGames = useCallback(async () => {
    setLoading(true);
    try {
      setGames(await api.getActiveGames());
    } catch {
      // ignore
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (visible) loadGames();
  }, [visible, loadGames]);

  return (
    <Modal visible={visible} animationType="slide" transparent onRequestClose={onClose}>
      <Pressable style={styles.modalOverlay} onPress={onClose}>
        <Pressable style={styles.modalContent} onPress={(e) => e.stopPropagation()}>
          <View style={styles.modalHeader}>
            <Text style={styles.modalTitle}>🎮 {t('browse.title')}</Text>
            <Pressable onPress={onClose}>
              <Text style={styles.closeButton}>✕</Text>
            </Pressable>
          </View>
          <ScrollView
            style={styles.modalScroll}
            refreshControl={
              <RefreshControl
                refreshing={loading}
                onRefresh={loadGames}
                tintColor={colors.primary}
              />
            }
          >
            {games.length === 0 ? (
              <Text style={styles.emptyText}>
                {loading ? t('browse.loading') : t('browse.noGames')}
              </Text>
            ) : (
              games.map((game, i) => (
                <Animated.View key={game.id} entering={SlideInRight.delay(i * 50)}>
                  <View style={styles.gameCard}>
                    <Text style={styles.gameCode}>{game.slug}</Text>
                    <Text style={styles.gamePlayers}>
                      {game.players.map((p) => p.username).join(', ')}
                    </Text>
                    <View style={styles.gameActions}>
                      {game.state === 'PENDING' ? (
                        <Pressable
                          style={styles.smallButton}
                          onPress={() => onSelectGame(game, false)}
                        >
                          <Text style={styles.smallButtonText}>{t('browse.join')}</Text>
                        </Pressable>
                      ) : (
                        <Pressable
                          style={styles.smallButtonOutline}
                          onPress={() => onSelectGame(game, true)}
                        >
                          <Text style={styles.smallButtonOutlineText}>{t('browse.watch')}</Text>
                        </Pressable>
                      )}
                    </View>
                  </View>
                </Animated.View>
              ))
            )}
          </ScrollView>
        </Pressable>
      </Pressable>
    </Modal>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0a1628',
  },
  settingsButton: {
    position: 'absolute',
    top: 16,
    right: 16,
    zIndex: 10,
  },
  keyboardView: {
    flex: 1,
  },
  scrollContent: {
    flexGrow: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 24,
  },
  emoji: {
    fontSize: 40,
    marginBottom: 8,
  },
  title: {
    fontSize: 28,
    fontWeight: '700',
    color: '#d4a84b',
    marginBottom: 4,
    fontFamily: Platform.select({ web: "'Pirata One', cursive", default: undefined }),
  },
  subtitle: {
    fontSize: 14,
    color: '#8899a6',
    fontStyle: 'italic',
    marginBottom: 32,
  },
  form: {
    width: '100%',
    maxWidth: 360,
    backgroundColor: '#0f1d2e',
    borderRadius: 12,
    padding: 24,
    borderWidth: 1,
    borderColor: '#1a2d42',
  },
  label: {
    fontSize: 13,
    fontWeight: '600',
    color: '#a0aec0',
    marginBottom: 6,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  input: {
    backgroundColor: '#0a1628',
    borderWidth: 1,
    borderColor: '#1a2d42',
    borderRadius: 8,
    padding: 12,
    fontSize: 16,
    color: '#e2e8f0',
    marginBottom: 16,
  },
  buttonRow: {
    flexDirection: 'row',
    gap: 12,
  },
  primaryButton: {
    flex: 1,
    backgroundColor: '#d4a84b',
    borderRadius: 8,
    padding: 14,
    alignItems: 'center',
  },
  primaryButtonText: {
    color: '#000',
    fontSize: 16,
    fontWeight: '600',
  },
  secondaryButton: {
    flex: 1,
    backgroundColor: '#1a2d42',
    borderRadius: 8,
    padding: 14,
    alignItems: 'center',
  },
  secondaryButtonText: {
    color: '#e2e8f0',
    fontSize: 15,
    fontWeight: '600',
  },
  buttonPressed: {
    opacity: 0.8,
    transform: [{ scale: 0.98 }],
  },
  error: {
    marginTop: 16,
    padding: 12,
    backgroundColor: 'rgba(239, 68, 68, 0.15)',
    borderRadius: 8,
    color: '#ef4444',
    fontSize: 14,
    textAlign: 'center',
  },
  // Modal
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0,0,0,0.7)',
    justifyContent: 'center',
    alignItems: 'center',
    padding: 24,
  },
  modalContent: {
    width: '100%',
    maxWidth: 400,
    maxHeight: '70%',
    backgroundColor: '#0f1d2e',
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#d4a84b33',
    overflow: 'hidden',
  },
  modalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#1a2d42',
  },
  modalTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#d4a84b',
  },
  closeButton: {
    fontSize: 20,
    color: '#8899a6',
    padding: 4,
  },
  modalScroll: {
    padding: 16,
  },
  emptyText: {
    color: '#8899a6',
    textAlign: 'center',
    padding: 32,
  },
  gameCard: {
    backgroundColor: '#0a1628',
    borderRadius: 8,
    padding: 12,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: '#1a2d42',
  },
  gameCode: {
    fontSize: 16,
    fontWeight: '700',
    color: '#d4a84b',
    letterSpacing: 2,
    marginBottom: 4,
  },
  gamePlayers: {
    fontSize: 13,
    color: '#8899a6',
    marginBottom: 12,
  },
  gameActions: {
    flexDirection: 'row',
  },
  smallButton: {
    backgroundColor: '#d4a84b',
    borderRadius: 6,
    paddingVertical: 8,
    paddingHorizontal: 16,
  },
  smallButtonText: {
    color: '#000',
    fontSize: 13,
    fontWeight: '600',
  },
  smallButtonOutline: {
    borderWidth: 1,
    borderColor: '#1a2d42',
    borderRadius: 6,
    paddingVertical: 8,
    paddingHorizontal: 16,
  },
  smallButtonOutlineText: {
    color: '#8899a6',
    fontSize: 13,
    fontWeight: '600',
  },
});
