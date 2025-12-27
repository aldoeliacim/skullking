import AsyncStorage from '@react-native-async-storage/async-storage';
import { useRouter } from 'expo-router';
import React, { useState, useCallback, useEffect } from 'react';
import { useTranslation } from 'react-i18next';
import {
  ActivityIndicator,
  Dimensions,
  KeyboardAvoidingView,
  Modal,
  Platform,
  Pressable,
  RefreshControl,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import Animated, {
  FadeIn,
  FadeInDown,
  FadeInUp,
  FadeOut,
  SlideInRight,
} from 'react-native-reanimated';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Button, Input, SettingsButton } from '../src/components';
import { GameHistoryItem, GameInfo, api } from '../src/services/api';
import {
  borderRadius,
  breakpoints,
  colors,
  shadows,
  spacing,
  typography,
} from '../src/styles/theme';

const PLAYER_NAME_KEY = '@skullking/playerName';

// Hook to get responsive values
function useResponsive() {
  const [dimensions, setDimensions] = useState(Dimensions.get('window'));

  useEffect(() => {
    const subscription = Dimensions.addEventListener('change', ({ window }) => {
      setDimensions(window);
    });
    return () => subscription.remove();
  }, []);

  const isDesktop = dimensions.width >= breakpoints.lg;
  const isTablet = dimensions.width >= breakpoints.md && dimensions.width < breakpoints.lg;
  const isMobile = dimensions.width < breakpoints.md;

  return { width: dimensions.width, height: dimensions.height, isDesktop, isTablet, isMobile };
}

// Game state badge component
function GameStateBadge({ state }: { state: string }) {
  const { t } = useTranslation();
  const stateConfig: Record<string, { label: string; color: string }> = {
    PENDING: { label: t('browse.statePending'), color: colors.warning },
    BIDDING: { label: t('browse.stateBidding'), color: colors.info },
    PICKING: { label: t('browse.statePlaying'), color: colors.success },
  };
  const config = stateConfig[state] || { label: state, color: colors.textMuted };

  return (
    <View style={[styles.badge, { backgroundColor: config.color + '30' }]}>
      <Text style={[styles.badgeText, { color: config.color }]}>{config.label}</Text>
    </View>
  );
}

// Browse Games Modal Component
function BrowseGamesModal({
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
  const [error, setError] = useState<string | null>(null);

  const loadGames = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.getActiveGames();
      setGames(data);
    } catch {
      setError(t('browse.error'));
    } finally {
      setLoading(false);
    }
  }, [t]);

  useEffect(() => {
    if (visible) {
      loadGames();
    }
  }, [visible, loadGames]);

  return (
    <Modal visible={visible} animationType="slide" transparent onRequestClose={onClose}>
      <Pressable style={styles.modalOverlay} onPress={onClose}>
        <Pressable style={styles.modalContainer} onPress={(e) => e.stopPropagation()}>
          <Animated.View entering={FadeIn.duration(200)} style={styles.modalContent}>
            <View style={styles.modalHeader}>
              <Text style={styles.modalTitle}>🎮 {t('browse.title')}</Text>
              <Pressable onPress={onClose} style={styles.closeButton}>
                <Text style={styles.closeButtonText}>✕</Text>
              </Pressable>
            </View>

            <ScrollView
              style={styles.modalScroll}
              contentContainerStyle={styles.modalScrollContent}
              refreshControl={
                <RefreshControl
                  refreshing={loading}
                  onRefresh={loadGames}
                  tintColor={colors.primary}
                />
              }
            >
              {loading && games.length === 0 ? (
                <View style={styles.centerContent}>
                  <ActivityIndicator size="large" color={colors.primary} />
                  <Text style={styles.loadingText}>{t('browse.loading')}</Text>
                </View>
              ) : error ? (
                <View style={styles.centerContent}>
                  <Text style={styles.errorText}>{error}</Text>
                  <Button title={t('browse.refresh')} onPress={loadGames} variant="outline" />
                </View>
              ) : games.length === 0 ? (
                <View style={styles.centerContent}>
                  <Text style={styles.emptyEmoji}>🏴‍☠️</Text>
                  <Text style={styles.emptyText}>{t('browse.noGames')}</Text>
                  <Text style={styles.emptySubtext}>{t('browse.noGamesHint')}</Text>
                </View>
              ) : (
                games.map((game, index) => (
                  <Animated.View
                    key={game.id}
                    entering={SlideInRight.delay(index * 50).duration(200)}
                  >
                    <View style={styles.gameCard}>
                      <View style={styles.gameCardHeader}>
                        <Text style={styles.gameCode}>{game.slug}</Text>
                        <GameStateBadge state={game.state} />
                      </View>
                      <View style={styles.gameCardInfo}>
                        <Text style={styles.gameInfoText}>
                          👥 {game.player_count} players
                          {game.spectator_count > 0 && ` • 👁 ${game.spectator_count} watching`}
                        </Text>
                        <Text style={styles.gamePlayersText}>
                          {game.players.map((p) => p.username).join(', ')}
                        </Text>
                      </View>
                      <View style={styles.gameCardActions}>
                        {game.state === 'PENDING' ? (
                          <Button
                            title="Join"
                            onPress={() => onSelectGame(game, false)}
                            size="sm"
                            fullWidth
                          />
                        ) : (
                          <Button
                            title={t('browse.watch')}
                            onPress={() => onSelectGame(game, true)}
                            variant="secondary"
                            size="sm"
                            fullWidth
                          />
                        )}
                      </View>
                    </View>
                  </Animated.View>
                ))
              )}
            </ScrollView>
          </Animated.View>
        </Pressable>
      </Pressable>
    </Modal>
  );
}

// History Modal Component
function HistoryModal({ visible, onClose }: { visible: boolean; onClose: () => void }) {
  const { t } = useTranslation();
  const [history, setHistory] = useState<GameHistoryItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadHistory = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.getGameHistory(20);
      setHistory(data);
    } catch {
      setError(t('history.error'));
    } finally {
      setLoading(false);
    }
  }, [t]);

  useEffect(() => {
    if (visible) {
      loadHistory();
    }
  }, [visible, loadHistory]);

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    return date.toLocaleDateString(undefined, {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  return (
    <Modal visible={visible} animationType="slide" transparent onRequestClose={onClose}>
      <Pressable style={styles.modalOverlay} onPress={onClose}>
        <Pressable style={styles.modalContainer} onPress={(e) => e.stopPropagation()}>
          <Animated.View entering={FadeIn.duration(200)} style={styles.modalContent}>
            <View style={styles.modalHeader}>
              <Text style={styles.modalTitle}>📜 {t('history.title')}</Text>
              <Pressable onPress={onClose} style={styles.closeButton}>
                <Text style={styles.closeButtonText}>✕</Text>
              </Pressable>
            </View>

            <ScrollView
              style={styles.modalScroll}
              contentContainerStyle={styles.modalScrollContent}
              refreshControl={
                <RefreshControl
                  refreshing={loading}
                  onRefresh={loadHistory}
                  tintColor={colors.primary}
                />
              }
            >
              {loading && history.length === 0 ? (
                <View style={styles.centerContent}>
                  <ActivityIndicator size="large" color={colors.primary} />
                  <Text style={styles.loadingText}>{t('history.loading')}</Text>
                </View>
              ) : error ? (
                <View style={styles.centerContent}>
                  <Text style={styles.errorText}>{error}</Text>
                  <Button title={t('browse.refresh')} onPress={loadHistory} variant="outline" />
                </View>
              ) : history.length === 0 ? (
                <View style={styles.centerContent}>
                  <Text style={styles.emptyEmoji}>🎯</Text>
                  <Text style={styles.emptyText}>{t('history.noGames')}</Text>
                  <Text style={styles.emptySubtext}>{t('history.noGamesHint')}</Text>
                </View>
              ) : (
                history.map((game, index) => (
                  <Animated.View
                    key={game.id}
                    entering={SlideInRight.delay(index * 50).duration(200)}
                  >
                    <View style={styles.historyCard}>
                      <View style={styles.historyCardHeader}>
                        <Text style={styles.historyWinner}>🏆 {game.winner}</Text>
                        <Text style={styles.historyDate}>{formatDate(game.completed_at)}</Text>
                      </View>
                      <View style={styles.historyScores}>
                        {game.final_scores.slice(0, 4).map((score, i) => (
                          <View key={score.player_id} style={styles.historyScoreRow}>
                            <Text style={styles.historyRank}>
                              {i === 0 ? '🥇' : i === 1 ? '🥈' : i === 2 ? '🥉' : `${i + 1}.`}
                            </Text>
                            <Text style={styles.historyPlayerName}>{score.username}</Text>
                            <Text style={styles.historyScore}>{score.score} pts</Text>
                          </View>
                        ))}
                      </View>
                    </View>
                  </Animated.View>
                ))
              )}
            </ScrollView>
          </Animated.View>
        </Pressable>
      </Pressable>
    </Modal>
  );
}

export default function HomeScreen(): React.JSX.Element {
  const { t } = useTranslation();
  const router = useRouter();
  const { isDesktop, isMobile } = useResponsive();

  const [playerName, setPlayerName] = useState('');
  const [gameCode, setGameCode] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showBrowse, setShowBrowse] = useState(false);
  const [showHistory, setShowHistory] = useState(false);

  // Load saved player name on mount
  useEffect(() => {
    const loadPlayerName = async (): Promise<void> => {
      const saved = await AsyncStorage.getItem(PLAYER_NAME_KEY);
      if (saved) {
        setPlayerName(saved);
      }
    };
    loadPlayerName();
  }, []);

  // Save player name when it changes
  const handleNameChange = useCallback((name: string) => {
    setPlayerName(name);
    setError(null);
    AsyncStorage.setItem(PLAYER_NAME_KEY, name);
  }, []);

  const validateName = useCallback((): boolean => {
    if (!playerName.trim()) {
      setError(t('login.errorEnterName'));
      return false;
    }
    return true;
  }, [playerName, t]);

  const handleCreateGame = useCallback(async () => {
    if (!validateName()) return;

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
  }, [playerName, validateName, router, t]);

  const handleJoinGame = useCallback(() => {
    if (!validateName()) return;

    if (!gameCode.trim()) {
      setError(t('login.errorEnterGameId'));
      return;
    }

    router.push({
      pathname: '/lobby/[id]',
      params: {
        id: gameCode.trim().toUpperCase(),
        playerName: playerName.trim(),
      },
    });
  }, [playerName, gameCode, validateName, router, t]);

  const handleSpectate = useCallback(() => {
    if (!gameCode.trim()) {
      setError(t('login.errorEnterGameId'));
      return;
    }

    router.push({
      pathname: '/game/[id]',
      params: {
        id: gameCode.trim().toUpperCase(),
        spectator: 'true',
      },
    });
  }, [gameCode, router, t]);

  const handleSelectGame = useCallback(
    (game: GameInfo, spectate: boolean) => {
      setShowBrowse(false);
      if (spectate) {
        router.push({
          pathname: '/game/[id]',
          params: { id: game.slug, spectator: 'true' },
        });
      } else {
        if (!validateName()) return;
        router.push({
          pathname: '/lobby/[id]',
          params: { id: game.slug, playerName: playerName.trim() },
        });
      }
    },
    [router, playerName, validateName],
  );

  return (
    <SafeAreaView style={styles.container} edges={['top', 'left', 'right']}>
      <SettingsButton style={styles.settingsButton} />

      <KeyboardAvoidingView
        style={styles.keyboardView}
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
      >
        <ScrollView
          contentContainerStyle={[styles.scrollContent, isDesktop && styles.scrollContentDesktop]}
          showsVerticalScrollIndicator={false}
          keyboardShouldPersistTaps="handled"
        >
          {/* Main content wrapper for desktop layout */}
          <View style={[styles.mainWrapper, isDesktop && styles.mainWrapperDesktop]}>
            {/* Left side - Header and Form */}
            <View style={[styles.leftSection, isDesktop && styles.leftSectionDesktop]}>
              {/* Header */}
              <Animated.View entering={FadeInUp.delay(100).duration(500)} style={styles.header}>
                <Text style={styles.emoji}>🏴‍☠️</Text>
                <Text style={[styles.title, isDesktop && styles.titleDesktop]}>
                  {t('app.title')}
                </Text>
                <Text style={styles.subtitle}>{t('app.subtitle')}</Text>
              </Animated.View>

              {/* Form Card */}
              <Animated.View
                entering={FadeInDown.delay(300).duration(500)}
                style={[styles.formCard, isDesktop && styles.formCardDesktop]}
              >
                <Input
                  label={t('login.yourName')}
                  placeholder={t('login.enterName')}
                  value={playerName}
                  onChangeText={handleNameChange}
                  autoCapitalize="words"
                  autoCorrect={false}
                  maxLength={20}
                />

                <Button
                  title={t('login.createGame')}
                  onPress={handleCreateGame}
                  loading={isLoading}
                  fullWidth
                  size="lg"
                />

                <View style={styles.divider}>
                  <View style={styles.dividerLine} />
                  <Text style={styles.dividerText}>{t('login.or')}</Text>
                  <View style={styles.dividerLine} />
                </View>

                <Input
                  label={t('login.gameCode')}
                  placeholder={t('login.enterGameId')}
                  value={gameCode}
                  onChangeText={(text) => {
                    setGameCode(text.toUpperCase());
                    setError(null);
                  }}
                  autoCapitalize="characters"
                  autoCorrect={false}
                  maxLength={4}
                />

                <View style={styles.buttonRow}>
                  <View style={styles.buttonFlex}>
                    <Button
                      title={t('login.joinGame')}
                      onPress={handleJoinGame}
                      variant="secondary"
                      fullWidth
                    />
                  </View>
                  <View style={styles.buttonFlex}>
                    <Button
                      title={t('login.spectateGame')}
                      onPress={handleSpectate}
                      variant="outline"
                      fullWidth
                    />
                  </View>
                </View>

                {error && (
                  <Animated.View entering={FadeInDown.duration(200)} exiting={FadeOut}>
                    <Text style={styles.error}>{error}</Text>
                  </Animated.View>
                )}
              </Animated.View>
            </View>

            {/* Right side - Quick Actions (on desktop) or Footer (on mobile) */}
            <Animated.View
              entering={FadeInDown.delay(500).duration(500)}
              style={[
                styles.rightSection,
                isDesktop && styles.rightSectionDesktop,
                isMobile && styles.rightSectionMobile,
              ]}
            >
              {/* Browse Games Card */}
              <Pressable
                style={({ pressed }) => [styles.actionCard, pressed && styles.actionCardPressed]}
                onPress={() => setShowBrowse(true)}
              >
                <Text style={styles.actionIcon}>🎮</Text>
                <View style={styles.actionCardContent}>
                  <Text style={styles.actionTitle}>{t('login.browseGames')}</Text>
                  <Text style={styles.actionDescription}>{t('browse.description')}</Text>
                </View>
                <Text style={styles.actionArrow}>→</Text>
              </Pressable>

              {/* History Card */}
              <Pressable
                style={({ pressed }) => [styles.actionCard, pressed && styles.actionCardPressed]}
                onPress={() => setShowHistory(true)}
              >
                <Text style={styles.actionIcon}>📜</Text>
                <View style={styles.actionCardContent}>
                  <Text style={styles.actionTitle}>{t('login.viewHistory')}</Text>
                  <Text style={styles.actionDescription}>{t('history.description')}</Text>
                </View>
                <Text style={styles.actionArrow}>→</Text>
              </Pressable>

              {/* Quick tip on desktop */}
              {isDesktop && (
                <View style={styles.tipCard}>
                  <Text style={styles.tipIcon}>💡</Text>
                  <Text style={styles.tipText}>
                    Tip: Share your 4-letter game code with friends to play together!
                  </Text>
                </View>
              )}
            </Animated.View>
          </View>
        </ScrollView>
      </KeyboardAvoidingView>

      {/* Modals */}
      <BrowseGamesModal
        visible={showBrowse}
        onClose={() => setShowBrowse(false)}
        onSelectGame={handleSelectGame}
      />
      <HistoryModal visible={showHistory} onClose={() => setShowHistory(false)} />
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background,
  },
  settingsButton: {
    position: 'absolute',
    top: spacing.md,
    right: spacing.md,
    zIndex: 10,
  },
  keyboardView: {
    flex: 1,
  },
  scrollContent: {
    flexGrow: 1,
    padding: spacing.lg,
    justifyContent: 'center',
  },
  scrollContentDesktop: {
    padding: spacing['2xl'],
  },
  mainWrapper: {
    width: '100%',
    maxWidth: 500,
    alignSelf: 'center',
  },
  mainWrapperDesktop: {
    maxWidth: 1000,
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: spacing['3xl'],
  },
  leftSection: {
    width: '100%',
  },
  leftSectionDesktop: {
    flex: 1,
    maxWidth: 480,
  },
  rightSection: {
    width: '100%',
    marginTop: spacing['2xl'],
    gap: spacing.md,
  },
  rightSectionDesktop: {
    flex: 1,
    marginTop: spacing['5xl'],
    maxWidth: 400,
  },
  rightSectionMobile: {
    marginTop: spacing.xl,
  },
  header: {
    alignItems: 'center',
    marginBottom: spacing['2xl'],
  },
  emoji: {
    fontSize: 64,
    marginBottom: spacing.sm,
  },
  title: {
    fontSize: typography.fontSize['4xl'],
    fontWeight: typography.fontWeight.bold,
    fontFamily: typography.fontFamilyDisplay,
    color: colors.accentGold,
    textAlign: 'center',
    marginBottom: spacing.xs,
    textShadowColor: 'rgba(0, 0, 0, 0.5)',
    textShadowOffset: { width: 2, height: 2 },
    textShadowRadius: 4,
    letterSpacing: 2,
  },
  titleDesktop: {
    fontSize: typography.fontSize['5xl'],
  },
  subtitle: {
    fontSize: typography.fontSize.md,
    fontFamily: typography.fontFamily,
    fontStyle: 'italic',
    color: colors.textMuted,
    textAlign: 'center',
  },
  formCard: {
    backgroundColor: colors.surface,
    borderRadius: borderRadius.lg,
    padding: spacing.xl,
    borderWidth: 1,
    borderColor: colors.border,
    ...shadows.md,
  },
  formCardDesktop: {
    padding: spacing['2xl'],
  },
  divider: {
    flexDirection: 'row',
    alignItems: 'center',
    marginVertical: spacing.lg,
  },
  dividerLine: {
    flex: 1,
    height: 1,
    backgroundColor: colors.border,
  },
  dividerText: {
    paddingHorizontal: spacing.md,
    fontSize: typography.fontSize.sm,
    color: colors.textMuted,
    textTransform: 'uppercase',
  },
  buttonRow: {
    flexDirection: 'row',
    gap: spacing.md,
  },
  buttonFlex: {
    flex: 1,
  },
  error: {
    marginTop: spacing.md,
    padding: spacing.md,
    backgroundColor: colors.error + '20',
    borderRadius: borderRadius.base,
    color: colors.error,
    fontSize: typography.fontSize.sm,
    textAlign: 'center',
  },
  // Action cards
  actionCard: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.surface,
    borderRadius: borderRadius.md,
    padding: spacing.lg,
    borderWidth: 1,
    borderColor: colors.border,
    ...shadows.sm,
  },
  actionCardPressed: {
    backgroundColor: colors.surfaceLight,
    transform: [{ scale: 0.98 }],
  },
  actionIcon: {
    fontSize: 28,
    marginRight: spacing.md,
  },
  actionCardContent: {
    flex: 1,
  },
  actionTitle: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text,
    marginBottom: 2,
  },
  actionDescription: {
    fontSize: typography.fontSize.sm,
    color: colors.textMuted,
  },
  actionArrow: {
    fontSize: typography.fontSize.xl,
    color: colors.primary,
    marginLeft: spacing.sm,
  },
  tipCard: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    backgroundColor: colors.primary + '15',
    borderRadius: borderRadius.md,
    padding: spacing.md,
    marginTop: spacing.md,
    borderWidth: 1,
    borderColor: colors.primary + '30',
  },
  tipIcon: {
    fontSize: 20,
    marginRight: spacing.sm,
  },
  tipText: {
    flex: 1,
    fontSize: typography.fontSize.sm,
    color: colors.textMuted,
    lineHeight: typography.fontSize.sm * 1.5,
  },
  // Modal styles
  modalOverlay: {
    flex: 1,
    backgroundColor: colors.overlay,
    justifyContent: 'center',
    alignItems: 'center',
    padding: spacing.lg,
  },
  modalContainer: {
    width: '100%',
    maxWidth: 500,
    maxHeight: '80%',
  },
  modalContent: {
    backgroundColor: colors.surface,
    borderRadius: borderRadius.lg,
    borderWidth: 1,
    borderColor: colors.borderGold,
    overflow: 'hidden',
    ...shadows.xl,
  },
  modalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: spacing.lg,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
  },
  modalTitle: {
    fontSize: typography.fontSize.xl,
    fontWeight: typography.fontWeight.bold,
    fontFamily: typography.fontFamilyDisplay,
    color: colors.primary,
  },
  closeButton: {
    width: 32,
    height: 32,
    borderRadius: borderRadius.full,
    backgroundColor: colors.surfaceLight,
    justifyContent: 'center',
    alignItems: 'center',
  },
  closeButtonText: {
    fontSize: typography.fontSize.lg,
    color: colors.textMuted,
  },
  modalScroll: {
    maxHeight: 400,
  },
  modalScrollContent: {
    padding: spacing.lg,
    gap: spacing.md,
  },
  centerContent: {
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: spacing['3xl'],
    gap: spacing.md,
  },
  loadingText: {
    fontSize: typography.fontSize.base,
    color: colors.textMuted,
  },
  errorText: {
    fontSize: typography.fontSize.base,
    color: colors.error,
    textAlign: 'center',
  },
  emptyEmoji: {
    fontSize: 48,
  },
  emptyText: {
    fontSize: typography.fontSize.lg,
    fontWeight: typography.fontWeight.medium,
    color: colors.text,
  },
  emptySubtext: {
    fontSize: typography.fontSize.sm,
    color: colors.textMuted,
  },
  // Game card
  gameCard: {
    backgroundColor: colors.backgroundLight,
    borderRadius: borderRadius.md,
    padding: spacing.md,
    borderWidth: 1,
    borderColor: colors.border,
  },
  gameCardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: spacing.sm,
  },
  gameCode: {
    fontSize: typography.fontSize.lg,
    fontWeight: typography.fontWeight.bold,
    fontFamily: typography.fontFamilyMono,
    color: colors.primary,
    letterSpacing: 2,
  },
  badge: {
    paddingHorizontal: spacing.sm,
    paddingVertical: 2,
    borderRadius: borderRadius.sm,
  },
  badgeText: {
    fontSize: typography.fontSize.xs,
    fontWeight: typography.fontWeight.semibold,
    textTransform: 'uppercase',
  },
  gameCardInfo: {
    marginBottom: spacing.md,
  },
  gameInfoText: {
    fontSize: typography.fontSize.sm,
    color: colors.textMuted,
    marginBottom: 2,
  },
  gamePlayersText: {
    fontSize: typography.fontSize.sm,
    color: colors.text,
  },
  gameCardActions: {
    flexDirection: 'row',
    gap: spacing.sm,
  },
  // History card
  historyCard: {
    backgroundColor: colors.backgroundLight,
    borderRadius: borderRadius.md,
    padding: spacing.md,
    borderWidth: 1,
    borderColor: colors.border,
  },
  historyCardHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: spacing.md,
    paddingBottom: spacing.sm,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
  },
  historyWinner: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.bold,
    color: colors.primary,
  },
  historyDate: {
    fontSize: typography.fontSize.xs,
    color: colors.textMuted,
  },
  historyScores: {
    gap: spacing.xs,
  },
  historyScoreRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: spacing.sm,
  },
  historyRank: {
    width: 24,
    fontSize: typography.fontSize.sm,
  },
  historyPlayerName: {
    flex: 1,
    fontSize: typography.fontSize.sm,
    color: colors.text,
  },
  historyScore: {
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.medium,
    color: colors.textMuted,
  },
});
