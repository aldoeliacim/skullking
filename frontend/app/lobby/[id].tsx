import * as Clipboard from 'expo-clipboard';
import { useLocalSearchParams, useRouter } from 'expo-router';
import React, { useEffect, useCallback, useState, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import { Alert, Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import Animated, { FadeInUp, FadeInDown, FadeIn } from 'react-native-reanimated';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Button, SettingsButton } from '../../src/components';
import { useGameStore } from '../../src/stores/gameStore';
import { borderRadius, colors, shadows, spacing, typography } from '../../src/styles/theme';

// Bot presets - simplified from separate type + difficulty
// Only rule-based bots actually use difficulty; RL and Random ignore it
type BotPreset = 'ai' | 'rule_hard' | 'rule_medium' | 'rule_easy' | 'random';

const BOT_PRESET_CONFIG: Record<BotPreset, { type: string; difficulty: string }> = {
  ai: { type: 'rl', difficulty: 'hard' },
  rule_hard: { type: 'rule_based', difficulty: 'hard' },
  rule_medium: { type: 'rule_based', difficulty: 'medium' },
  rule_easy: { type: 'rule_based', difficulty: 'easy' },
  random: { type: 'random', difficulty: 'easy' },
};

export default function LobbyScreen(): React.JSX.Element {
  const { t } = useTranslation();
  const router = useRouter();
  const params = useLocalSearchParams<{
    id: string;
    slug?: string;
    playerId?: string;
    playerName?: string;
  }>();

  const [copied, setCopied] = useState(false);
  const [selectedPreset, setSelectedPreset] = useState<BotPreset>('ai');
  const navigatingToGame = useRef(false);

  const {
    connectionState,
    players,
    playerId,
    phase,
    connect,
    disconnect,
    addBot,
    removeBot,
    startGame,
  } = useGameStore();

  const gameId = params.id || '';
  const gameCode = params.slug || gameId.substring(0, 8).toUpperCase();
  // Check if current player is the host (first non-bot player, or matches our playerId)
  const firstHumanPlayer = players.find((p) => !p.is_bot);
  const isHost = firstHumanPlayer?.id === playerId || players[0]?.id === playerId;
  const canStart = players.length >= 2 && isHost;
  const isConnected = connectionState === 'connected';

  // Generate stable player ID once
  const stablePlayerId = useRef(`player_${Date.now()}`);

  // Connect to game on mount
  useEffect(() => {
    const playerNameToUse = params.playerName || 'Player';
    connect(gameId, stablePlayerId.current, playerNameToUse);

    return () => {
      // Only disconnect if NOT navigating to game screen
      if (!navigatingToGame.current) {
        disconnect();
      }
    };
  }, [gameId, params.playerName, connect, disconnect]);

  // Navigate to game when it starts (don't disconnect - keep WebSocket alive)
  useEffect(() => {
    if (phase === 'BIDDING' || phase === 'PICKING') {
      // Mark that we're navigating to game, so cleanup doesn't disconnect
      navigatingToGame.current = true;
      router.replace({
        pathname: '/game/[id]',
        params: {
          id: gameId,
          playerId: stablePlayerId.current,
          playerName: params.playerName || 'Player',
        },
      });
    }
  }, [phase, gameId, router, params.playerName]);

  const handleCopyCode = useCallback(async () => {
    await Clipboard.setStringAsync(gameCode);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }, [gameCode]);

  const handleAddBot = useCallback(() => {
    const config = BOT_PRESET_CONFIG[selectedPreset];
    addBot(config.type, config.difficulty);
  }, [addBot, selectedPreset]);

  const handleFillWithBots = useCallback(() => {
    const botsNeeded = Math.max(0, 4 - players.length);
    const config = BOT_PRESET_CONFIG.ai; // Fill with AI bots
    for (let i = 0; i < botsNeeded; i++) {
      setTimeout(() => addBot(config.type, config.difficulty), i * 100);
    }
  }, [addBot, players.length]);

  const handleClearBots = useCallback(() => {
    players.filter((p) => p.is_bot).forEach((bot) => removeBot(bot.id));
  }, [players, removeBot]);

  const handleLeave = useCallback(() => {
    Alert.alert(t('lobby.leave'), 'Are you sure you want to leave?', [
      { text: 'Cancel', style: 'cancel' },
      {
        text: t('lobby.leave'),
        style: 'destructive',
        onPress: () => {
          disconnect();
          router.replace('/');
        },
      },
    ]);
  }, [disconnect, router, t]);

  const handleStartGame = useCallback(() => {
    if (canStart) {
      startGame();
    }
  }, [canStart, startGame]);

  // Simplified bot presets - combines type and difficulty into meaningful options
  const botPresets: Array<{ value: BotPreset; label: string; description: string }> = [
    { value: 'ai', label: t('lobby.botAI'), description: t('lobby.botAIDesc') },
    { value: 'rule_hard', label: t('lobby.botHard'), description: t('lobby.botHardDesc') },
    { value: 'rule_medium', label: t('lobby.botMedium'), description: t('lobby.botMediumDesc') },
    { value: 'rule_easy', label: t('lobby.botEasy'), description: t('lobby.botEasyDesc') },
    { value: 'random', label: t('lobby.botRandom'), description: t('lobby.botRandomDesc') },
  ];

  return (
    <SafeAreaView style={styles.container} edges={['top', 'left', 'right']}>
      {/* Settings button */}
      <SettingsButton style={styles.settingsButton} />

      <ScrollView contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false}>
        {/* Header */}
        <Animated.View entering={FadeInUp.duration(400)} style={styles.header}>
          <Text style={styles.title}>{t('lobby.title')}</Text>

          {/* Connection status */}
          <View style={styles.statusBadge}>
            <View
              style={[
                styles.statusDot,
                connectionState === 'connected' && styles.statusDotConnected,
                connectionState === 'connecting' && styles.statusDotConnecting,
                connectionState === 'reconnecting' && styles.statusDotReconnecting,
              ]}
            />
            <Text style={styles.statusText}>{connectionState}</Text>
          </View>
        </Animated.View>

        {/* Game Code */}
        <Animated.View entering={FadeInDown.delay(100).duration(400)} style={styles.codeSection}>
          <Text style={styles.codeLabel}>{t('lobby.shareCode')}</Text>
          <Pressable onPress={handleCopyCode} style={styles.codeContainer}>
            <Text style={styles.codeText}>{gameCode}</Text>
            <Text style={styles.copyButton}>{copied ? t('lobby.copied') : t('lobby.copy')}</Text>
          </Pressable>
        </Animated.View>

        {/* Players List */}
        <Animated.View entering={FadeInDown.delay(200).duration(400)} style={styles.section}>
          <Text style={styles.sectionTitle}>
            {t('lobby.players')} ({players.length}/8)
          </Text>

          <View style={styles.playersList}>
            {players.map((player, index) => (
              <Animated.View
                key={player.id}
                entering={FadeIn.delay(index * 50).duration(300)}
                style={styles.playerItem}
              >
                <View style={styles.playerInfo}>
                  <Text style={styles.playerEmoji}>{player.is_bot ? '🤖' : '👤'}</Text>
                  <View>
                    <Text style={styles.playerName}>
                      {player.username}
                      {player.id === playerId && ` (${t('lobby.you')})`}
                      {index === 0 && !player.is_bot && ` - ${t('lobby.host')}`}
                    </Text>
                    {player.is_bot && (
                      <Text style={styles.playerSubtitle}>
                        {player.bot_type} • {player.bot_difficulty}
                      </Text>
                    )}
                  </View>
                </View>

                {player.is_bot && isHost && (
                  <Pressable onPress={() => removeBot(player.id)} style={styles.removeButton}>
                    <Text style={styles.removeButtonText}>✕</Text>
                  </Pressable>
                )}
              </Animated.View>
            ))}

            {players.length === 0 && <Text style={styles.emptyText}>{t('game.waiting')}</Text>}
          </View>
        </Animated.View>

        {/* Bot Settings (Host only) */}
        {isHost && isConnected && (
          <Animated.View entering={FadeInDown.delay(300).duration(400)} style={styles.section}>
            <Text style={styles.sectionTitle}>{t('lobby.addOpponents')}</Text>

            <View style={styles.settingRow}>
              <Text style={styles.settingLabel}>{t('lobby.selectOpponent')}</Text>
              <View style={styles.presetButtons}>
                {botPresets.map((preset) => (
                  <Pressable
                    key={preset.value}
                    onPress={() => setSelectedPreset(preset.value)}
                    style={[
                      styles.presetButton,
                      selectedPreset === preset.value && styles.presetButtonSelected,
                    ]}
                  >
                    <Text
                      style={[
                        styles.presetButtonLabel,
                        selectedPreset === preset.value && styles.presetButtonLabelSelected,
                      ]}
                    >
                      {preset.label}
                    </Text>
                    <Text
                      style={[
                        styles.presetButtonDesc,
                        selectedPreset === preset.value && styles.presetButtonDescSelected,
                      ]}
                    >
                      {preset.description}
                    </Text>
                  </Pressable>
                ))}
              </View>
            </View>

            <Button
              title={t('lobby.addBot')}
              onPress={handleAddBot}
              variant="secondary"
              fullWidth
              disabled={players.length >= 8}
            />

            <View style={styles.quickActions}>
              <Button
                title={t('lobby.fillWithBots')}
                onPress={handleFillWithBots}
                variant="outline"
                size="sm"
              />
              <Button
                title={t('lobby.clearBots')}
                onPress={handleClearBots}
                variant="ghost"
                size="sm"
              />
            </View>
          </Animated.View>
        )}

        {/* Actions */}
        <Animated.View entering={FadeInDown.delay(400).duration(400)} style={styles.actions}>
          {isHost ? (
            <Button
              title={t('lobby.startGame')}
              onPress={handleStartGame}
              disabled={!canStart}
              fullWidth
              size="lg"
            />
          ) : (
            <Text style={styles.waitingText}>{t('lobby.waitingForHost')}</Text>
          )}

          {!canStart && isHost && <Text style={styles.hintText}>{t('lobby.needPlayers')}</Text>}

          <Button
            title={t('lobby.leave')}
            onPress={handleLeave}
            variant="ghost"
            style={styles.leaveButton}
          />
        </Animated.View>
      </ScrollView>
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
  scrollContent: {
    padding: spacing.lg,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: spacing.xl,
  },
  title: {
    fontSize: typography.fontSize['2xl'],
    fontWeight: typography.fontWeight.bold,
    fontFamily: typography.fontFamilyDisplay,
    color: colors.accentGold,
    textShadowColor: 'rgba(0, 0, 0, 0.5)',
    textShadowOffset: { width: 1, height: 1 },
    textShadowRadius: 2,
  },
  statusBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.surface,
    paddingHorizontal: spacing.sm,
    paddingVertical: spacing.xs,
    borderRadius: borderRadius.full,
  },
  statusDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    backgroundColor: colors.error,
    marginRight: spacing.xs,
  },
  statusDotConnected: {
    backgroundColor: colors.success,
  },
  statusDotConnecting: {
    backgroundColor: colors.warning,
  },
  statusDotReconnecting: {
    backgroundColor: colors.warning,
  },
  statusText: {
    fontSize: typography.fontSize.xs,
    color: colors.textMuted,
    textTransform: 'capitalize',
  },
  codeSection: {
    alignItems: 'center',
    marginBottom: spacing.xl,
  },
  codeLabel: {
    fontSize: typography.fontSize.sm,
    color: colors.textMuted,
    marginBottom: spacing.sm,
  },
  codeContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.surface,
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.md,
    borderRadius: borderRadius.lg,
    ...shadows.md,
  },
  codeText: {
    fontSize: typography.fontSize['3xl'],
    fontWeight: typography.fontWeight.bold,
    fontFamily: typography.fontFamilyDisplay,
    color: colors.accentGold,
    letterSpacing: 6,
    marginRight: spacing.md,
    textShadowColor: 'rgba(0, 0, 0, 0.4)',
    textShadowOffset: { width: 1, height: 1 },
    textShadowRadius: 2,
  },
  copyButton: {
    fontSize: typography.fontSize.sm,
    color: colors.primary,
    fontWeight: typography.fontWeight.medium,
  },
  section: {
    marginBottom: spacing.xl,
  },
  sectionTitle: {
    fontSize: typography.fontSize.lg,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text,
    marginBottom: spacing.md,
  },
  playersList: {
    backgroundColor: colors.surface,
    borderRadius: borderRadius.lg,
    padding: spacing.sm,
    ...shadows.sm,
  },
  playerItem: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: spacing.md,
    borderRadius: borderRadius.base,
  },
  playerInfo: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
  },
  playerEmoji: {
    fontSize: 24,
    marginRight: spacing.md,
  },
  playerName: {
    fontSize: typography.fontSize.base,
    fontWeight: typography.fontWeight.medium,
    color: colors.text,
  },
  playerSubtitle: {
    fontSize: typography.fontSize.xs,
    color: colors.textMuted,
    marginTop: 2,
  },
  removeButton: {
    width: 28,
    height: 28,
    borderRadius: 14,
    backgroundColor: colors.error + '30',
    justifyContent: 'center',
    alignItems: 'center',
  },
  removeButtonText: {
    color: colors.error,
    fontWeight: typography.fontWeight.bold,
  },
  emptyText: {
    textAlign: 'center',
    color: colors.textMuted,
    padding: spacing.xl,
    fontStyle: 'italic',
  },
  settingRow: {
    marginBottom: spacing.md,
  },
  settingLabel: {
    fontSize: typography.fontSize.sm,
    color: colors.textMuted,
    marginBottom: spacing.xs,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
  presetButtons: {
    gap: spacing.sm,
  },
  presetButton: {
    paddingHorizontal: spacing.md,
    paddingVertical: spacing.sm,
    borderRadius: borderRadius.base,
    backgroundColor: colors.surface,
    borderWidth: 1,
    borderColor: colors.border,
  },
  presetButtonSelected: {
    backgroundColor: colors.primary,
    borderColor: colors.accentGold,
    borderWidth: 2,
  },
  presetButtonLabel: {
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.medium,
    color: colors.text,
  },
  presetButtonLabelSelected: {
    color: colors.text,
    fontWeight: typography.fontWeight.bold,
  },
  presetButtonDesc: {
    fontSize: typography.fontSize.xs,
    color: colors.textMuted,
    marginTop: 2,
  },
  presetButtonDescSelected: {
    color: colors.textMuted,
  },
  quickActions: {
    flexDirection: 'row',
    justifyContent: 'center',
    gap: spacing.md,
    marginTop: spacing.md,
  },
  actions: {
    marginTop: spacing.lg,
    gap: spacing.md,
  },
  waitingText: {
    fontSize: typography.fontSize.lg,
    color: colors.textMuted,
    textAlign: 'center',
    fontStyle: 'italic',
  },
  hintText: {
    fontSize: typography.fontSize.sm,
    color: colors.textMuted,
    textAlign: 'center',
  },
  leaveButton: {
    marginTop: spacing.md,
  },
});
