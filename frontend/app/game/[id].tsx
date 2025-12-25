import { useLocalSearchParams, useRouter } from 'expo-router';
import React, { useEffect, useCallback, useState, useRef } from 'react';
import { useTranslation } from 'react-i18next';
import {
  ScrollView,
  type ScrollView as ScrollViewType,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import Animated, { FadeIn, FadeInUp, FadeInDown } from 'react-native-reanimated';
import { SafeAreaView } from 'react-native-safe-area-context';
import {
  AbilityModal,
  BiddingModal,
  Button,
  Hand,
  Scoreboard,
  SettingsButton,
  TigressModal,
  TrickArea,
} from '../../src/components';
import { type Card as CardType, useGameStore } from '../../src/stores/gameStore';
import { borderRadius, colors, shadows, spacing, typography } from '../../src/styles/theme';

export default function GameScreen(): React.JSX.Element {
  const { t } = useTranslation();
  const router = useRouter();
  const params = useLocalSearchParams<{
    id: string;
    spectator?: string;
    playerId?: string;
    playerName?: string;
  }>();

  const [selectedCard, setSelectedCard] = useState<CardType | null>(null);
  const [showTigressModal, setShowTigressModal] = useState(false);
  const logScrollRef = useRef<ScrollViewType>(null);

  const {
    connectionState,
    phase,
    players,
    playerId,
    currentRound,
    currentTrick: _currentTrick,
    hand,
    trickCards,
    pickingPlayerId,
    showBidding,
    showAbility,
    abilityData,
    trickWinner,
    isSpectator,
    connect,
    disconnect,
    placeBid,
    playCard,
    resolveAbility,
    clearTrickWinner,
    logs,
  } = useGameStore();

  const gameCode = params.id || '';
  const isMyTurn = pickingPlayerId === playerId && !isSpectator;
  const currentPlayer = players.find((p) => p.id === playerId);

  // Connect to game on mount (skip if already connected from lobby)
  useEffect(() => {
    const isSpectatorMode = params.spectator === 'true';

    // If we have a playerId from lobby and we're already connected, don't reconnect
    if (params.playerId && connectionState === 'connected') {
      return;
    }

    // Only connect if not already connected
    if (connectionState === 'disconnected') {
      const playerIdToUse =
        params.playerId || (isSpectatorMode ? `spectator_${Date.now()}` : `player_${Date.now()}`);
      const playerNameToUse = params.playerName || 'Player';
      connect(gameCode, playerIdToUse, playerNameToUse, isSpectatorMode);
    }

    return () => {
      disconnect();
    };
  }, [
    gameCode,
    params.spectator,
    params.playerId,
    params.playerName,
    connectionState,
    connect,
    disconnect,
  ]);

  // Navigate away when game ends
  useEffect(() => {
    if (phase === 'ENDED') {
      // Stay on game screen to show results
    }
  }, [phase]);

  // Clear trick winner after delay
  useEffect(() => {
    if (trickWinner) {
      const timer = setTimeout(clearTrickWinner, 2500);
      return () => clearTimeout(timer);
    }
    return undefined;
  }, [trickWinner, clearTrickWinner]);

  // Auto-scroll log to end when new logs arrive
  useEffect(() => {
    if (logs.length > 0 && logScrollRef.current) {
      logScrollRef.current.scrollToEnd({ animated: true });
    }
  }, [logs]);

  const handleCardPress = useCallback(
    (card: CardType) => {
      if (!isMyTurn) {
        return;
      }

      // Check for tigress card
      if (card.type === 'tigress' || card.id === 'tigress') {
        setSelectedCard(card);
        setShowTigressModal(true);
        return;
      }

      // Play the card directly
      playCard(card.id);
    },
    [isMyTurn, playCard],
  );

  const handleTigressChoice = useCallback(
    (choice: 'pirate' | 'escape') => {
      if (selectedCard) {
        playCard(selectedCard.id, choice);
        setShowTigressModal(false);
        setSelectedCard(null);
      }
    },
    [selectedCard, playCard],
  );

  const handleBid = useCallback(
    (bid: number) => {
      placeBid(bid);
    },
    [placeBid],
  );

  const handlePlayAgain = useCallback(() => {
    router.replace('/');
  }, [router]);

  // Render game over screen
  if (phase === 'ENDED') {
    const sortedPlayers = [...players].sort((a, b) => b.score - a.score);
    const winner = sortedPlayers[0];
    const isWinner = winner?.id === playerId;

    return (
      <SafeAreaView style={styles.container}>
        <SettingsButton style={styles.settingsButton} />
        <Animated.View entering={FadeIn.duration(500)} style={styles.gameOver}>
          <Text style={styles.gameOverEmoji}>{isWinner ? '🏆' : '🏴‍☠️'}</Text>
          <Text style={styles.gameOverTitle}>{t('results.title')}</Text>
          <Text style={styles.gameOverSubtitle}>
            {winner?.username} {t('results.wins')}
          </Text>

          <View style={styles.finalScores}>
            <Text style={styles.finalScoresTitle}>{t('results.finalScores')}</Text>
            {sortedPlayers.map((player, index) => (
              <View
                key={player.id}
                style={[
                  styles.finalScoreRow,
                  player.id === playerId && styles.finalScoreRowHighlight,
                ]}
              >
                <Text style={styles.finalScoreRank}>#{index + 1}</Text>
                <Text style={styles.finalScoreName}>{player.username}</Text>
                <Text style={styles.finalScoreValue}>{player.score}</Text>
              </View>
            ))}
          </View>

          <Button title={t('results.playAgain')} onPress={handlePlayAgain} size="lg" fullWidth />
        </Animated.View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container} edges={['top', 'left', 'right']}>
      {/* Header */}
      <Animated.View entering={FadeInUp.duration(300)} style={styles.header}>
        <View style={styles.headerLeft}>
          <Text style={styles.roundText}>
            {t('game.round')} {currentRound}/10
          </Text>
          {isSpectator && <Text style={styles.spectatorBadge}>👁️ Spectator</Text>}
        </View>

        <View style={styles.headerCenter}>
          <SettingsButton />
        </View>

        <View style={styles.headerRight}>
          {currentPlayer && (
            <View style={styles.playerStats}>
              <Text style={styles.scoreText}>
                {t('game.score')}: {currentPlayer.score}
              </Text>
              <Text style={styles.bidText}>
                {t('game.bid')}: {currentPlayer.bid ?? '-'} | {t('game.tricks')}:{' '}
                {currentPlayer.tricks_won}
              </Text>
            </View>
          )}
        </View>
      </Animated.View>

      {/* Main game area */}
      <ScrollView
        style={styles.mainArea}
        contentContainerStyle={styles.mainAreaContent}
        showsVerticalScrollIndicator={false}
      >
        {/* Trick Area */}
        <Animated.View entering={FadeIn.delay(100).duration(400)}>
          <TrickArea
            trickCards={trickCards}
            players={players}
            currentPlayerId={pickingPlayerId}
            winnerId={trickWinner?.playerId ?? null}
            winnerName={trickWinner?.playerName ?? null}
          />
        </Animated.View>

        {/* Turn indicator */}
        {isMyTurn && (
          <Animated.View entering={FadeIn.duration(300)} style={styles.turnIndicator}>
            <Text style={styles.turnText}>{t('game.yourTurn')}</Text>
          </Animated.View>
        )}

        {/* Scoreboard (compact) */}
        <Animated.View entering={FadeIn.delay(200).duration(400)} style={styles.scoreboardSection}>
          <Scoreboard
            players={players}
            currentPlayerId={playerId}
            currentRound={currentRound}
            compact
          />
        </Animated.View>
      </ScrollView>

      {/* Hand */}
      {!isSpectator && (
        <Animated.View entering={FadeInDown.delay(300).duration(400)} style={styles.handSection}>
          <Hand
            cards={hand}
            onCardPress={handleCardPress}
            trickCards={isMyTurn ? trickCards : []}
            disabled={!isMyTurn}
          />
        </Animated.View>
      )}

      {/* Game Log (collapsed by default) */}
      <View style={styles.logSection}>
        <ScrollView
          ref={logScrollRef}
          horizontal
          showsHorizontalScrollIndicator={false}
          contentContainerStyle={styles.logContent}
        >
          {logs.slice(-5).map((log, index) => (
            <Text key={log.timestamp + index} style={styles.logItem}>
              {log.message}
            </Text>
          ))}
        </ScrollView>
      </View>

      {/* Modals */}
      <BiddingModal visible={showBidding} maxBid={currentRound} hand={hand} onBid={handleBid} />

      <TigressModal visible={showTigressModal} onChoice={handleTigressChoice} />

      <AbilityModal
        visible={showAbility}
        abilityData={abilityData}
        players={players}
        onResolve={resolveAbility}
      />
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
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    padding: spacing.md,
    backgroundColor: colors.surface,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
  },
  headerLeft: {
    flex: 1,
    gap: spacing.xs,
  },
  headerCenter: {
    paddingHorizontal: spacing.sm,
  },
  roundText: {
    fontSize: typography.fontSize.xl,
    fontWeight: typography.fontWeight.bold,
    fontFamily: typography.fontFamilyDisplay,
    color: colors.accentGold,
    textShadowColor: 'rgba(0, 0, 0, 0.4)',
    textShadowOffset: { width: 1, height: 1 },
    textShadowRadius: 2,
  },
  spectatorBadge: {
    fontSize: typography.fontSize.xs,
    color: colors.warning,
  },
  headerRight: {
    flex: 1,
    alignItems: 'flex-end',
  },
  playerStats: {
    alignItems: 'flex-end',
  },
  scoreText: {
    fontSize: typography.fontSize.lg,
    fontWeight: typography.fontWeight.bold,
    color: colors.accentGold,
  },
  bidText: {
    fontSize: typography.fontSize.sm,
    color: colors.textMuted,
  },
  mainArea: {
    flex: 1,
  },
  mainAreaContent: {
    padding: spacing.md,
    gap: spacing.md,
  },
  turnIndicator: {
    alignSelf: 'center',
    backgroundColor: colors.primary,
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.sm,
    borderRadius: borderRadius.full,
    ...shadows.glow(colors.primary),
  },
  turnText: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.bold,
    color: colors.text,
  },
  scoreboardSection: {
    marginTop: spacing.md,
  },
  handSection: {
    backgroundColor: colors.surface,
    borderTopWidth: 1,
    borderTopColor: colors.border,
    paddingVertical: spacing.sm,
    paddingBottom: spacing.lg,
  },
  logSection: {
    backgroundColor: colors.backgroundDark,
    paddingVertical: spacing.xs,
    paddingHorizontal: spacing.md,
  },
  logContent: {
    gap: spacing.md,
    alignItems: 'center',
  },
  logItem: {
    fontSize: typography.fontSize.xs,
    color: colors.textDark,
    paddingHorizontal: spacing.sm,
    paddingVertical: spacing.xs,
    backgroundColor: colors.surface,
    borderRadius: borderRadius.sm,
  },

  // Game Over styles
  gameOver: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: spacing.xl,
  },
  gameOverEmoji: {
    fontSize: 80,
    marginBottom: spacing.lg,
  },
  gameOverTitle: {
    fontSize: typography.fontSize['5xl'],
    fontWeight: typography.fontWeight.bold,
    fontFamily: typography.fontFamilyDisplay,
    color: colors.accentGold,
    marginBottom: spacing.sm,
    textShadowColor: 'rgba(0, 0, 0, 0.5)',
    textShadowOffset: { width: 2, height: 2 },
    textShadowRadius: 4,
  },
  gameOverSubtitle: {
    fontSize: typography.fontSize.xl,
    fontFamily: typography.fontFamily,
    color: colors.text,
    marginBottom: spacing['3xl'],
  },
  finalScores: {
    width: '100%',
    maxWidth: 400,
    backgroundColor: colors.surface,
    borderRadius: borderRadius.lg,
    padding: spacing.lg,
    marginBottom: spacing.xl,
    ...shadows.lg,
  },
  finalScoresTitle: {
    fontSize: typography.fontSize.lg,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text,
    textAlign: 'center',
    marginBottom: spacing.md,
  },
  finalScoreRow: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: spacing.sm,
    paddingHorizontal: spacing.md,
    borderRadius: borderRadius.base,
  },
  finalScoreRowHighlight: {
    backgroundColor: colors.primary + '30',
  },
  finalScoreRank: {
    width: 30,
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.bold,
    color: colors.textMuted,
  },
  finalScoreName: {
    flex: 1,
    fontSize: typography.fontSize.base,
    color: colors.text,
  },
  finalScoreValue: {
    fontSize: typography.fontSize.lg,
    fontWeight: typography.fontWeight.bold,
    color: colors.accentGold,
  },
});
