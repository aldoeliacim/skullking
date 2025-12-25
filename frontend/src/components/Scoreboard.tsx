import React from 'react';
import { useTranslation } from 'react-i18next';
import { ScrollView, StyleSheet, Text, View, type ViewStyle } from 'react-native';
import Animated, {
  useAnimatedStyle,
  useSharedValue,
  withSpring,
  withSequence,
  withTiming,
} from 'react-native-reanimated';
import type { Player } from '../stores/gameStore';
import { borderRadius, colors, shadows, spacing, typography } from '../styles/theme';

interface ScoreboardProps {
  players: Player[];
  currentPlayerId: string | null;
  currentRound: number;
  style?: ViewStyle;
  compact?: boolean;
}

interface PlayerRowProps {
  player: Player;
  rank: number;
  isCurrentPlayer: boolean;
  isCurrentTurn: boolean;
  compact: boolean;
}

function PlayerRow({
  player,
  rank,
  isCurrentPlayer,
  isCurrentTurn,
  compact,
}: PlayerRowProps): React.JSX.Element {
  const { t } = useTranslation();
  const scale = useSharedValue(1);
  const prevScore = useSharedValue(player.score);

  // Animate on score change
  React.useEffect(() => {
    if (player.score !== prevScore.value) {
      const isPositive = player.score > prevScore.value;
      scale.value = withSequence(withSpring(1.1, { damping: 10 }), withSpring(1, { damping: 15 }));
      prevScore.value = player.score;
    }
  }, [player.score, prevScore, scale]);

  const animatedStyle = useAnimatedStyle(() => ({
    transform: [{ scale: scale.value }],
  }));

  const bidStatus = (): { text: string; color: string } => {
    if (player.bid === null) {
      return { text: '-', color: colors.textDark };
    }
    const diff = player.tricks_won - player.bid;
    if (diff === 0 && player.tricks_won > 0) {
      return { text: `${player.bid}`, color: colors.success };
    }
    if (diff === 0) {
      return { text: `${player.bid}`, color: colors.textMuted };
    }
    return { text: `${player.bid}`, color: diff > 0 ? colors.warning : colors.error };
  };

  const status = bidStatus();

  return (
    <Animated.View
      style={[
        styles.playerRow,
        isCurrentPlayer && styles.currentPlayerRow,
        isCurrentTurn && styles.currentTurnRow,
        compact && styles.playerRowCompact,
        animatedStyle,
      ]}
    >
      <View style={styles.rankContainer}>
        <Text style={styles.rank}>{rank}</Text>
      </View>

      <View style={styles.nameContainer}>
        <Text
          style={[styles.playerName, isCurrentPlayer && styles.currentPlayerName]}
          numberOfLines={1}
        >
          {player.username}
          {player.is_bot && ' 🤖'}
          {isCurrentPlayer && ` (${t('lobby.you')})`}
        </Text>
      </View>

      <View style={styles.statsContainer}>
        <View style={styles.stat}>
          <Text style={styles.statLabel}>{compact ? '' : t('game.bid')}</Text>
          <Text style={[styles.statValue, { color: status.color }]}>{status.text}</Text>
        </View>

        <View style={styles.stat}>
          <Text style={styles.statLabel}>{compact ? '' : t('game.tricks')}</Text>
          <Text style={styles.statValue}>{player.tricks_won}</Text>
        </View>

        <View style={[styles.stat, styles.scoreStat]}>
          <Text style={styles.statLabel}>{compact ? '' : t('game.score')}</Text>
          <Text style={[styles.scoreValue, player.score < 0 && styles.negativeScore]}>
            {player.score}
          </Text>
        </View>
      </View>
    </Animated.View>
  );
}

export function Scoreboard({
  players,
  currentPlayerId,
  currentRound,
  style,
  compact = false,
}: ScoreboardProps): React.JSX.Element {
  const { t } = useTranslation();

  // Sort players by score (descending)
  const sortedPlayers = [...players].sort((a, b) => b.score - a.score);

  return (
    <View style={[styles.container, compact && styles.containerCompact, style]}>
      {!compact && (
        <View style={styles.header}>
          <Text style={styles.title}>{t('scoreboard.title')}</Text>
          <Text style={styles.round}>
            {t('game.round')} {currentRound}
          </Text>
        </View>
      )}

      <ScrollView
        style={styles.scrollView}
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
      >
        {sortedPlayers.map((player, index) => (
          <PlayerRow
            key={player.id}
            player={player}
            rank={index + 1}
            isCurrentPlayer={player.id === currentPlayerId}
            isCurrentTurn={false}
            compact={compact}
          />
        ))}
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    backgroundColor: colors.surface,
    borderRadius: borderRadius.lg,
    padding: spacing.md,
    ...shadows.md,
  },
  containerCompact: {
    padding: spacing.sm,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: spacing.md,
    paddingBottom: spacing.sm,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
  },
  title: {
    fontSize: typography.fontSize.lg,
    fontWeight: typography.fontWeight.bold,
    color: colors.text,
  },
  round: {
    fontSize: typography.fontSize.sm,
    color: colors.textMuted,
  },
  scrollView: {
    maxHeight: 300,
  },
  scrollContent: {
    gap: spacing.xs,
  },
  playerRow: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.surfaceLight,
    borderRadius: borderRadius.base,
    padding: spacing.sm,
  },
  playerRowCompact: {
    padding: spacing.xs,
  },
  currentPlayerRow: {
    backgroundColor: colors.primary + '30',
    borderWidth: 1,
    borderColor: colors.primary,
  },
  currentTurnRow: {
    borderWidth: 2,
    borderColor: colors.accentGold,
  },
  rankContainer: {
    width: 24,
    alignItems: 'center',
  },
  rank: {
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.bold,
    color: colors.textMuted,
  },
  nameContainer: {
    flex: 1,
    marginLeft: spacing.sm,
  },
  playerName: {
    fontSize: typography.fontSize.base,
    color: colors.text,
  },
  currentPlayerName: {
    fontWeight: typography.fontWeight.semibold,
    color: colors.primary,
  },
  statsContainer: {
    flexDirection: 'row',
    gap: spacing.md,
  },
  stat: {
    alignItems: 'center',
    minWidth: 40,
  },
  scoreStat: {
    minWidth: 50,
  },
  statLabel: {
    fontSize: typography.fontSize.xs,
    color: colors.textDark,
    marginBottom: 2,
  },
  statValue: {
    fontSize: typography.fontSize.base,
    fontWeight: typography.fontWeight.medium,
    color: colors.text,
  },
  scoreValue: {
    fontSize: typography.fontSize.lg,
    fontWeight: typography.fontWeight.bold,
    color: colors.accentGold,
  },
  negativeScore: {
    color: colors.error,
  },
});

export default Scoreboard;
