import React from 'react';
import { useTranslation } from 'react-i18next';
import { StyleSheet, Text, View, type ViewStyle } from 'react-native';
import Animated, {
  useAnimatedStyle,
  useSharedValue,
  withSpring,
  withDelay,
  FadeIn,
} from 'react-native-reanimated';
import { type Player, type TrickCard, parseCard } from '../stores/gameStore';
import { borderRadius, colors, shadows, spacing, typography } from '../styles/theme';
import { Card } from './Card';

interface TrickAreaProps {
  trickCards: TrickCard[];
  players: Player[];
  currentPlayerId: string | null;
  winnerId?: string | null;
  winnerName?: string | null;
  style?: ViewStyle;
}

interface TrickCardDisplayProps {
  trickCard: TrickCard;
  playerName: string;
  index: number;
  isWinner: boolean;
}

function TrickCardDisplay({
  trickCard,
  playerName,
  index,
  isWinner,
}: TrickCardDisplayProps): React.JSX.Element {
  const scale = useSharedValue(0);

  React.useEffect(() => {
    // Entry animation with stagger
    scale.value = withDelay(index * 80, withSpring(1, { damping: 12, stiffness: 180 }));
  }, [index, scale]);

  const animatedStyle = useAnimatedStyle(() => ({
    transform: [{ scale: scale.value }],
    opacity: scale.value,
  }));

  const card = parseCard(trickCard.card_id);

  return (
    <Animated.View style={[styles.trickCardWrapper, animatedStyle]}>
      <Card card={card} size="small" showGlow={isWinner} />
      <Text style={[styles.playerName, isWinner && styles.winnerName]} numberOfLines={1}>
        {playerName}
      </Text>
      {trickCard.tigress_choice && (
        <View style={styles.tigressBadge}>
          <Text style={styles.tigressChoice}>
            {trickCard.tigress_choice === 'pirate' ? '🏴‍☠️' : '👻'}
          </Text>
        </View>
      )}
    </Animated.View>
  );
}

export function TrickArea({
  trickCards,
  players,
  currentPlayerId: _currentPlayerId,
  winnerId,
  winnerName,
  style,
}: TrickAreaProps): React.JSX.Element {
  const { t } = useTranslation();

  const getPlayerName = (playerId: string): string => {
    const player = players.find((p) => p.id === playerId);
    return player?.username || 'Unknown';
  };

  return (
    <View style={[styles.container, style]}>
      <Text style={styles.title}>{t('game.currentTrick')}</Text>

      <View style={styles.trickArea}>
        {trickCards.length === 0 ? (
          <Text style={styles.emptyText}>{t('game.waiting')}</Text>
        ) : (
          trickCards.map((tc, index) => (
            <TrickCardDisplay
              key={`${tc.card_id}-${tc.player_id}`}
              trickCard={tc}
              playerName={getPlayerName(tc.player_id)}
              index={index}
              isWinner={tc.player_id === winnerId}
            />
          ))
        )}
      </View>

      {winnerId && winnerName && (
        <Animated.View entering={FadeIn.duration(300)} style={styles.winnerBanner}>
          <Text style={styles.winnerText}>
            {winnerName} {t('game.wonTrick')}
          </Text>
        </Animated.View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    alignItems: 'center',
    paddingVertical: spacing.md,
  },
  title: {
    fontSize: typography.fontSize.sm,
    color: colors.textMuted,
    marginBottom: spacing.sm,
    textTransform: 'uppercase',
    letterSpacing: 1,
  },
  trickArea: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'center',
    alignItems: 'flex-start',
    gap: spacing.sm,
    minHeight: 120,
    paddingHorizontal: spacing.md,
  },
  emptyText: {
    fontSize: typography.fontSize.md,
    color: colors.textDark,
    fontStyle: 'italic',
  },
  trickCardWrapper: {
    alignItems: 'center',
  },
  playerName: {
    fontSize: typography.fontSize.xs,
    color: colors.textMuted,
    marginTop: spacing.xs,
    maxWidth: 70,
    textAlign: 'center',
  },
  winnerName: {
    color: colors.accentGold,
    fontWeight: typography.fontWeight.bold,
  },
  tigressBadge: {
    position: 'absolute',
    top: -4,
    right: -4,
    backgroundColor: colors.surface,
    borderRadius: 10,
    width: 20,
    height: 20,
    justifyContent: 'center',
    alignItems: 'center',
    ...shadows.sm,
  },
  tigressChoice: {
    fontSize: 12,
  },
  winnerBanner: {
    marginTop: spacing.md,
    paddingHorizontal: spacing.lg,
    paddingVertical: spacing.sm,
    backgroundColor: colors.accentGold,
    borderRadius: borderRadius.base,
    ...shadows.md,
  },
  winnerText: {
    fontSize: typography.fontSize.md,
    fontWeight: typography.fontWeight.bold,
    color: colors.backgroundDark,
    textAlign: 'center',
  },
});

export default TrickArea;
