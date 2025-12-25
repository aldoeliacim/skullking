import React, { useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { StyleSheet, Text, View, type ViewStyle } from 'react-native';
import Animated, {
  useAnimatedStyle,
  useSharedValue,
  withSpring,
  withTiming,
  withDelay,
  withSequence,
  Easing,
} from 'react-native-reanimated';
import type { Card as CardType, Player, TrickCard } from '../stores/gameStore';
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

// Parse card ID to card object (same as in gameStore)
function parseCard(cardId: string): CardType {
  const card: CardType = { id: cardId };

  if (
    cardId.startsWith('blue_') ||
    cardId.startsWith('yellow_') ||
    cardId.startsWith('green_') ||
    cardId.startsWith('purple_')
  ) {
    const parts = cardId.split('_');
    card.suit = parts[0] ?? '';
    card.number = parseInt(parts[1] ?? '0', 10);
    card.type = 'standard';
  } else if (cardId.startsWith('black_')) {
    const parts = cardId.split('_');
    card.suit = 'black';
    card.number = parseInt(parts[1] ?? '0', 10);
    card.type = 'black';
  } else if (cardId.startsWith('escape')) {
    card.type = 'escape';
  } else if (cardId.startsWith('pirate')) {
    card.type = 'pirate';
  } else if (cardId === 'skull_king') {
    card.type = 'skull_king';
  } else if (cardId.startsWith('mermaid')) {
    card.type = 'mermaid';
  } else if (cardId === 'tigress') {
    card.type = 'tigress';
  } else if (cardId === 'kraken') {
    card.type = 'kraken';
  } else if (cardId === 'white_whale') {
    card.type = 'white_whale';
  } else if (cardId.startsWith('loot')) {
    card.type = 'loot';
  }

  return card;
}

interface TrickCardDisplayProps {
  trickCard: TrickCard;
  playerName: string;
  index: number;
  totalCards: number;
  isWinner: boolean;
}

function TrickCardDisplay({
  trickCard,
  playerName,
  index,
  totalCards,
  isWinner,
}: TrickCardDisplayProps): React.JSX.Element {
  const scale = useSharedValue(0);
  const rotation = useSharedValue(-15 + Math.random() * 30);
  const glowOpacity = useSharedValue(0);

  React.useEffect(() => {
    // Entry animation
    scale.value = withDelay(index * 100, withSpring(1, { damping: 12, stiffness: 180 }));
    rotation.value = withDelay(index * 100, withSpring(0, { damping: 15 }));
  }, [index, scale, rotation]);

  React.useEffect(() => {
    if (isWinner) {
      // Winner glow animation
      glowOpacity.value = withSequence(
        withTiming(1, { duration: 300 }),
        withTiming(0.6, { duration: 500 }),
        withTiming(1, { duration: 500 }),
      );
    }
  }, [isWinner, glowOpacity]);

  const animatedStyle = useAnimatedStyle(() => ({
    transform: [{ scale: scale.value }, { rotate: `${rotation.value}deg` }],
  }));

  const glowStyle = useAnimatedStyle(() => ({
    opacity: glowOpacity.value,
  }));

  const card = parseCard(trickCard.card_id);

  // Calculate position in circular layout
  const angle = (index / Math.max(totalCards, 4)) * Math.PI * 2 - Math.PI / 2;
  const radius = 50;
  const x = Math.cos(angle) * radius;
  const y = Math.sin(angle) * radius;

  return (
    <Animated.View
      style={[
        styles.trickCardWrapper,
        {
          transform: [{ translateX: x }, { translateY: y }],
          zIndex: index,
        },
        animatedStyle,
      ]}
    >
      {isWinner && <Animated.View style={[styles.winnerGlow, glowStyle]} />}
      <Card card={card} size="small" showGlow={isWinner} />
      <Text style={[styles.playerName, isWinner && styles.winnerName]} numberOfLines={1}>
        {playerName}
      </Text>
      {trickCard.tigress_choice && (
        <Text style={styles.tigressChoice}>
          ({trickCard.tigress_choice === 'pirate' ? '🏴‍☠️' : '🏃'})
        </Text>
      )}
    </Animated.View>
  );
}

export function TrickArea({
  trickCards,
  players,
  currentPlayerId,
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
              key={tc.card_id}
              trickCard={tc}
              playerName={getPlayerName(tc.player_id)}
              index={index}
              totalCards={trickCards.length}
              isWinner={tc.player_id === winnerId}
            />
          ))
        )}
      </View>

      {winnerId && winnerName && (
        <Animated.View style={styles.winnerBanner}>
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
    width: 200,
    height: 180,
    justifyContent: 'center',
    alignItems: 'center',
    position: 'relative',
  },
  emptyText: {
    fontSize: typography.fontSize.md,
    color: colors.textDark,
    fontStyle: 'italic',
  },
  trickCardWrapper: {
    position: 'absolute',
    alignItems: 'center',
  },
  playerName: {
    fontSize: typography.fontSize.xs,
    color: colors.textMuted,
    marginTop: spacing.xs,
    maxWidth: 60,
    textAlign: 'center',
  },
  winnerName: {
    color: colors.accentGold,
    fontWeight: typography.fontWeight.bold,
  },
  tigressChoice: {
    fontSize: typography.fontSize.xs,
    marginTop: 2,
  },
  winnerGlow: {
    position: 'absolute',
    top: -8,
    left: -8,
    right: -8,
    bottom: -8,
    borderRadius: borderRadius.lg,
    backgroundColor: colors.accentGold,
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
