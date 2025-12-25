import React, { useCallback, useMemo } from 'react';
import { useTranslation } from 'react-i18next';
import { StyleSheet, Text, View, type ViewStyle } from 'react-native';
import Animated, {
  useAnimatedStyle,
  useSharedValue,
  withSpring,
  withDelay,
} from 'react-native-reanimated';
import type { Card as CardType } from '../stores/gameStore';
import { cardDimensions, colors, screen, spacing, typography } from '../styles/theme';
import { Card } from './Card';

interface HandProps {
  cards: CardType[];
  onCardPress?: (card: CardType) => void;
  selectedCardId?: string | null;
  validCardIds?: string[];
  disabled?: boolean;
  showLabel?: boolean;
  style?: ViewStyle;
}

export function Hand({
  cards,
  onCardPress,
  selectedCardId,
  validCardIds,
  disabled = false,
  showLabel = true,
  style,
}: HandProps): React.JSX.Element {
  const { t } = useTranslation();

  // Calculate card overlap based on number of cards and screen width
  const cardLayout = useMemo(() => {
    const maxWidth = screen.width - spacing.base * 2;
    const cardWidth = cardDimensions.width;
    const minOverlap = -cardWidth * 0.6; // Maximum 60% overlap
    const maxOverlap = -spacing.sm; // Minimum overlap (just a small gap)

    if (cards.length <= 1) {
      return { overlap: 0, containerWidth: cardWidth };
    }

    // Calculate ideal overlap to fit all cards
    const totalWidthNeeded = cardWidth * cards.length;
    const availableOverlap = (totalWidthNeeded - maxWidth) / (cards.length - 1);
    const overlap = Math.max(minOverlap, Math.min(maxOverlap, -availableOverlap));

    // Calculate container width
    const containerWidth = cardWidth + (cards.length - 1) * (cardWidth + overlap);

    return { overlap, containerWidth: Math.min(containerWidth, maxWidth) };
  }, [cards.length]);

  const handleCardPress = useCallback(
    (card: CardType) => {
      if (disabled) {
        return;
      }
      // Check if card is valid to play
      if (validCardIds && !validCardIds.includes(card.id)) {
        return;
      }
      onCardPress?.(card);
    },
    [disabled, validCardIds, onCardPress],
  );

  const isCardValid = useCallback(
    (cardId: string): boolean => {
      if (!validCardIds) {
        return true;
      }
      return validCardIds.includes(cardId);
    },
    [validCardIds],
  );

  return (
    <View style={[styles.container, style]}>
      {showLabel && (
        <Text style={styles.label}>
          {t('game.yourHand')} ({cards.length}{' '}
          {cards.length === 1 ? t('game.card') : t('game.cards')})
        </Text>
      )}

      <View style={[styles.cardsContainer, { width: cardLayout.containerWidth }]}>
        {cards.map((card, index) => (
          <Animated.View
            key={card.id}
            style={[
              styles.cardWrapper,
              {
                marginLeft: index === 0 ? 0 : cardLayout.overlap,
                zIndex: index,
              },
            ]}
          >
            <Card
              card={card}
              size="medium"
              selected={selectedCardId === card.id}
              disabled={disabled || !isCardValid(card.id)}
              onPress={() => handleCardPress(card)}
              animationDelay={index * 50}
            />
          </Animated.View>
        ))}
      </View>

      {cards.length === 0 && (
        <View style={styles.emptyHand}>
          <Text style={styles.emptyText}>{t('game.waiting')}</Text>
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    alignItems: 'center',
    paddingVertical: spacing.sm,
  },
  label: {
    fontSize: typography.fontSize.sm,
    color: colors.textMuted,
    marginBottom: spacing.sm,
    textTransform: 'uppercase',
    letterSpacing: 1,
  },
  cardsContainer: {
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'flex-end',
    minHeight: cardDimensions.height + spacing.md,
  },
  cardWrapper: {
    // Shadow for stacking effect
    shadowColor: '#000',
    shadowOffset: { width: -2, height: 0 },
    shadowOpacity: 0.2,
    shadowRadius: 4,
  },
  emptyHand: {
    height: cardDimensions.height,
    justifyContent: 'center',
    alignItems: 'center',
  },
  emptyText: {
    fontSize: typography.fontSize.md,
    color: colors.textDark,
    fontStyle: 'italic',
  },
});

export default Hand;
