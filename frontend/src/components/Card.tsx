import React, { useCallback } from 'react';
import {
  Image,
  type ImageSourcePropType,
  Pressable,
  StyleSheet,
  Text,
  View,
  type ViewStyle,
} from 'react-native';
import Animated, {
  useAnimatedStyle,
  useSharedValue,
  withSpring,
  withTiming,
  interpolate,
  runOnJS,
} from 'react-native-reanimated';
import type { Card as CardType } from '../stores/gameStore';
import { borderRadius, cardDimensions, colors, shadows, typography } from '../styles/theme';

// Card image mapping
const cardImages: Record<string, ImageSourcePropType> = {
  // We'll use placeholder for now - images would be loaded from assets
};

// Suit colors
const suitColors: Record<string, string> = {
  blue: colors.suitBlue,
  yellow: colors.suitYellow,
  green: colors.suitGreen,
  purple: colors.suitPurple,
  black: colors.suitBlack,
};

// Suit symbols
const suitSymbols: Record<string, string> = {
  blue: '🏴‍☠️',
  yellow: '⚓',
  green: '🗡️',
  purple: '🔮',
  black: '💀',
};

// Special card emojis
const specialEmojis: Record<string, string> = {
  escape: '🏃',
  pirate: '🏴‍☠️',
  skull_king: '👑',
  mermaid: '🧜‍♀️',
  tigress: '🎭',
  kraken: '🦑',
  white_whale: '🐋',
  loot: '💰',
};

interface CardProps {
  card: CardType;
  size?: 'small' | 'medium' | 'large';
  disabled?: boolean;
  selected?: boolean;
  faceDown?: boolean;
  onPress?: () => void;
  style?: ViewStyle;
  animationDelay?: number;
  showGlow?: boolean;
}

const AnimatedPressable = Animated.createAnimatedComponent(Pressable);

export function Card({
  card,
  size = 'medium',
  disabled = false,
  selected = false,
  faceDown = false,
  onPress,
  style,
  animationDelay = 0,
  showGlow = false,
}: CardProps): React.JSX.Element {
  const scale = useSharedValue(0);
  const rotation = useSharedValue(0);
  const translateY = useSharedValue(0);

  // Entry animation
  React.useEffect(() => {
    const timeout = setTimeout(() => {
      scale.value = withSpring(1, { damping: 15, stiffness: 150 });
      rotation.value = withSpring(0, { damping: 20 });
    }, animationDelay);

    return () => clearTimeout(timeout);
  }, [animationDelay, scale, rotation]);

  // Selection animation
  React.useEffect(() => {
    translateY.value = withSpring(selected ? -10 : 0, { damping: 15 });
  }, [selected, translateY]);

  const handlePress = useCallback(() => {
    if (disabled || !onPress) {
      return;
    }

    // Press animation
    scale.value = withSpring(0.95, { damping: 10 }, () => {
      scale.value = withSpring(1, { damping: 10 });
    });

    runOnJS(onPress)();
  }, [disabled, onPress, scale]);

  const animatedStyle = useAnimatedStyle(() => ({
    transform: [
      { scale: scale.value },
      { translateY: translateY.value },
      { rotateY: `${interpolate(rotation.value, [0, 1], [0, 180])}deg` },
    ],
    opacity: interpolate(scale.value, [0, 1], [0, 1]),
  }));

  // Determine card dimensions based on size
  const dimensions = {
    small: { width: cardDimensions.widthSmall, height: cardDimensions.heightSmall },
    medium: { width: cardDimensions.width, height: cardDimensions.height },
    large: { width: cardDimensions.widthLarge, height: cardDimensions.heightLarge },
  }[size];

  // Get card styling based on type
  const getCardBackground = (): string => {
    if (faceDown) {
      return colors.primaryDark;
    }
    if (card.suit) {
      return suitColors[card.suit] || colors.surface;
    }
    if (card.type === 'skull_king') {
      return colors.accentGold;
    }
    if (card.type === 'pirate' || card.type === 'tigress') {
      return colors.error;
    }
    if (card.type === 'mermaid') {
      return colors.accentBlue;
    }
    if (card.type === 'kraken') {
      return colors.accentPurple;
    }
    if (card.type === 'white_whale') {
      return colors.textMuted;
    }
    return colors.surface;
  };

  const renderCardContent = (): React.JSX.Element => {
    if (faceDown) {
      return (
        <View style={styles.cardBack}>
          <Text style={styles.cardBackText}>🏴‍☠️</Text>
        </View>
      );
    }

    // Standard suit card
    if (card.suit && card.number) {
      return (
        <View style={styles.suitCard}>
          <Text style={[styles.cardNumber, size === 'small' && styles.cardNumberSmall]}>
            {card.number}
          </Text>
          <Text style={[styles.suitSymbol, size === 'small' && styles.suitSymbolSmall]}>
            {suitSymbols[card.suit] || '?'}
          </Text>
        </View>
      );
    }

    // Special card
    return (
      <View style={styles.specialCard}>
        <Text style={[styles.specialEmoji, size === 'small' && styles.specialEmojiSmall]}>
          {specialEmojis[card.type || ''] || '?'}
        </Text>
        {size !== 'small' && card.name && (
          <Text style={styles.specialName} numberOfLines={2}>
            {card.name}
          </Text>
        )}
      </View>
    );
  };

  return (
    <AnimatedPressable
      onPress={handlePress}
      disabled={disabled}
      style={[
        styles.card,
        dimensions,
        { backgroundColor: getCardBackground() },
        disabled && styles.cardDisabled,
        selected && styles.cardSelected,
        showGlow && styles.cardGlow,
        animatedStyle,
        style,
      ]}
    >
      {renderCardContent()}
    </AnimatedPressable>
  );
}

const styles = StyleSheet.create({
  card: {
    borderRadius: borderRadius.md,
    borderWidth: 2,
    borderColor: colors.border,
    justifyContent: 'center',
    alignItems: 'center',
    ...shadows.md,
  },
  cardDisabled: {
    opacity: 0.5,
  },
  cardSelected: {
    borderColor: colors.accentGold,
    borderWidth: 3,
    ...shadows.glow(colors.accentGold),
  },
  cardGlow: {
    ...shadows.glow(colors.primary),
  },
  cardBack: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  cardBackText: {
    fontSize: typography.fontSize['2xl'],
  },
  suitCard: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 4,
  },
  cardNumber: {
    fontSize: typography.fontSize['2xl'],
    fontWeight: typography.fontWeight.bold,
    color: colors.text,
  },
  cardNumberSmall: {
    fontSize: typography.fontSize.lg,
  },
  suitSymbol: {
    fontSize: typography.fontSize.xl,
    marginTop: 2,
  },
  suitSymbolSmall: {
    fontSize: typography.fontSize.md,
  },
  specialCard: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 4,
  },
  specialEmoji: {
    fontSize: typography.fontSize['3xl'],
  },
  specialEmojiSmall: {
    fontSize: typography.fontSize.xl,
  },
  specialName: {
    fontSize: typography.fontSize.xs,
    color: colors.text,
    textAlign: 'center',
    marginTop: 2,
    fontWeight: typography.fontWeight.medium,
  },
});

export default Card;
