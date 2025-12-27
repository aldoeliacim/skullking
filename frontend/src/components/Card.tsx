import React, { useCallback } from 'react';
import { Image, Pressable, StyleSheet, Text, View, type ViewStyle } from 'react-native';
import Animated, {
  useAnimatedStyle,
  useSharedValue,
  withSpring,
  interpolate,
  runOnJS,
} from 'react-native-reanimated';
import type { Card as CardType } from '../stores/gameStore';
import { borderRadius, cardDimensions, colors, shadows, typography } from '../styles/theme';

// Base URL for card images
const CARD_IMAGE_BASE_URL = '/static/images/cards/';

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

  const renderCardContent = (): React.JSX.Element => {
    // Face down card
    if (faceDown) {
      return (
        <Image
          source={{ uri: `${CARD_IMAGE_BASE_URL}back.png` }}
          style={styles.cardImage}
          resizeMode="cover"
        />
      );
    }

    // Card with image
    if (card.image) {
      return (
        <View style={styles.imageContainer}>
          <Image
            source={{ uri: `${CARD_IMAGE_BASE_URL}${card.image}` }}
            style={styles.cardImage}
            resizeMode="cover"
          />
          {/* Number overlay for suit cards */}
          {card.number && (
            <View style={styles.numberOverlay}>
              <Text
                style={[
                  styles.overlayNumber,
                  size === 'small' && styles.overlayNumberSmall,
                  card.number === 14 && styles.overlayNumberBonus,
                ]}
              >
                {card.number}
              </Text>
            </View>
          )}
          {/* Bonus badge for 14 cards */}
          {card.number === 14 && (
            <View style={styles.bonusBadge}>
              <Text style={styles.bonusText}>+10</Text>
            </View>
          )}
        </View>
      );
    }

    // Fallback for cards without images
    return (
      <View style={styles.fallbackCard}>
        <Text style={styles.fallbackText}>{card.name || card.type || '?'}</Text>
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
    overflow: 'hidden',
    backgroundColor: colors.surface,
    ...shadows.md,
  },
  cardDisabled: {
    opacity: 0.35,
    borderColor: colors.textDark,
    borderWidth: 1,
  },
  cardSelected: {
    borderColor: colors.accentGold,
    borderWidth: 3,
    ...shadows.glow(colors.accentGold),
  },
  cardGlow: {
    ...shadows.glow(colors.primary),
  },
  imageContainer: {
    flex: 1,
    width: '100%',
    position: 'relative',
  },
  cardImage: {
    width: '100%',
    height: '100%',
  },
  numberOverlay: {
    position: 'absolute',
    top: 4,
    left: 4,
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
    borderRadius: 4,
    paddingHorizontal: 6,
    paddingVertical: 2,
  },
  overlayNumber: {
    fontSize: typography.fontSize.lg,
    fontWeight: typography.fontWeight.bold,
    color: colors.text,
  },
  overlayNumberSmall: {
    fontSize: typography.fontSize.sm,
  },
  overlayNumberBonus: {
    color: colors.accentGold,
  },
  bonusBadge: {
    position: 'absolute',
    bottom: 4,
    right: 4,
    backgroundColor: colors.accentGold,
    borderRadius: 4,
    paddingHorizontal: 4,
    paddingVertical: 2,
  },
  bonusText: {
    fontSize: typography.fontSize.xs,
    fontWeight: typography.fontWeight.bold,
    color: colors.backgroundDark,
  },
  fallbackCard: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 4,
  },
  fallbackText: {
    fontSize: typography.fontSize.sm,
    color: colors.text,
    textAlign: 'center',
  },
});

export default Card;
