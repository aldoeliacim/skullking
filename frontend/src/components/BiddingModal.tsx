import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Modal, Pressable, StyleSheet, Text, View } from 'react-native';
import Animated, {
  useAnimatedStyle,
  useSharedValue,
  withSpring,
  withSequence,
  FadeIn,
  FadeOut,
  SlideInDown,
} from 'react-native-reanimated';
import { borderRadius, colors, screen, shadows, spacing, typography } from '../styles/theme';

interface BiddingModalProps {
  visible: boolean;
  maxBid: number;
  onBid: (bid: number) => void;
}

interface BidButtonProps {
  value: number;
  selected: boolean;
  onPress: () => void;
  delay: number;
}

function BidButton({ value, selected, onPress, delay }: BidButtonProps): React.JSX.Element {
  const scale = useSharedValue(0);

  React.useEffect(() => {
    const timeout = setTimeout(() => {
      scale.value = withSpring(1, { damping: 12, stiffness: 200 });
    }, delay);
    return () => clearTimeout(timeout);
  }, [delay, scale]);

  const animatedStyle = useAnimatedStyle(() => ({
    transform: [{ scale: scale.value }],
  }));

  return (
    <Animated.View style={animatedStyle}>
      <Pressable
        onPress={onPress}
        style={({ pressed }) => [
          styles.bidButton,
          selected && styles.bidButtonSelected,
          pressed && styles.bidButtonPressed,
        ]}
      >
        <Text style={[styles.bidButtonText, selected && styles.bidButtonTextSelected]}>
          {value}
        </Text>
      </Pressable>
    </Animated.View>
  );
}

export function BiddingModal({ visible, maxBid, onBid }: BiddingModalProps): React.JSX.Element {
  const { t } = useTranslation();
  const [selectedBid, setSelectedBid] = useState<number | null>(null);

  const handleBidSelect = (bid: number): void => {
    setSelectedBid(bid);
  };

  const handleConfirm = (): void => {
    if (selectedBid !== null) {
      onBid(selectedBid);
      setSelectedBid(null);
    }
  };

  // Generate bid options (0 to maxBid)
  const bidOptions = Array.from({ length: Math.min(maxBid + 1, 11) }, (_, i) => i);

  // Calculate grid layout
  const columns = screen.isSmall ? 4 : 6;
  const rows = Math.ceil(bidOptions.length / columns);

  return (
    <Modal visible={visible} transparent animationType="none">
      <Animated.View
        entering={FadeIn.duration(200)}
        exiting={FadeOut.duration(200)}
        style={styles.overlay}
      >
        <Animated.View entering={SlideInDown.springify().damping(15)} style={styles.modal}>
          <Text style={styles.title}>{t('game.makeYourBid')}</Text>
          <Text style={styles.subtitle}>{t('game.bidQuestion')}</Text>

          <View style={styles.bidGrid}>
            {bidOptions.map((bid, index) => (
              <BidButton
                key={bid}
                value={bid}
                selected={selectedBid === bid}
                onPress={() => handleBidSelect(bid)}
                delay={index * 30}
              />
            ))}
          </View>

          <Pressable
            onPress={handleConfirm}
            disabled={selectedBid === null}
            style={({ pressed }) => [
              styles.confirmButton,
              selectedBid === null && styles.confirmButtonDisabled,
              pressed && styles.confirmButtonPressed,
            ]}
          >
            <Text
              style={[
                styles.confirmButtonText,
                selectedBid === null && styles.confirmButtonTextDisabled,
              ]}
            >
              {selectedBid !== null
                ? `${t('game.bid')}: ${selectedBid} ${selectedBid === 1 ? t('game.card') : t('game.cards')}`
                : t('game.makeYourBid')}
            </Text>
          </Pressable>
        </Animated.View>
      </Animated.View>
    </Modal>
  );
}

const styles = StyleSheet.create({
  overlay: {
    flex: 1,
    backgroundColor: colors.overlay,
    justifyContent: 'center',
    alignItems: 'center',
    padding: spacing.lg,
  },
  modal: {
    backgroundColor: colors.surface,
    borderRadius: borderRadius.xl,
    padding: spacing.xl,
    width: '100%',
    maxWidth: 400,
    ...shadows.xl,
  },
  title: {
    fontSize: typography.fontSize['2xl'],
    fontWeight: typography.fontWeight.bold,
    color: colors.text,
    textAlign: 'center',
    marginBottom: spacing.xs,
  },
  subtitle: {
    fontSize: typography.fontSize.base,
    color: colors.textMuted,
    textAlign: 'center',
    marginBottom: spacing.xl,
  },
  bidGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'center',
    gap: spacing.sm,
    marginBottom: spacing.xl,
  },
  bidButton: {
    width: 56,
    height: 56,
    borderRadius: borderRadius.base,
    backgroundColor: colors.surfaceLight,
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 2,
    borderColor: colors.border,
    ...shadows.sm,
  },
  bidButtonSelected: {
    backgroundColor: colors.primary,
    borderColor: colors.primary,
    ...shadows.glow(colors.primary),
  },
  bidButtonPressed: {
    transform: [{ scale: 0.95 }],
  },
  bidButtonText: {
    fontSize: typography.fontSize.xl,
    fontWeight: typography.fontWeight.bold,
    color: colors.text,
  },
  bidButtonTextSelected: {
    color: colors.text,
  },
  confirmButton: {
    backgroundColor: colors.primary,
    paddingVertical: spacing.md,
    paddingHorizontal: spacing.xl,
    borderRadius: borderRadius.base,
    ...shadows.md,
  },
  confirmButtonDisabled: {
    backgroundColor: colors.surfaceLight,
  },
  confirmButtonPressed: {
    transform: [{ scale: 0.98 }],
    opacity: 0.9,
  },
  confirmButtonText: {
    fontSize: typography.fontSize.lg,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text,
    textAlign: 'center',
  },
  confirmButtonTextDisabled: {
    color: colors.textDark,
  },
});

export default BiddingModal;
