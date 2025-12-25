import React from 'react';
import { useTranslation } from 'react-i18next';
import { Modal, Pressable, StyleSheet, Text, View } from 'react-native';
import Animated, { FadeIn, FadeOut, SlideInDown } from 'react-native-reanimated';
import { borderRadius, colors, shadows, spacing, typography } from '../styles/theme';

interface TigressModalProps {
  visible: boolean;
  onChoice: (choice: 'pirate' | 'escape') => void;
}

export function TigressModal({ visible, onChoice }: TigressModalProps): React.JSX.Element {
  const { t } = useTranslation();

  return (
    <Modal visible={visible} transparent animationType="none">
      <Animated.View
        entering={FadeIn.duration(200)}
        exiting={FadeOut.duration(200)}
        style={styles.overlay}
      >
        <Animated.View entering={SlideInDown.springify().damping(15)} style={styles.modal}>
          <Text style={styles.emoji}>🎭</Text>
          <Text style={styles.title}>{t('game.tigressChoice')}</Text>
          <Text style={styles.subtitle}>{t('game.tigressQuestion')}</Text>

          <View style={styles.choicesContainer}>
            <Pressable
              onPress={() => onChoice('pirate')}
              style={({ pressed }) => [
                styles.choiceButton,
                styles.pirateButton,
                pressed && styles.choiceButtonPressed,
              ]}
            >
              <Text style={styles.choiceEmoji}>🏴‍☠️</Text>
              <Text style={styles.choiceTitle}>{t('game.tigressPirate')}</Text>
              <Text style={styles.choiceDesc}>{t('game.tigressPirateDesc')}</Text>
            </Pressable>

            <Pressable
              onPress={() => onChoice('escape')}
              style={({ pressed }) => [
                styles.choiceButton,
                styles.escapeButton,
                pressed && styles.choiceButtonPressed,
              ]}
            >
              <Text style={styles.choiceEmoji}>🏃</Text>
              <Text style={styles.choiceTitle}>{t('game.tigressEscape')}</Text>
              <Text style={styles.choiceDesc}>{t('game.tigressEscapeDesc')}</Text>
            </Pressable>
          </View>
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
    alignItems: 'center',
    ...shadows.xl,
  },
  emoji: {
    fontSize: 48,
    marginBottom: spacing.md,
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
  choicesContainer: {
    flexDirection: 'row',
    gap: spacing.md,
    width: '100%',
  },
  choiceButton: {
    flex: 1,
    padding: spacing.md,
    borderRadius: borderRadius.lg,
    alignItems: 'center',
    borderWidth: 2,
    ...shadows.md,
  },
  pirateButton: {
    backgroundColor: colors.error + '20',
    borderColor: colors.error,
  },
  escapeButton: {
    backgroundColor: colors.info + '20',
    borderColor: colors.info,
  },
  choiceButtonPressed: {
    transform: [{ scale: 0.98 }],
    opacity: 0.9,
  },
  choiceEmoji: {
    fontSize: 32,
    marginBottom: spacing.sm,
  },
  choiceTitle: {
    fontSize: typography.fontSize.lg,
    fontWeight: typography.fontWeight.bold,
    color: colors.text,
    marginBottom: spacing.xs,
  },
  choiceDesc: {
    fontSize: typography.fontSize.sm,
    color: colors.textMuted,
    textAlign: 'center',
  },
});

export default TigressModal;
