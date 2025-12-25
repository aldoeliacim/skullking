import React, { useState, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import { Modal, Pressable, StyleSheet, Text, View } from 'react-native';
import Animated, { FadeIn, FadeOut, SlideInUp } from 'react-native-reanimated';
import { type LanguageCode, changeLanguage, getCurrentLanguage } from '../i18n';
import { colors, spacing, typography } from '../styles/theme';
import { Button } from './Button';

interface SettingsButtonProps {
  style?: object;
}

export function SettingsButton({ style }: SettingsButtonProps): React.JSX.Element {
  const { t } = useTranslation();
  const [visible, setVisible] = useState(false);
  const [currentLang, setCurrentLang] = useState<LanguageCode>(getCurrentLanguage());

  const toggleLanguage = useCallback(async () => {
    const newLang = currentLang === 'en' ? 'es' : 'en';
    await changeLanguage(newLang);
    setCurrentLang(newLang);
  }, [currentLang]);

  const handleClose = useCallback(() => {
    setVisible(false);
  }, []);

  return (
    <>
      <Pressable
        style={[styles.gearButton, style]}
        onPress={() => setVisible(true)}
        hitSlop={12}
        accessibilityLabel={t('settings.title')}
        accessibilityRole="button"
      >
        <Text style={styles.gearIcon}>⚙️</Text>
      </Pressable>

      <Modal visible={visible} transparent animationType="none" onRequestClose={handleClose}>
        <Animated.View
          entering={FadeIn.duration(200)}
          exiting={FadeOut.duration(150)}
          style={styles.overlay}
        >
          <Pressable style={styles.backdrop} onPress={handleClose} />
          <Animated.View entering={SlideInUp.duration(300)} style={styles.modal}>
            <View style={styles.header}>
              <Text style={styles.title}>{t('settings.title')}</Text>
              <Pressable onPress={handleClose} hitSlop={12}>
                <Text style={styles.closeIcon}>✕</Text>
              </Pressable>
            </View>

            <View style={styles.content}>
              <View style={styles.settingRow}>
                <Text style={styles.settingLabel}>{t('settings.language')}</Text>
                <Button
                  title={currentLang === 'en' ? '🇪🇸 Español' : '🇺🇸 English'}
                  onPress={toggleLanguage}
                  variant="secondary"
                  size="sm"
                />
              </View>

              <View style={styles.divider} />

              <View style={styles.settingRow}>
                <Text style={styles.settingLabel}>{t('settings.sound')}</Text>
                <Button title={t('settings.on')} onPress={() => {}} variant="outline" size="sm" />
              </View>

              <View style={styles.divider} />

              <View style={styles.settingRow}>
                <Text style={styles.settingLabel}>{t('settings.vibration')}</Text>
                <Button title={t('settings.on')} onPress={() => {}} variant="outline" size="sm" />
              </View>
            </View>

            <View style={styles.footer}>
              <Text style={styles.version}>v1.0.0</Text>
            </View>
          </Animated.View>
        </Animated.View>
      </Modal>
    </>
  );
}

const styles = StyleSheet.create({
  gearButton: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: colors.surface,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 1,
    borderColor: colors.border,
  },
  gearIcon: {
    fontSize: 20,
  },
  overlay: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  backdrop: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
  },
  modal: {
    width: '90%',
    maxWidth: 360,
    backgroundColor: colors.surface,
    borderRadius: 16,
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: colors.border,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    padding: spacing.lg,
    borderBottomWidth: 1,
    borderBottomColor: colors.border,
  },
  title: {
    fontSize: typography.fontSize.xl,
    fontWeight: typography.fontWeight.bold,
    color: colors.text,
  },
  closeIcon: {
    fontSize: 20,
    color: colors.textMuted,
  },
  content: {
    padding: spacing.lg,
  },
  settingRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  settingLabel: {
    fontSize: typography.fontSize.md,
    color: colors.text,
  },
  divider: {
    height: 1,
    backgroundColor: colors.border,
    marginVertical: spacing.md,
  },
  footer: {
    padding: spacing.md,
    alignItems: 'center',
    borderTopWidth: 1,
    borderTopColor: colors.border,
  },
  version: {
    fontSize: typography.fontSize.xs,
    color: colors.textMuted,
  },
});
