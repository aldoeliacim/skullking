import AsyncStorage from '@react-native-async-storage/async-storage';
import { useRouter } from 'expo-router';
import React, { useState, useCallback } from 'react';
import { useTranslation } from 'react-i18next';
import {
  Alert,
  KeyboardAvoidingView,
  Platform,
  ScrollView,
  StyleSheet,
  Text,
  View,
} from 'react-native';
import Animated, { FadeInUp, FadeInDown } from 'react-native-reanimated';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Button, Input } from '../src/components';
import { type LanguageCode, changeLanguage, getCurrentLanguage } from '../src/i18n';
import { api } from '../src/services/api';
import { colors, screen, spacing, typography } from '../src/styles/theme';

const PLAYER_NAME_KEY = '@skullking/playerName';

export default function HomeScreen(): React.JSX.Element {
  const { t } = useTranslation();
  const router = useRouter();

  const [playerName, setPlayerName] = useState('');
  const [gameCode, setGameCode] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [currentLang, setCurrentLang] = useState<LanguageCode>(getCurrentLanguage());

  // Load saved player name on mount
  React.useEffect(() => {
    const loadPlayerName = async (): Promise<void> => {
      const saved = await AsyncStorage.getItem(PLAYER_NAME_KEY);
      if (saved) {
        setPlayerName(saved);
      }
    };
    loadPlayerName();
  }, []);

  // Save player name when it changes
  const handleNameChange = useCallback((name: string) => {
    setPlayerName(name);
    setError(null);
    AsyncStorage.setItem(PLAYER_NAME_KEY, name);
  }, []);

  const validateName = useCallback((): boolean => {
    if (!playerName.trim()) {
      setError(t('login.errorEnterName'));
      return false;
    }
    return true;
  }, [playerName, t]);

  const handleCreateGame = useCallback(async () => {
    if (!validateName()) {
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const response = await api.createGame(playerName.trim());
      router.push({
        pathname: '/lobby/[id]',
        params: {
          id: response.game_id,
          slug: response.slug,
          playerId: 'host',
          playerName: playerName.trim(),
        },
      });
    } catch (err) {
      setError(err instanceof Error ? err.message : t('login.errorCreateFailed'));
    } finally {
      setIsLoading(false);
    }
  }, [playerName, validateName, router, t]);

  const handleJoinGame = useCallback(() => {
    if (!validateName()) {
      return;
    }

    if (!gameCode.trim()) {
      setError(t('login.errorEnterGameId'));
      return;
    }

    router.push({
      pathname: '/lobby/[id]',
      params: {
        id: gameCode.trim().toUpperCase(),
        playerName: playerName.trim(),
      },
    });
  }, [playerName, gameCode, validateName, router, t]);

  const handleSpectate = useCallback(() => {
    if (!gameCode.trim()) {
      setError(t('login.errorEnterGameId'));
      return;
    }

    router.push({
      pathname: '/game/[id]',
      params: {
        id: gameCode.trim().toUpperCase(),
        spectator: 'true',
      },
    });
  }, [gameCode, router, t]);

  const toggleLanguage = useCallback(async () => {
    const newLang = currentLang === 'en' ? 'es' : 'en';
    await changeLanguage(newLang);
    setCurrentLang(newLang);
  }, [currentLang]);

  return (
    <SafeAreaView style={styles.container} edges={['top', 'left', 'right']}>
      <KeyboardAvoidingView
        style={styles.keyboardView}
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
      >
        <ScrollView
          contentContainerStyle={styles.scrollContent}
          showsVerticalScrollIndicator={false}
          keyboardShouldPersistTaps="handled"
        >
          {/* Header */}
          <Animated.View entering={FadeInUp.delay(100).duration(500)} style={styles.header}>
            <Text style={styles.emoji}>🏴‍☠️</Text>
            <Text style={styles.title}>{t('app.title')}</Text>
            <Text style={styles.subtitle}>{t('app.subtitle')}</Text>
          </Animated.View>

          {/* Form */}
          <Animated.View entering={FadeInDown.delay(300).duration(500)} style={styles.form}>
            <Input
              label={t('login.yourName')}
              placeholder={t('login.enterName')}
              value={playerName}
              onChangeText={handleNameChange}
              autoCapitalize="words"
              autoCorrect={false}
              maxLength={20}
            />

            <Button
              title={t('login.createGame')}
              onPress={handleCreateGame}
              loading={isLoading}
              fullWidth
              size="lg"
            />

            <View style={styles.divider}>
              <View style={styles.dividerLine} />
              <Text style={styles.dividerText}>{t('login.or')}</Text>
              <View style={styles.dividerLine} />
            </View>

            <Input
              label={t('login.gameCode')}
              placeholder={t('login.enterGameId')}
              value={gameCode}
              onChangeText={(text) => {
                setGameCode(text.toUpperCase());
                setError(null);
              }}
              autoCapitalize="characters"
              autoCorrect={false}
              maxLength={8}
            />

            <View style={styles.buttonRow}>
              <View style={styles.buttonFlex}>
                <Button
                  title={t('login.joinGame')}
                  onPress={handleJoinGame}
                  variant="secondary"
                  fullWidth
                />
              </View>
              <View style={styles.buttonFlex}>
                <Button
                  title={t('login.spectateGame')}
                  onPress={handleSpectate}
                  variant="outline"
                  fullWidth
                />
              </View>
            </View>

            {error && (
              <Animated.View entering={FadeInDown.duration(200)}>
                <Text style={styles.error}>{error}</Text>
              </Animated.View>
            )}
          </Animated.View>

          {/* Footer */}
          <Animated.View entering={FadeInDown.delay(500).duration(500)} style={styles.footer}>
            <Button
              title={t('login.browseGames')}
              onPress={() => Alert.alert('Coming soon', 'Browse games feature')}
              variant="ghost"
              size="sm"
            />
            <Button
              title={t('login.viewHistory')}
              onPress={() => Alert.alert('Coming soon', 'Game history feature')}
              variant="ghost"
              size="sm"
            />
            <Button
              title={currentLang === 'en' ? '🇪🇸 Español' : '🇺🇸 English'}
              onPress={toggleLanguage}
              variant="ghost"
              size="sm"
            />
          </Animated.View>
        </ScrollView>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background,
  },
  keyboardView: {
    flex: 1,
  },
  scrollContent: {
    flexGrow: 1,
    padding: spacing.lg,
    justifyContent: 'center',
  },
  header: {
    alignItems: 'center',
    marginBottom: spacing['3xl'],
  },
  emoji: {
    fontSize: 64,
    marginBottom: spacing.md,
  },
  title: {
    fontSize: typography.fontSize['4xl'],
    fontWeight: typography.fontWeight.extrabold,
    color: colors.text,
    textAlign: 'center',
    marginBottom: spacing.xs,
  },
  subtitle: {
    fontSize: typography.fontSize.lg,
    color: colors.textMuted,
    textAlign: 'center',
  },
  form: {
    maxWidth: 400,
    alignSelf: 'center',
    width: '100%',
  },
  divider: {
    flexDirection: 'row',
    alignItems: 'center',
    marginVertical: spacing.lg,
  },
  dividerLine: {
    flex: 1,
    height: 1,
    backgroundColor: colors.border,
  },
  dividerText: {
    paddingHorizontal: spacing.md,
    fontSize: typography.fontSize.sm,
    color: colors.textMuted,
    textTransform: 'uppercase',
  },
  buttonRow: {
    flexDirection: 'row',
    gap: spacing.md,
  },
  buttonFlex: {
    flex: 1,
  },
  error: {
    marginTop: spacing.md,
    padding: spacing.md,
    backgroundColor: colors.error + '20',
    borderRadius: 8,
    color: colors.error,
    fontSize: typography.fontSize.sm,
    textAlign: 'center',
  },
  footer: {
    flexDirection: 'row',
    justifyContent: 'center',
    flexWrap: 'wrap',
    gap: spacing.md,
    marginTop: spacing['3xl'],
  },
});
