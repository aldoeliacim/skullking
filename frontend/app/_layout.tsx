import { Stack } from 'expo-router';
import * as SplashScreen from 'expo-splash-screen';
import { StatusBar } from 'expo-status-bar';
import React, { useEffect } from 'react';
import { I18nextProvider } from 'react-i18next';
import { StyleSheet, View } from 'react-native';
import { GestureHandlerRootView } from 'react-native-gesture-handler';
import i18n from '../src/i18n';
import { colors } from '../src/styles/theme';

// Prevent splash screen from auto-hiding
SplashScreen.preventAutoHideAsync();

export default function RootLayout(): React.JSX.Element {
  useEffect(() => {
    // Hide splash screen after app is ready
    const hideSplash = async (): Promise<void> => {
      await SplashScreen.hideAsync();
    };
    hideSplash();
  }, []);

  return (
    <I18nextProvider i18n={i18n}>
      <GestureHandlerRootView style={styles.container}>
        <View style={styles.container}>
          <StatusBar style="light" />
          <Stack
            screenOptions={{
              headerShown: false,
              contentStyle: styles.content,
              animation: 'slide_from_right',
            }}
          >
            <Stack.Screen name="index" />
            <Stack.Screen name="lobby/[id]" />
            <Stack.Screen name="game/[id]" />
          </Stack>
        </View>
      </GestureHandlerRootView>
    </I18nextProvider>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background,
  },
  content: {
    backgroundColor: colors.background,
  },
});
