import React, { useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Modal, Pressable, ScrollView, StyleSheet, Text, View } from 'react-native';
import Animated, { FadeIn, FadeOut, SlideInDown } from 'react-native-reanimated';
import { type AbilityData, type Card as CardType, parseCard } from '../stores/gameStore';
import { borderRadius, colors, shadows, spacing, typography } from '../styles/theme';
import { Card } from './Card';

interface AbilityModalProps {
  visible: boolean;
  abilityData: AbilityData | null;
  players: Array<{ id: string; username: string }>;
  onResolve: (data: Record<string, unknown>) => void;
}

interface PlayerOption {
  player_id: string;
  username: string;
}

export function AbilityModal({
  visible,
  abilityData,
  players: _players,
  onResolve,
}: AbilityModalProps): React.JSX.Element | null {
  const { t } = useTranslation();
  const [selectedCards, setSelectedCards] = useState<string[]>([]);
  const [selectedPlayer, setSelectedPlayer] = useState<string | null>(null);
  const [selectedBid, setSelectedBid] = useState<number | null>(null);

  if (!visible || !abilityData) {
    return null;
  }

  const { type, pirate: _pirate, data } = abilityData;

  const handleCardSelect = (cardId: string): void => {
    setSelectedCards((prev) => {
      if (prev.includes(cardId)) {
        return prev.filter((id) => id !== cardId);
      }
      return [...prev, cardId];
    });
  };

  const handleConfirmDiscard = (): void => {
    const mustDiscard = (data?.must_discard as number) || 2;
    if (selectedCards.length === mustDiscard) {
      onResolve({ discarded_cards: selectedCards });
      setSelectedCards([]);
    }
  };

  const handleConfirmPlayer = (): void => {
    if (selectedPlayer) {
      onResolve({ selected_player_id: selectedPlayer });
      setSelectedPlayer(null);
    }
  };

  const handleConfirmBid = (): void => {
    if (selectedBid !== null) {
      onResolve({ new_bid: selectedBid });
      setSelectedBid(null);
    }
  };

  const renderChooseStarter = (): React.JSX.Element => {
    const options = (data?.options as PlayerOption[]) || [];
    return (
      <>
        <Text style={styles.emoji}>🌹</Text>
        <Text style={styles.title}>{t('ability.rosie.title')}</Text>
        <Text style={styles.subtitle}>{t('ability.rosie.description')}</Text>
        <View style={styles.optionsContainer}>
          {options.map((option) => (
            <Pressable
              key={option.player_id}
              onPress={() => setSelectedPlayer(option.player_id)}
              style={({ pressed }) => [
                styles.optionButton,
                selectedPlayer === option.player_id && styles.optionButtonSelected,
                pressed && styles.optionButtonPressed,
              ]}
            >
              <Text
                style={[
                  styles.optionText,
                  selectedPlayer === option.player_id && styles.optionTextSelected,
                ]}
              >
                {option.username}
              </Text>
            </Pressable>
          ))}
        </View>
        <Pressable
          onPress={handleConfirmPlayer}
          disabled={!selectedPlayer}
          style={({ pressed }) => [
            styles.confirmButton,
            !selectedPlayer && styles.confirmButtonDisabled,
            pressed && styles.confirmButtonPressed,
          ]}
        >
          <Text
            style={[styles.confirmButtonText, !selectedPlayer && styles.confirmButtonTextDisabled]}
          >
            {t('common.confirm')}
          </Text>
        </Pressable>
      </>
    );
  };

  const renderDrawDiscard = (): React.JSX.Element => {
    // Use full hand if available, otherwise fall back to drawn cards
    const handCards = (data?.hand as number[]) || [];
    const drawnCards = (data?.drawn_cards as number[]) || [];
    const mustDiscard = (data?.must_discard as number) || 2;

    // Show full hand - player can choose ANY cards to discard
    const cardsToShow = handCards.length > 0 ? handCards : drawnCards;
    const parsedCards: CardType[] = cardsToShow.map((id) => parseCard(id));
    const drawnCardIds = new Set(drawnCards.map(String));

    const handleSkip = (): void => {
      // Skip by discarding the drawn cards (auto-select them)
      onResolve({ discarded_cards: drawnCards.slice(0, mustDiscard).map(String) });
      setSelectedCards([]);
    };

    return (
      <>
        <Text style={styles.emoji}>🃏</Text>
        <Text style={styles.title}>{t('ability.bendt.title')}</Text>
        <Text style={styles.subtitle}>{t('ability.bendt.description')}</Text>
        <ScrollView
          horizontal
          showsHorizontalScrollIndicator={false}
          contentContainerStyle={styles.cardsContainer}
        >
          {parsedCards.map((card) => (
            <View key={card.id} style={styles.cardWrapper}>
              <Card
                card={card}
                size="medium"
                selected={selectedCards.includes(card.id)}
                onPress={() => handleCardSelect(card.id)}
              />
              {drawnCardIds.has(card.id) && (
                <Text style={styles.newCardBadge}>{t('ability.newCard')}</Text>
              )}
            </View>
          ))}
        </ScrollView>
        <Text style={styles.selectionInfo}>
          {t('ability.selected')}: {selectedCards.length}/{mustDiscard}
        </Text>
        <View style={styles.buttonRow}>
          <Pressable
            onPress={handleSkip}
            style={({ pressed }) => [styles.skipButton, pressed && styles.skipButtonPressed]}
          >
            <Text style={styles.skipButtonText}>{t('ability.skip')}</Text>
          </Pressable>
          <Pressable
            onPress={handleConfirmDiscard}
            disabled={selectedCards.length !== mustDiscard}
            style={({ pressed }) => [
              styles.confirmButton,
              selectedCards.length !== mustDiscard && styles.confirmButtonDisabled,
              pressed && styles.confirmButtonPressed,
            ]}
          >
            <Text
              style={[
                styles.confirmButtonText,
                selectedCards.length !== mustDiscard && styles.confirmButtonTextDisabled,
              ]}
            >
              {t('ability.discard')}
            </Text>
          </Pressable>
        </View>
      </>
    );
  };

  const renderExtraBet = (): React.JSX.Element => {
    const options = (data?.options as number[]) || [0, 10, 20];
    return (
      <>
        <Text style={styles.emoji}>🎰</Text>
        <Text style={styles.title}>{t('ability.roatan.title')}</Text>
        <Text style={styles.subtitle}>{t('ability.roatan.description')}</Text>
        <View style={styles.bidOptionsContainer}>
          {options.map((bet) => (
            <Pressable
              key={bet}
              onPress={() => setSelectedBid(bet)}
              style={({ pressed }) => [
                styles.bidButton,
                selectedBid === bet && styles.bidButtonSelected,
                pressed && styles.bidButtonPressed,
              ]}
            >
              <Text
                style={[styles.bidButtonText, selectedBid === bet && styles.bidButtonTextSelected]}
              >
                {bet === 0 ? t('ability.noBet') : `+${bet}`}
              </Text>
            </Pressable>
          ))}
        </View>
        <Pressable
          onPress={handleConfirmBid}
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
            {t('common.confirm')}
          </Text>
        </Pressable>
      </>
    );
  };

  const renderViewDeck = (): React.JSX.Element => {
    const undealtCards = (data?.undealt_cards as number[]) || [];
    const parsedCards: CardType[] = undealtCards.map((id) => parseCard(id));

    return (
      <>
        <Text style={styles.emoji}>🔮</Text>
        <Text style={styles.title}>{t('ability.jade.title')}</Text>
        <Text style={styles.subtitle}>{t('ability.jade.description')}</Text>
        <ScrollView
          horizontal
          showsHorizontalScrollIndicator={false}
          contentContainerStyle={styles.cardsContainer}
        >
          {parsedCards.map((card) => (
            <Card key={card.id} card={card} size="small" disabled />
          ))}
        </ScrollView>
        <Pressable
          onPress={() => onResolve({})}
          style={({ pressed }) => [styles.confirmButton, pressed && styles.confirmButtonPressed]}
        >
          <Text style={styles.confirmButtonText}>{t('common.close')}</Text>
        </Pressable>
      </>
    );
  };

  const renderModifyBid = (): React.JSX.Element => {
    const currentBid = (data?.current_bid as number) || 0;
    const _tricksWon = (data?.tricks_won as number) || 0; // May be used for display
    // Harry's options are -1 (decrease), 0 (keep), 1 (increase)
    const options = (data?.options as number[]) || [-1, 0, 1];

    const getOptionLabel = (modifier: number): string => {
      if (modifier === -1) return t('ability.harry.decrease');
      if (modifier === 1) return t('ability.harry.increase');
      return t('ability.harry.keep');
    };

    const getOptionDescription = (modifier: number): string => {
      const newBid = Math.max(0, currentBid + modifier);
      return `${t('ability.harry.newBid')}: ${newBid}`;
    };

    return (
      <>
        <Text style={styles.emoji}>💪</Text>
        <Text style={styles.title}>{t('ability.harry.title')}</Text>
        <Text style={styles.subtitle}>{t('ability.harry.description')}</Text>
        <View style={styles.bidOptionsContainer}>
          {options.map((modifier) => (
            <Pressable
              key={modifier}
              onPress={() => setSelectedBid(modifier)}
              style={({ pressed }) => [
                styles.bidButton,
                selectedBid === modifier && styles.bidButtonSelected,
                modifier === 0 && styles.bidButtonCurrent,
                pressed && styles.bidButtonPressed,
              ]}
            >
              <Text
                style={[
                  styles.bidButtonText,
                  selectedBid === modifier && styles.bidButtonTextSelected,
                ]}
              >
                {getOptionLabel(modifier)}
              </Text>
              <Text style={styles.bidDescriptionText}>{getOptionDescription(modifier)}</Text>
            </Pressable>
          ))}
        </View>
        <Pressable
          onPress={handleConfirmBid}
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
            {t('common.confirm')}
          </Text>
        </Pressable>
      </>
    );
  };

  const renderContent = (): React.JSX.Element => {
    switch (type) {
      case 'choose_starter':
        return renderChooseStarter();
      case 'draw_discard':
        return renderDrawDiscard();
      case 'extra_bet':
        return renderExtraBet();
      case 'view_deck':
        return renderViewDeck();
      case 'modify_bid':
        return renderModifyBid();
      default:
        return (
          <>
            <Text style={styles.emoji}>⚔️</Text>
            <Text style={styles.title}>{t('ability.unknown.title')}</Text>
            <Text style={styles.subtitle}>{t('ability.unknown.description')}</Text>
            <Pressable
              onPress={() => onResolve({})}
              style={({ pressed }) => [
                styles.confirmButton,
                pressed && styles.confirmButtonPressed,
              ]}
            >
              <Text style={styles.confirmButtonText}>{t('common.continue')}</Text>
            </Pressable>
          </>
        );
    }
  };

  return (
    <Modal visible={visible} transparent animationType="none">
      <Animated.View
        entering={FadeIn.duration(200)}
        exiting={FadeOut.duration(200)}
        style={styles.overlay}
      >
        <Animated.View entering={SlideInDown.springify().damping(15)} style={styles.modal}>
          {renderContent()}
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
    maxWidth: 450,
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
    marginBottom: spacing.lg,
  },
  optionsContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: spacing.sm,
    justifyContent: 'center',
    marginBottom: spacing.lg,
  },
  optionButton: {
    paddingVertical: spacing.md,
    paddingHorizontal: spacing.lg,
    borderRadius: borderRadius.base,
    backgroundColor: colors.surfaceLight,
    borderWidth: 2,
    borderColor: colors.border,
  },
  optionButtonSelected: {
    backgroundColor: colors.primary,
    borderColor: colors.primary,
  },
  optionButtonPressed: {
    transform: [{ scale: 0.98 }],
  },
  optionText: {
    fontSize: typography.fontSize.base,
    fontWeight: typography.fontWeight.medium,
    color: colors.text,
  },
  optionTextSelected: {
    color: colors.text,
  },
  cardsContainer: {
    flexDirection: 'row',
    gap: spacing.sm,
    paddingVertical: spacing.sm,
    paddingHorizontal: spacing.xs,
    marginBottom: spacing.md,
  },
  selectionInfo: {
    fontSize: typography.fontSize.sm,
    color: colors.textMuted,
    marginBottom: spacing.md,
  },
  bidOptionsContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: spacing.sm,
    justifyContent: 'center',
    marginBottom: spacing.lg,
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
  },
  bidButtonSelected: {
    backgroundColor: colors.primary,
    borderColor: colors.primary,
  },
  bidButtonCurrent: {
    borderColor: colors.accentGold,
  },
  bidButtonPressed: {
    transform: [{ scale: 0.95 }],
  },
  bidButtonText: {
    fontSize: typography.fontSize.lg,
    fontWeight: typography.fontWeight.bold,
    color: colors.text,
  },
  bidButtonTextSelected: {
    color: colors.text,
  },
  bidDescriptionText: {
    fontSize: typography.fontSize.xs,
    color: colors.textMuted,
    marginTop: spacing.xs,
  },
  confirmButton: {
    backgroundColor: colors.primary,
    paddingVertical: spacing.md,
    paddingHorizontal: spacing.xl,
    borderRadius: borderRadius.base,
    minWidth: 150,
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
  buttonRow: {
    flexDirection: 'row',
    gap: spacing.md,
    justifyContent: 'center',
    flexWrap: 'wrap',
  },
  skipButton: {
    backgroundColor: colors.surfaceLight,
    paddingVertical: spacing.md,
    paddingHorizontal: spacing.lg,
    borderRadius: borderRadius.base,
    borderWidth: 1,
    borderColor: colors.border,
  },
  skipButtonPressed: {
    transform: [{ scale: 0.98 }],
    opacity: 0.8,
  },
  skipButtonText: {
    fontSize: typography.fontSize.base,
    color: colors.textMuted,
    textAlign: 'center',
  },
  cardWrapper: {
    position: 'relative',
  },
  newCardBadge: {
    position: 'absolute',
    top: -8,
    right: -8,
    backgroundColor: colors.success,
    color: colors.text,
    fontSize: typography.fontSize.xs,
    fontWeight: typography.fontWeight.bold,
    paddingHorizontal: spacing.xs,
    paddingVertical: 2,
    borderRadius: borderRadius.sm,
    overflow: 'hidden',
  },
});

export default AbilityModal;
