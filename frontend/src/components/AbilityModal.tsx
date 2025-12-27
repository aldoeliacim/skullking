import { useState } from 'react';
import { motion } from 'framer-motion';
import { useTranslation } from 'react-i18next';
import { Modal } from './Modal';
import { Card } from './Card';
import { useGameStore } from '../stores/gameStore';
import styles from './AbilityModal.module.css';

export function AbilityModal() {
  const { t } = useTranslation();
  const showAbility = useGameStore((s) => s.showAbility);
  const abilityData = useGameStore((s) => s.abilityData);
  const resolveAbility = useGameStore((s) => s.resolveAbility);
  const hand = useGameStore((s) => s.hand);

  const [selectedPlayer, setSelectedPlayer] = useState<string | null>(null);
  const [selectedCards, setSelectedCards] = useState<number[]>([]);
  const [selectedBet, setSelectedBet] = useState<number>(0);
  const [bidModifier, setBidModifier] = useState<number>(0);

  if (!abilityData) return null;

  const handleConfirm = () => {
    switch (abilityData.type) {
      case 'choose_starter':
        if (selectedPlayer) {
          resolveAbility({ chosen_player_id: selectedPlayer });
        }
        break;
      case 'draw_and_discard':
        resolveAbility({ discard_card_ids: selectedCards });
        break;
      case 'extra_bet':
        resolveAbility({ bet_amount: selectedBet });
        break;
      case 'view_deck':
        resolveAbility({});
        break;
      case 'modify_bid':
        resolveAbility({ modifier: bidModifier });
        break;
    }
    // Reset state
    setSelectedPlayer(null);
    setSelectedCards([]);
    setSelectedBet(0);
    setBidModifier(0);
  };

  const toggleCardSelection = (cardId: number) => {
    setSelectedCards((prev) =>
      prev.includes(cardId)
        ? prev.filter((id) => id !== cardId)
        : prev.length < 2
          ? [...prev, cardId]
          : prev
    );
  };

  const renderContent = () => {
    switch (abilityData.type) {
      case 'choose_starter':
        return (
          <>
            <p className={styles.description}>{t('ability.rosie.description')}</p>
            <div className={styles.playerList}>
              {abilityData.options?.map((playerId) => (
                <motion.button
                  key={playerId}
                  className={`${styles.playerButton} ${selectedPlayer === playerId ? styles.selected : ''}`}
                  onClick={() => setSelectedPlayer(playerId)}
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                >
                  {playerId}
                </motion.button>
              ))}
            </div>
            <button
              className={styles.confirmButton}
              onClick={handleConfirm}
              disabled={!selectedPlayer}
            >
              {t('common.confirm')}
            </button>
          </>
        );

      case 'draw_and_discard':
        const allCards = [...(abilityData.drawn_cards || []), ...hand];
        return (
          <>
            <p className={styles.description}>{t('ability.bendt.description')}</p>
            <div className={styles.cardGrid}>
              {allCards.map((cardId) => {
                const isDrawn = abilityData.drawn_cards?.includes(cardId);
                const isSelected = selectedCards.includes(cardId);
                return (
                  <div key={cardId} className={styles.cardWrapper}>
                    <Card
                      cardId={cardId}
                      size="small"
                      selected={isSelected}
                      onClick={() => toggleCardSelection(cardId)}
                    />
                    {isDrawn && <span className={styles.newBadge}>{t('ability.newCard')}</span>}
                  </div>
                );
              })}
            </div>
            <p className={styles.selectionCount}>
              {selectedCards.length}/2 selected
            </p>
            <div className={styles.buttonRow}>
              <button
                className={styles.skipButton}
                onClick={() => resolveAbility({ discard_card_ids: abilityData.drawn_cards })}
              >
                {t('ability.bendt.skip')}
              </button>
              <button
                className={styles.confirmButton}
                onClick={handleConfirm}
                disabled={selectedCards.length !== 2}
              >
                {t('ability.bendt.discard')}
              </button>
            </div>
          </>
        );

      case 'extra_bet':
        const betOptions = [0, 10, 20];
        return (
          <>
            <p className={styles.description}>{t('ability.roatan.description')}</p>
            <div className={styles.betGrid}>
              {betOptions.map((bet) => (
                <motion.button
                  key={bet}
                  className={`${styles.betButton} ${selectedBet === bet ? styles.selected : ''}`}
                  onClick={() => setSelectedBet(bet)}
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  {bet === 0 ? t('ability.roatan.noBet') : `+${bet}`}
                </motion.button>
              ))}
            </div>
            <button className={styles.confirmButton} onClick={handleConfirm}>
              {t('common.confirm')}
            </button>
          </>
        );

      case 'view_deck':
        return (
          <>
            <p className={styles.description}>{t('ability.jade.description')}</p>
            <div className={styles.deckCards}>
              {abilityData.deck_cards?.map((cardId) => (
                <Card key={cardId} cardId={cardId} size="small" disabled />
              ))}
            </div>
            <button className={styles.confirmButton} onClick={handleConfirm}>
              {t('ability.jade.gotIt')}
            </button>
          </>
        );

      case 'modify_bid':
        const currentBid = abilityData.current_bid || 0;
        return (
          <>
            <p className={styles.description}>{t('ability.harry.description')}</p>
            <div className={styles.modifierGrid}>
              {[-1, 0, 1].map((mod) => {
                const newBid = Math.max(0, currentBid + mod);
                const label = mod === -1 ? t('ability.harry.decrease') : mod === 1 ? t('ability.harry.increase') : t('ability.harry.keep');
                return (
                  <motion.button
                    key={mod}
                    className={`${styles.modifierButton} ${bidModifier === mod ? styles.selected : ''}`}
                    onClick={() => setBidModifier(mod)}
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                  >
                    <span className={styles.modLabel}>{label}</span>
                    <span className={styles.newBidValue}>{newBid}</span>
                  </motion.button>
                );
              })}
            </div>
            <button className={styles.confirmButton} onClick={handleConfirm}>
              {t('common.confirm')}
            </button>
          </>
        );

      default:
        return null;
    }
  };

  const getTitle = () => {
    switch (abilityData.type) {
      case 'choose_starter': return t('ability.rosie.title');
      case 'draw_and_discard': return t('ability.bendt.title');
      case 'extra_bet': return t('ability.roatan.title');
      case 'view_deck': return t('ability.jade.title');
      case 'modify_bid': return t('ability.harry.title');
      default: return 'Ability';
    }
  };

  const getIcon = () => {
    switch (abilityData.type) {
      case 'choose_starter': return '🌹';
      case 'draw_and_discard': return '🃏';
      case 'extra_bet': return '🎰';
      case 'view_deck': return '🔮';
      case 'modify_bid': return '💪';
      default: return '⚡';
    }
  };

  return (
    <Modal isOpen={showAbility} title={getTitle()} icon={getIcon()} closable={false}>
      {renderContent()}
    </Modal>
  );
}

export default AbilityModal;
