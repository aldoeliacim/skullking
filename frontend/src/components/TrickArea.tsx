import { motion, AnimatePresence } from 'framer-motion';
import { useTranslation } from 'react-i18next';
import { Card } from './Card';
import { useGameStore } from '../stores/gameStore';
import styles from './TrickArea.module.css';

export function TrickArea() {
  const { t } = useTranslation();
  const trickCards = useGameStore((s) => s.trickCards);
  const players = useGameStore((s) => s.players);
  const trickWinner = useGameStore((s) => s.trickWinner);

  const getPlayerName = (playerId: string) => {
    return players.find((p) => p.id === playerId)?.username || 'Unknown';
  };

  return (
    <div className={styles.container}>
      <AnimatePresence mode="popLayout">
        {trickCards.length === 0 ? (
          <motion.div
            key="empty"
            className={styles.empty}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            {t('game.waiting')}
          </motion.div>
        ) : (
          <div className={styles.cards}>
            {trickCards.map((tc, index) => (
              <motion.div
                key={`${tc.player_id}-${tc.card_id}`}
                className={`${styles.cardSlot} ${trickWinner?.playerId === tc.player_id ? styles.winner : ''}`}
                initial={{ opacity: 0, scale: 0.8, y: -30 }}
                animate={{ opacity: 1, scale: 1, y: 0 }}
                exit={{ opacity: 0, scale: 0.8 }}
                transition={{ delay: index * 0.1, type: 'spring', stiffness: 300 }}
              >
                <Card cardId={tc.card_id} size="small" disabled />
                <div className={styles.playerName}>{getPlayerName(tc.player_id)}</div>
                {tc.tigress_choice && (
                  <div className={styles.tigressBadge}>
                    {tc.tigress_choice === 'pirate' ? '🏴‍☠️' : '🏳️'}
                  </div>
                )}
              </motion.div>
            ))}
          </div>
        )}
      </AnimatePresence>

      {/* Winner announcement */}
      <AnimatePresence>
        {trickWinner && (
          <motion.div
            className={styles.winnerBanner}
            initial={{ opacity: 0, scale: 0.8 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.8 }}
          >
            {t('game.wonTrick', { player: trickWinner.playerName })}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export default TrickArea;
