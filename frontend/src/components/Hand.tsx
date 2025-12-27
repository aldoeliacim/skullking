import { useMemo } from 'react';
import { motion } from 'framer-motion';
import { Card } from './Card';
import { getValidCardIds } from '../utils/cardUtils';
import { useGameStore } from '../stores/gameStore';
import styles from './Hand.module.css';

interface HandProps {
  onCardClick?: (cardId: number) => void;
  disabled?: boolean;
}

export function Hand({ onCardClick, disabled = false }: HandProps) {
  const hand = useGameStore((s) => s.hand);
  const trickCards = useGameStore((s) => s.trickCards);
  const pickingPlayerId = useGameStore((s) => s.pickingPlayerId);
  const playerId = useGameStore((s) => s.playerId);

  const isMyTurn = pickingPlayerId === playerId;
  const validCards = useMemo(() => getValidCardIds(hand, trickCards), [hand, trickCards]);

  if (hand.length === 0) {
    return (
      <div className={styles.container}>
        <div className={styles.empty}>Waiting for cards...</div>
      </div>
    );
  }

  return (
    <motion.div
      className={styles.container}
      initial={{ opacity: 0, y: 30 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.2 }}
    >
      <div className={styles.cards}>
        {hand.map((cardId, index) => {
          const isValid = validCards.includes(cardId);
          const isDisabled = disabled || !isMyTurn || !isValid;

          return (
            <div key={cardId} className={styles.cardWrapper} style={{ zIndex: index }}>
              <Card
                cardId={cardId}
                disabled={isDisabled}
                onClick={() => !isDisabled && onCardClick?.(cardId)}
                delay={index * 0.05}
              />
            </div>
          );
        })}
      </div>
      <div className={styles.label}>
        {hand.length} {hand.length === 1 ? 'card' : 'cards'}
      </div>
    </motion.div>
  );
}

export default Hand;
