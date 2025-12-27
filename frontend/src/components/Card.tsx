import { motion } from 'framer-motion';
import { parseCard } from '../utils/cardUtils';
import { API_BASE_URL } from '../services/api';
import styles from './Card.module.css';

interface CardProps {
  cardId: number;
  size?: 'small' | 'medium' | 'large';
  disabled?: boolean;
  selected?: boolean;
  faceDown?: boolean;
  onClick?: () => void;
  showNumber?: boolean;
  delay?: number;
}

const CARD_IMAGE_BASE = `${API_BASE_URL}/static/images/cards/`;

export function Card({
  cardId,
  size = 'medium',
  disabled = false,
  selected = false,
  faceDown = false,
  onClick,
  showNumber = true,
  delay = 0,
}: CardProps) {
  const card = parseCard(cardId);

  if (!card) {
    return null;
  }

  const imageSrc = faceDown ? `${CARD_IMAGE_BASE}back.png` : `${CARD_IMAGE_BASE}${card.image}`;

  return (
    <motion.div
      className={`${styles.card} ${styles[size]} ${disabled ? styles.disabled : ''} ${selected ? styles.selected : ''}`}
      onClick={disabled ? undefined : onClick}
      initial={{ opacity: 0, y: 20, scale: 0.8 }}
      animate={{
        opacity: 1,
        y: selected ? -10 : 0,
        scale: 1,
      }}
      transition={{
        delay,
        type: 'spring',
        stiffness: 300,
        damping: 20,
      }}
      whileHover={disabled ? undefined : { y: -8, transition: { duration: 0.15 } }}
      whileTap={disabled ? undefined : { scale: 0.95 }}
    >
      <img src={imageSrc} alt={card.name} className={styles.image} draggable={false} />

      {/* Number overlay for suit cards */}
      {showNumber && card.number && !faceDown && (
        <div className={styles.number}>{card.number}</div>
      )}

      {/* Bonus badge for 14s */}
      {card.number === 14 && !faceDown && (
        <div className={styles.bonus}>+10</div>
      )}
    </motion.div>
  );
}

export default Card;
