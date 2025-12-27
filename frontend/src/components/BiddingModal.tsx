import { useState } from 'react';
import { motion } from 'framer-motion';
import { useTranslation } from 'react-i18next';
import { Modal } from './Modal';
import { Card } from './Card';
import { useGameStore } from '../stores/gameStore';
import styles from './BiddingModal.module.css';

export function BiddingModal() {
  const { t } = useTranslation();
  const showBidding = useGameStore((s) => s.showBidding);
  const hand = useGameStore((s) => s.hand);
  const currentRound = useGameStore((s) => s.currentRound);
  const placeBid = useGameStore((s) => s.placeBid);

  const [selectedBid, setSelectedBid] = useState<number | null>(null);

  const maxBid = currentRound;
  const bidOptions = Array.from({ length: maxBid + 1 }, (_, i) => i);

  const handleConfirm = () => {
    if (selectedBid !== null) {
      placeBid(selectedBid);
      setSelectedBid(null);
    }
  };

  return (
    <Modal isOpen={showBidding} title={t('bidding.title')} icon="🎯" closable={false}>
      <p className={styles.question}>{t('bidding.question')}</p>

      {/* Hand preview */}
      <div className={styles.handPreview}>
        {hand.map((cardId, index) => (
          <div key={cardId} className={styles.previewCard}>
            <Card cardId={cardId} size="small" disabled delay={index * 0.03} />
          </div>
        ))}
      </div>

      {/* Bid options */}
      <div className={styles.bidGrid}>
        {bidOptions.map((bid, index) => (
          <motion.button
            key={bid}
            className={`${styles.bidButton} ${selectedBid === bid ? styles.selected : ''}`}
            onClick={() => setSelectedBid(bid)}
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ delay: index * 0.03, type: 'spring', stiffness: 300 }}
            whileHover={{ scale: 1.05 }}
            whileTap={{ scale: 0.95 }}
          >
            {bid}
          </motion.button>
        ))}
      </div>

      {/* Confirm button */}
      <motion.button
        className={styles.confirmButton}
        onClick={handleConfirm}
        disabled={selectedBid === null}
        whileHover={selectedBid !== null ? { scale: 1.02 } : undefined}
        whileTap={selectedBid !== null ? { scale: 0.98 } : undefined}
      >
        {selectedBid !== null
          ? `${t('bidding.confirm')} (${selectedBid} ${selectedBid === 1 ? t('bidding.trick') : t('bidding.tricks')})`
          : t('bidding.confirm')}
      </motion.button>
    </Modal>
  );
}

export default BiddingModal;
