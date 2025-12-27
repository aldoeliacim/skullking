import { motion } from 'framer-motion';
import { useTranslation } from 'react-i18next';
import { Modal } from './Modal';
import { useGameStore } from '../stores/gameStore';
import styles from './TigressModal.module.css';

export function TigressModal() {
  const { t } = useTranslation();
  const showTigress = useGameStore((s) => s.showTigress);
  const confirmTigress = useGameStore((s) => s.confirmTigress);
  const cancelTigress = useGameStore((s) => s.cancelTigress);

  return (
    <Modal
      isOpen={showTigress}
      onClose={cancelTigress}
      title={t('tigress.title')}
      icon="🎭"
    >
      <p className={styles.question}>{t('tigress.question')}</p>

      <div className={styles.choices}>
        <motion.button
          className={`${styles.choiceButton} ${styles.pirate}`}
          onClick={() => confirmTigress('pirate')}
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
        >
          <span className={styles.choiceIcon}>🏴‍☠️</span>
          <span className={styles.choiceLabel}>{t('tigress.pirate')}</span>
          <span className={styles.choiceDesc}>{t('tigress.pirateDesc')}</span>
        </motion.button>

        <motion.button
          className={`${styles.choiceButton} ${styles.escape}`}
          onClick={() => confirmTigress('escape')}
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
        >
          <span className={styles.choiceIcon}>🏳️</span>
          <span className={styles.choiceLabel}>{t('tigress.escape')}</span>
          <span className={styles.choiceDesc}>{t('tigress.escapeDesc')}</span>
        </motion.button>
      </div>
    </Modal>
  );
}

export default TigressModal;
