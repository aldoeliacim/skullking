import { useState } from 'react';
import { motion } from 'framer-motion';
import { useTranslation } from 'react-i18next';
import { Modal } from './Modal';
import { changeLanguage, getCurrentLanguage } from '../i18n';
import styles from './Settings.module.css';

interface SettingsButtonProps {
  className?: string;
}

export function SettingsButton({ className = '' }: SettingsButtonProps) {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <>
      <motion.button
        className={`${styles.settingsButton} ${className}`}
        onClick={() => setIsOpen(true)}
        whileHover={{ scale: 1.1 }}
        whileTap={{ scale: 0.9 }}
        aria-label="Settings"
      >
        ⚙️
      </motion.button>

      <SettingsModal isOpen={isOpen} onClose={() => setIsOpen(false)} />
    </>
  );
}

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
}

function SettingsModal({ isOpen, onClose }: SettingsModalProps) {
  const { t } = useTranslation();
  const currentLang = getCurrentLanguage();
  const [sound, setSound] = useState(true);
  const [vibration, setVibration] = useState(true);

  const handleLanguageChange = (lang: 'en' | 'es') => {
    changeLanguage(lang);
  };

  return (
    <Modal isOpen={isOpen} onClose={onClose} title={t('settings.title')} icon="⚙️">
      <div className={styles.settings}>
        {/* Language */}
        <div className={styles.settingRow}>
          <span className={styles.settingLabel}>{t('settings.language')}</span>
          <div className={styles.toggleGroup}>
            <button
              className={`${styles.toggleButton} ${currentLang === 'en' ? styles.active : ''}`}
              onClick={() => handleLanguageChange('en')}
            >
              🇺🇸 English
            </button>
            <button
              className={`${styles.toggleButton} ${currentLang === 'es' ? styles.active : ''}`}
              onClick={() => handleLanguageChange('es')}
            >
              🇪🇸 Español
            </button>
          </div>
        </div>

        {/* Sound */}
        <div className={styles.settingRow}>
          <span className={styles.settingLabel}>{t('settings.sound')}</span>
          <div className={styles.toggleGroup}>
            <button
              className={`${styles.toggleButton} ${sound ? styles.active : ''}`}
              onClick={() => setSound(true)}
            >
              {t('settings.on')}
            </button>
            <button
              className={`${styles.toggleButton} ${!sound ? styles.active : ''}`}
              onClick={() => setSound(false)}
            >
              {t('settings.off')}
            </button>
          </div>
        </div>

        {/* Vibration */}
        <div className={styles.settingRow}>
          <span className={styles.settingLabel}>{t('settings.vibration')}</span>
          <div className={styles.toggleGroup}>
            <button
              className={`${styles.toggleButton} ${vibration ? styles.active : ''}`}
              onClick={() => setVibration(true)}
            >
              {t('settings.on')}
            </button>
            <button
              className={`${styles.toggleButton} ${!vibration ? styles.active : ''}`}
              onClick={() => setVibration(false)}
            >
              {t('settings.off')}
            </button>
          </div>
        </div>

        {/* Version */}
        <div className={styles.version}>v1.0.0</div>
      </div>
    </Modal>
  );
}

export default SettingsButton;
