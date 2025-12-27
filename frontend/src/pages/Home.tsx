import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useTranslation } from 'react-i18next';
import { Button, SettingsButton, Modal } from '../components';
import { api } from '../services/api';
import type { GameInfo } from '../types/game';
import styles from './Home.module.css';

const PLAYER_NAME_KEY = 'skullking_playerName';

interface HomeProps {
  onCreateGame: (gameId: string, playerId: string, playerName: string) => void;
  onJoinGame: (gameId: string, playerName: string) => void;
  onSpectate: (gameId: string) => void;
}

export function Home({ onCreateGame, onJoinGame, onSpectate }: HomeProps) {
  const { t } = useTranslation();
  const [playerName, setPlayerName] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showBrowse, setShowBrowse] = useState(false);

  // Load saved name
  useEffect(() => {
    const saved = localStorage.getItem(PLAYER_NAME_KEY);
    if (saved) setPlayerName(saved);
  }, []);

  const handleNameChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const name = e.target.value;
    setPlayerName(name);
    setError(null);
    localStorage.setItem(PLAYER_NAME_KEY, name);
  };

  const handleCreateGame = async () => {
    if (!playerName.trim()) {
      setError(t('home.errorEnterName'));
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const response = await api.createGame();
      const playerId = `player_${Date.now()}`;
      onCreateGame(response.game_id, playerId, playerName.trim());
    } catch (err) {
      setError(err instanceof Error ? err.message : t('home.errorCreateFailed'));
    } finally {
      setIsLoading(false);
    }
  };

  const handleSelectGame = (game: GameInfo, spectate: boolean) => {
    setShowBrowse(false);
    if (spectate) {
      onSpectate(game.slug);
    } else {
      if (!playerName.trim()) {
        setError(t('home.errorEnterName'));
        return;
      }
      onJoinGame(game.slug, playerName.trim());
    }
  };

  return (
    <div className={styles.container}>
      <SettingsButton className={styles.settingsButton} />

      <motion.div
        className={styles.content}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        {/* Header */}
        <div className={styles.header}>
          <span className={styles.emoji}>🏴‍☠️</span>
          <h1 className={styles.title}>{t('app.title')}</h1>
          <p className={styles.subtitle}>{t('app.subtitle')}</p>
        </div>

        {/* Form */}
        <div className={styles.form}>
          <label className={styles.label}>{t('home.yourName')}</label>
          <input
            type="text"
            className={styles.input}
            placeholder={t('home.enterName')}
            value={playerName}
            onChange={handleNameChange}
            maxLength={20}
            autoComplete="off"
          />

          <div className={styles.buttonRow}>
            <Button
              variant="primary"
              size="lg"
              onClick={handleCreateGame}
              loading={isLoading}
              fullWidth
            >
              {t('home.createGame')}
            </Button>
            <Button
              variant="secondary"
              size="lg"
              onClick={() => setShowBrowse(true)}
              fullWidth
            >
              {t('home.browseGames')}
            </Button>
          </div>

          {error && (
            <motion.div
              className={styles.error}
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
            >
              {error}
            </motion.div>
          )}
        </div>
      </motion.div>

      {/* Browse Modal */}
      <BrowseModal
        isOpen={showBrowse}
        onClose={() => setShowBrowse(false)}
        onSelectGame={handleSelectGame}
      />
    </div>
  );
}

interface BrowseModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSelectGame: (game: GameInfo, spectate: boolean) => void;
}

function BrowseModal({ isOpen, onClose, onSelectGame }: BrowseModalProps) {
  const { t } = useTranslation();
  const [games, setGames] = useState<GameInfo[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (isOpen) {
      setLoading(true);
      api.getActiveGames()
        .then(setGames)
        .catch(() => {})
        .finally(() => setLoading(false));
    }
  }, [isOpen]);

  return (
    <Modal isOpen={isOpen} onClose={onClose} title={t('browse.title')} icon="🎮">
      <div className={styles.browseContent}>
        {loading ? (
          <p className={styles.browseEmpty}>{t('browse.loading')}</p>
        ) : games.length === 0 ? (
          <p className={styles.browseEmpty}>{t('browse.noGames')}</p>
        ) : (
          <AnimatePresence>
            {games.map((game, index) => (
              <motion.div
                key={game.id}
                className={styles.gameCard}
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.05 }}
              >
                <div className={styles.gameInfo}>
                  <span className={styles.gameCode}>{game.slug}</span>
                  <span className={styles.gamePlayers}>
                    {game.players.map((p) => p.username).join(', ')}
                  </span>
                  <span className={styles.gameState}>
                    {game.state === 'PENDING' ? t('browse.inLobby') : t('browse.inProgress')}
                  </span>
                </div>
                <div className={styles.gameActions}>
                  {game.state === 'PENDING' ? (
                    <Button size="sm" onClick={() => onSelectGame(game, false)}>
                      {t('browse.join')}
                    </Button>
                  ) : (
                    <Button size="sm" variant="outline" onClick={() => onSelectGame(game, true)}>
                      {t('browse.watch')}
                    </Button>
                  )}
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
        )}
      </div>
    </Modal>
  );
}

export default Home;
