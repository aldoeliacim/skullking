import { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { useTranslation } from 'react-i18next';
import { Button, SettingsButton } from '../components';
import { useGameStore } from '../stores/gameStore';
import styles from './Lobby.module.css';

interface LobbyProps {
  onLeave: () => void;
  onGameStart: () => void;
}

export function Lobby({ onLeave, onGameStart }: LobbyProps) {
  const { t } = useTranslation();
  const [copied, setCopied] = useState(false);

  const gameId = useGameStore((s) => s.gameId);
  const playerId = useGameStore((s) => s.playerId);
  const players = useGameStore((s) => s.players);
  const isHost = useGameStore((s) => s.isHost);
  const phase = useGameStore((s) => s.phase);
  const connectionState = useGameStore((s) => s.connectionState);
  const addBot = useGameStore((s) => s.addBot);
  const removeBot = useGameStore((s) => s.removeBot);
  const clearBots = useGameStore((s) => s.clearBots);
  const startGame = useGameStore((s) => s.startGame);

  // Watch for game start
  useEffect(() => {
    if (phase === 'BIDDING' || phase === 'PICKING') {
      onGameStart();
    }
  }, [phase, onGameStart]);

  const handleCopyCode = async () => {
    if (gameId) {
      await navigator.clipboard.writeText(gameId.slice(0, 4).toUpperCase());
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const handleAddBot = (difficulty: 'easy' | 'medium' | 'hard') => {
    const typeMap = { easy: 'random', medium: 'rule_based', hard: 'rl' } as const;
    addBot(typeMap[difficulty], difficulty);
  };

  const handleFillBots = () => {
    const needed = 4 - players.length;
    for (let i = 0; i < needed; i++) {
      addBot('rl', 'hard');
    }
  };

  const canStart = players.length >= 2 && isHost;
  const gameCode = gameId?.slice(0, 4).toUpperCase() || '----';

  return (
    <div className={styles.container}>
      <SettingsButton className={styles.settingsButton} />

      <motion.div
        className={styles.content}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        {/* Header */}
        <div className={styles.header}>
          <h1 className={styles.title}>{t('lobby.title')}</h1>
          <div className={`${styles.connectionBadge} ${styles[connectionState]}`}>
            {connectionState}
          </div>
        </div>

        {/* Game Code */}
        <motion.div
          className={styles.codeSection}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
        >
          <p className={styles.codeLabel}>{t('lobby.shareCode')}</p>
          <div className={styles.codeRow}>
            <span className={styles.code}>{gameCode}</span>
            <Button size="sm" variant="outline" onClick={handleCopyCode}>
              {copied ? t('lobby.copied') : t('lobby.copy')}
            </Button>
          </div>
        </motion.div>

        {/* Players */}
        <motion.div
          className={styles.playersSection}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
        >
          <h2 className={styles.sectionTitle}>{t('lobby.players')}</h2>
          <div className={styles.playerList}>
            {players.map((player, index) => (
              <motion.div
                key={player.id}
                className={`${styles.playerCard} ${player.id === playerId ? styles.me : ''}`}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.05 }}
              >
                <span className={styles.playerIcon}>
                  {player.is_bot ? '🤖' : '👤'}
                </span>
                <span className={styles.playerName}>
                  {player.username}
                  {player.id === playerId && (
                    <span className={styles.youBadge}>{t('lobby.you')}</span>
                  )}
                  {index === 0 && !player.is_bot && (
                    <span className={styles.hostBadge}>{t('lobby.host')}</span>
                  )}
                </span>
                {player.is_bot && isHost && (
                  <button
                    className={styles.removeButton}
                    onClick={() => removeBot(player.id)}
                  >
                    ✕
                  </button>
                )}
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* Bot Controls (Host Only) */}
        {isHost && (
          <motion.div
            className={styles.botSection}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.3 }}
          >
            <h2 className={styles.sectionTitle}>{t('lobby.addBot')}</h2>
            <div className={styles.botButtons}>
              <Button size="sm" variant="outline" onClick={() => handleAddBot('easy')}>
                {t('lobby.botEasy')}
              </Button>
              <Button size="sm" variant="outline" onClick={() => handleAddBot('medium')}>
                {t('lobby.botMedium')}
              </Button>
              <Button size="sm" variant="outline" onClick={() => handleAddBot('hard')}>
                {t('lobby.botHard')}
              </Button>
            </div>
            <div className={styles.botActions}>
              <Button size="sm" variant="secondary" onClick={handleFillBots}>
                {t('lobby.fillBots')}
              </Button>
              <Button size="sm" variant="secondary" onClick={clearBots}>
                {t('lobby.clearBots')}
              </Button>
            </div>
          </motion.div>
        )}

        {/* Actions */}
        <motion.div
          className={styles.actions}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
        >
          {isHost ? (
            <>
              <Button
                size="lg"
                onClick={startGame}
                disabled={!canStart}
                fullWidth
              >
                {canStart ? t('lobby.startGame') : t('lobby.needPlayers')}
              </Button>
            </>
          ) : (
            <p className={styles.waitingText}>{t('lobby.waitingHost')}</p>
          )}
          <Button size="lg" variant="danger" onClick={onLeave} fullWidth>
            {t('lobby.leave')}
          </Button>
        </motion.div>
      </motion.div>
    </div>
  );
}

export default Lobby;
