import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { useTranslation } from 'react-i18next';
import {
  Hand,
  TrickArea,
  Scoreboard,
  SettingsButton,
  BiddingModal,
  TigressModal,
  AbilityModal,
} from '../components';
import { useGameStore } from '../stores/gameStore';
import styles from './Game.module.css';

interface GameProps {
  onLeave: () => void;
}

export function Game({ onLeave }: GameProps) {
  const { t } = useTranslation();
  const [showMobileScoreboard, setShowMobileScoreboard] = useState(false);
  const logRef = useRef<HTMLDivElement>(null);

  const phase = useGameStore((s) => s.phase);
  const currentRound = useGameStore((s) => s.currentRound);
  const playerId = useGameStore((s) => s.playerId);
  const pickingPlayerId = useGameStore((s) => s.pickingPlayerId);
  const players = useGameStore((s) => s.players);
  const isSpectator = useGameStore((s) => s.isSpectator);
  const logs = useGameStore((s) => s.logs);
  const showResults = useGameStore((s) => s.showResults);
  const playCard = useGameStore((s) => s.playCard);

  const isMyTurn = pickingPlayerId === playerId;

  // Auto-scroll log
  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [logs]);

  // Get winner for results
  const sortedPlayers = [...players].sort((a, b) => b.score - a.score);
  const winner = sortedPlayers[0];
  const isWinner = winner?.id === playerId;

  const formatTime = (timestamp: number) => {
    const date = new Date(timestamp);
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  };

  const handleCardClick = (cardId: number) => {
    if (phase === 'PICKING' && isMyTurn && !isSpectator) {
      playCard(cardId);
    }
  };

  const getPhaseText = () => {
    switch (phase) {
      case 'BIDDING':
        return t('game.biddingPhase');
      case 'PICKING':
        return t('game.playingPhase');
      default:
        return phase;
    }
  };

  return (
    <div className={styles.container}>
      <SettingsButton className={styles.settingsButton} />

      {/* Header */}
      <header className={styles.header}>
        <div className={styles.roundInfo}>
          <span className={styles.roundText}>
            {t('game.round', { current: currentRound })}
          </span>
          <span className={styles.phaseText}>{getPhaseText()}</span>
        </div>

        {isMyTurn && phase === 'PICKING' && !isSpectator && (
          <motion.div
            className={styles.turnIndicator}
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ type: 'spring', stiffness: 300 }}
          >
            {t('game.yourTurn')}
          </motion.div>
        )}
      </header>

      {/* Main area */}
      <main className={styles.mainArea}>
        {/* Center area */}
        <div className={styles.centerArea}>
          {/* Trick area */}
          <div className={styles.trickSection}>
            <TrickArea />
          </div>

          {/* Game log */}
          <div className={styles.logSection}>
            <div className={styles.logHeader}>Game Log</div>
            <div className={styles.logContent} ref={logRef}>
              {logs.map((entry) => (
                <div key={entry.id} className={styles.logEntry}>
                  <span className={styles.logTime}>{formatTime(entry.timestamp)}</span>
                  {entry.message}
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Sidebar */}
        <aside className={styles.sidebar}>
          <div className={styles.scoreboardWrapper}>
            <Scoreboard />
          </div>

          {isSpectator && (
            <div className={styles.spectatorBadge}>
              👁️ {t('game.spectator')}
            </div>
          )}
        </aside>
      </main>

      {/* Mobile scoreboard toggle */}
      <button
        className={styles.mobileScoreToggle}
        onClick={() => setShowMobileScoreboard(true)}
      >
        📊
      </button>

      {/* Mobile scoreboard overlay */}
      <AnimatePresence>
        {showMobileScoreboard && (
          <motion.div
            className={styles.mobileScoreboard}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setShowMobileScoreboard(false)}
          >
            <motion.div
              className={styles.mobileScoreContent}
              initial={{ scale: 0.9, y: 20 }}
              animate={{ scale: 1, y: 0 }}
              exit={{ scale: 0.9, y: 20 }}
              onClick={(e) => e.stopPropagation()}
            >
              <Scoreboard />
              <button
                className={styles.closeScoreboard}
                onClick={() => setShowMobileScoreboard(false)}
              >
                {t('common.close')}
              </button>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Hand area */}
      {!isSpectator && (
        <div className={styles.handArea}>
          <Hand onCardClick={handleCardClick} disabled={phase !== 'PICKING'} />
        </div>
      )}

      {/* Modals */}
      <BiddingModal />
      <TigressModal />
      <AbilityModal />

      {/* Results overlay */}
      <AnimatePresence>
        {showResults && winner && (
          <motion.div
            className={styles.resultsOverlay}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <motion.div
              className={styles.resultsContent}
              initial={{ scale: 0.8, y: 30 }}
              animate={{ scale: 1, y: 0 }}
              exit={{ scale: 0.8, y: 30 }}
              transition={{ type: 'spring', stiffness: 200 }}
            >
              <h1 className={styles.resultsTitle}>{t('results.title')}</h1>
              <p className={styles.resultsWinner}>
                <span className="trophy">🏆</span>
                {isWinner
                  ? t('results.youWon')
                  : t('results.winner', { player: winner.username })}
              </p>
              <div className={styles.resultsScoreboard}>
                <Scoreboard />
              </div>
              <button className={styles.playAgainButton} onClick={onLeave}>
                {t('results.playAgain')}
              </button>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

export default Game;
