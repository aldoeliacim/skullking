import { motion } from 'framer-motion';
import { useTranslation } from 'react-i18next';
import { useGameStore } from '../stores/gameStore';
import styles from './Scoreboard.module.css';

interface ScoreboardProps {
  compact?: boolean;
}

export function Scoreboard({ compact = false }: ScoreboardProps) {
  const { t } = useTranslation();
  const players = useGameStore((s) => s.players);
  const playerId = useGameStore((s) => s.playerId);
  const lootAlliances = useGameStore((s) => s.lootAlliances);

  // Sort by score descending
  const sortedPlayers = [...players].sort((a, b) => b.score - a.score);

  return (
    <div className={`${styles.container} ${compact ? styles.compact : ''}`}>
      {!compact && <h3 className={styles.title}>{t('game.score')}</h3>}

      <div className={styles.table}>
        <div className={styles.header}>
          <span className={styles.rank}>#</span>
          <span className={styles.name}>Player</span>
          <span className={styles.stat}>{t('game.bid')}</span>
          <span className={styles.stat}>{t('game.tricks')}</span>
          <span className={styles.score}>{t('game.score')}</span>
        </div>

        {sortedPlayers.map((player, index) => {
          const isMe = player.id === playerId;
          const isAllied = Object.values(lootAlliances).includes(player.id) ||
            Object.keys(lootAlliances).includes(player.id);
          const bidMade = player.bid !== null && player.tricks_won === player.bid;
          const bidOver = player.bid !== null && player.tricks_won > player.bid;

          return (
            <motion.div
              key={player.id}
              className={`${styles.row} ${isMe ? styles.me : ''} ${isAllied ? styles.allied : ''}`}
              initial={{ opacity: 0, x: -20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: index * 0.05 }}
            >
              <span className={styles.rank}>
                {index === 0 && '🥇'}
                {index === 1 && '🥈'}
                {index === 2 && '🥉'}
                {index > 2 && index + 1}
              </span>
              <span className={styles.name}>
                {player.username}
                {player.is_bot && ' 🤖'}
                {isMe && <span className={styles.youBadge}>{t('lobby.you')}</span>}
              </span>
              <span className={`${styles.stat} ${bidMade ? styles.bidMade : bidOver ? styles.bidOver : ''}`}>
                {player.bid ?? '-'}
              </span>
              <span className={styles.stat}>{player.tricks_won}</span>
              <motion.span
                className={styles.score}
                key={player.score}
                initial={{ scale: 1.2 }}
                animate={{ scale: 1 }}
              >
                {player.score}
              </motion.span>
            </motion.div>
          );
        })}
      </div>
    </div>
  );
}

export default Scoreboard;
