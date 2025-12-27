import { useState, useCallback, useEffect } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import { Home, Lobby, Game } from './pages';
import { useGameStore } from './stores/gameStore';
import './styles/theme.css';

type Screen = 'home' | 'lobby' | 'game';

export function App() {
  const [screen, setScreen] = useState<Screen>('home');
  const connect = useGameStore((s) => s.connect);
  const disconnect = useGameStore((s) => s.disconnect);
  const reset = useGameStore((s) => s.reset);
  const phase = useGameStore((s) => s.phase);

  // Update screen based on game phase
  useEffect(() => {
    if (phase === 'BIDDING' || phase === 'PICKING') {
      setScreen('game');
    } else if (phase === 'ENDED') {
      // Stay on game screen to show results
    }
  }, [phase]);

  const handleCreateGame = useCallback(
    (gameId: string, playerId: string, playerName: string) => {
      connect(gameId, playerId, playerName);
      setScreen('lobby');
    },
    [connect]
  );

  const handleJoinGame = useCallback(
    (gameId: string, playerName: string) => {
      const playerId = `player_${Date.now()}`;
      connect(gameId, playerId, playerName);
      setScreen('lobby');
    },
    [connect]
  );

  const handleSpectate = useCallback(
    (gameId: string) => {
      const spectatorId = `spectator_${Date.now()}`;
      connect(gameId, spectatorId, 'Spectator', true);
      setScreen('game');
    },
    [connect]
  );

  const handleGameStart = useCallback(() => {
    setScreen('game');
  }, []);

  const handleLeave = useCallback(() => {
    disconnect();
    reset();
    setScreen('home');
  }, [disconnect, reset]);

  return (
    <AnimatePresence mode="wait">
      {screen === 'home' && (
        <motion.div
          key="home"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.3 }}
        >
          <Home
            onCreateGame={handleCreateGame}
            onJoinGame={handleJoinGame}
            onSpectate={handleSpectate}
          />
        </motion.div>
      )}

      {screen === 'lobby' && (
        <motion.div
          key="lobby"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.3 }}
        >
          <Lobby onLeave={handleLeave} onGameStart={handleGameStart} />
        </motion.div>
      )}

      {screen === 'game' && (
        <motion.div
          key="game"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.3 }}
        >
          <Game onLeave={handleLeave} />
        </motion.div>
      )}
    </AnimatePresence>
  );
}

export default App;
