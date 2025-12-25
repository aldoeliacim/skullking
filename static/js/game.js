// Skull King Game Client
class SkullKingGame {
    constructor() {
        this.ws = null;
        this.gameId = null;
        this.playerId = null;
        this.username = null;
        this.gameState = null;
        this.isHost = false;
        this.isSpectator = false;
        this.selectedCardId = null; // For tap-to-confirm on mobile
        this.isTouchDevice = this.detectTouchDevice();

        // WebSocket reconnection state
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000; // Start with 1 second
        this.reconnectTimer = null;
        this.intentionalClose = false;

        this.initializeEventListeners();
        this.initializeI18n();
    }

    // Detect if device uses touch as primary input
    detectTouchDevice() {
        return window.matchMedia('(hover: none) and (pointer: coarse)').matches;
    }

    async initializeI18n() {
        await window.i18n.init();
        this.updateLangFlag();

        window.addEventListener('localeChanged', () => {
            this.updateLangFlag();
            this.updateDynamicContent();
        });
    }

    initializeSounds() {
        // Initialize sound manager on first interaction
        if (window.soundManager) {
            window.soundManager.init();
            this.updateSoundIcon();
        }
    }

    toggleSound() {
        if (window.soundManager) {
            window.soundManager.toggle();
            this.updateSoundIcon();
            window.soundManager.click();
        }
    }

    updateSoundIcon() {
        const enabled = window.soundManager?.enabled;
        const icon = enabled ? '🔊' : '🔇';

        const soundIcon = document.getElementById('sound-icon');
        const soundIconGame = document.getElementById('sound-icon-game');
        const soundBtn = document.getElementById('sound-toggle');
        const soundBtnGame = document.getElementById('sound-toggle-game');

        if (soundIcon) soundIcon.textContent = icon;
        if (soundIconGame) soundIconGame.textContent = icon;
        if (soundBtn) soundBtn.classList.toggle('muted', !enabled);
        if (soundBtnGame) soundBtnGame.classList.toggle('muted', !enabled);
    }

    playSound(soundName) {
        if (window.soundManager) {
            window.soundManager[soundName]?.();
        }
    }

    playCardSound(cardId) {
        if (!window.soundManager) return;

        // Special card sounds based on card ID
        // Card IDs: 1=Skull King, 2=White Whale, 3=Kraken, 4-5=Mermaids, 6-10=Pirates
        // 67-71=Escapes, 72=Tigress, 73-74=Loot
        if (cardId === 1) {
            window.soundManager.skullKing();
        } else if (cardId === 3) {
            window.soundManager.kraken();
        } else if (cardId >= 4 && cardId <= 10) {
            // Pirates and mermaids
            window.soundManager.specialCard();
        } else if (cardId === 72) {
            // Tigress
            window.soundManager.specialCard();
        } else {
            window.soundManager.cardPlay();
        }
    }

    playDealSounds(cardCount) {
        if (!window.soundManager) return;

        // Play staggered card deal sounds
        for (let i = 0; i < Math.min(cardCount, 10); i++) {
            setTimeout(() => {
                window.soundManager.cardDeal();
            }, i * 60);
        }
    }

    updateLangFlag() {
        const locale = window.i18n.getLocale().toUpperCase();
        // Update both login screen and game screen language flags
        const flag = document.getElementById('lang-flag');
        const flagGame = document.getElementById('lang-flag-game');
        if (flag) {
            flag.textContent = locale;
        }
        if (flagGame) {
            flagGame.textContent = locale;
        }
    }

    updateDynamicContent() {
        // Only update game screen if we're actually in a game (not lobby)
        if (this.gameState && this.isGameInProgress()) {
            this.updateGameScreen();
        } else if (this.gameState) {
            // Update lobby if we have game state but game hasn't started
            this.updateLobby();
        }
    }

    isGameInProgress() {
        // Game is in progress if state is not PENDING/LOBBY
        const state = this.gameState?.state;
        return state && state !== 'PENDING' && state !== 'LOBBY';
    }

    initializeEventListeners() {
        // Login screen
        document.getElementById('create-game-btn').addEventListener('click', () => this.createGame());
        document.getElementById('join-game-btn').addEventListener('click', () => this.joinGame());
        document.getElementById('spectate-game-btn').addEventListener('click', () => this.spectateGame());
        document.getElementById('username-input').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.createGame();
        });
        document.getElementById('game-id-input').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.joinGame();
        });

        // Lobby screen
        document.getElementById('add-bot-btn').addEventListener('click', () => this.addBot());
        document.getElementById('fill-bots-btn').addEventListener('click', () => this.fillWithBots());
        document.getElementById('clear-bots-btn').addEventListener('click', () => this.clearBots());
        document.getElementById('start-game-btn').addEventListener('click', () => this.startGame());
        document.getElementById('copy-game-id-btn').addEventListener('click', () => this.copyGameId());
        document.getElementById('leave-lobby-btn').addEventListener('click', () => this.leaveLobby());

        // Results screen
        document.getElementById('new-game-btn').addEventListener('click', () => this.returnToLogin());

        // Language switcher (both login and game screens)
        document.getElementById('lang-toggle')?.addEventListener('click', () => {
            window.i18n.toggleLocale();
            this.updateLangFlag();
        });
        document.getElementById('lang-toggle-game')?.addEventListener('click', () => {
            window.i18n.toggleLocale();
            this.updateLangFlag();
        });

        // Sound toggle (both login and game screens)
        document.getElementById('sound-toggle')?.addEventListener('click', () => {
            this.initializeSounds();
            this.toggleSound();
        });
        document.getElementById('sound-toggle-game')?.addEventListener('click', () => {
            this.initializeSounds();
            this.toggleSound();
        });

        // Initialize sounds on first user interaction
        document.addEventListener('click', () => this.initializeSounds(), { once: true });

        // Game history modal
        document.getElementById('history-btn')?.addEventListener('click', () => this.showHistory());
        document.getElementById('close-history-btn')?.addEventListener('click', () => this.hideHistory());
        document.getElementById('history-modal')?.addEventListener('click', (e) => {
            if (e.target.id === 'history-modal') this.hideHistory();
        });

        // Browse games modal
        document.getElementById('browse-games-btn')?.addEventListener('click', () => this.showBrowseGames());
        document.getElementById('close-browse-btn')?.addEventListener('click', () => this.hideBrowseGames());
        document.getElementById('refresh-games-btn')?.addEventListener('click', () => this.refreshActiveGames());
        document.getElementById('browse-games-modal')?.addEventListener('click', (e) => {
            if (e.target.id === 'browse-games-modal') this.hideBrowseGames();
        });

        // Game log toggle
        document.getElementById('log-toggle')?.addEventListener('click', () => {
            document.getElementById('game-log')?.classList.toggle('collapsed');
        });

        // Rules modal (both login and game screens)
        document.getElementById('rules-toggle')?.addEventListener('click', () => {
            document.getElementById('rules-modal')?.classList.remove('hidden');
        });
        document.getElementById('rules-toggle-game')?.addEventListener('click', () => {
            document.getElementById('rules-modal')?.classList.remove('hidden');
        });
        document.getElementById('rules-close').addEventListener('click', () => {
            document.getElementById('rules-modal').classList.add('hidden');
        });
        document.getElementById('rules-modal').addEventListener('click', (e) => {
            if (e.target.id === 'rules-modal') {
                document.getElementById('rules-modal').classList.add('hidden');
            }
        });

        // Advanced rules image zoom toggle
        const advancedRulesImg = document.getElementById('advanced-rules-img');
        if (advancedRulesImg) {
            advancedRulesImg.addEventListener('click', () => {
                if (advancedRulesImg.classList.contains('zoomed')) {
                    advancedRulesImg.classList.remove('zoomed');
                    document.querySelector('.advanced-rules-overlay')?.remove();
                } else {
                    const overlay = document.createElement('div');
                    overlay.className = 'advanced-rules-overlay';
                    overlay.addEventListener('click', () => {
                        advancedRulesImg.classList.remove('zoomed');
                        overlay.remove();
                    });
                    document.body.appendChild(overlay);
                    advancedRulesImg.classList.add('zoomed');
                }
            });
        }

        // Scoreboard toggle (legacy panel)
        document.getElementById('scoreboard-toggle')?.addEventListener('click', () => {
            document.getElementById('scoreboard-panel')?.classList.toggle('hidden');
        });
        document.getElementById('scoreboard-close')?.addEventListener('click', () => {
            document.getElementById('scoreboard-panel')?.classList.add('hidden');
        });

        // Scoreboard dropdown bar toggle
        document.getElementById('scoreboard-toggle-bar')?.addEventListener('click', () => {
            const bar = document.getElementById('scoreboard-bar');
            const dropdown = document.getElementById('scoreboard-dropdown');
            if (bar && dropdown) {
                bar.classList.toggle('expanded');
                dropdown.classList.toggle('hidden');
            }
        });

    }

    async createGame() {
        const username = document.getElementById('username-input').value.trim();

        if (!username) {
            this.showError('login', window.i18n.t('login.errorEnterName'));
            return;
        }

        this.username = username;
        this.isHost = true;

        try {
            const response = await fetch('/games', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ lobby_id: 'default' })
            });

            if (!response.ok) {
                throw new Error(window.i18n.t('login.errorCreateFailed'));
            }

            const data = await response.json();
            this.gameId = data.game_id;
            this.playerId = this.generatePlayerId();

            this.connectWebSocket();
        } catch (error) {
            this.showError('login', `${window.i18n.t('login.errorCreateFailed')}: ${error.message}`);
        }
    }

    async joinGame() {
        const username = document.getElementById('username-input').value.trim();
        const gameId = document.getElementById('game-id-input').value.trim();

        if (!username) {
            this.showError('login', window.i18n.t('login.errorEnterName'));
            return;
        }

        if (!gameId) {
            this.showError('login', window.i18n.t('login.errorEnterGameId'));
            return;
        }

        this.username = username;
        this.gameId = gameId;
        this.playerId = this.generatePlayerId();
        this.isHost = false;
        this.isSpectator = false;

        this.connectWebSocket();
    }

    spectateGame() {
        const gameId = document.getElementById('game-id-input').value.trim();

        if (!gameId) {
            this.showError('login', window.i18n.t('login.errorEnterGameId'));
            return;
        }

        this.username = 'Spectator';
        this.gameId = gameId;
        this.playerId = this.generatePlayerId();
        this.isHost = false;
        this.isSpectator = true;

        this.connectWebSocket();
    }

    connectWebSocket() {
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const endpoint = this.isSpectator ? 'spectate' : 'join';
        const idParam = this.isSpectator ? 'spectator_id' : 'player_id';
        const wsUrl = `${wsProtocol}//${window.location.host}/games/${endpoint}?game_id=${this.gameId}&${idParam}=${this.playerId}&username=${encodeURIComponent(this.username)}`;

        // Clear any existing reconnect timer
        if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
            this.reconnectTimer = null;
        }

        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
            console.log('WebSocket connected');
            // Reset reconnection state on successful connect
            this.reconnectAttempts = 0;
            this.reconnectDelay = 1000;
            this.hideReconnecting();

            if (this.isSpectator) {
                this.switchScreen('game');
                document.body.classList.add('spectator-mode');
            } else {
                this.switchScreen('lobby');
                this.updateLobby();
            }
        };

        this.ws.onmessage = (event) => {
            const message = JSON.parse(event.data);
            this.handleMessage(message);
        };

        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };

        this.ws.onclose = (event) => {
            console.log('WebSocket closed:', event.code, event.reason);

            // Don't reconnect if intentionally closed or game ended
            if (this.intentionalClose || this.gameEnded ||
                (this.gameState && this.gameState.state === 'ENDED')) {
                return;
            }

            // Attempt to reconnect
            if (this.reconnectAttempts < this.maxReconnectAttempts) {
                this.reconnectAttempts++;
                const delay = Math.min(this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1), 30000);
                console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);

                this.showReconnecting(this.reconnectAttempts, this.maxReconnectAttempts);

                this.reconnectTimer = setTimeout(() => {
                    this.connectWebSocket();
                }, delay);
            } else {
                console.log('Max reconnection attempts reached');
                this.hideReconnecting();
                this.showError('lobby', window.i18n.t('lobby.errorConnectionLost'));
            }
        };
    }

    showReconnecting(attempt, max) {
        let indicator = document.getElementById('reconnect-indicator');
        if (!indicator) {
            indicator = document.createElement('div');
            indicator.id = 'reconnect-indicator';
            indicator.className = 'reconnect-indicator';
            document.body.appendChild(indicator);
        }
        indicator.innerHTML = `&#128268; Reconnecting... (${attempt}/${max})`;
        indicator.classList.add('visible');
    }

    hideReconnecting() {
        const indicator = document.getElementById('reconnect-indicator');
        if (indicator) {
            indicator.classList.remove('visible');
        }
    }

    handleMessage(message) {
        console.log('[handleMessage] Received:', message.command, message);

        switch (message.command) {
            case 'INIT':
                this.gameState = message.content.game;
                if (message.content.is_spectator) {
                    this.isSpectator = true;
                    document.body.classList.add('spectator-mode');
                    this.updateGameScreen();
                } else {
                    this.updateLobby();
                }
                break;
            case 'JOINED':
                this.addLog(window.i18n.t('log.playerJoined', { username: message.content.username }), 'info', '&#128100;');
                break;
            case 'SPECTATOR_JOINED':
                this.updateSpectatorCount(message.content.spectator_count);
                this.addLog(window.i18n.t('log.spectatorJoined', {
                    username: message.content.username,
                    count: message.content.spectator_count
                }) || `${message.content.username} is now watching (${message.content.spectator_count} spectators)`, 'info', '&#128065;');
                break;
            case 'SPECTATOR_LEFT':
                this.updateSpectatorCount(message.content.spectator_count);
                this.addLog(window.i18n.t('log.spectatorLeft', {
                    count: message.content.spectator_count
                }) || `A spectator left (${message.content.spectator_count} spectators)`, 'info', '&#128065;');
                break;
            case 'LEFT':
            case 'PLAYER_LEFT':
                this.addLog(window.i18n.t('log.playerLeft', { username: message.content.username }));
                break;
            case 'GAME_STATE':
                this.gameState = message.content;
                // Process bids if they come as an array
                if (this.gameState.bids && Array.isArray(this.gameState.bids)) {
                    const bidsMap = {};
                    this.gameState.bids.forEach(b => {
                        if (b.bid !== undefined) {
                            bidsMap[b.player_id] = b.bid;
                        }
                    });
                    this.gameState.bids = bidsMap;
                    // Update player objects
                    if (this.gameState.players) {
                        this.gameState.players.forEach(player => {
                            if (bidsMap[player.id] !== undefined) {
                                player.bid = bidsMap[player.id];
                            }
                        });
                    }
                }
                if (this.gameState.state === 'PENDING') {
                    this.updateLobby();
                } else {
                    this.updateGameScreen();
                }
                break;
            case 'STARTED':
                this.addLog(window.i18n.t('log.gameStarted', { count: message.content.player_count }), 'round', '&#127919;');
                this.switchScreen('game');
                break;
            case 'DEAL':
                console.log('[DEAL] Received cards:', message.content.cards, 'for round:', message.content.round);
                // Store dealt cards
                this.gameState = this.gameState || {};
                this.gameState.hand = message.content.cards;
                this.gameState.current_round = message.content.round;
                // Reset bids and tricks for new round
                this.gameState.bids = {};
                this.gameState.trick_cards = [];
                if (this.gameState.players) {
                    this.gameState.players.forEach(p => {
                        p.bid = null;
                        p.tricks_won = 0;
                    });
                }
                console.log('[DEAL] Updated gameState.hand:', this.gameState.hand);
                this.updateGameScreen();
                this.playDealSounds(message.content.cards.length);
                this.addLog(window.i18n.t('log.cardsDealt', { round: message.content.round }), 'round', '&#127183;');
                break;
            case 'START_BIDDING':
                this.gameState = this.gameState || {};
                this.gameState.state = 'BIDDING';
                this.gameState.current_round = message.content.round;
                // Reset trick state for new round
                this.gameState.current_trick = 0;
                this.gameState.trick_cards = [];
                // Hide any lingering winner label from previous round
                const winnerLabelBid = document.getElementById('trick-winner');
                if (winnerLabelBid) winnerLabelBid.classList.add('hidden');
                // First update game screen so cards are visible
                this.updateGameScreen();
                // Then show bidding UI
                this.showBiddingUI(message.content);
                this.playSound('notify');
                break;
            case 'BADE':
                // Another player made their bid - update their bid value
                if (this.gameState?.players) {
                    const player = this.gameState.players.find(p => p.id === message.content.player_id);
                    if (player) {
                        player.bid = message.content.bid;
                    }
                }
                // Also store in bids map
                this.gameState = this.gameState || {};
                this.gameState.bids = this.gameState.bids || {};
                this.gameState.bids[message.content.player_id] = message.content.bid;

                const bidPlayerName = this.gameState?.players?.find(p => p.id === message.content.player_id)?.username || 'Player';
                this.addLog(window.i18n.t('log.playerBid', {
                    player: message.content.player_id === this.playerId ? 'You' : bidPlayerName,
                    bid: message.content.bid
                }), 'bid');
                this.playSound('bidPlaced');
                this.updateGameScreen();
                break;
            case 'END_BIDDING':
                // All bids are in
                this.gameState = this.gameState || {};
                this.gameState.state = 'PICKING';
                // Bids come as array: [{player_id, bid}, ...]
                // Convert to map for easy lookup
                this.gameState.bids = {};
                if (message.content.bids) {
                    message.content.bids.forEach(b => {
                        this.gameState.bids[b.player_id] = b.bid;
                    });
                }
                // Update player bid values
                if (this.gameState.players) {
                    this.gameState.players.forEach(player => {
                        const playerBid = this.gameState.bids[player.id];
                        if (playerBid !== undefined) {
                            player.bid = playerBid;
                        }
                    });
                }
                this.addLog(window.i18n.t('log.biddingComplete'), 'bid', '&#10004;');
                this.updateGameScreen();
                break;
            case 'START_PICKING':
                console.log('[START_PICKING] Picking player:', message.content.picking_player_id, 'Trick:', message.content.trick);
                this.gameState = this.gameState || {};
                this.gameState.state = 'PICKING';
                this.gameState.picking_player_id = message.content.picking_player_id;

                // Clear trick cards when starting a NEW trick (prevents race condition with animation timer)
                const newTrickNumber = message.content.trick;
                const oldTrickNumber = this.gameState.current_trick || 0;
                if (newTrickNumber > oldTrickNumber) {
                    console.log('[START_PICKING] New trick', newTrickNumber, '- clearing old trick cards');
                    this.gameState.trick_cards = [];
                    // Hide winner label if still visible
                    const winnerLabel = document.getElementById('trick-winner');
                    if (winnerLabel) winnerLabel.classList.add('hidden');
                    // Remove any collection animations
                    const trickCardsContainer = document.getElementById('trick-cards');
                    if (trickCardsContainer) {
                        trickCardsContainer.classList.remove('collecting', 'collect-top', 'collect-left', 'collect-right', 'collect-bottom');
                    }
                }
                this.gameState.current_trick = newTrickNumber;

                console.log('[START_PICKING] Is my turn:', message.content.picking_player_id === this.playerId);
                // Play your turn sound if it's our turn
                if (message.content.picking_player_id === this.playerId) {
                    this.playSound('yourTurn');
                }
                this.updateGameScreen();
                break;
            case 'PICKED':
                console.log('[PICKED] Player:', message.content.player_id, 'played card:', message.content.card_id, 'tigress_choice:', message.content.tigress_choice);
                // A card was played
                this.gameState.trick_cards = this.gameState.trick_cards || [];
                this.gameState.trick_cards.push({
                    player_id: message.content.player_id,
                    card_id: message.content.card_id,
                    tigress_choice: message.content.tigress_choice || null
                });
                // Remove from hand if it's our card
                if (message.content.player_id === this.playerId && this.gameState.hand) {
                    console.log('[PICKED] Removing card from my hand. Before:', this.gameState.hand);
                    this.gameState.hand = this.gameState.hand.filter(c => c !== message.content.card_id);
                    console.log('[PICKED] After removal:', this.gameState.hand);
                }
                // Play card sound - special sounds for special cards
                this.playCardSound(message.content.card_id);
                this.updateGameScreen();
                break;
            case 'NEXT_TRICK':
                this.gameState.picking_player_id = message.content.picking_player_id;
                // Play your turn sound if it's now our turn
                if (message.content.picking_player_id === this.playerId) {
                    this.playSound('yourTurn');
                }
                this.updateGameScreen();
                break;
            case 'ANNOUNCE_TRICK_WINNER':
                this.handleTrickWinner(message.content);
                break;
            case 'ANNOUNCE_SCORES':
                this.handleScoresAnnounced(message.content);
                break;
            case 'BID_PHASE':
                this.showBiddingUI(message.content);
                break;
            case 'TRICK_COMPLETE':
                this.handleTrickComplete(message.content);
                break;
            case 'ROUND_COMPLETE':
                this.handleRoundComplete(message.content);
                break;
            case 'END_GAME':
            case 'GAME_OVER':
                this.handleGameOver(message.content);
                break;
            case 'REPORT_ERROR':
            case 'ERROR':
                this.showError('lobby', message.content.error || message.content.message || window.i18n.t('lobby.errorOccurred'));
                break;
            // Pirate ability handlers
            case 'ABILITY_TRIGGERED':
                this.handleAbilityTriggered(message.content);
                break;
            case 'ABILITY_RESOLVED':
                this.handleAbilityResolved(message.content);
                break;
            case 'SHOW_DECK':
                this.handleShowDeck(message.content);
                break;
        }
    }

    updateLobby() {
        if (!this.gameState) return;

        // Show game code
        const shortCode = (this.gameState.id || this.gameId).substring(0, 8);
        document.getElementById('lobby-game-id').textContent = shortCode;

        // Update players list
        const playersList = document.getElementById('lobby-players');
        playersList.innerHTML = '';

        const players = this.gameState.players || [];
        players.forEach((player, index) => {
            const li = document.createElement('li');
            if (player.id === this.playerId) {
                li.classList.add('is-you');
            }

            const isFirstPlayer = index === 0;
            const initial = player.username.charAt(0).toUpperCase();

            li.innerHTML = `
                <div class="player-info">
                    <div class="player-avatar ${player.is_bot ? 'bot' : ''}">${player.is_bot ? '🤖' : initial}</div>
                    <span class="player-name">${player.username}</span>
                    <div class="player-badges">
                        ${player.id === this.playerId ? `<span class="badge badge-you">${window.i18n.t('lobby.you')}</span>` : ''}
                        ${player.is_bot ? `<span class="badge badge-bot">${window.i18n.t('lobby.bot')}</span>` : ''}
                        ${isFirstPlayer ? `<span class="badge badge-host">${window.i18n.t('lobby.host')}</span>` : ''}
                    </div>
                </div>
                ${player.is_bot && this.isHost ? `<button class="remove-player-btn" data-player-id="${player.id}" title="Remove">✕</button>` : ''}
            `;

            playersList.appendChild(li);
        });

        // Add event listeners for remove buttons
        playersList.querySelectorAll('.remove-player-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const playerId = e.target.dataset.playerId;
                this.removeBot(playerId);
            });
        });

        // Update player count
        document.getElementById('player-count').textContent = players.length;

        // Show/hide config panel based on host status
        const configPanel = document.getElementById('config-panel');
        if (configPanel) {
            configPanel.style.display = this.isHost ? 'block' : 'none';
        }

        // Update start button state
        const startBtn = document.getElementById('start-game-btn');
        const startHint = document.getElementById('start-hint');
        const canStart = players.length >= 2 && players.length <= 8;

        startBtn.disabled = !canStart || !this.isHost;
        startHint.style.display = canStart ? 'none' : 'block';

        if (!this.isHost) {
            startBtn.innerHTML = `<span>${window.i18n.t('lobby.waitingForHost')}</span>`;
        }
    }

    updateGameScreen() {
        if (!this.gameState || !this.isGameInProgress()) return;

        console.log('[updateGameScreen] State:', this.gameState.state, 'Round:', this.gameState.current_round, 'Hand:', this.gameState.hand?.length, 'cards');

        // Switch to game screen only if game is actually in progress
        if (!document.getElementById('game-screen').classList.contains('active')) {
            this.switchScreen('game');
        }

        // Hide bidding modal when not in bidding state
        const biddingModal = document.getElementById('bidding-modal');
        if (biddingModal && this.gameState.state !== 'BIDDING') {
            biddingModal.classList.add('hidden');
        }

        // Update round info
        document.getElementById('current-round').textContent = this.gameState.current_round || 1;
        document.getElementById('game-status-text').textContent = this.getStatusText();

        // Update player score
        const myPlayer = this.gameState.players?.find(p => p.id === this.playerId);
        document.getElementById('player-score').textContent = myPlayer?.score || 0;

        // Update player's bid and tricks won
        const myBid = myPlayer?.bid ?? this.gameState.bids?.[this.playerId] ?? null;
        document.getElementById('player-bid').textContent = myBid !== null ? myBid : '-';
        document.getElementById('player-tricks').textContent = myPlayer?.tricks_won || 0;

        // Update turn indicator
        const isMyTurn = this.gameState.picking_player_id === this.playerId ||
                        this.gameState.current_player_id === this.playerId;
        const turnIndicator = document.getElementById('your-turn-indicator');
        if (turnIndicator) {
            if (isMyTurn && this.gameState.state === 'PICKING') {
                turnIndicator.classList.remove('hidden');
            } else {
                turnIndicator.classList.add('hidden');
            }
        }

        // Update opponents
        this.updateOpponents();

        // Update hand
        this.updateHand();

        // Update trick area
        this.updateTrickArea();

        // Update scoreboard
        this.updateScoreboard();
    }

    updateOpponents() {
        // Clear all position containers
        const topContainer = document.getElementById('opponents-top');
        const leftContainer = document.getElementById('opponents-left');
        const rightContainer = document.getElementById('opponents-right');
        const legacyContainer = document.getElementById('opponents-container');

        if (topContainer) topContainer.innerHTML = '';
        if (leftContainer) leftContainer.innerHTML = '';
        if (rightContainer) rightContainer.innerHTML = '';
        if (legacyContainer) legacyContainer.innerHTML = '';

        const players = this.gameState.players || [];
        const opponents = players.filter(p => p.id !== this.playerId);

        // Get current round for card count
        const currentRound = this.gameState.current_round || 1;

        // Distribute opponents: 1=top, 2=top+left or top+right, 3=left+top+right
        const positions = this.getOpponentPositions(opponents.length);

        opponents.forEach((player, index) => {
            const div = this.createOpponentCard(player, currentRound);
            const position = positions[index];

            // Determine container based on position
            let container;
            if (position === 'top' && topContainer) container = topContainer;
            else if (position === 'left' && leftContainer) container = leftContainer;
            else if (position === 'right' && rightContainer) container = rightContainer;
            else container = legacyContainer;

            if (container) container.appendChild(div);
        });

        // Update turn arrow
        this.updateTurnArrow();
    }

    getOpponentPositions(count) {
        // Distribute opponents around the table
        switch (count) {
            case 1: return ['top'];
            case 2: return ['left', 'right'];
            case 3: return ['left', 'top', 'right'];
            case 4: return ['left', 'top', 'top', 'right'];
            case 5: return ['left', 'top', 'top', 'top', 'right'];
            case 6: return ['left', 'left', 'top', 'top', 'right', 'right'];
            case 7: return ['left', 'left', 'top', 'top', 'top', 'right', 'right'];
            default: return Array(count).fill('top');
        }
    }

    createOpponentCard(player, currentRound) {
        const div = document.createElement('div');
        div.className = 'opponent-card';
        div.dataset.playerId = player.id;

        // Check if it's this opponent's turn to pick
        const isCurrentTurn = this.gameState.picking_player_id === player.id ||
            this.gameState.current_player_id === player.id;
        if (isCurrentTurn) {
            div.classList.add('current-turn');
        }

        // Get bid from player object or from gameState.bids
        const playerBid = player.bid ?? this.gameState.bids?.[player.id] ?? null;
        const bidDisplay = playerBid !== null && playerBid !== undefined ? playerBid : '-';

        // Check for score animation
        const scoreChange = this.lastScoreChanges?.[player.id];
        let scoreClass = 'stat-value';
        if (scoreChange !== undefined) {
            scoreClass += scoreChange > 0 ? ' score-up' : ' score-down';
        }

        // Calculate hand size (cards remaining)
        const handSize = player.hand_size ?? currentRound;

        // Create face-down cards for opponent's hand
        const cardBacks = Array(handSize).fill(0).map(() =>
            '<div class="card-back"></div>'
        ).join('');

        div.innerHTML = `
            <div class="opponent-name">
                ${player.is_bot ? '🤖 ' : ''}${player.username}
            </div>
            <div class="opponent-stats">
                <div class="stat">
                    <span class="${scoreClass}">${player.score || 0}</span>
                    <span class="stat-label">${window.i18n.t('game.score')}</span>
                </div>
                <div class="stat">
                    <span class="stat-value">${bidDisplay}</span>
                    <span class="stat-label">${window.i18n.t('game.bid')}</span>
                </div>
                <div class="stat">
                    <span class="stat-value">${player.tricks_won || 0}</span>
                    <span class="stat-label">${window.i18n.t('game.tricks')}</span>
                </div>
            </div>
            <div class="opponent-hand" style="--card-count: ${handSize}">${cardBacks}</div>
        `;

        return div;
    }

    updateTurnArrow() {
        // Turn indicator is now handled by highlighting the opponent card
        // Just hide the arrow element completely
        const arrow = document.getElementById('turn-arrow');
        if (arrow) {
            arrow.classList.add('hidden');
        }
    }

    updateHand() {
        const hand = this.gameState.hand || [];
        const handContainer = document.getElementById('player-hand');
        handContainer.innerHTML = '';

        console.log('[updateHand] Rendering hand with', hand.length, 'cards:', hand);

        document.getElementById('hand-count').textContent = hand.length;

        // Set CSS variable for card fan centering
        handContainer.style.setProperty('--total', hand.length);

        // Note: Cards are also shown in bidding modal, but we still render hand

        // Check if it's our turn and what's playable
        const isMyTurn = this.gameState.picking_player_id === this.playerId;
        const playableCards = isMyTurn ? this.getPlayableCards(hand) : new Set(hand);

        // Add/remove your-turn class on player area
        const playerArea = document.querySelector('.player-area');
        if (playerArea) {
            playerArea.classList.toggle('your-turn', isMyTurn);
        }

        hand.forEach((cardId, index) => {
            // Convert card ID to card object for display
            const card = this.cardIdToCard(cardId);
            const cardElement = this.createCardElement(card);
            cardElement.dataset.cardId = cardId; // Store actual ID

            // Mark unplayable cards
            if (!playableCards.has(cardId)) {
                cardElement.classList.add('unplayable');
            }

            cardElement.addEventListener('click', (e) => {
                console.log('[card click] Card clicked:', cardId);
                this.handleCardClick(cardId, cardElement, e);
            });
            handContainer.appendChild(cardElement);
        });

        // Check if hand is scrollable on touch devices
        if (this.isTouchDevice && hand.length > 5) {
            handContainer.classList.add('scrollable');
        } else {
            handContainer.classList.remove('scrollable');
        }

        console.log('[updateHand] Done rendering hand');
    }

    // Handle card click with tap-to-confirm on touch devices
    handleCardClick(cardId, cardElement, event) {
        // Desktop: play immediately
        if (!this.isTouchDevice) {
            this.playCard(cardId);
            return;
        }

        // Touch device: tap-to-confirm behavior
        if (this.selectedCardId === cardId) {
            // Second tap - play the card
            this.clearCardSelection();
            this.playCard(cardId);
        } else {
            // First tap - select the card
            this.selectCard(cardId, cardElement);
        }
    }

    // Select a card (for tap-to-confirm)
    selectCard(cardId, cardElement) {
        this.clearCardSelection();
        this.selectedCardId = cardId;
        cardElement.classList.add('touch-selected');
        this.playSound('click');

        // Clear selection after 3 seconds if no second tap
        this.selectionTimeout = setTimeout(() => {
            this.clearCardSelection();
        }, 3000);
    }

    // Clear current card selection
    clearCardSelection() {
        if (this.selectionTimeout) {
            clearTimeout(this.selectionTimeout);
            this.selectionTimeout = null;
        }
        this.selectedCardId = null;
        document.querySelectorAll('.hand-cards .card.touch-selected').forEach(el => {
            el.classList.remove('touch-selected');
        });
    }

    // Get the suit of a card ID
    getCardSuit(cardId) {
        // Special cards (can always be played)
        if (cardId <= 10) return 'special'; // King, Whale, Kraken, Mermaids, Pirates
        if (cardId >= 67) return 'special'; // Escapes, Tigress, Loot

        // Suit cards
        if (cardId >= 11 && cardId <= 24) return 'roger';  // Black/Trump
        if (cardId >= 25 && cardId <= 38) return 'parrot'; // Green
        if (cardId >= 39 && cardId <= 52) return 'map';    // Purple
        if (cardId >= 53 && cardId <= 66) return 'chest';  // Yellow

        return 'unknown';
    }

    // Determine which cards are playable based on game rules
    getPlayableCards(hand) {
        const playable = new Set();
        const trickCards = this.gameState.trick_cards || [];

        // If we're leading (no cards played), all cards are playable
        if (trickCards.length === 0) {
            hand.forEach(id => playable.add(id));
            return playable;
        }

        // Find the lead suit (first non-special card played)
        let leadSuit = null;
        for (const tc of trickCards) {
            const cardId = tc.card_id || tc.card;
            const suit = this.getCardSuit(cardId);
            if (suit !== 'special') {
                leadSuit = suit;
                break;
            }
        }

        // If no lead suit established (all specials), all cards are playable
        if (!leadSuit) {
            hand.forEach(id => playable.add(id));
            return playable;
        }

        // Check if we have cards of the lead suit
        const suitCards = hand.filter(id => this.getCardSuit(id) === leadSuit);

        if (suitCards.length > 0) {
            // Must follow suit - only suit cards and specials are playable
            hand.forEach(id => {
                const suit = this.getCardSuit(id);
                if (suit === leadSuit || suit === 'special') {
                    playable.add(id);
                }
            });
        } else {
            // Can't follow suit - all cards are playable
            hand.forEach(id => playable.add(id));
        }

        return playable;
    }

    // Pirate names for card IDs 6-10 (matching backend PIRATE_IDENTITY order)
    // PIRATE1=Rosie, PIRATE2=Bendt, PIRATE3=Roatan(rascal), PIRATE4=Jade(juanita), PIRATE5=Harry
    pirateNames = ['rosie', 'bendt', 'rascal', 'juanita', 'harry'];

    // Convert numeric card ID to card object for display
    cardIdToCard(cardId) {
        // Card ID mapping based on app/models/card.py
        // 1 = Skull King, 2 = Whale, 3 = Kraken
        // 4-5 = Mermaids, 6-10 = Pirates
        // 11-24 = Roger (trump) 1-14
        // 25-38 = Parrot 1-14
        // 39-52 = Map 1-14
        // 53-66 = Chest 1-14
        // 67-71 = Escape 1-5
        // 72 = Tigress, 73-74 = Loot

        if (cardId === 1) return { type: 'king', number: null, image: 'skullking.png' };
        if (cardId === 2) return { type: 'whale', number: null, image: 'whale.png' };
        if (cardId === 3) return { type: 'kraken', number: null, image: 'kraken.png' };
        if (cardId >= 4 && cardId <= 5) return { type: 'mermaid', number: null, image: 'siren.png' };
        if (cardId >= 6 && cardId <= 10) {
            const pirateIndex = cardId - 6;
            return { type: 'pirate', number: null, image: `${this.pirateNames[pirateIndex]}.png` };
        }
        if (cardId >= 11 && cardId <= 24) return { type: 'roger', number: cardId - 10, image: 'black.png' };
        if (cardId >= 25 && cardId <= 38) return { type: 'parrot', number: cardId - 24, image: 'green.png' };
        if (cardId >= 39 && cardId <= 52) return { type: 'map', number: cardId - 38, image: 'purple.png' };
        if (cardId >= 53 && cardId <= 66) return { type: 'chest', number: cardId - 52, image: 'yellow.png' };
        if (cardId >= 67 && cardId <= 71) return { type: 'escape', number: null, image: 'flee.png' };
        if (cardId === 72) return { type: 'tigress', number: null, image: 'tigress.png' };
        if (cardId >= 73 && cardId <= 74) return { type: 'loot', number: null, image: 'loot.png' };

        return { type: 'unknown', number: cardId, image: 'back.png' };
    }

    updateTrickArea() {
        const trickCards = this.gameState.trick_cards || [];
        const container = document.getElementById('trick-cards');
        container.innerHTML = '';

        // Set CSS variable for fan centering
        container.style.setProperty('--total', trickCards.length || 1);

        trickCards.forEach(({ player_id, card_id, player_name, card, tigress_choice }) => {
            const wrapper = document.createElement('div');
            wrapper.className = 'trick-card-wrapper';
            wrapper.dataset.playerId = player_id;

            // Handle both formats: {card_id} from server or {card} object
            const cardData = card || this.cardIdToCard(card_id);
            const cardElement = this.createCardElement(cardData);

            // Add Tigress choice badge if this is Tigress
            if (card_id === 72 && tigress_choice) {
                const choiceBadge = document.createElement('div');
                choiceBadge.className = `card-choice-badge ${tigress_choice}`;
                choiceBadge.innerHTML = tigress_choice === 'pirate'
                    ? '<span class="choice-icon">🏴‍☠️</span><span class="choice-text">Pirate</span>'
                    : '<span class="choice-icon">👻</span><span class="choice-text">Escape</span>';
                cardElement.appendChild(choiceBadge);
            }

            const label = document.createElement('div');
            label.className = 'trick-player-name';
            // Find player name from ID if not provided
            const playerName = player_name || this.gameState.players?.find(p => p.id === player_id)?.username || 'Player';
            label.textContent = playerName;

            wrapper.appendChild(cardElement);
            wrapper.appendChild(label);
            container.appendChild(wrapper);
        });
    }

    createCardElement(card) {
        const div = document.createElement('div');
        div.className = 'card';

        // Add card type/suit class
        const cardClass = this.getCardClass(card);
        if (cardClass) {
            div.classList.add(cardClass);
        }

        // Use image-based rendering if image is available
        if (card.image) {
            div.classList.add('card-image');

            const img = document.createElement('img');
            img.src = `/static/images/cards/${card.image}`;
            img.alt = this.formatCardType(card.type);
            img.className = 'card-img';
            img.draggable = false;
            div.appendChild(img);

            // Add number overlay for suit cards
            if (card.number) {
                const numberOverlay = document.createElement('div');
                numberOverlay.className = 'card-number-overlay';
                numberOverlay.textContent = card.number;
                div.appendChild(numberOverlay);
            }

            // Add bonus badge for 14 cards
            const bonusPoints = this.getCardBonus(card);
            if (bonusPoints > 0) {
                const bonus = document.createElement('div');
                bonus.className = 'card-bonus';
                bonus.textContent = `+${bonusPoints}`;
                div.appendChild(bonus);
            }
        } else {
            // Fallback to emoji-based rendering
            const number = document.createElement('div');
            number.className = 'card-number';
            number.textContent = this.getCardDisplay(card);

            const type = document.createElement('div');
            type.className = 'card-type';
            type.textContent = this.formatCardType(card.type);

            // Add suit icon for numbered cards
            const suitIcon = this.getSuitIcon(card);
            if (suitIcon) {
                const icon = document.createElement('div');
                icon.className = 'card-suit-icon';
                icon.textContent = suitIcon;
                div.appendChild(icon);
            }

            div.appendChild(number);
            div.appendChild(type);

            // Add bonus badge for 14 cards
            const bonusPoints = this.getCardBonus(card);
            if (bonusPoints > 0) {
                const bonus = document.createElement('div');
                bonus.className = 'card-bonus';
                bonus.textContent = `+${bonusPoints}`;
                div.appendChild(bonus);
            }
        }

        return div;
    }

    getCardBonus(card) {
        // 14s give bonus points when captured
        if (card.number === 14) {
            if (card.type === 'roger') {
                return 20; // Black 14 gives +20
            }
            return 10; // Other 14s give +10
        }
        return 0;
    }

    getSuitIcon(card) {
        if (!card.number) return null; // Special cards don't have suit icons

        switch (card.type) {
            case 'parrot': return '🦜';
            case 'chest': return '📦';
            case 'map': return '🗺️';
            case 'roger': return '🏴‍☠️';
            default: return null;
        }
    }

    getCardClass(card) {
        if (!card.type) return '';

        const type = card.type.toLowerCase();

        // Special cards
        if (type === 'king' || type === 'skull_king') return 'king';
        if (type === 'pirate') return 'pirate';
        if (type === 'mermaid') return 'mermaid';
        if (type === 'escape') return 'escape';
        if (type === 'scary_mary' || type === 'tigress') return 'tigress';
        if (type === 'kraken') return 'kraken';
        if (type === 'whale') return 'whale';
        if (type === 'loot') return 'loot';

        // Suit cards
        if (type === 'chest' || type === 'yellow' || type.includes('yellow')) return 'yellow';
        if (type === 'parrot' || type === 'green' || type.includes('green')) return 'green';
        if (type === 'map' || type === 'purple' || type.includes('purple')) return 'purple';
        if (type === 'roger' || type === 'black' || type.includes('black')) return 'black';

        return '';
    }

    getCardDisplay(card) {
        if (!card.type) return '?';

        const type = card.type.toLowerCase();

        if (type === 'king' || type === 'skull_king') return '💀';
        if (type === 'pirate') return '🏴‍☠️';
        if (type === 'mermaid') return '🧜‍♀️';
        if (type === 'escape') return '🏳️';
        if (type === 'scary_mary' || type === 'tigress') return '🎭';
        if (type === 'kraken') return '🦑';
        if (type === 'whale') return '🐋';
        if (type === 'loot') return '💰';

        return card.number || '?';
    }

    formatCardType(type) {
        if (!type) return '';
        return type.replace(/_/g, ' ').toLowerCase();
    }

    getStatusText() {
        const state = this.gameState.state;

        switch (state) {
            case 'BIDDING':
                return window.i18n.t('game.biddingPhase');
            case 'PICKING':
                return window.i18n.t('game.playingTricks');
            case 'ENDED':
                return window.i18n.t('game.gameOver');
            default:
                return window.i18n.t('game.inProgress');
        }
    }

    showBiddingUI(data) {
        const modal = document.getElementById('bidding-modal');
        modal.classList.remove('hidden');

        // Max bid is the round number (round 1 = max bid 1, round 10 = max bid 10)
        const maxBid = data.round || data.max_bid || this.gameState?.current_round || 1;
        const buttonsContainer = document.getElementById('bid-buttons');
        buttonsContainer.innerHTML = '';

        // Remove old hint if exists
        const contentDiv = modal.querySelector('.bidding-content');
        const hint = contentDiv.querySelector('.cards-hint');
        if (hint) hint.remove();

        // Render bid buttons
        for (let i = 0; i <= maxBid; i++) {
            const button = document.createElement('button');
            button.className = 'bid-btn';
            button.textContent = i;
            button.addEventListener('click', () => this.makeBid(i));
            buttonsContainer.appendChild(button);
        }

        // Render hand cards in the modal
        this.renderBidHandPreview();
    }

    renderBidHandPreview() {
        const bidHandContainer = document.getElementById('bid-hand-cards');
        if (!bidHandContainer) return;

        bidHandContainer.innerHTML = '';
        const hand = this.gameState?.hand || [];

        // Set CSS variable for card fan centering
        bidHandContainer.style.setProperty('--total', hand.length);

        hand.forEach((cardId) => {
            const card = this.cardIdToCard(cardId);
            const cardElement = this.createCardElement(card);
            cardElement.dataset.cardId = cardId;
            bidHandContainer.appendChild(cardElement);
        });
    }

    makeBid(bid) {
        this.sendMessage('BID', { bid });

        // Animate cards from modal to hand area
        this.animateCardsToHand();
    }

    animateCardsToHand() {
        const modal = document.getElementById('bidding-modal');
        const bidCards = document.querySelectorAll('#bid-hand-cards .card');
        const handContainer = document.getElementById('player-hand');

        if (!bidCards.length || !handContainer) {
            modal.classList.add('hidden');
            return;
        }

        // Get target position (hand area)
        const handRect = handContainer.getBoundingClientRect();
        const targetX = handRect.left + handRect.width / 2;
        const targetY = handRect.top + handRect.height / 2;

        // Clone cards and animate them
        const flyingCards = [];
        bidCards.forEach((card, index) => {
            const rect = card.getBoundingClientRect();
            const clone = card.cloneNode(true);

            // Position clone at card's current position
            clone.classList.add('flying-to-hand');
            clone.style.left = `${rect.left}px`;
            clone.style.top = `${rect.top}px`;
            clone.style.width = `${rect.width}px`;
            clone.style.height = `${rect.height}px`;

            document.body.appendChild(clone);
            flyingCards.push(clone);

            // Animate to target with stagger
            setTimeout(() => {
                clone.style.left = `${targetX - rect.width / 2 + (index - bidCards.length / 2) * 20}px`;
                clone.style.top = `${targetY - rect.height / 2}px`;
                clone.style.opacity = '0';
                clone.style.transform = 'scale(1.2)';
            }, 50 + index * 30);
        });

        // Hide modal immediately
        modal.classList.add('hidden');

        // Clean up flying cards after animation
        setTimeout(() => {
            flyingCards.forEach(card => card.remove());
        }, 700);
    }

    playCard(cardId) {
        console.log('[playCard] Called with cardId:', cardId);
        console.log('[playCard] Current game state:', this.gameState?.state);
        console.log('[playCard] Picking player:', this.gameState?.picking_player_id);
        console.log('[playCard] My player ID:', this.playerId);
        console.log('[playCard] Is my turn:', this.gameState?.picking_player_id === this.playerId);
        console.log('[playCard] My hand:', this.gameState?.hand);

        // Tigress (card_id 72) requires player to choose pirate or escape
        if (cardId === 72) {
            console.log('[playCard] Tigress card - showing choice modal');
            this.showTigressChoice(cardId);
            return;
        }
        console.log('[playCard] Sending PICK command');
        this.sendMessage('PICK', { card_id: cardId });
    }

    showTigressChoice(cardId) {
        const modal = document.getElementById('tigress-modal');
        modal.classList.remove('hidden');

        const pirateBtn = document.getElementById('tigress-pirate');
        const escapeBtn = document.getElementById('tigress-escape');

        // Remove old listeners
        const newPirateBtn = pirateBtn.cloneNode(true);
        const newEscapeBtn = escapeBtn.cloneNode(true);
        pirateBtn.parentNode.replaceChild(newPirateBtn, pirateBtn);
        escapeBtn.parentNode.replaceChild(newEscapeBtn, escapeBtn);

        // Add new listeners
        newPirateBtn.addEventListener('click', () => {
            modal.classList.add('hidden');
            this.sendMessage('PICK', { card_id: cardId, tigress_choice: 'pirate' });
        });

        newEscapeBtn.addEventListener('click', () => {
            modal.classList.add('hidden');
            this.sendMessage('PICK', { card_id: cardId, tigress_choice: 'escape' });
        });
    }

    handleTrickWinner(data) {
        // Update tricks won for the winner
        const winner = this.gameState?.players?.find(p => p.id === data.winner_player_id);
        if (winner) {
            winner.tricks_won = (winner.tricks_won || 0) + 1;
        }

        // Play special sounds for notable wins
        // Mermaid captures Skull King (40 point bonus with mermaid card 4-5)
        if (data.bonus_points >= 40 && data.winner_card_id >= 4 && data.winner_card_id <= 5) {
            window.soundManager?.mermaidCapture();
        } else if (data.winner_player_id === this.playerId) {
            this.playSound('trickWon');
        } else {
            this.playSound('trickLost');
        }

        // Mark the winning card in the trick area
        const trickCards = document.querySelectorAll('.trick-card-wrapper');
        trickCards.forEach(wrapper => {
            const playerId = wrapper.dataset.playerId;
            if (playerId === data.winner_player_id) {
                wrapper.classList.add('winner');
            }
        });

        const winnerLabel = document.getElementById('trick-winner');
        const winnerName = data.winner_name || winner?.username || 'Unknown';

        winnerLabel.textContent = `${winnerName} ${window.i18n.t('game.wonTrick')}`;
        winnerLabel.classList.remove('hidden');

        this.addLog(window.i18n.t('log.trickWon', { winner: winnerName }));
        this.updateGameScreen();

        // NOTE: Trick cards are cleared by START_PICKING when next trick begins
        // No setTimeout here - all state changes are server-driven
    }

    // Determine which direction cards should fly to based on winner position
    getCollectDirection(winnerId) {
        if (winnerId === this.playerId) {
            return 'collect-bottom';
        }

        const players = this.gameState?.players || [];
        const myIndex = players.findIndex(p => p.id === this.playerId);
        const winnerIndex = players.findIndex(p => p.id === winnerId);
        const playerCount = players.length;

        if (myIndex === -1 || winnerIndex === -1) {
            return 'collect-top';
        }

        // Calculate relative position
        const relativePos = (winnerIndex - myIndex + playerCount) % playerCount;
        const halfCount = playerCount / 2;

        if (relativePos === 0) {
            return 'collect-bottom';
        } else if (relativePos < halfCount) {
            return 'collect-left';
        } else if (relativePos > halfCount) {
            return 'collect-right';
        } else {
            return 'collect-top';
        }
    }

    handleTrickComplete(data) {
        // Delegate to handleTrickWinner - same logic
        this.handleTrickWinner(data);
    }

    handleScoresAnnounced(data) {
        console.log('[handleScoresAnnounced] Round:', data.round, 'Scores:', data.scores);

        // Update player scores
        if (this.gameState?.players && data.scores) {
            data.scores.forEach(scoreInfo => {
                const player = this.gameState.players.find(p => p.id === scoreInfo.player_id);
                if (player) {
                    player.score = scoreInfo.total_score;
                    player.tricks_won = 0; // Reset for next round
                    player.bid = null; // Reset bid
                }
            });
        }

        this.addLog(window.i18n.t('log.roundComplete', { round: data.round }));
        this.updateGameScreen();

        // Show round summary overlay (dismissible by click)
        this.showRoundSummary(data.round, data.scores);
    }

    showRoundSummary(roundNumber, scores) {
        // Remove any existing summary
        const existingSummary = document.querySelector('.round-summary-overlay');
        if (existingSummary) existingSummary.remove();

        // Create summary overlay
        const overlay = document.createElement('div');
        overlay.className = 'round-summary-overlay';

        const sortedScores = [...(scores || [])].sort((a, b) => b.total_score - a.total_score);
        const myScore = scores?.find(s => s.player_id === this.playerId);

        overlay.innerHTML = `
            <div class="round-summary-content">
                <h3>Round ${roundNumber} Complete!</h3>
                <div class="round-summary-scores">
                    ${sortedScores.slice(0, 4).map((s, i) => {
                        const player = this.gameState?.players?.find(p => p.id === s.player_id);
                        const isMe = s.player_id === this.playerId;
                        const deltaClass = s.score_delta >= 0 ? 'positive' : 'negative';
                        const deltaSign = s.score_delta >= 0 ? '+' : '';
                        return `
                            <div class="summary-player ${isMe ? 'is-you' : ''}">
                                <span class="summary-rank">${i + 1}</span>
                                <span class="summary-name">${player?.username || 'Player'}${isMe ? ' ⭐' : ''}</span>
                                <span class="summary-delta ${deltaClass}">${deltaSign}${s.score_delta}</span>
                                <span class="summary-total">${s.total_score}</span>
                            </div>
                        `;
                    }).join('')}
                </div>
                ${myScore ? `<div class="your-round-result ${myScore.score_delta >= 0 ? 'positive' : 'negative'}">
                    Your score: ${myScore.score_delta >= 0 ? '+' : ''}${myScore.score_delta}
                </div>` : ''}
            </div>
        `;

        document.body.appendChild(overlay);

        // Click to dismiss (no auto-timeout - sequential state flow)
        overlay.addEventListener('click', () => {
            overlay.classList.add('fade-out');
            setTimeout(() => overlay.remove(), 300);
        });
    }

    handleRoundComplete(data) {
        this.playSound('roundComplete');
        this.addLog(window.i18n.t('log.roundComplete', { round: data.round_number || data.round }));
    }

    handleGameOver(data) {
        this.playSound('gameOver');
        // Get players from either leaderboard (server) or final_scores format
        let players = data.leaderboard || data.final_scores || [];

        // Convert leaderboard format to expected format if needed
        if (players.length > 0 && players[0].player_id && !players[0].id) {
            players = players.map(p => ({
                id: p.player_id,
                username: p.username || this.gameState?.players?.find(gp => gp.id === p.player_id)?.username || 'Player',
                score: p.score
            }));
        }

        players.sort((a, b) => b.score - a.score);

        // Store final results to prevent loss on screen switch
        this.finalResults = players;

        // Add log entry
        if (players[0]) {
            this.addLog(window.i18n.t('log.gameWon', { winner: players[0].username }) || `${players[0].username} won the game!`);
        }

        this.switchScreen('results');

        // Update winner announcement
        const winnerAnnouncement = document.getElementById('winner-announcement');
        if (winnerAnnouncement && players[0]) {
            const isYouWinner = players[0].id === this.playerId;
            winnerAnnouncement.innerHTML = isYouWinner
                ? `<span class="winner-crown">👑</span> ${window.i18n.t('results.youWon') || 'You Won!'}`
                : `<span class="winner-crown">👑</span> ${players[0].username} ${window.i18n.t('results.wins') || 'Wins!'}`;
        }

        // Update podium with animations
        if (players[0]) {
            const place1 = document.getElementById('place-1');
            place1.innerHTML = `
                <div class="podium-rank">🥇</div>
                <div class="podium-name">${players[0].username}${players[0].id === this.playerId ? ' ⭐' : ''}</div>
                <div class="podium-score">${players[0].score}</div>
            `;
            place1.classList.add('winner-glow');
        }
        if (players[1]) {
            document.getElementById('place-2').innerHTML = `
                <div class="podium-rank">🥈</div>
                <div class="podium-name">${players[1].username}${players[1].id === this.playerId ? ' ⭐' : ''}</div>
                <div class="podium-score">${players[1].score}</div>
            `;
        }
        if (players[2]) {
            document.getElementById('place-3').innerHTML = `
                <div class="podium-rank">🥉</div>
                <div class="podium-name">${players[2].username}${players[2].id === this.playerId ? ' ⭐' : ''}</div>
                <div class="podium-score">${players[2].score}</div>
            `;
        }
        // Handle 4th place
        if (players[3]) {
            let place4 = document.getElementById('place-4');
            if (!place4) {
                const podium = document.querySelector('.podium');
                place4 = document.createElement('div');
                place4.id = 'place-4';
                place4.className = 'podium-place fourth';
                podium.appendChild(place4);
            }
            place4.innerHTML = `
                <div class="podium-rank">4</div>
                <div class="podium-name">${players[3].username}${players[3].id === this.playerId ? ' ⭐' : ''}</div>
                <div class="podium-score">${players[3].score}</div>
            `;
        }

        // Update table
        const tbody = document.getElementById('final-scores-body');
        tbody.innerHTML = '';

        players.forEach((player, index) => {
            const tr = document.createElement('tr');
            const isYou = player.id === this.playerId;
            if (isYou) tr.classList.add('is-you');
            if (index === 0) tr.classList.add('winner');

            tr.innerHTML = `
                <td><strong>${index === 0 ? '👑' : index + 1}</strong></td>
                <td>${player.username}${isYou ? ` <span class="you-badge">(${window.i18n.t('lobby.you')})</span>` : ''}</td>
                <td><strong>${player.score}</strong></td>
            `;
            tbody.appendChild(tr);
        });

        // Mark game as ended to prevent WebSocket reconnection issues
        this.gameEnded = true;
    }

    // ============================================
    // Pirate Ability Handlers
    // ============================================

    handleAbilityTriggered(data) {
        // Show ability prompt modal based on ability type
        const abilityType = data.ability_type;
        const pirateType = data.pirate_type;

        console.log('[ABILITY_TRIGGERED] Type:', abilityType, 'Pirate:', pirateType);

        switch (abilityType) {
            case 'choose_starter':
                // Rosie - choose who starts next trick
                this.showRosieModal(data.options);
                break;
            case 'draw_discard':
                // Bendt - draw 2, discard 2
                this.showBendtModal(data.drawn_cards, data.must_discard);
                break;
            case 'extra_bet':
                // Roatán - declare extra bet
                this.showRoatanModal(data.options);
                break;
            case 'view_deck':
                // Jade - view undealt cards (handled by SHOW_DECK)
                break;
            case 'modify_bid':
                // Harry - shown at end of round
                this.showHarryModal();
                break;
        }
    }

    handleAbilityResolved(data) {
        // Hide any ability modal and show confirmation
        this.hideAbilityModal();
        this.addLog(window.i18n.t('log.abilityResolved', { ability: data.ability_type }) || `${data.ability_type} ability resolved`, 'ability', '⚔️');
    }

    handleShowDeck(data) {
        // Jade's ability - show undealt cards
        this.showJadeModal(data.undealt_cards);
    }

    showRosieModal(options) {
        const modal = this.createAbilityModal('rosie-modal', 'Rosie\'s Ability');
        const content = modal.querySelector('.ability-content');

        content.innerHTML = `
            <p class="ability-description">${window.i18n.t('ability.rosie.description') || 'Choose who will start the next trick:'}</p>
            <div class="ability-options">
                ${options.map(opt => `
                    <button class="ability-option" data-player-id="${opt.player_id}">
                        ${opt.username}
                    </button>
                `).join('')}
            </div>
        `;

        content.querySelectorAll('.ability-option').forEach(btn => {
            btn.addEventListener('click', () => {
                this.sendMessage('RESOLVE_ROSIE', { chosen_player_id: btn.dataset.playerId });
                this.hideAbilityModal();
            });
        });

        document.body.appendChild(modal);
    }

    showBendtModal(drawnCards, mustDiscard) {
        const modal = this.createAbilityModal('bendt-modal', 'Bendt\'s Ability');
        const content = modal.querySelector('.ability-content');

        content.innerHTML = `
            <p class="ability-description">${window.i18n.t('ability.bendt.description') || `You drew ${drawnCards.length} cards. Select ${mustDiscard} cards to discard:`}</p>
            <div class="ability-cards drawn-cards">
                <p class="cards-label">${window.i18n.t('ability.bendt.drawnCards') || 'Drawn cards:'}</p>
                <div class="card-row">
                    ${drawnCards.map(cardId => this.renderCardForAbility(cardId, true)).join('')}
                </div>
            </div>
            <div class="ability-cards hand-cards">
                <p class="cards-label">${window.i18n.t('ability.bendt.yourHand') || 'Your hand:'}</p>
                <div class="card-row">
                    ${(this.gameState?.hand || []).filter(c => !drawnCards.includes(c)).map(cardId => this.renderCardForAbility(cardId, true)).join('')}
                </div>
            </div>
            <p class="discard-count">Selected: <span id="discard-count">0</span> / ${mustDiscard}</p>
            <button class="ability-confirm" disabled>Confirm Discard</button>
        `;

        const selectedCards = [];
        content.querySelectorAll('.ability-card').forEach(card => {
            card.addEventListener('click', () => {
                const cardId = parseInt(card.dataset.cardId);
                const idx = selectedCards.indexOf(cardId);
                if (idx >= 0) {
                    selectedCards.splice(idx, 1);
                    card.classList.remove('selected');
                } else if (selectedCards.length < mustDiscard) {
                    selectedCards.push(cardId);
                    card.classList.add('selected');
                }
                document.getElementById('discard-count').textContent = selectedCards.length;
                content.querySelector('.ability-confirm').disabled = selectedCards.length !== mustDiscard;
            });
        });

        content.querySelector('.ability-confirm').addEventListener('click', () => {
            this.sendMessage('RESOLVE_BENDT', { discard_cards: selectedCards });
            this.hideAbilityModal();
        });

        document.body.appendChild(modal);
    }

    showRoatanModal(options) {
        const modal = this.createAbilityModal('roatan-modal', 'Roatán\'s Ability');
        const content = modal.querySelector('.ability-content');

        content.innerHTML = `
            <p class="ability-description">${window.i18n.t('ability.roatan.description') || 'Choose your extra bet amount:'}</p>
            <div class="ability-options">
                ${options.map(amount => `
                    <button class="ability-option" data-amount="${amount}">
                        ${amount === 0 ? 'No Bet' : `+${amount} points`}
                    </button>
                `).join('')}
            </div>
            <p class="ability-hint">${window.i18n.t('ability.roatan.hint') || 'If you win this trick and make your bid, you get the bonus. Otherwise, lose the points!'}</p>
        `;

        content.querySelectorAll('.ability-option').forEach(btn => {
            btn.addEventListener('click', () => {
                this.sendMessage('RESOLVE_ROATAN', { extra_bet: parseInt(btn.dataset.amount) });
                this.hideAbilityModal();
            });
        });

        document.body.appendChild(modal);
    }

    showJadeModal(undealtCards) {
        const modal = this.createAbilityModal('jade-modal', 'Jade\'s Ability');
        const content = modal.querySelector('.ability-content');

        content.innerHTML = `
            <p class="ability-description">${window.i18n.t('ability.jade.description') || 'These cards were not dealt this round:'}</p>
            <div class="ability-cards deck-cards">
                <div class="card-row">
                    ${undealtCards.map(cardId => this.renderCardForAbility(cardId, false)).join('')}
                </div>
            </div>
            <button class="ability-confirm">Got it!</button>
        `;

        content.querySelector('.ability-confirm').addEventListener('click', () => {
            this.sendMessage('RESOLVE_JADE', {});
            this.hideAbilityModal();
        });

        document.body.appendChild(modal);
    }

    showHarryModal() {
        const modal = this.createAbilityModal('harry-modal', 'Harry\'s Ability');
        const content = modal.querySelector('.ability-content');

        content.innerHTML = `
            <p class="ability-description">${window.i18n.t('ability.harry.description') || 'Adjust your bid by ±1:'}</p>
            <div class="ability-options">
                <button class="ability-option" data-modifier="-1">−1</button>
                <button class="ability-option" data-modifier="0">No change</button>
                <button class="ability-option" data-modifier="1">+1</button>
            </div>
        `;

        content.querySelectorAll('.ability-option').forEach(btn => {
            btn.addEventListener('click', () => {
                this.sendMessage('RESOLVE_HARRY', { modifier: parseInt(btn.dataset.modifier) });
                this.hideAbilityModal();
            });
        });

        document.body.appendChild(modal);
    }

    createAbilityModal(id, title) {
        // Remove any existing ability modal
        this.hideAbilityModal();

        const modal = document.createElement('div');
        modal.id = id;
        modal.className = 'modal ability-modal';
        modal.innerHTML = `
            <div class="modal-content ability-modal-content">
                <h2 class="ability-title">${title}</h2>
                <div class="ability-content"></div>
            </div>
        `;

        return modal;
    }

    hideAbilityModal() {
        document.querySelectorAll('.ability-modal').forEach(modal => modal.remove());
    }

    renderCardForAbility(cardId, selectable) {
        const cardInfo = this.getCardInfo(cardId);
        return `
            <div class="ability-card ${selectable ? 'selectable' : ''}" data-card-id="${cardId}">
                <div class="card ${cardInfo.suitClass}">${cardInfo.display}</div>
            </div>
        `;
    }

    addBot() {
        const botType = document.getElementById('bot-type-select').value;
        const difficulty = document.getElementById('bot-difficulty-select').value;

        this.sendMessage('ADD_BOT', { bot_type: botType, difficulty: difficulty });
    }

    fillWithBots() {
        const currentPlayers = this.gameState?.players?.length || 1;
        const botsToAdd = 4 - currentPlayers;

        // Send all bot additions immediately (server handles sequentially)
        for (let i = 0; i < botsToAdd; i++) {
            this.addBot();
        }
    }

    clearBots() {
        this.sendMessage('CLEAR_BOTS', {});
    }

    removeBot(playerId) {
        this.sendMessage('REMOVE_BOT', { player_id: playerId });
    }

    startGame() {
        this.sendMessage('START_GAME', {});
    }

    leaveLobby() {
        this.intentionalClose = true;
        if (this.ws) {
            this.ws.close();
        }
        this.returnToLogin();
    }

    sendMessage(command, content) {
        console.log('[sendMessage] Command:', command, 'Content:', content);
        console.log('[sendMessage] WebSocket exists:', !!this.ws);
        console.log('[sendMessage] WebSocket readyState:', this.ws?.readyState, '(OPEN=1)');

        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            const message = {
                command,
                game_id: this.gameId,
                player_id: this.playerId,
                content
            };
            console.log('[sendMessage] Sending:', JSON.stringify(message));
            this.ws.send(JSON.stringify(message));
        } else {
            console.error('[sendMessage] WebSocket not ready! Cannot send message.');
        }
    }

    copyGameId() {
        const gameId = document.getElementById('lobby-game-id').textContent;
        navigator.clipboard.writeText(gameId).then(() => {
            const btn = document.getElementById('copy-game-id-btn');
            const originalHTML = btn.innerHTML;
            btn.innerHTML = '✓';
            setTimeout(() => {
                btn.innerHTML = originalHTML;
            }, 2000);
        });
    }

    addLog(message, type = 'info', icon = null) {
        const logContainer = document.getElementById('log-messages');
        const entry = document.createElement('div');
        entry.className = `log-entry log-${type}`;

        const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

        // Default icons based on type
        const defaultIcons = {
            'info': '',
            'bid': '&#128176;',
            'play': '&#127183;',
            'win': '&#127942;',
            'score': '&#10004;',
            'round': '&#128161;'
        };

        const displayIcon = icon || defaultIcons[type] || '';

        if (displayIcon) {
            entry.innerHTML = `<span class="log-icon">${displayIcon}</span><span class="log-text">[${time}] ${message}</span>`;
        } else {
            entry.innerHTML = `<span class="log-text">[${time}] ${message}</span>`;
        }

        logContainer.appendChild(entry);
        logContainer.scrollTop = logContainer.scrollHeight;
    }

    updateScoreboard() {
        const tbody = document.getElementById('scoreboard-body');
        const dropdownBody = document.getElementById('scoreboard-dropdown-body');

        if (!this.gameState?.players) return;

        // Sort players by score
        const sortedPlayers = [...this.gameState.players].sort((a, b) => (b.score || 0) - (a.score || 0));

        // Update both scoreboards
        [tbody, dropdownBody].forEach(container => {
            if (!container) return;
            container.innerHTML = '';

            sortedPlayers.forEach((player, index) => {
                const tr = document.createElement('tr');
                const isYou = player.id === this.playerId;
                const isCurrentTurn = player.id === this.gameState.picking_player_id;
                const bid = player.bid ?? this.gameState.bids?.[player.id] ?? null;
                const tricksWon = player.tricks_won || 0;

                if (isYou) tr.classList.add('is-you', 'current-player');
                if (isCurrentTurn) tr.classList.add('current-turn');

                // Bid progress indicator
                if (bid !== null) {
                    if (tricksWon === bid) {
                        tr.classList.add('bid-match');
                    } else if (tricksWon > bid) {
                        tr.classList.add('bid-over');
                    }
                }

                // Dropdown uses compact format (no rank column)
                if (container === dropdownBody) {
                    tr.innerHTML = `
                        <td>${player.is_bot ? '🤖 ' : ''}${player.username}${isYou ? ' ⭐' : ''}</td>
                        <td>${bid !== null ? bid : '-'}</td>
                        <td>${tricksWon}</td>
                        <td><strong>${player.score || 0}</strong></td>
                    `;
                } else {
                    tr.innerHTML = `
                        <td>${index + 1}</td>
                        <td>${player.is_bot ? '&#129302; ' : ''}${player.username}${isYou ? ' &#11088;' : ''}</td>
                        <td>${bid !== null ? bid : '-'}</td>
                        <td>${tricksWon}</td>
                        <td><strong>${player.score || 0}</strong></td>
                    `;
                }
                container.appendChild(tr);
            });
        });
    }

    switchScreen(screenName) {
        document.querySelectorAll('.screen').forEach(screen => {
            screen.classList.remove('active');
        });
        document.getElementById(`${screenName}-screen`).classList.add('active');

        // Hide/show login screen floating buttons based on current screen
        const langBtn = document.getElementById('lang-toggle');
        const rulesBtn = document.getElementById('rules-toggle');
        const soundBtn = document.getElementById('sound-toggle');
        const isGameScreen = screenName === 'game';

        if (langBtn) langBtn.style.display = isGameScreen ? 'none' : '';
        if (rulesBtn) rulesBtn.style.display = isGameScreen ? 'none' : '';
        if (soundBtn) soundBtn.style.display = isGameScreen ? 'none' : '';
    }

    showError(screen, message) {
        const errorElement = document.getElementById(`${screen}-error`);
        if (errorElement) {
            errorElement.textContent = message;
            setTimeout(() => {
                errorElement.textContent = '';
            }, 5000);
        }
        // Also show as toast
        this.showToast(message, 'error');
    }

    showToast(message, type = 'info', duration = 3000) {
        const container = document.getElementById('toast-container');
        if (!container) return;

        const icons = {
            success: '&#10004;',
            error: '&#10060;',
            info: '&#8505;',
            warning: '&#9888;'
        };

        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        toast.innerHTML = `
            <span class="toast-icon">${icons[type] || icons.info}</span>
            <span class="toast-message">${message}</span>
        `;

        container.appendChild(toast);

        // Auto remove after duration
        setTimeout(() => {
            toast.classList.add('toast-out');
            setTimeout(() => toast.remove(), 300);
        }, duration);
    }

    setButtonLoading(button, loading = true) {
        if (loading) {
            button.classList.add('loading');
            button.disabled = true;
        } else {
            button.classList.remove('loading');
            button.disabled = false;
        }
    }

    returnToLogin() {
        this.intentionalClose = true;
        if (this.reconnectTimer) {
            clearTimeout(this.reconnectTimer);
            this.reconnectTimer = null;
        }
        if (this.ws) {
            this.ws.close();
        }
        this.ws = null;
        this.gameId = null;
        this.playerId = null;
        this.gameState = null;
        this.isHost = false;
        this.reconnectAttempts = 0;
        this.intentionalClose = false; // Reset for next game
        this.hideReconnecting();
        this.switchScreen('login');
    }

    // ============================================
    //  Game History
    // ============================================

    async showHistory() {
        const modal = document.getElementById('history-modal');
        const list = document.getElementById('history-list');

        modal.classList.remove('hidden');
        list.innerHTML = '<p class="loading">' + (window.i18n?.t('history.loading') || 'Loading...') + '</p>';

        try {
            const response = await fetch('/games/history?limit=10');
            const data = await response.json();

            if (data.games && data.games.length > 0) {
                list.innerHTML = data.games.map(game => this.renderHistoryItem(game)).join('');
            } else {
                list.innerHTML = '<p class="empty">' + (window.i18n?.t('history.noGames') || 'No completed games yet') + '</p>';
            }
        } catch (error) {
            console.error('Failed to load game history:', error);
            list.innerHTML = '<p class="empty">' + (window.i18n?.t('history.error') || 'Failed to load history') + '</p>';
        }
    }

    renderHistoryItem(game) {
        const date = new Date(game.created_at);
        const duration = this.formatDuration(game.duration_seconds);
        const playerCount = game.players?.length || 0;

        return `
            <div class="history-item" data-game-id="${game.game_id}">
                <div class="history-item-icon">&#127942;</div>
                <div class="history-item-info">
                    <div class="history-item-title">${game.winner_username} won!</div>
                    <div class="history-item-meta">${playerCount} players - ${game.total_rounds} rounds - ${duration}</div>
                </div>
                <div class="history-item-score">
                    <div class="history-item-winner">${game.players?.[0]?.score || 0} pts</div>
                    <div class="history-item-date">${date.toLocaleDateString()}</div>
                </div>
            </div>
        `;
    }

    formatDuration(seconds) {
        if (!seconds) return '0m';
        const mins = Math.floor(seconds / 60);
        const secs = seconds % 60;
        if (mins === 0) return `${secs}s`;
        return `${mins}m ${secs}s`;
    }

    hideHistory() {
        document.getElementById('history-modal')?.classList.add('hidden');
    }

    // ============================================
    //  Browse Active Games
    // ============================================

    async showBrowseGames() {
        const modal = document.getElementById('browse-games-modal');
        modal?.classList.remove('hidden');
        await this.refreshActiveGames();
    }

    hideBrowseGames() {
        document.getElementById('browse-games-modal')?.classList.add('hidden');
    }

    async refreshActiveGames() {
        const list = document.getElementById('active-games-list');
        if (!list) return;

        list.innerHTML = '<p class="loading">' + (window.i18n?.t('browse.loading') || 'Loading...') + '</p>';

        try {
            const response = await fetch('/games/active');
            const data = await response.json();

            if (data.games && data.games.length > 0) {
                list.innerHTML = data.games.map(game => this.renderActiveGameItem(game)).join('');

                // Add click handlers to spectate buttons
                list.querySelectorAll('.spectate-btn').forEach(btn => {
                    btn.addEventListener('click', () => {
                        const gameId = btn.dataset.gameId;
                        this.hideBrowseGames();
                        this.spectateGameById(gameId);
                    });
                });
            } else {
                list.innerHTML = '<p class="empty">' + (window.i18n?.t('browse.noGames') || 'No active games right now') + '</p>';
            }
        } catch (error) {
            console.error('Failed to load active games:', error);
            list.innerHTML = '<p class="empty">' + (window.i18n?.t('browse.error') || 'Failed to load games') + '</p>';
        }
    }

    renderActiveGameItem(game) {
        const stateLabel = this.getGameStateLabel(game.state);
        const playerInfo = game.player_names.join(', ') + (game.bot_count > 0 ? ` + ${game.bot_count} bots` : '');
        const spectatorInfo = game.spectator_count > 0 ? `&#128065; ${game.spectator_count}` : '';

        return `
            <div class="active-game-item">
                <div class="game-item-info">
                    <div class="game-item-players">
                        <span class="player-count">${game.player_count} players</span>
                        <span class="player-names">${playerInfo}</span>
                    </div>
                    <div class="game-item-status">
                        <span class="game-state ${game.state.toLowerCase()}">${stateLabel}</span>
                        ${game.current_round > 0 ? `<span class="round-info">Round ${game.current_round}/10</span>` : ''}
                        ${spectatorInfo ? `<span class="spectator-info">${spectatorInfo}</span>` : ''}
                    </div>
                </div>
                <button class="btn btn-ghost spectate-btn" data-game-id="${game.game_id}">
                    <span>&#128065;</span> Watch
                </button>
            </div>
        `;
    }

    getGameStateLabel(state) {
        const labels = {
            'PENDING': window.i18n?.t('browse.statePending') || 'In Lobby',
            'BIDDING': window.i18n?.t('browse.stateBidding') || 'Bidding',
            'PICKING': window.i18n?.t('browse.statePlaying') || 'Playing'
        };
        return labels[state] || state;
    }

    spectateGameById(gameId) {
        this.username = 'Spectator';
        this.gameId = gameId;
        this.playerId = this.generatePlayerId();
        this.isHost = false;
        this.isSpectator = true;

        this.connectWebSocket();
    }

    updateSpectatorCount(count) {
        const countElement = document.getElementById('spectator-count');
        const valueElement = document.getElementById('spectator-count-value');

        if (countElement && valueElement) {
            valueElement.textContent = count;
            if (count > 0) {
                countElement.classList.remove('hidden');
            } else {
                countElement.classList.add('hidden');
            }
        }
    }

    generatePlayerId() {
        return 'player_' + Math.random().toString(36).substr(2, 9);
    }
}

// Initialize game when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.game = new SkullKingGame();
});
