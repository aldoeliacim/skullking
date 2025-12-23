// Skull King Game Client
class SkullKingGame {
    constructor() {
        this.ws = null;
        this.gameId = null;
        this.playerId = null;
        this.username = null;
        this.gameState = null;
        this.isHost = false;

        this.initializeEventListeners();
        this.initializeI18n();
    }

    async initializeI18n() {
        await window.i18n.init();
        this.updateLangFlag();

        window.addEventListener('localeChanged', () => {
            this.updateLangFlag();
            this.updateDynamicContent();
        });
    }

    updateLangFlag() {
        const flag = document.getElementById('lang-flag');
        if (flag) {
            flag.textContent = window.i18n.getLocale().toUpperCase();
        }
    }

    updateDynamicContent() {
        if (this.gameState) {
            this.updateGameScreen();
        }
    }

    initializeEventListeners() {
        // Login screen
        document.getElementById('create-game-btn').addEventListener('click', () => this.createGame());
        document.getElementById('join-game-btn').addEventListener('click', () => this.joinGame());
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

        // Language switcher
        document.getElementById('lang-toggle').addEventListener('click', () => {
            window.i18n.toggleLocale();
        });

        // Game log toggle
        document.getElementById('log-toggle').addEventListener('click', () => {
            document.getElementById('game-log').classList.toggle('collapsed');
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

        this.connectWebSocket();
    }

    connectWebSocket() {
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${wsProtocol}//${window.location.host}/games/join?game_id=${this.gameId}&player_id=${this.playerId}&username=${encodeURIComponent(this.username)}`;

        this.ws = new WebSocket(wsUrl);

        this.ws.onopen = () => {
            console.log('WebSocket connected');
            this.switchScreen('lobby');
            this.updateLobby();
        };

        this.ws.onmessage = (event) => {
            const message = JSON.parse(event.data);
            this.handleMessage(message);
        };

        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.showError('login', window.i18n.t('login.errorConnection'));
        };

        this.ws.onclose = (event) => {
            console.log('WebSocket closed:', event.code, event.reason);
            // Don't show error if game ended normally or we're on results screen
            if (this.gameState && this.gameState.state !== 'ENDED' && !this.gameEnded) {
                this.showError('lobby', window.i18n.t('lobby.errorConnectionLost'));
            }
        };
    }

    handleMessage(message) {
        console.log('Received:', message);

        switch (message.command) {
            case 'INIT':
                this.gameState = message.content.game;
                this.updateLobby();
                break;
            case 'JOINED':
                this.addLog(window.i18n.t('log.playerJoined', { username: message.content.username }));
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
                this.addLog(window.i18n.t('log.gameStarted', { count: message.content.player_count }));
                this.switchScreen('game');
                break;
            case 'DEAL':
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
                this.updateGameScreen();
                this.addLog(window.i18n.t('log.cardsDealt', { round: message.content.round }));
                break;
            case 'START_BIDDING':
                this.gameState = this.gameState || {};
                this.gameState.state = 'BIDDING';
                this.gameState.current_round = message.content.round;
                // First update game screen so cards are visible
                this.updateGameScreen();
                // Then show bidding UI
                this.showBiddingUI(message.content);
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
                }));
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
                this.addLog(window.i18n.t('log.biddingComplete'));
                this.updateGameScreen();
                break;
            case 'START_PICKING':
                this.gameState = this.gameState || {};
                this.gameState.state = 'PICKING';
                this.gameState.picking_player_id = message.content.picking_player_id;
                this.gameState.current_trick = message.content.trick;
                this.updateGameScreen();
                break;
            case 'PICKED':
                // A card was played
                this.gameState.trick_cards = this.gameState.trick_cards || [];
                this.gameState.trick_cards.push({
                    player_id: message.content.player_id,
                    card_id: message.content.card_id
                });
                // Remove from hand if it's our card
                if (message.content.player_id === this.playerId && this.gameState.hand) {
                    this.gameState.hand = this.gameState.hand.filter(c => c !== message.content.card_id);
                }
                this.updateGameScreen();
                break;
            case 'NEXT_TRICK':
                this.gameState.picking_player_id = message.content.picking_player_id;
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
        if (!this.gameState) return;

        // Switch to game screen
        if (!document.getElementById('game-screen').classList.contains('active')) {
            this.switchScreen('game');
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
    }

    updateOpponents() {
        const container = document.getElementById('opponents-container');
        container.innerHTML = '';

        const players = this.gameState.players || [];
        const opponents = players.filter(p => p.id !== this.playerId);

        opponents.forEach(player => {
            const div = document.createElement('div');
            div.className = 'opponent-card';

            if (this.gameState.current_player_id === player.id) {
                div.classList.add('current-turn');
            }

            // Get bid from player object or from gameState.bids
            const playerBid = player.bid ?? this.gameState.bids?.[player.id] ?? null;
            const bidDisplay = playerBid !== null && playerBid !== undefined ? playerBid : '-';

            div.innerHTML = `
                <div class="opponent-name">
                    ${player.is_bot ? '🤖 ' : ''}${player.username}
                </div>
                <div class="opponent-stats">
                    <div class="stat">
                        <span class="stat-value">${player.score || 0}</span>
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
            `;

            container.appendChild(div);
        });
    }

    updateHand() {
        const hand = this.gameState.hand || [];
        const handContainer = document.getElementById('player-hand');
        handContainer.innerHTML = '';

        document.getElementById('hand-count').textContent = hand.length;

        hand.forEach((cardId, index) => {
            // Convert card ID to card object for display
            const card = this.cardIdToCard(cardId);
            const cardElement = this.createCardElement(card);
            cardElement.dataset.cardId = cardId; // Store actual ID
            cardElement.addEventListener('click', () => this.playCard(cardId));
            handContainer.appendChild(cardElement);
        });
    }

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

        if (cardId === 1) return { type: 'king', number: null };
        if (cardId === 2) return { type: 'whale', number: null };
        if (cardId === 3) return { type: 'kraken', number: null };
        if (cardId >= 4 && cardId <= 5) return { type: 'mermaid', number: null };
        if (cardId >= 6 && cardId <= 10) return { type: 'pirate', number: null };
        if (cardId >= 11 && cardId <= 24) return { type: 'roger', number: cardId - 10 };
        if (cardId >= 25 && cardId <= 38) return { type: 'parrot', number: cardId - 24 };
        if (cardId >= 39 && cardId <= 52) return { type: 'map', number: cardId - 38 };
        if (cardId >= 53 && cardId <= 66) return { type: 'chest', number: cardId - 52 };
        if (cardId >= 67 && cardId <= 71) return { type: 'escape', number: null };
        if (cardId === 72) return { type: 'tigress', number: null };
        if (cardId >= 73 && cardId <= 74) return { type: 'loot', number: null };

        return { type: 'unknown', number: cardId };
    }

    updateTrickArea() {
        const trickCards = this.gameState.trick_cards || [];
        const container = document.getElementById('trick-cards');
        container.innerHTML = '';

        trickCards.forEach(({ player_id, card_id, player_name, card }) => {
            const wrapper = document.createElement('div');
            wrapper.className = 'trick-card-wrapper';

            // Handle both formats: {card_id} from server or {card} object
            const cardData = card || this.cardIdToCard(card_id);
            const cardElement = this.createCardElement(cardData);

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

        // Add bonus badge for 14 cards
        const bonusPoints = this.getCardBonus(card);
        if (bonusPoints > 0) {
            const bonus = document.createElement('div');
            bonus.className = 'card-bonus';
            bonus.textContent = `+${bonusPoints}`;
            div.appendChild(bonus);
        }

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

        // Add hint about cards being visible
        const contentDiv = modal.querySelector('.bidding-content');
        let hint = contentDiv.querySelector('.cards-hint');
        if (!hint) {
            hint = document.createElement('div');
            hint.className = 'cards-hint';
            hint.innerHTML = `<span>&#128071;</span> ${window.i18n.t('game.cardsVisibleBelow') || 'Your cards are visible below'}`;
            contentDiv.querySelector('p').after(hint);
        }

        for (let i = 0; i <= maxBid; i++) {
            const button = document.createElement('button');
            button.className = 'bid-btn';
            button.textContent = i;
            button.addEventListener('click', () => this.makeBid(i));
            buttonsContainer.appendChild(button);
        }
    }

    makeBid(bid) {
        this.sendMessage('BID', { bid });
        document.getElementById('bidding-modal').classList.add('hidden');
    }

    playCard(cardId) {
        this.sendMessage('PICK', { card_id: cardId });
    }

    handleTrickWinner(data) {
        // Update tricks won for the winner
        const winner = this.gameState?.players?.find(p => p.id === data.winner_player_id);
        if (winner) {
            winner.tricks_won = (winner.tricks_won || 0) + 1;
        }

        const winnerLabel = document.getElementById('trick-winner');
        const winnerName = winner?.username || 'Unknown';

        winnerLabel.textContent = `${winnerName} ${window.i18n.t('game.wonTrick')}`;
        winnerLabel.classList.remove('hidden');

        this.addLog(window.i18n.t('log.trickWon', { winner: winnerName }));

        // Update UI immediately to show new trick count
        this.updateGameScreen();

        setTimeout(() => {
            winnerLabel.classList.add('hidden');
            // Clear trick cards
            this.gameState.trick_cards = [];
            this.updateGameScreen();
        }, 2000);
    }

    handleTrickComplete(data) {
        // Update tricks won for the winner
        if (data.winner_player_id && this.gameState?.players) {
            const winner = this.gameState.players.find(p => p.id === data.winner_player_id);
            if (winner) {
                winner.tricks_won = (winner.tricks_won || 0) + 1;
            }
        }

        const winnerLabel = document.getElementById('trick-winner');
        const winnerName = data.winner_name || 'Unknown';
        winnerLabel.textContent = `${winnerName} ${window.i18n.t('game.wonTrick')}`;
        winnerLabel.classList.remove('hidden');

        this.addLog(window.i18n.t('log.trickWon', { winner: winnerName }));

        // Update UI immediately
        this.updateGameScreen();

        setTimeout(() => {
            winnerLabel.classList.add('hidden');
            // Clear trick cards
            this.gameState.trick_cards = [];
            this.updateGameScreen();
        }, 2000);
    }

    handleScoresAnnounced(data) {
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
    }

    handleRoundComplete(data) {
        this.addLog(window.i18n.t('log.roundComplete', { round: data.round_number || data.round }));
    }

    handleGameOver(data) {
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

    addBot() {
        const botType = document.getElementById('bot-type-select').value;
        const difficulty = document.getElementById('bot-difficulty-select').value;

        this.sendMessage('ADD_BOT', { bot_type: botType, difficulty: difficulty });
    }

    fillWithBots() {
        const currentPlayers = this.gameState?.players?.length || 1;
        const botsToAdd = 4 - currentPlayers;

        for (let i = 0; i < botsToAdd; i++) {
            setTimeout(() => this.addBot(), i * 100);
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
        if (this.ws) {
            this.ws.close();
        }
        this.returnToLogin();
    }

    sendMessage(command, content) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({
                command,
                game_id: this.gameId,
                player_id: this.playerId,
                content
            }));
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

    addLog(message) {
        const logContainer = document.getElementById('log-messages');
        const entry = document.createElement('div');
        entry.className = 'log-entry';
        const time = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        entry.textContent = `[${time}] ${message}`;
        logContainer.appendChild(entry);
        logContainer.scrollTop = logContainer.scrollHeight;
    }

    switchScreen(screenName) {
        document.querySelectorAll('.screen').forEach(screen => {
            screen.classList.remove('active');
        });
        document.getElementById(`${screenName}-screen`).classList.add('active');
    }

    showError(screen, message) {
        const errorElement = document.getElementById(`${screen}-error`);
        if (errorElement) {
            errorElement.textContent = message;
            setTimeout(() => {
                errorElement.textContent = '';
            }, 5000);
        }
    }

    returnToLogin() {
        if (this.ws) {
            this.ws.close();
        }
        this.ws = null;
        this.gameId = null;
        this.playerId = null;
        this.gameState = null;
        this.isHost = false;
        this.switchScreen('login');
    }

    generatePlayerId() {
        return 'player_' + Math.random().toString(36).substr(2, 9);
    }
}

// Initialize game when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.game = new SkullKingGame();
});
