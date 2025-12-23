// Skull King Game Client
class SkullKingGame {
    constructor() {
        this.ws = null;
        this.gameId = null;
        this.playerId = null;
        this.username = null;
        this.gameState = null;

        this.initializeEventListeners();
    }

    initializeEventListeners() {
        // Login screen
        document.getElementById('create-game-btn').addEventListener('click', () => this.createGame());
        document.getElementById('join-game-btn').addEventListener('click', () => this.joinGame());
        document.getElementById('username-input').addEventListener('keypress', (e) => {
            if (e.key === 'Enter') this.createGame();
        });

        // Lobby screen
        document.getElementById('add-bot-btn').addEventListener('click', () => this.addBot());
        document.getElementById('start-game-btn').addEventListener('click', () => this.startGame());
        document.getElementById('copy-game-id-btn').addEventListener('click', () => this.copyGameId());

        // Results screen
        document.getElementById('new-game-btn').addEventListener('click', () => this.returnToLogin());
    }

    async createGame() {
        const username = document.getElementById('username-input').value.trim();

        if (!username) {
            this.showError('login', 'Please enter your name');
            return;
        }

        this.username = username;

        try {
            const response = await fetch('/games', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ lobby_id: 'default' })
            });

            if (!response.ok) {
                throw new Error('Failed to create game');
            }

            const data = await response.json();
            this.gameId = data.game_id;
            this.playerId = this.generatePlayerId();

            this.connectWebSocket();
        } catch (error) {
            this.showError('login', 'Failed to create game: ' + error.message);
        }
    }

    async joinGame() {
        const username = document.getElementById('username-input').value.trim();
        const gameId = document.getElementById('game-id-input').value.trim();

        if (!username) {
            this.showError('login', 'Please enter your name');
            return;
        }

        if (!gameId) {
            this.showError('login', 'Please enter a game ID');
            return;
        }

        this.username = username;
        this.gameId = gameId;
        this.playerId = this.generatePlayerId();

        this.connectWebSocket();
    }

    connectWebSocket() {
        const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${wsProtocol}//${window.location.host}/games/join?game_id=${this.gameId}&player_id=${this.playerId}&username=${this.username}`;

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
            this.showError('login', 'Connection error');
        };

        this.ws.onclose = () => {
            console.log('WebSocket closed');
            if (this.gameState && this.gameState.state !== 'ENDED') {
                this.showError('lobby', 'Connection lost');
            }
        };
    }

    handleMessage(message) {
        console.log('Received message:', message);

        switch (message.command) {
            case 'INIT':
                this.gameState = message.content.game;
                this.updateLobby();
                break;
            case 'JOINED':
                this.addLog(`${message.content.username} joined the game`);
                break;
            case 'PLAYER_LEFT':
                this.addLog(`${message.content.username} left the game`);
                break;
            case 'GAME_STATE':
                this.gameState = message.content;
                this.updateGameScreen();
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
            case 'GAME_OVER':
                this.handleGameOver(message.content);
                break;
            case 'ERROR':
                this.showError('lobby', message.content.message || 'An error occurred');
                break;
        }
    }

    updateLobby() {
        if (!this.gameState) return;

        document.getElementById('lobby-game-id').textContent = this.gameState.id || this.gameId;

        const playersList = document.getElementById('lobby-players');
        playersList.innerHTML = '';

        const players = this.gameState.players || [];
        players.forEach(player => {
            const li = document.createElement('li');
            li.innerHTML = `
                <span class="player-name">${player.username}${player.id === this.playerId ? ' (You)' : ''}</span>
                ${player.is_bot ? '<span class="player-badge">BOT</span>' : ''}
            `;
            playersList.appendChild(li);
        });

        document.getElementById('player-count').textContent = players.length;

        const startBtn = document.getElementById('start-game-btn');
        startBtn.disabled = players.length < 2;
    }

    updateGameScreen() {
        if (!this.gameState) return;

        // Switch to game screen if not already there
        if (!document.getElementById('game-screen').classList.contains('active')) {
            this.switchScreen('game');
        }

        // Update round info
        document.getElementById('current-round').textContent = this.gameState.current_round || 1;
        document.getElementById('game-status-text').textContent = this.getStatusText();

        // Update players
        this.updatePlayers();

        // Update hand
        this.updateHand();

        // Update trick area
        this.updateTrickArea();
    }

    updatePlayers() {
        const container = document.getElementById('players-container');
        container.innerHTML = '';

        const players = this.gameState.players || [];
        players.forEach(player => {
            const div = document.createElement('div');
            div.className = 'player-card';

            if (player.id === this.playerId) {
                div.classList.add('is-you');
            }

            div.innerHTML = `
                <div class="player-header">
                    <span class="player-name-display">${player.username}</span>
                    <span class="player-score">${player.score || 0}</span>
                </div>
                <div class="player-stats">
                    <span>Bid: ${player.bid !== undefined ? player.bid : '-'}</span>
                    <span>Tricks: ${player.tricks_won || 0}</span>
                </div>
            `;

            container.appendChild(div);
        });
    }

    updateHand() {
        const hand = this.gameState.hand || [];
        const handContainer = document.getElementById('player-hand');
        handContainer.innerHTML = '';

        document.getElementById('hand-count').textContent = `${hand.length} card${hand.length !== 1 ? 's' : ''}`;

        hand.forEach((card, index) => {
            const cardElement = this.createCardElement(card);
            cardElement.addEventListener('click', () => this.playCard(index));
            handContainer.appendChild(cardElement);
        });
    }

    updateTrickArea() {
        const trickCards = this.gameState.trick_cards || [];
        const container = document.getElementById('trick-cards');
        container.innerHTML = '';

        trickCards.forEach(({player_name, card}) => {
            const wrapper = document.createElement('div');
            wrapper.style.textAlign = 'center';

            const cardElement = this.createCardElement(card);
            cardElement.style.cursor = 'default';

            const label = document.createElement('div');
            label.textContent = player_name;
            label.style.marginTop = '5px';
            label.style.fontSize = '0.9em';

            wrapper.appendChild(cardElement);
            wrapper.appendChild(label);
            container.appendChild(wrapper);
        });
    }

    createCardElement(card) {
        const div = document.createElement('div');
        div.className = 'card';

        // Add card type class
        if (card.type) {
            div.classList.add(card.type.toLowerCase().replace('_', '-'));
        }

        const number = document.createElement('div');
        number.className = 'card-number';
        number.textContent = card.number || '?';

        const type = document.createElement('div');
        type.className = 'card-type';
        type.textContent = this.formatCardType(card.type);

        div.appendChild(number);
        div.appendChild(type);

        return div;
    }

    formatCardType(type) {
        if (!type) return '';
        return type.replace('_', ' ').toLowerCase();
    }

    getStatusText() {
        const state = this.gameState.state;

        switch (state) {
            case 'BIDDING':
                return 'Bidding Phase';
            case 'PICKING':
                return 'Playing Tricks';
            case 'ENDED':
                return 'Game Over';
            default:
                return 'In Progress';
        }
    }

    showBiddingUI(data) {
        const biddingArea = document.getElementById('bidding-area');
        biddingArea.classList.remove('hidden');

        const maxBid = data.max_bid || 10;
        const buttonsContainer = document.getElementById('bid-buttons');
        buttonsContainer.innerHTML = '';

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
        document.getElementById('bidding-area').classList.add('hidden');
    }

    playCard(cardIndex) {
        this.sendMessage('PLAY_CARD', { card_index: cardIndex });
    }

    handleTrickComplete(data) {
        this.addLog(`Trick won by ${data.winner_name}`);
        // Update UI after short delay to show trick result
        setTimeout(() => this.updateGameScreen(), 1500);
    }

    handleRoundComplete(data) {
        this.addLog(`Round ${data.round_number} complete!`);
    }

    handleGameOver(data) {
        this.switchScreen('results');

        const tbody = document.getElementById('final-scores-body');
        tbody.innerHTML = '';

        const players = data.final_scores || [];
        players.sort((a, b) => b.score - a.score);

        players.forEach((player, index) => {
            const tr = document.createElement('tr');

            let badge = `<span class="rank-badge">${index + 1}</span>`;
            if (index === 0) badge = `<span class="rank-badge gold">🏆</span>`;
            else if (index === 1) badge = `<span class="rank-badge silver">🥈</span>`;
            else if (index === 2) badge = `<span class="rank-badge bronze">🥉</span>`;

            tr.innerHTML = `
                <td>${badge}</td>
                <td>${player.username}${player.id === this.playerId ? ' (You)' : ''}</td>
                <td><strong>${player.score}</strong></td>
            `;
            tbody.appendChild(tr);
        });
    }

    addBot() {
        this.sendMessage('ADD_BOT', {});
    }

    startGame() {
        this.sendMessage('START_GAME', {});
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
            const originalText = btn.textContent;
            btn.textContent = '✓ Copied!';
            setTimeout(() => {
                btn.textContent = originalText;
            }, 2000);
        });
    }

    addLog(message) {
        const logContainer = document.getElementById('log-messages');
        const entry = document.createElement('div');
        entry.className = 'log-entry';
        entry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
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
        this.gameId = null;
        this.playerId = null;
        this.gameState = null;
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
