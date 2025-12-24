/**
 * Sound effects manager for Skull King using Web Audio API
 * Creates synthesized sounds without external audio files
 */
class SoundManager {
    constructor() {
        this.enabled = this.getStoredPreference();
        this.audioContext = null;
        this.masterGain = null;
        this.volume = 0.3;
    }

    /**
     * Initialize audio context (must be called after user interaction)
     */
    init() {
        if (this.audioContext) return;

        try {
            this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
            this.masterGain = this.audioContext.createGain();
            this.masterGain.gain.value = this.volume;
            this.masterGain.connect(this.audioContext.destination);
        } catch (e) {
            console.warn('Web Audio API not supported:', e);
            this.enabled = false;
        }
    }

    /**
     * Resume audio context if suspended (required by browsers)
     */
    async resume() {
        if (this.audioContext?.state === 'suspended') {
            await this.audioContext.resume();
        }
    }

    /**
     * Get stored sound preference
     */
    getStoredPreference() {
        const stored = localStorage.getItem('skullking-sound');
        return stored !== 'false'; // Default to enabled
    }

    /**
     * Toggle sound on/off
     */
    toggle() {
        this.enabled = !this.enabled;
        localStorage.setItem('skullking-sound', this.enabled.toString());
        return this.enabled;
    }

    /**
     * Set volume (0-1)
     */
    setVolume(value) {
        this.volume = Math.max(0, Math.min(1, value));
        if (this.masterGain) {
            this.masterGain.gain.value = this.volume;
        }
    }

    /**
     * Play a tone with given frequency and duration
     */
    playTone(frequency, duration, type = 'sine', attack = 0.01, decay = 0.1) {
        if (!this.enabled || !this.audioContext) return;

        this.resume();

        const oscillator = this.audioContext.createOscillator();
        const gainNode = this.audioContext.createGain();

        oscillator.type = type;
        oscillator.frequency.value = frequency;

        // ADSR envelope
        const now = this.audioContext.currentTime;
        gainNode.gain.setValueAtTime(0, now);
        gainNode.gain.linearRampToValueAtTime(1, now + attack);
        gainNode.gain.linearRampToValueAtTime(0.3, now + attack + decay);
        gainNode.gain.linearRampToValueAtTime(0, now + duration);

        oscillator.connect(gainNode);
        gainNode.connect(this.masterGain);

        oscillator.start(now);
        oscillator.stop(now + duration);
    }

    /**
     * Play noise burst (for clicks, shuffles)
     */
    playNoise(duration, filterFreq = 1000) {
        if (!this.enabled || !this.audioContext) return;

        this.resume();

        const bufferSize = this.audioContext.sampleRate * duration;
        const buffer = this.audioContext.createBuffer(1, bufferSize, this.audioContext.sampleRate);
        const data = buffer.getChannelData(0);

        for (let i = 0; i < bufferSize; i++) {
            data[i] = Math.random() * 2 - 1;
        }

        const noise = this.audioContext.createBufferSource();
        noise.buffer = buffer;

        const filter = this.audioContext.createBiquadFilter();
        filter.type = 'lowpass';
        filter.frequency.value = filterFreq;

        const gainNode = this.audioContext.createGain();
        const now = this.audioContext.currentTime;
        gainNode.gain.setValueAtTime(0.5, now);
        gainNode.gain.linearRampToValueAtTime(0, now + duration);

        noise.connect(filter);
        filter.connect(gainNode);
        gainNode.connect(this.masterGain);

        noise.start(now);
    }

    // ================== Game Sound Effects ==================

    /**
     * Card play sound - soft thud
     */
    cardPlay() {
        this.playNoise(0.08, 400);
        this.playTone(150, 0.1, 'sine', 0.005, 0.05);
    }

    /**
     * Card deal sound - lighter click
     */
    cardDeal() {
        this.playNoise(0.05, 800);
    }

    /**
     * Bid placed sound - confirmation beep
     */
    bidPlaced() {
        this.playTone(440, 0.1, 'sine', 0.01, 0.05);
        setTimeout(() => this.playTone(550, 0.15, 'sine', 0.01, 0.08), 80);
    }

    /**
     * Your turn notification
     */
    yourTurn() {
        this.playTone(523, 0.12, 'sine', 0.01, 0.05); // C5
        setTimeout(() => this.playTone(659, 0.12, 'sine', 0.01, 0.05), 100); // E5
        setTimeout(() => this.playTone(784, 0.2, 'sine', 0.01, 0.1), 200); // G5
    }

    /**
     * Trick won sound - triumphant
     */
    trickWon() {
        this.playTone(392, 0.15, 'triangle', 0.01, 0.08); // G4
        setTimeout(() => this.playTone(494, 0.15, 'triangle', 0.01, 0.08), 100); // B4
        setTimeout(() => this.playTone(587, 0.25, 'triangle', 0.01, 0.15), 200); // D5
    }

    /**
     * Trick lost sound - descending
     */
    trickLost() {
        this.playTone(400, 0.12, 'sawtooth', 0.01, 0.08);
        setTimeout(() => this.playTone(350, 0.15, 'sawtooth', 0.01, 0.1), 100);
    }

    /**
     * Round complete sound
     */
    roundComplete() {
        const notes = [523, 587, 659, 784]; // C5, D5, E5, G5
        notes.forEach((freq, i) => {
            setTimeout(() => this.playTone(freq, 0.2, 'sine', 0.01, 0.1), i * 120);
        });
    }

    /**
     * Game over / victory fanfare
     */
    gameOver() {
        const notes = [392, 494, 587, 784, 988]; // G4, B4, D5, G5, B5
        notes.forEach((freq, i) => {
            setTimeout(() => this.playTone(freq, 0.3, 'triangle', 0.02, 0.15), i * 150);
        });
    }

    /**
     * Special card played (pirates, skull king, etc)
     */
    specialCard() {
        this.playTone(200, 0.15, 'sawtooth', 0.01, 0.05);
        this.playNoise(0.1, 600);
        setTimeout(() => this.playTone(300, 0.2, 'sawtooth', 0.01, 0.1), 50);
    }

    /**
     * Skull King played - dramatic
     */
    skullKing() {
        this.playTone(100, 0.3, 'sawtooth', 0.02, 0.1);
        setTimeout(() => this.playTone(150, 0.25, 'sawtooth', 0.01, 0.1), 100);
        setTimeout(() => this.playTone(200, 0.4, 'sawtooth', 0.01, 0.2), 200);
        this.playNoise(0.15, 300);
    }

    /**
     * Mermaid captures Skull King
     */
    mermaidCapture() {
        const notes = [659, 784, 988, 1175]; // E5, G5, B5, D6
        notes.forEach((freq, i) => {
            setTimeout(() => this.playTone(freq, 0.2, 'sine', 0.01, 0.1), i * 80);
        });
    }

    /**
     * Kraken played - destruction
     */
    kraken() {
        this.playNoise(0.3, 200);
        this.playTone(80, 0.4, 'sawtooth', 0.02, 0.2);
        setTimeout(() => this.playTone(60, 0.3, 'sawtooth', 0.01, 0.15), 150);
    }

    /**
     * Error / invalid action
     */
    error() {
        this.playTone(200, 0.15, 'square', 0.01, 0.05);
        setTimeout(() => this.playTone(150, 0.2, 'square', 0.01, 0.1), 100);
    }

    /**
     * Button click
     */
    click() {
        this.playTone(800, 0.05, 'sine', 0.005, 0.02);
    }

    /**
     * Toast notification
     */
    notify() {
        this.playTone(600, 0.08, 'sine', 0.01, 0.03);
        setTimeout(() => this.playTone(800, 0.1, 'sine', 0.01, 0.05), 60);
    }
}

// Create global sound manager instance
window.soundManager = new SoundManager();
