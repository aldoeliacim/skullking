/**
 * Lightweight i18n (internationalization) utility for Skull King
 * Supports English (en) and Spanish (es)
 */
class I18n {
    constructor() {
        this.translations = {};
        this.currentLocale = this.getStoredLocale() || this.detectLocale();
        this.fallbackLocale = 'en';
        this.loadedLocales = new Set();
    }

    /**
     * Get stored locale from localStorage
     */
    getStoredLocale() {
        return localStorage.getItem('skullking-locale');
    }

    /**
     * Detect user's preferred locale from browser
     */
    detectLocale() {
        const browserLang = navigator.language || navigator.userLanguage;
        const lang = browserLang.split('-')[0];
        return ['en', 'es'].includes(lang) ? lang : 'en';
    }

    /**
     * Load translations for a locale
     */
    async loadLocale(locale) {
        if (this.loadedLocales.has(locale)) {
            return;
        }

        try {
            const response = await fetch(`/static/locales/${locale}.json`);
            if (!response.ok) {
                throw new Error(`Failed to load locale: ${locale}`);
            }
            this.translations[locale] = await response.json();
            this.loadedLocales.add(locale);
        } catch (error) {
            console.error(`Error loading locale ${locale}:`, error);
            // Fall back to English if loading fails
            if (locale !== this.fallbackLocale) {
                await this.loadLocale(this.fallbackLocale);
            }
        }
    }

    /**
     * Initialize i18n - load current locale and apply translations
     */
    async init() {
        await this.loadLocale(this.currentLocale);
        if (this.currentLocale !== this.fallbackLocale) {
            await this.loadLocale(this.fallbackLocale);
        }
        this.applyTranslations();
        this.updateDocumentLang();
    }

    /**
     * Get a translation by key path (e.g., "login.enterName")
     * Supports interpolation with {variable} syntax
     */
    t(keyPath, params = {}) {
        const keys = keyPath.split('.');
        let value = this.translations[this.currentLocale];

        // Try current locale
        for (const key of keys) {
            if (value && typeof value === 'object') {
                value = value[key];
            } else {
                value = undefined;
                break;
            }
        }

        // Fall back to fallback locale
        if (value === undefined && this.currentLocale !== this.fallbackLocale) {
            value = this.translations[this.fallbackLocale];
            for (const key of keys) {
                if (value && typeof value === 'object') {
                    value = value[key];
                } else {
                    value = undefined;
                    break;
                }
            }
        }

        // Return key if translation not found
        if (value === undefined) {
            console.warn(`Missing translation: ${keyPath}`);
            return keyPath;
        }

        // Interpolate parameters
        if (typeof value === 'string' && Object.keys(params).length > 0) {
            return value.replace(/\{(\w+)\}/g, (match, key) => {
                return params[key] !== undefined ? params[key] : match;
            });
        }

        return value;
    }

    /**
     * Apply translations to all elements with data-i18n attribute
     */
    applyTranslations() {
        // Translate text content
        document.querySelectorAll('[data-i18n]').forEach(element => {
            const key = element.getAttribute('data-i18n');
            element.textContent = this.t(key);
        });

        // Translate placeholders
        document.querySelectorAll('[data-i18n-placeholder]').forEach(element => {
            const key = element.getAttribute('data-i18n-placeholder');
            element.placeholder = this.t(key);
        });

        // Translate title
        const titleKey = document.querySelector('title')?.getAttribute('data-i18n');
        if (titleKey) {
            document.title = this.t(titleKey);
        }
    }

    /**
     * Update document lang attribute
     */
    updateDocumentLang() {
        document.documentElement.lang = this.currentLocale;
    }

    /**
     * Switch to a different locale
     */
    async setLocale(locale) {
        if (!['en', 'es'].includes(locale)) {
            console.warn(`Unsupported locale: ${locale}`);
            return;
        }

        this.currentLocale = locale;
        localStorage.setItem('skullking-locale', locale);

        await this.loadLocale(locale);
        this.applyTranslations();
        this.updateDocumentLang();

        // Dispatch event for dynamic content updates
        window.dispatchEvent(new CustomEvent('localeChanged', { detail: { locale } }));
    }

    /**
     * Get current locale
     */
    getLocale() {
        return this.currentLocale;
    }

    /**
     * Toggle between available locales
     */
    async toggleLocale() {
        const newLocale = this.currentLocale === 'en' ? 'es' : 'en';
        await this.setLocale(newLocale);
    }
}

// Create global i18n instance
window.i18n = new I18n();
