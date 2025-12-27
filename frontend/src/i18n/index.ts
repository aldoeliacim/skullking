import i18n from 'i18next';
import { initReactI18next } from 'react-i18next';
import LanguageDetector from 'i18next-browser-languagedetector';

import en from './en.json';
import es from './es.json';

const STORAGE_KEY = 'skullking_language';

i18n
  .use(LanguageDetector)
  .use(initReactI18next)
  .init({
    resources: {
      en: { translation: en },
      es: { translation: es },
    },
    fallbackLng: 'en',
    supportedLngs: ['en', 'es'],
    interpolation: {
      escapeValue: false,
    },
    detection: {
      order: ['localStorage', 'navigator'],
      lookupLocalStorage: STORAGE_KEY,
      caches: ['localStorage'],
    },
  });

export const changeLanguage = (lang: 'en' | 'es') => {
  i18n.changeLanguage(lang);
  localStorage.setItem(STORAGE_KEY, lang);
};

export const getCurrentLanguage = (): 'en' | 'es' => {
  return (i18n.language?.substring(0, 2) as 'en' | 'es') || 'en';
};

export default i18n;
