import React, { createContext, useState, useContext, useEffect, useCallback } from 'react';
import { translations, TranslationKey } from '../translations';

interface Language {
    code: string;
    name: string;
}

interface LanguageContextType {
    selectedLanguage: string;
    setSelectedLanguage: (language: string) => void;
    t: (key: TranslationKey, replacements?: { [key: string]: string }) => string;
    languages: Language[];
}

const LanguageContext = createContext<LanguageContextType | undefined>(undefined);

export const languages: Language[] = [
    { code: 'en', name: 'English' },
    { code: 'hi', name: 'हिन्दी' },
    { code: 'bn', name: 'বাংলা' },
    { code: 'te', name: 'తెలుగు' },
    { code: 'mr', name: 'मराठी' },
    { code: 'ta', name: 'தமிழ்' },
    { code: 'gu', name: 'ગુજરાતી' },
    { code: 'kn', name: 'ಕನ್ನಡ' },
    { code: 'ml', name: 'മലയാളം' },
    { code: 'pa', name: 'ਪੰਜਾਬੀ' },
];

export const LanguageProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
    const [selectedLanguage, setSelectedLanguage] = useState<string>(() => {
        return localStorage.getItem('appLanguage') || 'en';
    });

    useEffect(() => {
        localStorage.setItem('appLanguage', selectedLanguage);
    }, [selectedLanguage]);

    const t = useCallback((key: TranslationKey, replacements: { [key: string]: string } = {}): string => {
        const langDict = translations[selectedLanguage] || translations['en'];
        let text = langDict[key] || translations['en'][key] || key;

        for (const placeholder in replacements) {
            text = text.replace(`{${placeholder}}`, replacements[placeholder]);
        }

        return text;
    }, [selectedLanguage]);

    return (
        <LanguageContext.Provider value={{ selectedLanguage, setSelectedLanguage, t, languages }}>
            {children}
        </LanguageContext.Provider>
    );
};

export const useLanguage = () => {
    const context = useContext(LanguageContext);
    if (context === undefined) {
        throw new Error('useLanguage must be used within a LanguageProvider');
    }
    return context;
};
