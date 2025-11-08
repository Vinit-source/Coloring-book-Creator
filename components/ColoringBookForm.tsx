import React, { useState } from 'react';
import { useLanguage } from '../contexts/LanguageContext';

interface ColoringBookFormProps {
    onSubmit: (theme: string, name: string) => void;
    isLoading: boolean;
}

const ColoringBookForm: React.FC<ColoringBookFormProps> = ({ onSubmit, isLoading }) => {
    const [theme, setTheme] = useState('');
    const [name, setName] = useState('');
    const { t } = useLanguage();

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        if (theme.trim() && name.trim()) {
            onSubmit(theme, name);
        }
    };

    return (
        <div className="w-full max-w-lg bg-white/70 backdrop-blur-sm p-8 rounded-2xl shadow-lg border border-gray-200">
            <h2 className="text-3xl font-bold font-display text-center text-blue-900 mb-2">{t('createTitle')}</h2>
            <p className="text-center text-gray-600 mb-6">{t('createSubtitle')}</p>
            <form onSubmit={handleSubmit} className="space-y-6">
                <div>
                    <label htmlFor="theme" className="block text-sm font-medium text-gray-700 mb-1">{t('themeLabel')}</label>
                    <input
                        type="text"
                        id="theme"
                        value={theme}
                        onChange={(e) => setTheme(e.target.value)}
                        placeholder={t('themePlaceholder')}
                        className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-pink-500 focus:border-pink-500 transition"
                        required
                    />
                </div>
                <div>
                    <label htmlFor="name" className="block text-sm font-medium text-gray-700 mb-1">{t('nameLabel')}</label>
                    <input
                        type="text"
                        id="name"
                        value={name}
                        onChange={(e) => setName(e.target.value)}
                        placeholder={t('namePlaceholder')}
                        className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-pink-500 focus:border-pink-500 transition"
                        required
                    />
                </div>
                <button
                    type="submit"
                    disabled={isLoading}
                    className="w-full bg-pink-500 text-white font-bold py-3 px-4 rounded-lg hover:bg-pink-600 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-pink-500 disabled:bg-pink-300 disabled:cursor-not-allowed transition-all duration-300 transform hover:scale-105"
                >
                    {isLoading ? t('creatingButton') : t('generateButton')}
                </button>
            </form>
        </div>
    );
};

export default ColoringBookForm;