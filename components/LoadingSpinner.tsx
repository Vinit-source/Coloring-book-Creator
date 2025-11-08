import React, { useState, useEffect, useMemo } from 'react';
import { useLanguage } from '../contexts/LanguageContext';

interface LoadingSpinnerProps {
    theme: string;
}

const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({ theme }) => {
    const { t } = useLanguage();
    
    const messages = useMemo(() => [
        t('loadingMessage1'),
        t('loadingMessage2', { theme }),
        t('loadingMessage3'),
        t('loadingMessage4', { theme }),
        t('loadingMessage5'),
        t('loadingMessage6', { theme }),
        t('loadingMessage7'),
        t('loadingMessage8'),
    ], [t, theme]);

    const [message, setMessage] = useState(messages[0]);

    useEffect(() => {
        const intervalId = setInterval(() => {
            setMessage(messages[Math.floor(Math.random() * messages.length)]);
        }, 2000);

        return () => clearInterval(intervalId);
    }, [messages]);


    return (
        <div className="flex flex-col items-center justify-center text-center p-8">
            <div className="w-16 h-16 border-8 border-dashed rounded-full animate-spin border-pink-500"></div>
            <h2 className="text-2xl font-display text-blue-800 mt-6">{t('generatingTitle')}</h2>
            <p className="text-gray-600 mt-2">{message}</p>
        </div>
    );
};

export default LoadingSpinner;