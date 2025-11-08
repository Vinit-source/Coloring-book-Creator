import React, { createContext, useState, useContext, useEffect, useCallback } from 'react';

const API_KEY_STORAGE_KEY = 'geminiApiKey';

interface ApiKeyContextType {
    apiKey: string | null;
    isKeySet: boolean;
    saveApiKey: (key: string) => void;
}

const ApiKeyContext = createContext<ApiKeyContextType | undefined>(undefined);

export const ApiKeyProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
    const [apiKey, setApiKey] = useState<string | null>(null);
    const [isKeySet, setIsKeySet] = useState(false);

    useEffect(() => {
        const storedKey = localStorage.getItem(API_KEY_STORAGE_KEY);
        if (storedKey) {
            setApiKey(storedKey);
            setIsKeySet(true);
        }
    }, []);

    const saveApiKey = useCallback((key: string) => {
        if (key && key.trim()) {
            localStorage.setItem(API_KEY_STORAGE_KEY, key);
            setApiKey(key);
            setIsKeySet(true);
        }
    }, []);

    return (
        <ApiKeyContext.Provider value={{ apiKey, isKeySet, saveApiKey }}>
            {children}
        </ApiKeyContext.Provider>
    );
};

export const useApiKey = () => {
    const context = useContext(ApiKeyContext);
    if (context === undefined) {
        throw new Error('useApiKey must be used within an ApiKeyProvider');
    }
    return context;
};
