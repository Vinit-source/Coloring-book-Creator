import React, { useState, useMemo } from 'react';
import ColoringBookForm from './components/ColoringBookForm';
import GeneratedPages from './components/GeneratedPages';
import LoadingSpinner from './components/LoadingSpinner';
import Chatbot from './components/Chatbot';
import { generateColoringPages } from './services/geminiService';

const App: React.FC = () => {
    const [theme, setTheme] = useState('');
    const [childName, setChildName] = useState('');
    const [generatedImages, setGeneratedImages] = useState<string[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [appState, setAppState] = useState<'form' | 'loading' | 'results'>('form');
    
    // Generate a unique session ID for the chatbot
    const sessionId = useMemo(() => Date.now().toString(), []);

    const handleFormSubmit = async (newTheme: string, newName: string) => {
        setTheme(newTheme);
        setChildName(newName);
        setAppState('loading');
        setIsLoading(true);
        setError(null);

        try {
            const images = await generateColoringPages(newTheme);
            setGeneratedImages(images);
            setAppState('results');
        } catch (err: any) {
            setError(err.message || "An unknown error occurred.");
            setAppState('form');
        } finally {
            setIsLoading(false);
        }
    };

    const handleReset = () => {
        setTheme('');
        setChildName('');
        setGeneratedImages([]);
        setError(null);
        setAppState('form');
    };

    const renderContent = () => {
        switch (appState) {
            case 'loading':
                return <LoadingSpinner theme={theme} />;
            case 'results':
                return <GeneratedPages images={generatedImages} theme={theme} childName={childName} onReset={handleReset} />;
            case 'form':
            default:
                return (
                    <>
                        {error && (
                            <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded-lg relative mb-6" role="alert">
                                <strong className="font-bold">Oh no! </strong>
                                <span className="block sm:inline">{error}</span>
                            </div>
                        )}
                        <ColoringBookForm onSubmit={handleFormSubmit} isLoading={isLoading} />
                    </>
                );
        }
    };

    return (
        <div className="min-h-screen w-full bg-gradient-to-br from-blue-100 via-purple-100 to-pink-100 font-sans flex items-center justify-center p-4 relative">
            <div className="absolute top-0 left-0 w-full h-full bg-[url('data:image/svg+xml,%3Csvg%20width%3D%2260%22%20height%3D%2260%22%20viewBox%3D%220%200%2060%2060%22%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%3E%3Cg%20fill%3D%22none%22%20fill-rule%3D%22evenodd%22%3E%3Cg%20fill%3D%22%239C92AC%22%20fill-opacity%3D%220.07%22%3E%3Cpath%20d%3D%22M36%2034v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6%2034v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6%204V0H4v4H0v2h4v4h2V6h4V4H6z%22/%3E%3C%2Fg%3E%3C%2Fg%3E%3C%2Fsvg%3E')] opacity-50"></div>
            <main className="z-10 w-full">
                <div className="flex flex-col items-center">
                    {renderContent()}
                </div>
            </main>
            <Chatbot sessionId={sessionId} />
        </div>
    );
};

export default App;
