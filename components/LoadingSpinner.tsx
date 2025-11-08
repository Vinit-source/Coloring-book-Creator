import React, { useState, useEffect, useMemo } from 'react';

interface LoadingSpinnerProps {
    theme: string;
}

const generateMessages = (theme: string): string[] => {
    // Capitalize the first letter of the theme for better display
    const capitalizedTheme = theme.charAt(0).toUpperCase() + theme.slice(1);
    return [
        "Warming up the AI crayons...",
        `Teaching robots how to draw ${capitalizedTheme}...`,
        "Sharpening our digital pencils...",
        `Dreaming up wonderful scenes about ${capitalizedTheme}...`,
        "Mixing the perfect shades of black and white...",
        `Searching for the cutest ${theme.toLowerCase()}...`,
        "Making sure the lines are extra thick for coloring...",
        "Preparing your personalized art studio...",
    ];
};


const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({ theme }) => {
    const messages = useMemo(() => generateMessages(theme), [theme]);
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
            <h2 className="text-2xl font-display text-blue-800 mt-6">Generating Your Masterpiece!</h2>
            <p className="text-gray-600 mt-2">{message}</p>
        </div>
    );
};

export default LoadingSpinner;
