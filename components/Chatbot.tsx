import React, { useState, useRef, useEffect } from 'react';
import { ChatIcon } from './icons/ChatIcon';
import { CloseIcon } from './icons/CloseIcon';
import { SendIcon } from './icons/SendIcon';
import { getChatResponse } from '../services/geminiService';
import { ChatMessage } from '../types';

interface ChatbotProps {
    sessionId: string;
}

const Chatbot: React.FC<ChatbotProps> = ({ sessionId }) => {
    const [isOpen, setIsOpen] = useState(false);
    const [messages, setMessages] = useState<ChatMessage[]>([
        { id: 'initial', role: 'bot', text: "Hi! I'm your friendly coloring assistant. Ask me anything!" }
    ]);
    const [inputValue, setInputValue] = useState('');
    const [isTyping, setIsTyping] = useState(false);
    const messagesEndRef = useRef<HTMLDivElement>(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(scrollToBottom, [messages]);

    const handleSendMessage = async () => {
        if (inputValue.trim() === '') return;

        const userMessage: ChatMessage = { id: Date.now().toString(), role: 'user', text: inputValue };
        setMessages(prev => [...prev, userMessage]);
        setInputValue('');
        setIsTyping(true);

        try {
            const botResponseText = await getChatResponse(sessionId, inputValue);
            const botMessage: ChatMessage = { id: (Date.now() + 1).toString(), role: 'bot', text: botResponseText };
            setMessages(prev => [...prev, botMessage]);
        } catch (error: any) {
            const errorMessage: ChatMessage = { id: (Date.now() + 1).toString(), role: 'bot', text: error.message };
            setMessages(prev => [...prev, errorMessage]);
        } finally {
            setIsTyping(false);
        }
    };

    return (
        <>
            <button
                onClick={() => setIsOpen(!isOpen)}
                className="fixed bottom-6 right-6 bg-blue-600 text-white w-16 h-16 rounded-full shadow-lg flex items-center justify-center transform hover:scale-110 transition-transform duration-300 z-50"
                aria-label="Toggle Chatbot"
            >
                {isOpen ? <CloseIcon className="w-8 h-8"/> : <ChatIcon className="w-8 h-8"/>}
            </button>
            
            {isOpen && (
                <div className="fixed bottom-24 right-6 w-full max-w-sm h-[60vh] bg-white rounded-2xl shadow-2xl flex flex-col z-40 overflow-hidden border">
                    <header className="bg-blue-600 text-white p-4 font-bold text-lg">
                        Coloring Assistant
                    </header>
                    <div className="flex-1 p-4 overflow-y-auto bg-gray-50">
                        {messages.map((msg) => (
                            <div key={msg.id} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'} mb-3`}>
                                <div className={`px-4 py-2 rounded-xl max-w-xs ${msg.role === 'user' ? 'bg-blue-500 text-white' : 'bg-gray-200 text-gray-800'}`}>
                                    {msg.text}
                                </div>
                            </div>
                        ))}
                        {isTyping && (
                             <div className="flex justify-start mb-3">
                                <div className="px-4 py-2 rounded-xl max-w-xs bg-gray-200 text-gray-800">
                                    <div className="flex items-center space-x-1">
                                        <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce [animation-delay:-0.3s]"></div>
                                        <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce [animation-delay:-0.15s]"></div>
                                        <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce"></div>
                                    </div>
                                </div>
                            </div>
                        )}
                        <div ref={messagesEndRef} />
                    </div>
                    <div className="p-4 bg-white border-t">
                        <div className="flex items-center space-x-2">
                            <input
                                type="text"
                                value={inputValue}
                                onChange={(e) => setInputValue(e.target.value)}
                                onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
                                placeholder="Ask a question..."
                                className="flex-1 px-4 py-2 border rounded-full focus:ring-2 focus:ring-blue-500 focus:outline-none"
                            />
                            <button onClick={handleSendMessage} disabled={isTyping} className="bg-blue-600 text-white p-3 rounded-full hover:bg-blue-700 disabled:bg-blue-300 transition">
                                <SendIcon className="w-5 h-5"/>
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </>
    );
};

export default Chatbot;
