import React, { useState } from 'react';
import { useApiKey } from '../contexts/ApiKeyContext';
import { useLanguage } from '../contexts/LanguageContext';

interface ApiKeyModalProps {
  isOpen: boolean;
}

const ApiKeyModal: React.FC<ApiKeyModalProps> = ({ isOpen }) => {
  const { saveApiKey } = useApiKey();
  const { t } = useLanguage();
  const [key, setKey] = useState('');

  const handleSave = () => {
    if (key.trim()) {
      saveApiKey(key.trim());
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-gray-900/80 z-50 flex items-center justify-center p-4 backdrop-blur-sm">
      <div className="bg-white rounded-2xl shadow-2xl max-w-lg w-full p-8 text-center transform transition-all scale-100 opacity-100">
        <h2 className="text-3xl font-bold font-display text-blue-900 mb-4">{t('apiKeyModalTitle')}</h2>
        
        <p className="text-gray-600 mb-4">{t('apiKeyModalDisclaimer')}</p>
        <p className="text-sm text-gray-500 mb-6">{t('apiKeyModalFreeTier')}</p>

        <div className="text-left mb-6">
          <label htmlFor="apiKeyInput" className="block text-sm font-medium text-gray-700 mb-1">
            {t('apiKeyModalInputLabel')}
          </label>
          <input
            id="apiKeyInput"
            type="password"
            value={key}
            onChange={(e) => setKey(e.target.value)}
            placeholder={t('apiKeyModalInputPlaceholder')}
            className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-pink-500 focus:border-pink-500 transition"
          />
        </div>

        <a 
          href="https://aistudio.google.com/app/apikey" 
          target="_blank" 
          rel="noopener noreferrer"
          className="text-sm text-blue-600 hover:underline mb-6 block"
        >
          {t('apiKeyModalGetAKeyLink')}
        </a>
        
        <button
          onClick={handleSave}
          disabled={!key.trim()}
          className="w-full bg-pink-500 text-white font-bold py-3 px-4 rounded-lg hover:bg-pink-600 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-pink-500 disabled:bg-pink-300 disabled:cursor-not-allowed transition-all duration-300 transform hover:scale-105"
        >
          {t('apiKeyModalSaveButton')}
        </button>
      </div>
    </div>
  );
};

export default ApiKeyModal;
