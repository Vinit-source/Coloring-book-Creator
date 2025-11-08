import React, { useState, useEffect } from 'react';
import { CloseIcon } from './icons/CloseIcon';
import LanguageSelector from './LanguageSelector';
import { useApiKey } from '../contexts/ApiKeyContext';
import { useLanguage } from '../contexts/LanguageContext';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
}

const SettingsModal: React.FC<SettingsModalProps> = ({ isOpen, onClose }) => {
  const { t } = useLanguage();
  const { apiKey, saveApiKey } = useApiKey();
  const [currentKey, setCurrentKey] = useState(apiKey || '');

  useEffect(() => {
    setCurrentKey(apiKey || '');
  }, [apiKey, isOpen]);

  const handleSave = () => {
    saveApiKey(currentKey.trim());
    onClose();
  };

  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 bg-black/60 z-50 flex items-center justify-center p-4"
      aria-modal="true"
      role="dialog"
    >
      <div className="relative bg-white rounded-2xl shadow-2xl max-w-md w-full p-8 text-left transform transition-all scale-100 opacity-100">
        <div className="flex justify-between items-center mb-6">
          <h3 className="text-2xl font-bold font-display text-blue-900">{t('settingsModalTitle')}</h3>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600"
            aria-label="Close"
          >
            <CloseIcon className="w-6 h-6" />
          </button>
        </div>

        <div className="space-y-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">{t('settingsLanguageLabel')}</label>
            <LanguageSelector />
          </div>
          <div>
            <label htmlFor="settingsApiKeyInput" className="block text-sm font-medium text-gray-700 mb-1">
              {t('settingsApiKeyLabel')}
            </label>
            <input
              id="settingsApiKeyInput"
              type="password"
              value={currentKey}
              onChange={(e) => setCurrentKey(e.target.value)}
              placeholder={t('apiKeyModalInputPlaceholder')}
              className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-pink-500 focus:border-pink-500 transition"
            />
          </div>
        </div>

        <button
          onClick={handleSave}
          className="mt-8 w-full bg-pink-500 text-white font-bold py-3 px-4 rounded-lg hover:bg-pink-600 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-pink-500 transition-all duration-300 transform hover:scale-105"
        >
          {t('settingsSaveButton')}
        </button>
      </div>
    </div>
  );
};

export default SettingsModal;