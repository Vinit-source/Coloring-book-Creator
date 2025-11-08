import React from 'react';
import { CloseIcon } from './icons/CloseIcon';
import { useLanguage } from '../contexts/LanguageContext';

interface UpgradeModalProps {
  isOpen: boolean;
  onClose: () => void;
  onUpgrade: () => void;
}

const UpgradeModal: React.FC<UpgradeModalProps> = ({ isOpen, onClose, onUpgrade }) => {
  const { t } = useLanguage();
  if (!isOpen) return null;

  return (
    <div
      className="fixed inset-0 bg-black/60 z-50 flex items-center justify-center p-4"
      aria-modal="true"
      role="dialog"
    >
      <div className="relative bg-white rounded-2xl shadow-2xl max-w-md w-full p-8 text-center transform transition-all scale-100 opacity-100">
        <button
          onClick={onClose}
          className="absolute top-4 right-4 text-gray-400 hover:text-gray-600"
          aria-label="Close"
        >
          <CloseIcon className="w-6 h-6" />
        </button>

        <div className="mx-auto flex items-center justify-center h-16 w-16 rounded-full bg-yellow-100 mb-4">
          <svg className="h-10 w-10 text-yellow-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 3v4M3 5h4M6 17v4m-2-2h4m5-14l-3 3m5 5l3 3m-5-13l3 3m-5 5l-3 3m10-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
        </div>

        <h3 className="text-2xl font-bold font-display text-blue-900 mb-2">{t('upgradeTitle')}</h3>
        <p className="text-gray-600 mb-1">{t('upgradeSubtitle1')}</p>
        <p className="text-gray-600 mb-6">{t('upgradeSubtitle2')}</p>
        
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-6">
            <p className="text-blue-900 font-bold text-lg">{t('proPlan')}</p>
            <p className="text-3xl font-bold text-pink-500 my-2">₹2000</p>
            <p className="text-blue-800" dangerouslySetInnerHTML={{ __html: t('planBenefit') }}></p>
        </div>

        <button
          onClick={onUpgrade}
          className="w-full bg-pink-500 text-white font-bold py-3 px-4 rounded-lg hover:bg-pink-600 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-pink-500 transition-all duration-300 transform hover:scale-105"
        >
          {t('upgradeButton')}
        </button>
        <p className="text-xs text-gray-400 mt-3">{t('paymentProcessor')}</p>

      </div>
    </div>
  );
};

export default UpgradeModal;