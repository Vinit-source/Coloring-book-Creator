import React from 'react';
import { usePdfGenerator } from '../hooks/usePdfGenerator';
import { useLanguage } from '../contexts/LanguageContext';

interface GeneratedPagesProps {
    images: string[];
    theme: string;
    childName: string;
    onReset: () => void;
}

const GeneratedPages: React.FC<GeneratedPagesProps> = ({ images, theme, childName, onReset }) => {
    const { downloadPdf } = usePdfGenerator();
    const { t } = useLanguage();

    const handleDownload = () => {
        downloadPdf(images, theme, childName, t('coverTitle'), t('coverByline'));
    };
    
    return (
        <div className="w-full max-w-4xl bg-white/80 backdrop-blur-sm p-8 rounded-2xl shadow-lg border border-gray-200">
            <h2 className="text-3xl font-bold font-display text-center text-blue-900 mb-4">{t('resultsTitle')}</h2>
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4 mb-6">
                {images.map((img, index) => (
                    <div key={index} className="aspect-[4/3] bg-white p-2 rounded-lg shadow-md">
                        <img 
                            src={`data:image/png;base64,${img}`} 
                            alt={`Coloring page ${index + 1}`}
                            className="w-full h-full object-contain"
                        />
                    </div>
                ))}
                <div className="aspect-[4/3] bg-gray-100 p-2 rounded-lg shadow-md flex items-center justify-center text-center">
                   <p className="text-gray-500 font-semibold">{t('customCover')}</p>
                </div>
            </div>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
                 <button
                    onClick={handleDownload}
                    className="w-full sm:w-auto bg-green-500 text-white font-bold py-3 px-6 rounded-lg hover:bg-green-600 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-green-500 transition-all duration-300 transform hover:scale-105"
                >
                    {t('downloadButton')}
                </button>
                <button
                    onClick={onReset}
                    className="w-full sm:w-auto bg-gray-500 text-white font-bold py-3 px-6 rounded-lg hover:bg-gray-600 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-gray-500 transition-all duration-300"
                >
                    {t('createAnotherButton')}
                </button>
            </div>
        </div>
    );
};

export default GeneratedPages;