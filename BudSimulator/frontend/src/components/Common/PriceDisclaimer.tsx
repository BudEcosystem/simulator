import React from 'react';

export const PriceDisclaimer: React.FC = () => {
  return (
    <div className="bg-gray-900/80 backdrop-blur-sm border border-yellow-500/20 rounded-lg p-4 mb-4">
      <div className="flex items-start">
        <svg className="w-5 h-5 text-yellow-500 mt-0.5 mr-2 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
          <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
        </svg>
        <div className="text-sm text-gray-300">
          <p className="font-semibold mb-1 text-yellow-400">Pricing Disclaimer</p>
          <p className="text-gray-400">
            The price indicators shown are approximate values intended for relative comparison only. 
            They do not represent actual costs and may vary extensively based on vendor, region, 
            and configuration. Future versions will include detailed pricing from on-premise vendors 
            and cloud providers.
          </p>
        </div>
      </div>
    </div>
  );
}; 