import React, { useState } from 'react';
import { Info } from 'lucide-react';

interface SpecificationTooltipProps {
  content: string;
  className?: string;
}

export const SpecificationTooltip: React.FC<SpecificationTooltipProps> = ({
  content,
  className = ''
}) => {
  const [isVisible, setIsVisible] = useState(false);

  return (
    <div className={`relative inline-block ${className}`}>
      <button
        onMouseEnter={() => setIsVisible(true)}
        onMouseLeave={() => setIsVisible(false)}
        onFocus={() => setIsVisible(true)}
        onBlur={() => setIsVisible(false)}
        className="ml-1 text-gray-400 hover:text-gray-600 focus:outline-none"
        type="button"
      >
        <Info className="w-4 h-4" />
      </button>
      
      {isVisible && (
        <div className="absolute z-10 w-64 p-2 mt-1 text-sm text-white bg-gray-900 rounded-md shadow-lg -left-32 transform">
          <div className="relative">
            {content}
            <div className="absolute -top-1 left-1/2 transform -translate-x-1/2 w-2 h-2 bg-gray-900 rotate-45"></div>
          </div>
        </div>
      )}
    </div>
  );
}; 