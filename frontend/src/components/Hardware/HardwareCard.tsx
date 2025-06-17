import React from 'react';
import { Zap, HardDrive, Activity, Cpu, DollarSign } from 'lucide-react';
import { Hardware } from '../../types/hardware';

interface HardwareCardProps {
  hardware: Hardware;
  onSelect: () => void;
}

export const HardwareCard: React.FC<HardwareCardProps> = ({ hardware, onSelect }) => {
  const formatNumber = (num: number) => {
    if (num >= 1000) {
      return (num / 1000).toFixed(1) + 'K';
    }
    return num.toLocaleString();
  };

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'gpu':
        return 'bg-green-100 text-green-700';
      case 'cpu':
        return 'bg-blue-100 text-blue-700';
      case 'accelerator':
        return 'bg-purple-100 text-purple-700';
      case 'asic':
        return 'bg-orange-100 text-orange-700';
      default:
        return 'bg-gray-100 text-gray-700';
    }
  };

  return (
    <div
      onClick={onSelect}
      className="bg-gray-900/50 rounded-lg shadow-md hover:shadow-lg transition-all duration-200 cursor-pointer border border-gray-800 hover:border-purple-500/50 p-6"
    >
      {/* Header */}
      <div className="flex items-start justify-between mb-4">
        <div className="flex-1">
          <h3 className="font-semibold text-lg text-white mb-1 line-clamp-2">
            {hardware.name}
          </h3>
          {hardware.manufacturer && (
            <p className="text-sm text-gray-400">{hardware.manufacturer}</p>
          )}
        </div>
        <span className={`text-xs font-medium px-2 py-1 rounded-full ${getTypeColor(hardware.type)}`}>
          {hardware.type.toUpperCase()}
        </span>
      </div>

      {/* Key Specifications */}
      <div className="space-y-3 mb-4">
        <div className="flex items-center text-sm">
          <Zap className="w-4 h-4 mr-2 text-yellow-500" />
          <span className="text-gray-400">Performance:</span>
          <span className="ml-auto font-medium text-white">{formatNumber(hardware.flops)} TFLOPS</span>
        </div>
        
        <div className="flex items-center text-sm">
          <HardDrive className="w-4 h-4 mr-2 text-blue-500" />
          <span className="text-gray-400">Memory:</span>
          <span className="ml-auto font-medium text-white">{hardware.memory_size} GB</span>
        </div>
        
        <div className="flex items-center text-sm">
          <Activity className="w-4 h-4 mr-2 text-green-500" />
          <span className="text-gray-400">Bandwidth:</span>
          <span className="ml-auto font-medium text-white">{formatNumber(hardware.memory_bw)} GB/s</span>
        </div>
        
        {hardware.power && (
          <div className="flex items-center text-sm">
            <Cpu className="w-4 h-4 mr-2 text-red-500" />
            <span className="text-gray-400">Power:</span>
            <span className="ml-auto font-medium text-white">{hardware.power} W</span>
          </div>
        )}
      </div>

      {/* Pricing Information */}
      {(hardware.min_on_prem_price || hardware.clouds.length > 0) && (
        <div className="border-t pt-3 mb-4">
          {hardware.min_on_prem_price && (
            <div className="flex items-center text-sm">
              <DollarSign className="w-4 h-4 mr-2 text-green-500" />
              <span className="text-gray-400">From:</span>
              <span className="ml-auto font-medium text-green-400">
                ${hardware.min_on_prem_price.toLocaleString()}
              </span>
            </div>
          )}
          
          {hardware.clouds.length > 0 && (
            <div className="text-xs text-gray-500 mt-1">
              Available on {hardware.clouds.length} cloud{hardware.clouds.length > 1 ? 's' : ''}
            </div>
          )}
        </div>
      )}

      {/* Performance Indicator */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          {hardware.real_values && (
            <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-green-100 text-green-800">
              Verified
            </span>
          )}
          {hardware.source === 'json_import' && (
            <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
              Imported
            </span>
          )}
        </div>
        
        <button className="text-purple-400 hover:text-purple-300 text-sm font-medium">
          See More →
        </button>
      </div>

      {/* Description Preview */}
      {hardware.description && (
        <div className="mt-3 pt-3 border-t">
          <p className="text-xs text-gray-500 line-clamp-2">
            {hardware.description}
          </p>
        </div>
      )}
    </div>
  );
}; 