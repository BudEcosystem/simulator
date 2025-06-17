import React, { useEffect, useState } from 'react';
import { Cpu, TrendingUp, Zap, AlertCircle } from 'lucide-react';
import { Hardware } from '../../types/hardware';
import { hardwareAPI, buildHardwareParams } from '../../services/hardwareAPI';

interface FeaturedHardwareProps {
  onHardwareClick: (hardwareName: string) => void;
}

export const FeaturedHardware: React.FC<FeaturedHardwareProps> = ({ onHardwareClick }) => {
  const [hardware, setHardware] = useState<Hardware[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchFeaturedHardware();
  }, []);

  const fetchFeaturedHardware = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const params = buildHardwareParams({
        type: 'gpu', // Focus on GPUs and accelerators for featured section
        sortBy: 'flops',
        sortOrder: 'desc',
        limit: 6
      });
      
      const data = await hardwareAPI.filter(params);
      setHardware(data);
    } catch (error) {
      console.error('Error fetching featured hardware:', error);
      setError('Failed to load featured hardware');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="mb-12">
        <div className="animate-pulse">
          <div className="h-8 bg-gray-700 rounded w-48 mb-6"></div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {[...Array(6)].map((_, i) => (
              <div key={i} className="h-40 bg-gray-700 rounded-lg"></div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="mb-12">
        <h2 className="text-2xl font-bold mb-6 flex items-center text-white">
          <Cpu className="w-6 h-6 mr-2 text-purple-600" />
          Popular AI Hardware
        </h2>
        <div className="text-center py-8">
          <AlertCircle className="w-12 h-12 text-red-500 mx-auto mb-4" />
          <p className="text-gray-400 mb-4">{error}</p>
          <button
            onClick={fetchFeaturedHardware}
            className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700"
          >
            Try Again
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="mb-12">
      <h2 className="text-2xl font-bold mb-6 flex items-center text-white">
        <Cpu className="w-6 h-6 mr-2 text-purple-600" />
        Popular AI Hardware
      </h2>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {hardware.map((hw) => (
          <button
            key={hw.name}
            onClick={() => onHardwareClick(hw.name)}
            className="bg-gray-900/50 rounded-lg shadow-md p-4 hover:shadow-lg transition-all duration-200 text-left border border-gray-800 hover:border-purple-500/50"
          >
            <div className="flex items-start justify-between mb-3">
              <h3 className="font-semibold text-lg text-white line-clamp-2 flex-1 mr-2">
                {hw.name}
              </h3>
              <span className="text-xs bg-purple-600/20 text-purple-400 px-2 py-1 rounded-full whitespace-nowrap">
                {hw.type.toUpperCase()}
              </span>
            </div>
            
            <div className="space-y-2 text-sm text-gray-400">
              <div className="flex items-center">
                <Zap className="w-4 h-4 mr-2 text-yellow-500" />
                <span className="text-white">{hw.flops.toLocaleString()} TFLOPS</span>
              </div>
              <div className="flex items-center">
                <TrendingUp className="w-4 h-4 mr-2 text-blue-500" />
                <span className="text-white">{hw.memory_size} GB Memory</span>
              </div>
              {hw.manufacturer && (
                <div className="text-xs text-gray-500 mt-2">{hw.manufacturer}</div>
              )}
            </div>

            {/* Quick performance indicator */}
            <div className="mt-3 pt-3 border-t border-gray-700">
              <div className="flex items-center justify-between">
                <span className="text-xs text-gray-500">
                  {hw.memory_bw.toLocaleString()} GB/s bandwidth
                </span>
                {hw.min_on_prem_price && (
                  <span className="text-xs font-medium text-green-400">
                    From ${hw.min_on_prem_price.toLocaleString()}
                  </span>
                )}
              </div>
            </div>
          </button>
        ))}
      </div>
      
      <div className="text-center mt-6">
        <button
          onClick={() => onHardwareClick('')}
          className="text-purple-400 hover:text-purple-300 font-medium transition-colors"
        >
          View All Hardware →
        </button>
      </div>
    </div>
  );
}; 