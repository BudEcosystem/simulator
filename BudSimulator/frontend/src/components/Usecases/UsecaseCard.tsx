import React from 'react';
import { Briefcase, Clock, Zap, Tag, TrendingUp } from 'lucide-react';
import { Usecase } from '../../types/usecase';

interface UsecaseCardProps {
  usecase: Usecase;
  onClick: () => void;
}

export const UsecaseCard: React.FC<UsecaseCardProps> = ({ usecase, onClick }) => {
  const getLatencyProfileColor = (profile?: string) => {
    switch (profile) {
      case 'real-time':
        return 'text-red-400 bg-red-900/20 border-red-500/20';
      case 'interactive':
        return 'text-orange-400 bg-orange-900/20 border-orange-500/20';
      case 'responsive':
        return 'text-blue-400 bg-blue-900/20 border-blue-500/20';
      case 'batch':
        return 'text-green-400 bg-green-900/20 border-green-500/20';
      default:
        return 'text-gray-400 bg-gray-800 border-gray-600';
    }
  };

  const getLatencyProfileIcon = (profile?: string) => {
    switch (profile) {
      case 'real-time':
        return <Zap className="w-3 h-3" />;
      case 'interactive':
      case 'responsive':
        return <Clock className="w-3 h-3" />;
      case 'batch':
        return <TrendingUp className="w-3 h-3" />;
      default:
        return null;
    }
  };

  return (
    <div
      onClick={onClick}
      className="bg-gray-900/50 rounded-lg border border-gray-800 p-6 hover:shadow-lg hover:shadow-purple-500/25 transition-all cursor-pointer hover:border-purple-500/50"
    >
      <div className="flex items-start justify-between mb-4">
        <div className="flex-1">
          <h3 className="text-lg font-semibold text-white mb-1">{usecase.name}</h3>
          <div className="flex items-center gap-2 text-sm text-gray-400">
            <Briefcase className="w-4 h-4" />
            <span>{usecase.industry}</span>
          </div>
        </div>
        {usecase.latency_profile && (
          <div className={`flex items-center gap-1 px-2 py-1 rounded-full text-xs font-medium border ${getLatencyProfileColor(usecase.latency_profile)}`}>
            {getLatencyProfileIcon(usecase.latency_profile)}
            <span className="capitalize">{usecase.latency_profile}</span>
          </div>
        )}
      </div>

      <p className="text-gray-300 text-sm mb-4 line-clamp-2">{usecase.description}</p>

      <div className="space-y-3">
        {/* Performance Characteristics */}
        <div className="flex items-center gap-4 text-sm">
          <div className="flex items-center gap-1 text-gray-400">
            <Zap className="w-4 h-4" />
            <span>TTFT: {usecase.ttft_min}-{usecase.ttft_max}s</span>
          </div>
          <div className="flex items-center gap-1 text-gray-400">
            <Clock className="w-4 h-4" />
            <span>E2E: {usecase.e2e_min}-{usecase.e2e_max}s</span>
          </div>
        </div>

        {/* Tags */}
        {usecase.tags && usecase.tags.length > 0 && (
          <div className="flex items-start gap-2">
            <Tag className="w-4 h-4 text-gray-500 mt-0.5" />
            <div className="flex flex-wrap gap-1">
              {(usecase.tags || []).slice(0, 3).map((tag, index) => (
                <span
                  key={index}
                  className="px-2 py-0.5 bg-gray-800 text-gray-300 rounded text-xs"
                >
                  {tag}
                </span>
              ))}
              {(usecase.tags || []).length > 3 && (
                <span className="px-2 py-0.5 text-gray-500 text-xs">
                  +{(usecase.tags || []).length - 3} more
                </span>
              )}
            </div>
          </div>
        )}

        {/* Token Information */}
        <div className="flex items-center gap-4 text-xs text-gray-500">
          <span>Input: {usecase.input_tokens_min}-{usecase.input_tokens_max} tokens</span>
          <span>Output: {usecase.output_tokens_min}-{usecase.output_tokens_max} tokens</span>
        </div>
      </div>
    </div>
  );
}; 