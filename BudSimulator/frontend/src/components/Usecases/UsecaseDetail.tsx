import React, { useState, useEffect } from 'react';
import { 
  ArrowLeft, Briefcase, Clock, Zap, Tag, TrendingUp, 
  AlertCircle, Loader2, Target, Cpu, 
  Package, GitBranch, Activity, CheckCircle, XCircle
} from 'lucide-react';
import { Usecase } from '../../types/usecase';
import { usecaseAPI } from '../../services/usecaseAPI';
import { OptimizedHardwareRecommendations } from './OptimizedHardwareRecommendations';

interface UsecaseDetailProps {
  usecaseId: string;
  onBack: () => void;
}

export const UsecaseDetail: React.FC<UsecaseDetailProps> = ({ usecaseId, onBack }) => {
  const [usecase, setUsecase] = useState<Usecase | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  


  useEffect(() => {
    const fetchUsecase = async () => {
      setLoading(true);
      setError(null);
      
      try {
        const data = await usecaseAPI.getUsecase(usecaseId);
        setUsecase(data);
      } catch (err) {
        setError('Failed to load usecase details. Please try again later.');
        console.error('Error fetching usecase:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchUsecase();
  }, [usecaseId]);


  const getLatencyProfileColor = (profile?: string) => {
    switch (profile) {
      case 'real-time':
        return 'text-red-600 bg-red-50 border-red-200';
      case 'interactive':
        return 'text-orange-600 bg-orange-50 border-orange-200';
      case 'responsive':
        return 'text-blue-600 bg-blue-50 border-blue-200';
      case 'batch':
        return 'text-green-600 bg-green-50 border-green-200';
      default:
        return 'text-gray-600 bg-gray-50 border-gray-200';
    }
  };


  const formatNumber = (num: number | null) => {
    if (num === null) return 'N/A';
    return num.toLocaleString();
  };

  const formatPercentage = (num: number) => {
    return `${(num * 100).toFixed(1)}%`;
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <Loader2 className="w-8 h-8 animate-spin text-purple-600 mx-auto mb-4" />
          <p className="text-gray-600">Loading usecase details...</p>
        </div>
      </div>
    );
  }

  if (error || !usecase) {
    return (
      <div className="min-h-screen bg-gray-50 px-4 py-8">
        <div className="max-w-4xl mx-auto">
          <button
            onClick={onBack}
            className="flex items-center gap-2 text-gray-600 hover:text-gray-900 mb-6"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to usecases
          </button>
          
          <div className="bg-red-50 border border-red-200 rounded-lg p-6 flex items-center gap-3">
            <AlertCircle className="w-6 h-6 text-red-600 flex-shrink-0" />
            <div>
              <h3 className="text-lg font-medium text-red-900">Error loading usecase</h3>
              <p className="text-red-700 mt-1">{error || 'Usecase not found'}</p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-black text-white pt-16">
      <div className="max-w-6xl mx-auto px-4 py-8">
        {/* Back Button */}
        <button
          onClick={onBack}
          className="flex items-center gap-2 text-gray-400 hover:text-white transition-colors mb-6"
        >
          <ArrowLeft className="w-4 h-4" />
          Back to usecases
        </button>

        {/* Header */}
        <div className="bg-gray-900/50 rounded-lg shadow-sm border border-gray-800 p-6 mb-6">
          <div className="flex items-start justify-between mb-4">
            <div>
              <h1 className="text-2xl font-bold text-white mb-2">{usecase.name}</h1>
              <div className="flex items-center gap-4 text-sm">
                <div className="flex items-center gap-1 text-gray-400">
                  <Briefcase className="w-4 h-4" />
                  <span>{usecase.industry}</span>
                </div>
                <div className={`flex items-center gap-1 ${usecase.is_active ? 'text-green-400' : 'text-gray-500'}`}>
                  {usecase.is_active ? <CheckCircle className="w-4 h-4" /> : <XCircle className="w-4 h-4" />}
                  <span>{usecase.is_active ? 'Active' : 'Inactive'}</span>
                </div>
              </div>
            </div>
            {usecase.latency_profile && (
              <div className={`px-3 py-1.5 rounded-full text-sm font-medium border ${getLatencyProfileColor(usecase.latency_profile)}`}>
                {usecase.latency_profile}
              </div>
            )}
          </div>
          <p className="text-gray-300">{usecase.description}</p>
        </div>

        {/* Performance Characteristics */}
        <div className="bg-gray-900/50 rounded-lg shadow-sm border border-gray-800 p-6 mb-6">
          <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <Zap className="w-5 h-5 text-purple-500" />
            Performance Characteristics
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-gray-800/50 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <Activity className="w-4 h-4 text-gray-400" />
                <span className="text-sm font-medium text-gray-300">Time to First Token</span>
              </div>
              <p className="text-2xl font-bold text-white">
                {usecase.ttft_min === usecase.ttft_max ? `${usecase.ttft_min}s` : `${usecase.ttft_min}-${usecase.ttft_max}s`}
              </p>
              <p className="text-xs text-gray-500 mt-1">Response start time</p>
            </div>
            
            <div className="bg-gray-800/50 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <Clock className="w-4 h-4 text-gray-400" />
                <span className="text-sm font-medium text-gray-300">End-to-End Latency</span>
              </div>
              <p className="text-2xl font-bold text-white">
                {usecase.e2e_min === usecase.e2e_max ? `${usecase.e2e_min}s` : `${usecase.e2e_min}-${usecase.e2e_max}s`}
              </p>
              <p className="text-xs text-gray-500 mt-1">Total response time</p>
            </div>
            
            <div className="bg-gray-800/50 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <TrendingUp className="w-4 h-4 text-gray-400" />
                <span className="text-sm font-medium text-gray-300">Inter-token Latency</span>
              </div>
              <p className="text-2xl font-bold text-white">
                {usecase.inter_token_min === usecase.inter_token_max ? `${usecase.inter_token_min}s` : `${usecase.inter_token_min}-${usecase.inter_token_max}s`}
              </p>
              <p className="text-xs text-gray-500 mt-1">Between tokens</p>
            </div>
          </div>
        </div>

        {/* Token Configuration */}
        <div className="bg-gray-900/50 rounded-lg shadow-sm border border-gray-800 p-6 mb-6">
          <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <Package className="w-5 h-5 text-purple-500" />
            Token Configuration
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div>
              <h3 className="text-sm font-medium text-gray-300 mb-3">Input Tokens</h3>
              <div className="bg-gray-800/50 rounded-lg p-3">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-400">Range:</span>
                  <span className="text-lg font-semibold text-white">
                    {usecase.input_tokens_min === usecase.input_tokens_max ? 
                      usecase.input_tokens_min : 
                      `${usecase.input_tokens_min} - ${usecase.input_tokens_max}`
                    }
                  </span>
                </div>
              </div>
            </div>
            
            <div>
              <h3 className="text-sm font-medium text-gray-300 mb-3">Output Tokens</h3>
              <div className="bg-gray-800/50 rounded-lg p-3">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-400">Range:</span>
                  <span className="text-lg font-semibold text-white">
                    {usecase.output_tokens_min === usecase.output_tokens_max ? 
                      usecase.output_tokens_min : 
                      `${usecase.output_tokens_min} - ${usecase.output_tokens_max}`
                    }
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Configuration Details */}
        <div className="bg-gray-900/50 rounded-lg shadow-sm border border-gray-800 p-6 mb-6">
          <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <Target className="w-5 h-5 text-purple-500" />
            Configuration Details
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="bg-gray-800/50 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <Package className="w-4 h-4 text-gray-400" />
                <span className="text-sm font-medium text-gray-300">Batch Size</span>
              </div>
              <p className="text-2xl font-bold text-white">{usecase.batch_size}</p>
              <p className="text-xs text-gray-500 mt-1">Concurrent requests</p>
            </div>
            
            <div className="bg-gray-800/50 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <GitBranch className="w-4 h-4 text-gray-400" />
                <span className="text-sm font-medium text-gray-300">Beam Size</span>
              </div>
              <p className="text-2xl font-bold text-white">{usecase.beam_size}</p>
              <p className="text-xs text-gray-500 mt-1">Search width</p>
            </div>

            <div className="bg-gray-800/50 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <CheckCircle className="w-4 h-4 text-gray-400" />
                <span className="text-sm font-medium text-gray-300">Source</span>
              </div>
              <p className="text-lg font-medium text-white capitalize">{usecase.source}</p>
              <p className="text-xs text-gray-500 mt-1">Data origin</p>
            </div>
          </div>
        </div>

        {/* Optimized Hardware Recommendations */}
        <OptimizedHardwareRecommendations usecaseId={usecaseId} usecase={usecase} />

        {/* Tags */}
        {usecase.tags && usecase.tags.length > 0 && (
          <div className="bg-gray-900/50 rounded-lg shadow-sm border border-gray-800 p-6 mb-6">
            <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
              <Tag className="w-5 h-5 text-purple-500" />
              Tags
            </h2>
            <div className="flex flex-wrap gap-2">
              {(usecase.tags || []).map((tag, index) => (
                <span
                  key={index}
                  className="px-3 py-1.5 bg-gray-800 text-gray-300 rounded-full text-sm"
                >
                  {tag}
                </span>
              ))}
            </div>
          </div>
        )}

        {/* Metadata */}
        <div className="bg-gray-800/50 rounded-lg p-4 text-sm text-gray-400">
          <div className="flex justify-between">
            <span>Unique ID: {usecase.unique_id}</span>
            <span>Created: {new Date(usecase.created_at).toLocaleDateString()}</span>
            <span>Updated: {new Date(usecase.updated_at).toLocaleDateString()}</span>
          </div>
        </div>
      </div>
    </div>
  );
}; 