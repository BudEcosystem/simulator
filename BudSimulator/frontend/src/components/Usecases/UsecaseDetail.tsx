import React, { useState, useEffect } from 'react';
import { 
  ArrowLeft, Briefcase, Clock, Zap, Tag, TrendingUp, 
  AlertCircle, Loader2, BarChart3, Target, Cpu, 
  Package, GitBranch, Activity, CheckCircle, XCircle,
  DollarSign, ChevronDown, ChevronUp, Star, Filter
} from 'lucide-react';
import { Usecase } from '../../types/usecase';
import { usecaseAPI } from '../../services/usecaseAPI';

interface UsecaseDetailProps {
  usecaseId: string;
  onBack: () => void;
}

// New interfaces for recommendations
interface HardwareOption {
  hardware_name: string;
  nodes_required: number;
  memory_per_chip: number;
  utilization: number;
  price_per_hour: number;
  total_cost_per_hour: number;
  optimality: string;
}

interface BatchConfiguration {
  batch_size: number;
  memory_required_gb: number;
  meets_slo: boolean;
  estimated_ttft: number;
  estimated_e2e: number;
  hardware_options: HardwareOption[];
}

interface ModelRecommendation {
  model_id: string;
  parameter_count: number;
  model_type: string;
  attention_type: string;
  batch_configurations: BatchConfiguration[];
}

interface CategoryRecommendation {
  model_category: string;
  recommended_models: ModelRecommendation[];
}

interface RecommendationsResponse {
  usecase: Usecase;
  recommendations: CategoryRecommendation[];
}

export const UsecaseDetail: React.FC<UsecaseDetailProps> = ({ usecaseId, onBack }) => {
  const [usecase, setUsecase] = useState<Usecase | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  // New state for recommendations
  const [recommendations, setRecommendations] = useState<RecommendationsResponse | null>(null);
  const [loadingRecommendations, setLoadingRecommendations] = useState(false);
  const [recommendationsError, setRecommendationsError] = useState<string | null>(null);
  const [showRecommendations, setShowRecommendations] = useState(false);
  const [selectedBatchSize, setSelectedBatchSize] = useState(1);
  const [selectedCategory, setSelectedCategory] = useState('8B');
  const [selectedManufacturer, setSelectedManufacturer] = useState('all');

  // Helper to extract manufacturer from hardware name
  const getManufacturer = (hardwareName: string) => {
    const name = hardwareName.toLowerCase();
    if (name.includes('nvidia') || name.includes('a100') || name.includes('h100') || name.includes('rtx')) return 'NVIDIA';
    if (name.includes('intel') || name.includes('gaudi')) return 'Intel';
    if (name.includes('amd')) return 'AMD';
    if (name.includes('tpu')) return 'Google';
    return 'Other';
  };

  // Get unique manufacturers from the current recommendations
  const availableManufacturers = React.useMemo(() => {
    if (!recommendations) return ['all'];
    const allHardware = recommendations.recommendations
      .flatMap(cat => cat.recommended_models)
      .flatMap(model => model.batch_configurations)
      .flatMap(config => config.hardware_options);
    
    const manufacturers = new Set(allHardware.map(hw => getManufacturer(hw.hardware_name)));
    return ['all', ...Array.from(manufacturers)];
  }, [recommendations]);

  useEffect(() => {
    const fetchUsecase = async () => {
      setLoading(true);
      setError(null);
      
      try {
        const data = await usecaseAPI.getUsecase(usecaseId);
        setUsecase(data);
        
        // Automatically fetch recommendations after usecase is loaded
        if (data) {
          fetchRecommendations(data);
        }
      } catch (err) {
        setError('Failed to load usecase details. Please try again later.');
        console.error('Error fetching usecase:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchUsecase();
  }, [usecaseId]);

  const fetchRecommendations = async (usecaseData?: Usecase) => {
    const targetUsecase = usecaseData || usecase;
    if (!targetUsecase) return;
    
    setLoadingRecommendations(true);
    setRecommendationsError(null);
    
    try {
      const response = await fetch(`/api/usecases/${usecaseId}/recommendations`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          batch_sizes: [1, 8, 16, 32, 64],
          model_categories: ['3B', '8B', '32B', '72B', '200B+'],
          precision: 'fp16',
          include_pricing: true
        })
      });
      
      if (!response.ok) {
        throw new Error('Failed to fetch recommendations');
      }
      
      const data = await response.json();
      setRecommendations(data);
      
      // Set the first available category as the default if not already set
      if (data.recommendations && data.recommendations.length > 0) {
        const availableCategories = data.recommendations.map((cat: CategoryRecommendation) => cat.model_category);
        if (!availableCategories.includes(selectedCategory)) {
          setSelectedCategory(availableCategories[0]);
        }
      }
      
      setShowRecommendations(true);
    } catch (err) {
      setRecommendationsError('Failed to load recommendations. Please try again later.');
      console.error('Error fetching recommendations:', err);
    } finally {
      setLoadingRecommendations(false);
    }
  };

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

  const getSLOComplianceColor = (meets_slo: boolean) => {
    return meets_slo ? 'text-green-400' : 'text-red-400';
  };

  const getOptimalityColor = (optimality: string) => {
    switch (optimality) {
      case 'optimal':
        return 'bg-green-500/20 text-green-400 border-green-500/30';
      case 'good':
        return 'bg-yellow-500/20 text-yellow-400 border-yellow-500/30';
      case 'ok':
        return 'bg-orange-500/20 text-orange-400 border-orange-500/30';
      default:
        return 'bg-gray-500/20 text-gray-400 border-gray-500/30';
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

        {/* NEW: Model & Hardware Recommendations Section */}
        <div className="bg-gray-900/50 rounded-lg shadow-sm border border-gray-800 p-6 mb-6">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-lg font-semibold text-white flex items-center gap-2">
              <BarChart3 className="w-5 h-5 text-purple-500" />
              Model & Hardware Recommendations
            </h2>
            
            {showRecommendations && recommendations && (
              <button
                onClick={() => setShowRecommendations(false)}
                className="text-gray-400 hover:text-white transition-colors text-sm"
              >
                Hide Recommendations
              </button>
            )}
          </div>

          {recommendationsError && (
            <div className="bg-red-900/20 border border-red-500/20 rounded-lg p-4 mb-4">
              <div className="flex items-center gap-2">
                <AlertCircle className="w-5 h-5 text-red-500" />
                <span className="text-red-300">{recommendationsError}</span>
              </div>
            </div>
          )}

          {loadingRecommendations && (
            <div className="text-center py-8">
              <Loader2 className="w-8 h-8 animate-spin text-purple-500 mx-auto mb-4" />
              <p className="text-gray-400">
                Analyzing {usecase.name} requirements and generating personalized recommendations...
              </p>
            </div>
          )}

          {showRecommendations && recommendations && (
            <div className="space-y-6">
              {/* Controls */}
              <div className="grid md:grid-cols-4 gap-4 p-4 bg-gray-800/30 rounded-lg">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Model Category
                  </label>
                  <select 
                    value={selectedCategory}
                    onChange={(e) => setSelectedCategory(e.target.value)}
                    className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white"
                  >
                    {recommendations?.recommendations?.length > 0 
                      ? recommendations.recommendations.map(cat => (
                          <option key={cat.model_category} value={cat.model_category}>
                            {cat.model_category} Parameters
                          </option>
                        ))
                      : ['3B', '8B', '32B', '72B', '200B+'].map(cat => (
                          <option key={cat} value={cat}>{cat} Parameters</option>
                        ))
                    }
                  </select>
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Batch Size
                  </label>
                  <select 
                    value={selectedBatchSize}
                    onChange={(e) => setSelectedBatchSize(Number(e.target.value))}
                    className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white"
                  >
                    {[1, 8, 16, 32, 64].map(size => (
                      <option key={size} value={size}>Batch Size {size}</option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Hardware Manufacturer
                  </label>
                  <select 
                    value={selectedManufacturer}
                    onChange={(e) => setSelectedManufacturer(e.target.value)}
                    className="w-full bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-white"
                  >
                    {availableManufacturers.map(m => (
                      <option key={m} value={m}>{m === 'all' ? 'All Manufacturers' : m}</option>
                    ))}
                  </select>
                </div>

                <div className="flex items-end">
                  <div className="w-full text-center">
                    <p className="text-sm text-gray-400 mb-1">Recommendations for</p>
                    <p className="text-white font-medium">{usecase.name}</p>
                  </div>
                </div>
              </div>

              {/* Recommendations Grid */}
              <div className="space-y-6">
                {recommendations.recommendations
                  ?.filter(cat => cat.model_category === selectedCategory)
                  ?.map(category => (
                    <div key={category.model_category} className="space-y-4">
                      <h3 className="text-xl font-semibold text-purple-400">
                        {category.model_category} Parameter Models
                      </h3>
                      
                      {category.recommended_models.map(model => {
                        const batchConfig = model.batch_configurations.find(
                          config => config.batch_size === selectedBatchSize
                        );
                        
                        return (
                          <div key={model.model_id} className="bg-gray-800/50 rounded-lg border border-gray-700 p-5">
                            {/* Model Header */}
                            <div className="flex items-center justify-between mb-4">
                              <div>
                                <h4 className="text-lg font-semibold text-white">{model.model_id}</h4>
                                <p className="text-gray-400 text-sm">
                                  {(model.parameter_count / 1e9).toFixed(1)}B parameters • {model.model_type}
                                  {model.attention_type && ` • ${model.attention_type.toUpperCase()}`}
                                </p>
                              </div>
                              
                              {batchConfig && (
                                <div className={`flex items-center gap-2 ${getSLOComplianceColor(batchConfig.meets_slo)}`}>
                                  {batchConfig.meets_slo ? (
                                    <CheckCircle className="w-5 h-5" />
                                  ) : (
                                    <XCircle className="w-5 h-5" />
                                  )}
                                  <span className="font-medium">
                                    {batchConfig.meets_slo ? 'Meets SLO' : 'SLO Risk'}
                                  </span>
                                </div>
                              )}
                            </div>

                            {/* Performance Metrics */}
                            {batchConfig && (
                              <div className="grid md:grid-cols-4 gap-3 mb-4">
                                <div className="bg-gray-900/50 rounded-lg p-3">
                                  <div className="flex items-center gap-2 mb-1">
                                    <Activity className="w-3 h-3 text-yellow-500" />
                                    <span className="text-xs text-gray-400">TTFT</span>
                                  </div>
                                  <p className="text-lg font-semibold text-white">
                                    {batchConfig.estimated_ttft.toFixed(2)}s
                                  </p>
                                  <p className="text-xs text-gray-500">
                                    Target: ≤{usecase.ttft_max}s
                                  </p>
                                </div>
                                
                                <div className="bg-gray-900/50 rounded-lg p-3">
                                  <div className="flex items-center gap-2 mb-1">
                                    <Clock className="w-3 h-3 text-blue-500" />
                                    <span className="text-xs text-gray-400">E2E</span>
                                  </div>
                                  <p className="text-lg font-semibold text-white">
                                    {batchConfig.estimated_e2e.toFixed(2)}s
                                  </p>
                                  <p className="text-xs text-gray-500">
                                    Target: ≤{usecase.e2e_max}s
                                  </p>
                                </div>
                                
                                <div className="bg-gray-900/50 rounded-lg p-3">
                                  <div className="flex items-center gap-2 mb-1">
                                    <Cpu className="w-3 h-3 text-purple-500" />
                                    <span className="text-xs text-gray-400">Memory</span>
                                  </div>
                                  <p className="text-lg font-semibold text-white">
                                    {batchConfig.memory_required_gb.toFixed(1)} GB
                                  </p>
                                  <p className="text-xs text-gray-500">Required</p>
                                </div>

                                <div className="bg-gray-900/50 rounded-lg p-3">
                                  <div className="flex items-center gap-2 mb-1">
                                    <Package className="w-3 h-3 text-green-500" />
                                    <span className="text-xs text-gray-400">Batch</span>
                                  </div>
                                  <p className="text-lg font-semibold text-white">
                                    {selectedBatchSize}
                                  </p>
                                  <p className="text-xs text-gray-500">Concurrent</p>
                                </div>
                              </div>
                            )}

                            {/* Hardware Options */}
                            {batchConfig && batchConfig.hardware_options && (
                              <div className="space-y-3">
                                <h5 className="text-sm font-medium text-gray-300 flex items-center gap-2">
                                  <Cpu className="w-4 h-4" />
                                  Recommended Hardware
                                </h5>
                                
                                <div className="grid gap-3">
                                  {batchConfig.hardware_options
                                    .filter(hw => selectedManufacturer === 'all' || getManufacturer(hw.hardware_name) === selectedManufacturer)
                                    .slice(0, 3)
                                    .map((hw, idx) => (
                                    <div key={idx} className="bg-gray-900/30 rounded-lg p-4 border border-gray-600">
                                      <div className="flex items-center justify-between mb-3">
                                        <div className="flex-1">
                                          <h6 className="font-medium text-white text-sm">{hw.hardware_name}</h6>
                                          <p className="text-xs text-gray-400">
                                            {hw.nodes_required} node{hw.nodes_required > 1 ? 's' : ''} • 
                                            {hw.memory_per_chip} GB per chip
                                          </p>
                                        </div>
                                        
                                        <div className="text-right">
                                          <p className="text-lg font-semibold text-green-400">
                                            ${hw.total_cost_per_hour?.toFixed(2)}/hr
                                          </p>
                                          <div className={`text-xs px-2 py-1 rounded-full border ${getOptimalityColor(hw.optimality)}`}>
                                            {hw.optimality}
                                          </div>
                                        </div>
                                      </div>
                                      
                                      <div className="flex items-center justify-between text-xs">
                                        <span className="text-gray-400">
                                          {hw.utilization?.toFixed(1)}% utilization
                                        </span>
                                        <div className="w-24 bg-gray-700 rounded-full h-2">
                                          <div 
                                            className="bg-gradient-to-r from-purple-500 to-pink-500 h-2 rounded-full"
                                            style={{ width: `${Math.min(100, hw.utilization)}%` }}
                                          />
                                        </div>
                                      </div>
                                    </div>
                                  ))}
                                </div>
                              </div>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  ))}
              </div>
            </div>
          )}

          {!showRecommendations && !loadingRecommendations && !recommendationsError && (
            <div className="text-center py-8">
              <BarChart3 className="w-12 h-12 text-gray-600 mx-auto mb-4" />
              <p className="text-gray-400 mb-4">
                No recommendations available at the moment
              </p>
              <button
                onClick={() => fetchRecommendations()}
                className="bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white px-4 py-2 rounded-lg font-medium transition-all duration-200 flex items-center gap-2 mx-auto"
              >
                <TrendingUp className="w-4 h-4" />
                Retry Recommendations
              </button>
            </div>
          )}
        </div>

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