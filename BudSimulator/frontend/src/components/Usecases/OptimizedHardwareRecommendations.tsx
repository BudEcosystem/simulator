import React, { useState, useEffect } from 'react';
import { 
  Cpu, 
  Server, 
  DollarSign, 
  Zap, 
  Activity,
  Clock,
  CheckCircle,
  AlertCircle,
  ChevronRight,
  Filter,
  Layers,
  TrendingUp,
  Package,
  Gauge,
  Info,
  ExternalLink
} from 'lucide-react';

interface OptimizationConfig {
  model_id: string;
  model_size: string;
  hardware_type: string;
  num_nodes: number;
  parallelism: string;
  batch_size: number;
  achieved_ttft: number;
  achieved_e2e: number;
  required_ttft: number;
  required_e2e: number;
  meets_slo: boolean;
  cost_per_hour: number;
  cost_per_request: number;
  throughput: number;
  utilization: number;
  efficiency_score: number;
}

interface OptimizationResponse {
  usecase: any;
  configurations: OptimizationConfig[];
  optimization_mode: string;
  summary: {
    total_configurations_found: number;
    configurations_returned: number;
    model_sizes_evaluated: string[];
    hardware_types_found: string[];
    batch_sizes_evaluated: number[];
    min_cost_per_request: number | null;
    min_ttft_achieved: number | null;
    max_throughput: number | null;
  };
}

interface Props {
  usecaseId: string;
  usecase: any;
}

export const OptimizedHardwareRecommendations: React.FC<Props> = ({ usecaseId, usecase }) => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [optimizationData, setOptimizationData] = useState<OptimizationResponse | null>(null);
  const [selectedBatchSize, setSelectedBatchSize] = useState(1);
  const [selectedModelSize, setSelectedModelSize] = useState('all');
  const [optimizationMode, setOptimizationMode] = useState<'cost' | 'performance' | 'balanced'>('balanced');
  const [expandedConfig, setExpandedConfig] = useState<string | null>(null);

  const modelSizes = ['1B', '3B', '8B', '32B', '70B'];
  const batchSizes = [1, 4, 8, 16];

  useEffect(() => {
    fetchOptimizedConfigurations();
  }, [usecaseId, selectedBatchSize, optimizationMode]);

  const fetchOptimizedConfigurations = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`/api/usecases/${usecaseId}/optimize-hardware`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          batch_sizes: [selectedBatchSize],
          model_sizes: selectedModelSize === 'all' ? modelSizes : [selectedModelSize],
          max_results: 20,
          optimization_mode: optimizationMode
        })
      });

      if (!response.ok) {
        throw new Error('Failed to fetch optimized configurations');
      }

      const data = await response.json();
      setOptimizationData(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  const getModelSizeColor = (size: string) => {
    const colors: Record<string, string> = {
      '1B': 'text-green-400 bg-green-400/10 border-green-400/20',
      '3B': 'text-blue-400 bg-blue-400/10 border-blue-400/20',
      '8B': 'text-purple-400 bg-purple-400/10 border-purple-400/20',
      '32B': 'text-orange-400 bg-orange-400/10 border-orange-400/20',
      '70B': 'text-red-400 bg-red-400/10 border-red-400/20',
      '100B+': 'text-pink-400 bg-pink-400/10 border-pink-400/20'
    };
    return colors[size] || 'text-gray-400 bg-gray-400/10 border-gray-400/20';
  };

  const formatCost = (cost: number) => {
    if (cost < 0.01) return `$${(cost * 1000).toFixed(2)}/1K`;
    if (cost < 1) return `$${cost.toFixed(3)}`;
    return `$${cost.toFixed(2)}`;
  };

  const getSLOStatus = (achieved: number, required: number) => {
    const ratio = achieved / required;
    if (ratio <= 0.8) return { color: 'text-green-400', icon: CheckCircle, label: 'Excellent' };
    if (ratio <= 0.95) return { color: 'text-yellow-400', icon: AlertCircle, label: 'Good' };
    return { color: 'text-red-400', icon: AlertCircle, label: 'Tight' };
  };

  const getHardwareIcon = (type: string) => {
    if (type.includes('A100') || type.includes('H100')) return '🎮';
    if (type.includes('TPU')) return '🧠';
    if (type.includes('MI300')) return '🔴';
    return '💻';
  };

  if (loading) {
    return (
      <div className="bg-gray-800/50 rounded-xl p-8 border border-gray-700">
        <div className="flex items-center justify-center space-x-3">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-yellow-400"></div>
          <span className="text-gray-300">Bud is finding the most optimal hardware configuration for you...</span>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-red-900/20 border border-red-700 rounded-xl p-6">
        <div className="flex items-center gap-3">
          <AlertCircle className="w-6 h-6 text-red-400" />
          <div>
            <h4 className="text-red-400 font-semibold">Optimization Error</h4>
            <p className="text-gray-300 text-sm mt-1">{error}</p>
          </div>
        </div>
      </div>
    );
  }

  const configurations = optimizationData?.configurations || [];
  const summary = optimizationData?.summary;

  return (
    <div className="space-y-6">
      {/* Header with Controls */}
      <div className="bg-gray-800/50 rounded-xl p-6 border border-gray-700">
        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
          <div>
            <h3 className="text-xl font-bold text-white flex items-center gap-2">
              <Cpu className="w-6 h-6 text-yellow-400" />
              Optimized Hardware Configurations
            </h3>
            <p className="text-gray-400 text-sm mt-1">
              AI-powered recommendations using Bud Simulator performance modeling
            </p>
          </div>

          {/* Control Filters */}
          <div className="flex flex-wrap gap-3">
            {/* Batch Size Selector */}
            <div className="flex items-center gap-2 bg-gray-900/50 rounded-lg p-1">
              <Package className="w-4 h-4 text-gray-400 ml-2" />
              <select
                value={selectedBatchSize}
                onChange={(e) => setSelectedBatchSize(Number(e.target.value))}
                className="bg-transparent text-white text-sm px-2 py-1 focus:outline-none"
              >
                {batchSizes.map(size => (
                  <option key={size} value={size}>Batch {size}</option>
                ))}
              </select>
            </div>

            {/* Model Size Filter */}
            <div className="flex items-center gap-2 bg-gray-900/50 rounded-lg p-1">
              <Layers className="w-4 h-4 text-gray-400 ml-2" />
              <select
                value={selectedModelSize}
                onChange={(e) => setSelectedModelSize(e.target.value)}
                className="bg-transparent text-white text-sm px-2 py-1 focus:outline-none"
              >
                <option value="all">All Sizes</option>
                {modelSizes.map(size => (
                  <option key={size} value={size}>{size} Model</option>
                ))}
              </select>
            </div>

            {/* Optimization Mode */}
            <div className="flex bg-gray-900/50 rounded-lg p-1">
              <button
                onClick={() => setOptimizationMode('cost')}
                className={`px-3 py-1 text-sm rounded-md transition-colors ${
                  optimizationMode === 'cost'
                    ? 'bg-yellow-500/20 text-yellow-400'
                    : 'text-gray-400 hover:text-white'
                }`}
              >
                <DollarSign className="w-4 h-4 inline mr-1" />
                Cost
              </button>
              <button
                onClick={() => setOptimizationMode('performance')}
                className={`px-3 py-1 text-sm rounded-md transition-colors ${
                  optimizationMode === 'performance'
                    ? 'bg-yellow-500/20 text-yellow-400'
                    : 'text-gray-400 hover:text-white'
                }`}
              >
                <Zap className="w-4 h-4 inline mr-1" />
                Speed
              </button>
              <button
                onClick={() => setOptimizationMode('balanced')}
                className={`px-3 py-1 text-sm rounded-md transition-colors ${
                  optimizationMode === 'balanced'
                    ? 'bg-yellow-500/20 text-yellow-400'
                    : 'text-gray-400 hover:text-white'
                }`}
              >
                <Gauge className="w-4 h-4 inline mr-1" />
                Balanced
              </button>
            </div>
          </div>
        </div>

        {/* Summary Stats */}
        {summary && summary.total_configurations_found > 0 && (
          <div className="mt-4 pt-4 border-t border-gray-700 grid grid-cols-2 md:grid-cols-4 gap-4">
            <div>
              <p className="text-gray-400 text-xs">Min Cost/Request</p>
              <p className="text-white font-semibold">
                {summary.min_cost_per_request ? formatCost(summary.min_cost_per_request) : 'N/A'}
              </p>
            </div>
            <div>
              <p className="text-gray-400 text-xs">Best TTFT</p>
              <p className="text-white font-semibold">
                {summary.min_ttft_achieved ? `${summary.min_ttft_achieved.toFixed(2)}s` : 'N/A'}
              </p>
            </div>
            <div>
              <p className="text-gray-400 text-xs">Max Throughput</p>
              <p className="text-white font-semibold">
                {summary.max_throughput ? `${summary.max_throughput.toFixed(1)} req/s` : 'N/A'}
              </p>
            </div>
            <div>
              <p className="text-gray-400 text-xs">Valid Configs</p>
              <p className="text-white font-semibold">
                {summary.configurations_returned} found
              </p>
            </div>
          </div>
        )}
      </div>

      {/* Configuration Cards */}
      {configurations.length === 0 ? (
        <div className="bg-gray-800/30 rounded-xl p-8 border border-gray-700 text-center">
          <Cpu className="w-12 h-12 text-gray-600 mx-auto mb-4" />
          <p className="text-gray-400">No configurations found that meet the SLO requirements.</p>
          <p className="text-gray-500 text-sm mt-2">
            Try adjusting the batch size or relaxing the SLO constraints.
          </p>
        </div>
      ) : (
        <div className="space-y-4">
          {configurations.map((config, idx) => {
            const configId = `${config.model_id}-${config.hardware_type}-${idx}`;
            const isExpanded = expandedConfig === configId;
            const ttftStatus = getSLOStatus(config.achieved_ttft, config.required_ttft);
            const e2eStatus = getSLOStatus(config.achieved_e2e, config.required_e2e);

            return (
              <div
                key={configId}
                className={`bg-gray-800/50 rounded-xl border transition-all ${
                  idx === 0 ? 'border-yellow-500/50 shadow-lg shadow-yellow-500/10' : 'border-gray-700'
                }`}
              >
                {/* Main Configuration Info */}
                <div
                  className="p-6 cursor-pointer"
                  onClick={() => setExpandedConfig(isExpanded ? null : configId)}
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      {/* Model and Hardware Info */}
                      <div className="flex items-center gap-4 mb-3">
                        <span className={`px-3 py-1 rounded-full text-sm font-medium border ${getModelSizeColor(config.model_size)}`}>
                          {config.model_size} Model
                        </span>
                        <div className="flex items-center gap-2 text-white">
                          <span className="text-2xl">{getHardwareIcon(config.hardware_type)}</span>
                          <span className="font-semibold">{config.hardware_type}</span>
                          {config.num_nodes > 1 && (
                            <span className="text-gray-400">× {config.num_nodes} nodes</span>
                          )}
                        </div>
                      </div>

                      {/* Performance Metrics Grid */}
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        <div>
                          <div className="flex items-center gap-2 mb-1">
                            <Activity className="w-4 h-4 text-yellow-400" />
                            <span className="text-xs text-gray-400">TTFT</span>
                            <ttftStatus.icon className={`w-3 h-3 ${ttftStatus.color}`} />
                          </div>
                          <p className="text-lg font-semibold text-white">
                            {config.achieved_ttft.toFixed(2)}s
                          </p>
                          <p className="text-xs text-gray-500">
                            Target ≤{config.required_ttft}s
                          </p>
                        </div>

                        <div>
                          <div className="flex items-center gap-2 mb-1">
                            <Clock className="w-4 h-4 text-blue-400" />
                            <span className="text-xs text-gray-400">E2E</span>
                            <e2eStatus.icon className={`w-3 h-3 ${e2eStatus.color}`} />
                          </div>
                          <p className="text-lg font-semibold text-white">
                            {config.achieved_e2e.toFixed(2)}s
                          </p>
                          <p className="text-xs text-gray-500">
                            Target ≤{config.required_e2e}s
                          </p>
                        </div>

                        <div>
                          <div className="flex items-center gap-2 mb-1">
                            <DollarSign className="w-4 h-4 text-green-400" />
                            <span className="text-xs text-gray-400">Cost/Request</span>
                          </div>
                          <p className="text-lg font-semibold text-white">
                            {formatCost(config.cost_per_request)}
                          </p>
                          <p className="text-xs text-gray-500">
                            ${config.cost_per_hour.toFixed(2)}/hr
                          </p>
                        </div>

                        <div>
                          <div className="flex items-center gap-2 mb-1">
                            <TrendingUp className="w-4 h-4 text-purple-400" />
                            <span className="text-xs text-gray-400">Throughput</span>
                          </div>
                          <p className="text-lg font-semibold text-white">
                            {config.throughput.toFixed(1)}
                          </p>
                          <p className="text-xs text-gray-500">req/sec</p>
                        </div>
                      </div>
                    </div>

                    {/* Ranking Badge */}
                    {idx === 0 && (
                      <div className="ml-4 bg-yellow-500/20 text-yellow-400 px-3 py-1 rounded-full text-sm font-medium">
                        Best {optimizationMode === 'cost' ? 'Value' : optimizationMode === 'performance' ? 'Speed' : 'Overall'}
                      </div>
                    )}

                    {/* Expand Arrow */}
                    <ChevronRight className={`w-5 h-5 text-gray-400 ml-4 transition-transform ${
                      isExpanded ? 'rotate-90' : ''
                    }`} />
                  </div>
                </div>

                {/* Expanded Details */}
                {isExpanded && (
                  <div className="px-6 pb-6 border-t border-gray-700 pt-4">
                    <div className="grid md:grid-cols-2 gap-6">
                      {/* Left Column - Technical Details */}
                      <div className="space-y-4">
                        <div>
                          <h4 className="text-sm font-semibold text-gray-300 mb-2 flex items-center gap-2">
                            <Server className="w-4 h-4" />
                            Configuration Details
                          </h4>
                          <div className="bg-gray-900/50 rounded-lg p-4 space-y-2 text-sm">
                            <div className="flex justify-between">
                              <span className="text-gray-400">Model</span>
                              <span className="text-white font-mono">{config.model_id}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-gray-400">Parallelism</span>
                              <span className="text-white font-mono">{config.parallelism}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-gray-400">Batch Size</span>
                              <span className="text-white">{config.batch_size}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-gray-400">Utilization</span>
                              <span className="text-white">{(config.utilization * 100).toFixed(1)}%</span>
                            </div>
                          </div>
                        </div>
                      </div>

                      {/* Right Column - Deployment Guide */}
                      <div className="space-y-4">
                        <div>
                          <h4 className="text-sm font-semibold text-gray-300 mb-2 flex items-center gap-2">
                            <Info className="w-4 h-4" />
                            Deployment Guide
                          </h4>
                          <div className="bg-gray-900/50 rounded-lg p-4 space-y-3">
                            <p className="text-sm text-gray-300">
                              Deploy {config.model_size} model on {config.num_nodes} × {config.hardware_type} 
                              {config.num_nodes > 1 ? ' nodes' : ' node'} with {config.parallelism} parallelization.
                            </p>
                            <div className="flex items-center gap-2 text-xs text-blue-400">
                              <ExternalLink className="w-3 h-3" />
                              <a href="#" className="hover:underline">View deployment instructions</a>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}

      {/* Info Box */}
      <div className="bg-blue-900/20 border border-blue-700/50 rounded-xl p-4">
        <div className="flex items-start gap-3">
          <Info className="w-5 h-5 text-blue-400 mt-0.5" />
          <div className="text-sm">
            <p className="text-blue-300 font-medium mb-1">About Bud Optimization</p>
            <p className="text-gray-300">
              These recommendations use advanced performance modeling to predict actual latencies and throughput
              based on hardware capabilities, memory bandwidth, and parallelization strategies.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};