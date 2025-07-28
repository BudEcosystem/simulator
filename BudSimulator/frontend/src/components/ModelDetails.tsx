import React, { useState, useEffect } from 'react';
import { ArrowLeft, Database, CheckCircle, AlertCircle, ExternalLink, Info, Cpu, HardDrive, Settings, BarChart3, FileText, Zap } from 'lucide-react';
import { ModelLogoWithFallback } from '../AIMemoryCalculator';

interface ModelAnalysisEval {
  name: string;
  score: number;
}

interface ModelAnalysis {
  description: string;
  advantages: string[];
  disadvantages: string[];
  usecases: string[];
  evals: ModelAnalysisEval[];
}

interface ModelConfig {
  hidden_size?: number;
  num_hidden_layers?: number;
  num_attention_heads?: number;
  num_key_value_heads?: number;
  intermediate_size?: number;
  vocab_size?: number;
  max_position_embeddings?: number;
  activation_function?: string;
  model_type?: string;
  torch_dtype?: string;
  num_parameters?: number;
  [key: string]: any; // For other config properties
}

interface ModelDetails {
  model_id: string;
  name: string;
  author: string | null;
  model_type: string;
  attention_type: string | null;
  parameter_count: number | null;
  logo?: string;
  source: string;
  in_model_dict: boolean;
  in_database: boolean;
  analysis?: ModelAnalysis | null;
  config?: ModelConfig;
}

interface ModelDetailsProps {
  modelId: string;
  onBack: () => void;
}

const ModelDetails: React.FC<ModelDetailsProps> = ({ modelId, onBack }) => {
  const [details, setDetails] = useState<ModelDetails | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState('');
  const [activeTab, setActiveTab] = useState('overview');

  useEffect(() => {
    const fetchDetails = async () => {
      setIsLoading(true);
      setError('');
      try {
        const response = await fetch(`http://localhost:8000/api/models/${encodeURIComponent(modelId)}`);
        if (!response.ok) {
          throw new Error('Failed to fetch model details');
        }
        const data = await response.json();
        setDetails(data);
      } catch (err) {
        setError('Failed to load model details. Please try again later.');
        console.error('Error fetching model details:', err);
      } finally {
        setIsLoading(false);
      }
    };

    fetchDetails();
  }, [modelId]);

  if (isLoading) {
    return (
      <div className="min-h-screen bg-black text-white pt-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="text-center py-16">
            <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-purple-500 mx-auto"></div>
            <p className="mt-4 text-gray-400">Loading model details...</p>
          </div>
        </div>
      </div>
    );
  }

  if (error || !details) {
    return (
      <div className="min-h-screen bg-black text-white pt-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="text-center py-16">
            <AlertCircle className="w-12 h-12 text-red-500 mx-auto mb-4" />
            <p className="text-red-400">{error || 'Failed to load model details'}</p>
            <button
              onClick={onBack}
              className="mt-6 bg-purple-600 hover:bg-purple-700 text-white px-6 py-3 rounded-lg transition-colors"
            >
              Go Back
            </button>
          </div>
        </div>
      </div>
    );
  }

  const formatParameterCount = (count: number) => {
    if (count >= 1e9) return `${(count / 1e9).toFixed(1)}B`;
    if (count >= 1e6) return `${(count / 1e6).toFixed(1)}M`;
    if (count >= 1e3) return `${(count / 1e3).toFixed(1)}K`;
    return count.toString();
  };

  const tabs = [
    { id: 'overview', label: 'Overview', icon: Info },
    { id: 'analysis', label: 'Analysis', icon: BarChart3, disabled: !details.analysis },
    { id: 'technical', label: 'Technical', icon: Settings },
  ];

  return (
    <div className="min-h-screen bg-black text-white pt-16">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Header */}
        <div className="flex items-center space-x-4 mb-8">
          <button
            onClick={onBack}
            className="text-gray-400 hover:text-white transition-colors"
          >
            <ArrowLeft className="w-5 h-5" />
          </button>
          <h1 className="text-3xl font-bold">Model Details</h1>
        </div>

        {/* Hero Section */}
        <div className="bg-gray-900/50 rounded-2xl p-8 border border-gray-800 mb-8">
          <div className="flex items-start justify-between">
            <div className="flex items-center space-x-6">
              <ModelLogoWithFallback logo={details.logo} modelId={details.model_id} size="lg" />
              <div>
                <h2 className="text-3xl font-bold mb-2">{details.name}</h2>
                <p className="text-gray-400 mb-4">{details.model_id}</p>
                <div className="flex items-center space-x-6 text-sm">
                  <div className="flex items-center space-x-2">
                    <Cpu className="w-4 h-4 text-purple-400" />
                    <span className="text-purple-400">
                      {details.parameter_count ? formatParameterCount(details.parameter_count) : 'Unknown'} parameters
                    </span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <Database className="w-4 h-4 text-blue-400" />
                    <span className="text-blue-400">{details.model_type}</span>
                  </div>
                  {details.attention_type && (
                    <div className="flex items-center space-x-2">
                      <Zap className="w-4 h-4 text-green-400" />
                      <span className="text-green-400">{details.attention_type.toUpperCase()}</span>
                    </div>
                  )}
                </div>
              </div>
            </div>
            <a
              href={`https://huggingface.co/${details.model_id}`}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center space-x-2 text-purple-400 hover:text-purple-300 transition-colors"
            >
              <span>View on HuggingFace</span>
              <ExternalLink className="w-4 h-4" />
            </a>
          </div>
        </div>

        {/* Tabs */}
        <div className="mb-8">
          <div className="border-b border-gray-800">
            <nav className="-mb-px flex space-x-8">
              {tabs.map((tab) => {
                const Icon = tab.icon;
                return (
                  <button
                    key={tab.id}
                    onClick={() => !tab.disabled && setActiveTab(tab.id)}
                    className={`py-4 px-1 border-b-2 font-medium text-sm flex items-center space-x-2 transition-colors ${
                      activeTab === tab.id
                        ? 'border-purple-500 text-purple-400'
                        : tab.disabled
                        ? 'border-transparent text-gray-600 cursor-not-allowed'
                        : 'border-transparent text-gray-400 hover:text-gray-300 hover:border-gray-300'
                    }`}
                    disabled={tab.disabled}
                  >
                    <Icon className="w-4 h-4" />
                    <span>{tab.label}</span>
                  </button>
                );
              })}
            </nav>
          </div>
        </div>

        {/* Tab Content */}
        {activeTab === 'overview' && (
          <div className="grid lg:grid-cols-3 gap-8">
            {/* Basic Info */}
            <div className="lg:col-span-2 space-y-6">
              <div className="bg-gray-900/50 rounded-2xl p-6 border border-gray-800">
                <h3 className="text-xl font-semibold mb-4">Basic Information</h3>
                <div className="grid md:grid-cols-2 gap-4">
                  <div>
                    <label className="text-sm text-gray-400">Author</label>
                    <p className="text-white">{details.author || 'Unknown'}</p>
                  </div>
                  <div>
                    <label className="text-sm text-gray-400">Source</label>
                    <p className="text-white capitalize">{details.source}</p>
                  </div>
                  <div>
                    <label className="text-sm text-gray-400">Model Type</label>
                    <p className="text-white">{details.model_type}</p>
                  </div>
                  <div>
                    <label className="text-sm text-gray-400">Attention Type</label>
                    <p className="text-white">{details.attention_type || 'N/A'}</p>
                  </div>
                </div>
              </div>

              {details.analysis?.description && (
                <div className="bg-gray-900/50 rounded-2xl p-6 border border-gray-800">
                  <h3 className="text-xl font-semibold mb-4">Description</h3>
                  <p className="text-gray-300 leading-relaxed">{details.analysis.description}</p>
                </div>
              )}
            </div>

            {/* Quick Stats */}
            <div className="space-y-6">
              <div className="bg-gray-900/50 rounded-2xl p-6 border border-gray-800">
                <h3 className="text-xl font-semibold mb-4">Quick Stats</h3>
                <div className="space-y-4">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Parameters</span>
                    <span className="text-white font-medium">
                      {details.parameter_count ? formatParameterCount(details.parameter_count) : 'Unknown'}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">In Model Dict</span>
                    <span className={details.in_model_dict ? 'text-green-400' : 'text-gray-400'}>
                      {details.in_model_dict ? 'Yes' : 'No'}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">In Database</span>
                    <span className={details.in_database ? 'text-green-400' : 'text-gray-400'}>
                      {details.in_database ? 'Yes' : 'No'}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Analysis Available</span>
                    <span className={details.analysis ? 'text-green-400' : 'text-gray-400'}>
                      {details.analysis ? 'Yes' : 'No'}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'analysis' && details.analysis && (
          <div className="grid lg:grid-cols-2 gap-8">
            {/* Left Column */}
            <div className="space-y-8">
              {/* Advantages & Disadvantages */}
              <div className="grid grid-cols-1 gap-6">
                <div className="bg-gray-900/50 rounded-2xl p-6 border border-gray-800">
                  <h3 className="text-xl font-semibold mb-4 text-green-400">Advantages</h3>
                  <ul className="space-y-3">
                    {details.analysis.advantages.map((advantage, index) => (
                      <li key={index} className="flex items-start space-x-2">
                        <CheckCircle className="w-5 h-5 text-green-500 mt-1 flex-shrink-0" />
                        <span className="text-gray-300">{advantage}</span>
                      </li>
                    ))}
                  </ul>
                </div>

                <div className="bg-gray-900/50 rounded-2xl p-6 border border-gray-800">
                  <h3 className="text-xl font-semibold mb-4 text-red-400">Disadvantages</h3>
                  <ul className="space-y-3">
                    {details.analysis.disadvantages.map((disadvantage, index) => (
                      <li key={index} className="flex items-start space-x-2">
                        <AlertCircle className="w-5 h-5 text-red-500 mt-1 flex-shrink-0" />
                        <span className="text-gray-300">{disadvantage}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>

            {/* Right Column */}
            <div className="space-y-8">
              {/* Use Cases */}
              <div className="bg-gray-900/50 rounded-2xl p-6 border border-gray-800">
                <h3 className="text-xl font-semibold mb-4">Use Cases</h3>
                <ul className="space-y-3">
                  {details.analysis.usecases.map((usecase, index) => (
                    <li key={index} className="flex items-start space-x-2">
                      <span className="text-purple-400 font-bold">•</span>
                      <span className="text-gray-300">{usecase}</span>
                    </li>
                  ))}
                </ul>
              </div>

              {/* Evaluation Scores */}
              {details.analysis.evals.length > 0 && (
                <div className="bg-gray-900/50 rounded-2xl p-6 border border-gray-800">
                  <h3 className="text-xl font-semibold mb-4">Evaluation Scores</h3>
                  <div className="space-y-3">
                    {details.analysis.evals.map((evaluation, index) => (
                      <div key={index} className="flex items-center justify-between p-3 bg-gray-800/50 rounded-lg">
                        <span className="text-white font-medium">{evaluation.name}</span>
                        <div className="flex items-center space-x-3">
                          <div className="w-24 bg-gray-700 rounded-full h-2">
                            <div
                              className="bg-gradient-to-r from-purple-500 to-pink-500 h-2 rounded-full"
                              style={{ width: `${Math.min(100, (evaluation.score / 5) * 100)}%` }}
                            ></div>
                          </div>
                          <span className="text-purple-400 font-bold min-w-[3rem] text-right">
                            {evaluation.score.toFixed(1)}
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        {activeTab === 'technical' && (
          <div className="grid lg:grid-cols-2 gap-8">
            {/* Architecture Details */}
            <div className="bg-gray-900/50 rounded-2xl p-6 border border-gray-800">
              <h3 className="text-xl font-semibold mb-4">Architecture</h3>
              <div className="space-y-4">
                {details.config?.hidden_size && (
                  <div className="flex justify-between">
                    <span className="text-gray-400">Hidden Size</span>
                    <span className="text-white">{details.config.hidden_size.toLocaleString()}</span>
                  </div>
                )}
                {details.config?.num_hidden_layers && (
                  <div className="flex justify-between">
                    <span className="text-gray-400">Hidden Layers</span>
                    <span className="text-white">{details.config.num_hidden_layers}</span>
                  </div>
                )}
                {details.config?.num_attention_heads && (
                  <div className="flex justify-between">
                    <span className="text-gray-400">Attention Heads</span>
                    <span className="text-white">{details.config.num_attention_heads}</span>
                  </div>
                )}
                {details.config?.num_key_value_heads && (
                  <div className="flex justify-between">
                    <span className="text-gray-400">KV Heads</span>
                    <span className="text-white">{details.config.num_key_value_heads}</span>
                  </div>
                )}
                {details.config?.intermediate_size && (
                  <div className="flex justify-between">
                    <span className="text-gray-400">Intermediate Size</span>
                    <span className="text-white">{details.config.intermediate_size.toLocaleString()}</span>
                  </div>
                )}
                {details.config?.vocab_size && (
                  <div className="flex justify-between">
                    <span className="text-gray-400">Vocabulary Size</span>
                    <span className="text-white">{details.config.vocab_size.toLocaleString()}</span>
                  </div>
                )}
                {details.config?.max_position_embeddings && (
                  <div className="flex justify-between">
                    <span className="text-gray-400">Max Position</span>
                    <span className="text-white">{details.config.max_position_embeddings.toLocaleString()}</span>
                  </div>
                )}
              </div>
            </div>

            {/* Other Technical Details */}
            <div className="bg-gray-900/50 rounded-2xl p-6 border border-gray-800">
              <h3 className="text-xl font-semibold mb-4">Configuration</h3>
              <div className="space-y-4">
                {details.config?.activation_function && (
                  <div className="flex justify-between">
                    <span className="text-gray-400">Activation</span>
                    <span className="text-white">{details.config.activation_function}</span>
                  </div>
                )}
                {details.config?.torch_dtype && (
                  <div className="flex justify-between">
                    <span className="text-gray-400">Data Type</span>
                    <span className="text-white">{details.config.torch_dtype}</span>
                  </div>
                )}
                {details.config?.model_type && (
                  <div className="flex justify-between">
                    <span className="text-gray-400">Model Type</span>
                    <span className="text-white">{details.config.model_type}</span>
                  </div>
                )}
                {details.config?.num_parameters && (
                  <div className="flex justify-between">
                    <span className="text-gray-400">Total Parameters</span>
                    <span className="text-white">{formatParameterCount(details.config.num_parameters)}</span>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* No Analysis Available Message */}
        {activeTab === 'analysis' && !details.analysis && (
          <div className="text-center py-16">
            <FileText className="w-16 h-16 text-gray-600 mx-auto mb-4" />
            <h3 className="text-xl font-semibold text-gray-300 mb-2">No Analysis Available</h3>
            <p className="text-gray-500">This model doesn't have detailed analysis information yet.</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default ModelDetails; 