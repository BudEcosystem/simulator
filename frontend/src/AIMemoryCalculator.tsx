import React, { useState, useEffect } from 'react';
import { Calculator, Zap, BarChart3, Settings, Home, ArrowRight, ArrowLeft, Cpu, HardDrive, Database, Eye, TrendingUp, GitCompare, ChevronDown, ChevronUp, Info, CheckCircle, AlertCircle, Sparkles, Loader2, ExternalLink, RefreshCw } from 'lucide-react';
import ModelDetails from './components/ModelDetails';
import { FeaturedHardware } from './components/Hardware/FeaturedHardware';
import { HardwareList } from './components/Hardware/HardwareList';
import { HardwareDetail } from './components/Hardware/HardwareDetail';
import { HardwareRecommendation } from './types/hardware';
import { hardwareAPI } from './services/hardwareAPI';

// =============================================================================
// TYPE DEFINITIONS
// =============================================================================

interface ModelConfig {
  model_id: string;
  model_type: string;
  attention_type: string;
  parameter_count: number;
  architecture: string;
  logo?: string;
  config: {
    hidden_size: number;
    num_hidden_layers: number;
    num_attention_heads: number;
    num_key_value_heads: number;
    intermediate_size: number;
    vocab_size: number;
    max_position_embeddings: number;
    activation_function: string;
  };
  metadata: {
    downloads: number;
    likes: number;
    size_gb: number;
    tags: string[];
  };
}

interface CalculationResults {
  model_type: string;
  attention_type: string;
  precision: string;
  parameter_count: number;
  memory_breakdown: {
    weight_memory_gb: number;
    kv_cache_gb: number;
    activation_memory_gb: number;
    state_memory_gb: number;
    image_memory_gb: number;
    extra_work_gb: number;
  };
  total_memory_gb: number;
  recommendations: {
    recommended_gpu_memory_gb: number;
    can_fit_24gb_gpu: boolean;
    can_fit_80gb_gpu: boolean;
    min_gpu_memory_gb: number;
  };
}

interface UserConfig {
  precision: string;
  batchSize: number;
  seqLength: number;
  numImages: number;
  includeGradients: boolean;
  decodeLength: number;
}

interface ComparisonItem {
  id: number;
  model_id: string;
  model_name: string;
  logo?: string;
  config: UserConfig;
  results: CalculationResults;
}

interface AnalysisInsights {
  memory_per_token_bytes: number;
  efficiency_rating: string;
  recommendations: string[];
}

interface AnalysisData {
  [key: string]: {
    total_memory_gb: number;
    kv_cache_gb: number;
    kv_cache_percent: number;
  };
}

interface AnalysisResults {
  model_id: string;
  attention_type: string;
  analysis: AnalysisData;
  insights: AnalysisInsights;
}

interface PopularModel {
  model_id: string;
  name: string;
  parameters: string;
  model_type: string;
  attention_type: string;
  downloads: number;
  likes: number;
  description: string;
  logo?: string;
}

interface ComparePayloadItem {
  model_id: string;
  precision: string;
  batch_size: number;
  seq_length: number;
}

interface ModelSummary {
  model_id: string;
  name: string;
  author: string | null;
  model_type: string;
  attention_type: string | null;
  parameter_count: number | null;
  source: string;
  in_model_dict: boolean;
  in_database: boolean;
  logo?: string;
}

// Enhanced response for gated model support
interface ValidateResponse {
  valid: boolean;
  error?: string;
  error_code?: string;
  model_id?: string;
  requires_config?: boolean;
  config_submission_url?: string;
}

interface ConfigSubmitResponse {
  success: boolean;
  message: string;
  model_id: string;
  validation?: {
    valid: boolean;
    model_type: string;
    attention_type: string;
    parameter_count: number;
  };
  error_code?: string;
  missing_fields?: string[];
}

// =============================================================================
// LOGO COMPONENT
// =============================================================================

const ModelLogo: React.FC<{ logo?: string; modelId: string; size?: 'sm' | 'md' | 'lg' }> = ({ logo, modelId, size = 'md' }) => {
  const sizeClasses = {
    sm: 'w-8 h-8',
    md: 'w-12 h-12',
    lg: 'w-16 h-16'
  };

  const iconSizeClasses = {
    sm: 'w-4 h-4',
    md: 'w-6 h-6',
    lg: 'w-8 h-8'
  };

  if (logo) {
    return (
      <img 
        src={logo} 
        alt={modelId}
        className={`${sizeClasses[size]} rounded-xl object-cover`}
        onError={(e) => {
          // If image fails to load, replace with default
          e.currentTarget.style.display = 'none';
          e.currentTarget.nextElementSibling?.classList.remove('hidden');
        }}
      />
    );
  }

  // Default logo
  return (
    <div className={`${sizeClasses[size]} bg-gradient-to-br from-purple-500 to-pink-500 rounded-xl flex items-center justify-center`}>
      <Database className={`${iconSizeClasses[size]} text-white`} />
    </div>
  );
};

// Hidden fallback for failed image loads
export const ModelLogoWithFallback: React.FC<{ logo?: string; modelId: string; size?: 'sm' | 'md' | 'lg' }> = ({ logo, modelId, size = 'md' }) => {
  const sizeClasses = {
    sm: 'w-8 h-8',
    md: 'w-12 h-12',
    lg: 'w-16 h-16'
  };

  const iconSizeClasses = {
    sm: 'w-4 h-4',
    md: 'w-6 h-6',
    lg: 'w-8 h-8'
  };

  return (
    <div className="relative">
      {logo && (
        <img 
          src={logo} 
          alt={modelId}
          className={`${sizeClasses[size]} rounded-xl object-cover`}
          onError={(e) => {
            e.currentTarget.style.display = 'none';
            e.currentTarget.nextElementSibling?.classList.remove('hidden');
          }}
        />
      )}
      <div className={`${logo ? 'hidden' : ''} ${sizeClasses[size]} bg-gradient-to-br from-purple-500 to-pink-500 rounded-xl flex items-center justify-center`}>
        <Database className={`${iconSizeClasses[size]} text-white`} />
      </div>
    </div>
  );
};

const AIMemoryCalculator = () => {
  const [currentScreen, setCurrentScreen] = useState('home');
  const [currentStep, setCurrentStep] = useState(1); // For calculator flow
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null);
  
  // Hardware-related states
  const [selectedHardwareName, setSelectedHardwareName] = useState<string | null>(null);
  const [hardwareRecommendations, setHardwareRecommendations] = useState<HardwareRecommendation[]>([]);
  
  // States for different screens
  const [modelUrl, setModelUrl] = useState('');
  const [modelConfig, setModelConfig] = useState<ModelConfig | null>(null);
  const [userConfig, setUserConfig] = useState<UserConfig>({
    precision: 'fp16',
    batchSize: 1,
    seqLength: 2048,
    numImages: 0,
    includeGradients: false,
    decodeLength: 0
  });
  const [results, setResults] = useState<CalculationResults | null>(null);
  const [comparisonResults, setComparisonResults] = useState<ComparisonItem[]>([]);
  const [analysisResults, setAnalysisResults] = useState<AnalysisResults | null>(null);
  const [popularModels, setPopularModels] = useState<PopularModel[]>([]);
  
  // Loading states
  const [isValidating, setIsValidating] = useState(false);
  const [isCalculating, setIsCalculating] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isLoadingPopular, setIsLoadingPopular] = useState(false);
  
  // Error states
  const [validationError, setValidationError] = useState('');
  const [calculationError, setCalculationError] = useState('');
  
  // Gated model support states
  const [showConfigInput, setShowConfigInput] = useState(false);
  const [gatedModelId, setGatedModelId] = useState<string | null>(null);
  const [configJson, setConfigJson] = useState('');
  const [configSubmissionError, setConfigSubmissionError] = useState('');
  const [isSubmittingConfig, setIsSubmittingConfig] = useState(false);
  
  // Pending action for continuing flow after config submission
  interface PendingAction {
    type: 'calculate' | 'analyze' | 'compare' | 'validate';
    params?: any;
  }
  const [pendingAction, setPendingAction] = useState<PendingAction | null>(null);

  // API Base URL - replace with your actual backend URL
  const API_BASE = 'http://localhost:8000/api';

  // =============================================================================
  // API ENDPOINTS DEFINITION
  // =============================================================================
  
  /*
  Required REST API Endpoints:

  1. POST /api/models/validate
     Body: { "model_url": "meta-llama/Llama-2-7b-hf" }
     Response: { "valid": true/false, "error": "error message if invalid" }

  2. GET /api/models/{model_id}/config
     Response: {
       "model_id": "meta-llama/Llama-2-7b-hf",
       "model_type": "decoder-only",
       "attention_type": "gqa",
       "parameter_count": 7000000000,
       "architecture": "LlamaForCausalLM",
       "config": {
         "hidden_size": 4096,
         "num_hidden_layers": 32,
         "num_attention_heads": 32,
         "num_key_value_heads": 8,
         "intermediate_size": 11008,
         "vocab_size": 32000,
         "max_position_embeddings": 4096,
         "activation_function": "silu"
       },
       "metadata": {
         "downloads": 1500000,
         "likes": 25000,
         "size_gb": 13.5,
         "tags": ["text-generation", "llama"]
       }
     }

  3. POST /api/models/calculate
     Body: {
       "model_id": "meta-llama/Llama-2-7b-hf",
       "precision": "fp16",
       "batch_size": 1,
       "seq_length": 2048,
       "num_images": 0,
       "include_gradients": false,
       "decode_length": 0
     }
     Response: {
       "model_type": "decoder-only",
       "attention_type": "gqa",
       "precision": "fp16",
       "parameter_count": 7000000000,
       "memory_breakdown": {
         "weight_memory_gb": 13.5,
         "kv_cache_gb": 0.5,
         "activation_memory_gb": 2.1,
         "state_memory_gb": 0.0,
         "image_memory_gb": 0.0,
         "extra_work_gb": 0.5
       },
       "total_memory_gb": 16.6,
       "recommendations": {
         "recommended_gpu_memory_gb": 24,
         "can_fit_24gb_gpu": true,
         "can_fit_80gb_gpu": true,
         "min_gpu_memory_gb": 16.6
       }
     }

  4. POST /api/models/compare
     Body: {
       "models": [
         {
           "model_id": "meta-llama/Llama-2-7b-hf",
           "precision": "fp16",
           "batch_size": 1,
           "seq_length": 2048
         },
         {
           "model_id": "mistralai/Mistral-7B-v0.1",
           "precision": "fp16",
           "batch_size": 1,
           "seq_length": 2048
         }
       ]
     }
     Response: {
       "comparisons": [
         {
           "model_id": "meta-llama/Llama-2-7b-hf",
           "model_name": "Llama 2 7B",
           "total_memory_gb": 16.6,
           "memory_breakdown": {...},
           "recommendations": {...}
         },
         {
           "model_id": "mistralai/Mistral-7B-v0.1",
           "model_name": "Mistral 7B",
           "total_memory_gb": 15.2,
           "memory_breakdown": {...},
           "recommendations": {...}
         }
       ]
     }

  5. POST /api/models/analyze
     Body: {
       "model_id": "meta-llama/Llama-2-7b-hf",
       "precision": "fp16",
       "batch_size": 1,
       "sequence_lengths": [1024, 4096, 16384, 32768]
     }
     Response: {
       "model_id": "meta-llama/Llama-2-7b-hf",
       "attention_type": "gqa",
       "analysis": {
         "1024": {
           "total_memory_gb": 14.2,
           "kv_cache_gb": 0.25,
           "kv_cache_percent": 1.8
         },
         "4096": {
           "total_memory_gb": 16.6,
           "kv_cache_gb": 1.0,
           "kv_cache_percent": 6.0
         },
         "16384": {
           "total_memory_gb": 26.8,
           "kv_cache_gb": 4.0,
           "kv_cache_percent": 14.9
         },
         "32768": {
           "total_memory_gb": 47.2,
           "kv_cache_gb": 8.0,
           "kv_cache_percent": 16.9
         }
       },
       "insights": {
         "memory_per_token_bytes": 256,
         "efficiency_rating": "high",
         "recommendations": [
           "GQA provides 4x KV cache compression compared to MHA",
           "Consider using sliding window attention for very long sequences"
         ]
       }
     }

  6. GET /api/models/popular
     Response: {
       "models": [
         {
           "model_id": "meta-llama/Llama-2-7b-hf",
           "name": "Llama 2 7B",
           "parameters": "7B",
           "model_type": "decoder-only",
           "attention_type": "gqa",
           "downloads": 1500000,
           "likes": 25000,
           "description": "High-performance language model"
         },
         {
           "model_id": "mistralai/Mistral-7B-v0.1",
           "name": "Mistral 7B",
           "parameters": "7B",
           "model_type": "decoder-only",
           "attention_type": "gqa",
           "downloads": 800000,
           "likes": 15000,
           "description": "Efficient 7B parameter model"
         }
       ]
     }
  */

  // =============================================================================
  // API FUNCTIONS
  // =============================================================================

  const validateModelUrl = async (url: string, continueWithAction: boolean = false): Promise<ValidateResponse> => {
    setIsValidating(true);
    setValidationError('');
    
    try {
      const response = await fetch(`${API_BASE}/models/validate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ model_url: url }),
      });
      
      const data: ValidateResponse = await response.json();
      
      if (!response.ok) {
        throw new Error(data.error || 'Validation failed');
      }
      
      if (!data.valid) {
        // Check if it's a gated model
        if (data.error_code === 'MODEL_GATED') {
          setGatedModelId(data.model_id || url);
          setShowConfigInput(true);
          
          // Store the pending action if needed
          if (continueWithAction && pendingAction) {
            // Keep the existing pending action
          } else if (continueWithAction) {
            // Store a validate action to continue after config submission
            setPendingAction({ type: 'validate', params: { url } });
          }
        }
        
        setValidationError(data.error || 'Invalid model URL');
        return data;
      }
      
      return data;
    } catch (error) {
      const errorResponse: ValidateResponse = {
        valid: false,
        error: error instanceof Error ? error.message : 'An unknown error occurred'
      };
      setValidationError(errorResponse.error || '');
      return errorResponse;
    } finally {
      setIsValidating(false);
    }
  };

  const fetchModelConfig = async (modelId: string) => {
    try {
      const response = await fetch(`${API_BASE}/models/${encodeURIComponent(modelId)}/config`);
      
      const data = await response.json();
      
      if (!response.ok) {
        // Check if it's a gated model error (can be in data or data.detail)
        const errorData = data.detail || data;
        if (errorData.error_code === 'MODEL_GATED' || response.status === 403) {
          setGatedModelId(modelId);
          setShowConfigInput(true);
          setPendingAction({ type: 'validate', params: { url: modelId } });
          return null;
        }
        throw new Error(errorData.error || data.error || 'Failed to fetch model configuration');
      }
      
      setModelConfig(data);
      return data;
    } catch (error) {
      console.error("Error fetching model config:", error);
      setValidationError(error instanceof Error ? error.message : 'An unknown error occurred');
      return null;
    }
  };
  
  const submitConfig = async () => {
    if (!gatedModelId) return;
    
    setIsSubmittingConfig(true);
    setConfigSubmissionError('');
    
    try {
      // Parse the config JSON
      let config;
      try {
        config = JSON.parse(configJson);
      } catch (e) {
        throw new Error('Invalid JSON format. Please paste a valid config.json');
      }
      
      const response = await fetch(`${API_BASE}/models/config/submit`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model_uri: gatedModelId,
          config: config
        }),
      });
      
      const data: ConfigSubmitResponse = await response.json();
      
      if (!response.ok || !data.success) {
        if (data.error_code === 'INVALID_CONFIG' && data.missing_fields) {
          throw new Error(`Missing required fields: ${data.missing_fields.join(', ')}`);
        }
        throw new Error(data.message || 'Failed to submit configuration');
      }
      
      // Config saved successfully, hide the input
      setShowConfigInput(false);
      setConfigJson('');
      setGatedModelId(null);
      
      // Continue with the pending action
      if (pendingAction) {
        switch (pendingAction.type) {
          case 'validate':
            // Fetch the model config and continue
            await fetchModelConfig(data.model_id);
            setCurrentStep(2);
            break;
            
          case 'calculate':
            // Fetch config first, then calculate
            await fetchModelConfig(data.model_id);
            await calculateMemory();
            break;
            
          case 'analyze':
            // Fetch config first, then analyze
            await fetchModelConfig(data.model_id);
            await analyzeModel();
            break;
            
          case 'compare':
            // Handle comparison case
            if (pendingAction.params) {
              await compareModels(pendingAction.params);
            }
            break;
        }
        
        setPendingAction(null);
      }
      
    } catch (error) {
      setConfigSubmissionError(error instanceof Error ? error.message : 'Failed to submit configuration');
    } finally {
      setIsSubmittingConfig(false);
    }
  };

  const calculateMemory = async () => {
    if (!modelConfig) {
      setCalculationError("Model configuration is not loaded.");
      return;
    }

    setIsCalculating(true);
    setCalculationError('');
    
    try {
      const response = await fetch(`${API_BASE}/models/calculate`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model_id: modelConfig.model_id,
          precision: userConfig.precision,
          batch_size: userConfig.batchSize,
          seq_length: userConfig.seqLength,
          num_images: userConfig.numImages,
          include_gradients: userConfig.includeGradients,
          decode_length: userConfig.decodeLength
        }),
      });
      
      const data = await response.json();
      
      if (!response.ok) {
        // Check if it's a gated model error (can be in data or data.detail)
        const errorData = data.detail || data;
        if (errorData.error_code === 'MODEL_GATED' || response.status === 403) {
          setGatedModelId(modelConfig.model_id);
          setShowConfigInput(true);
          setPendingAction({ 
            type: 'calculate',
            params: userConfig 
          });
          setIsCalculating(false);
          return;
        }
        throw new Error(errorData.error || data.error || 'Calculation failed');
      }
      
      setResults(data);
      setCurrentScreen('results');
    } catch (error) {
      if (error instanceof Error) {
        setCalculationError(error.message);
      } else {
        setCalculationError('An unknown error occurred');
      }
    } finally {
      setIsCalculating(false);
    }
  };

  const compareModels = async (models: ComparePayloadItem[]) => {
    try {
      const response = await fetch(`${API_BASE}/models/compare`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ models }),
      });
      
      if (!response.ok) {
        throw new Error('Comparison failed');
      }
      
      const data = await response.json();
      setComparisonResults(data.comparisons);
    } catch (error) {
      console.error("Error comparing models:", error);
    }
  };

  const analyzeModel = async () => {
    if (!modelConfig) return;

    setIsAnalyzing(true);
    
    try {
      const response = await fetch(`${API_BASE}/models/analyze`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model_id: modelConfig.model_id,
          precision: userConfig.precision,
          batch_size: userConfig.batchSize,
          sequence_lengths: [1024, 4096, 16384, 32768]
        }),
      });
      
      const data = await response.json();
      
      if (!response.ok) {
        // Check if it's a gated model error (can be in data or data.detail)
        const errorData = data.detail || data;
        if (errorData.error_code === 'MODEL_GATED' || response.status === 403) {
          setGatedModelId(modelConfig.model_id);
          setShowConfigInput(true);
          setPendingAction({ 
            type: 'analyze',
            params: { precision: userConfig.precision, batch_size: userConfig.batchSize }
          });
          setIsAnalyzing(false);
          return;
        }
        throw new Error(errorData.error || data.error || 'Analysis failed');
      }
      
      setAnalysisResults(data);
      setCurrentScreen('analysis');
    } catch (error) {
      console.error('Analysis error:', error);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const fetchPopularModels = async () => {
    setIsLoadingPopular(true);
    
    try {
      const response = await fetch(`${API_BASE}/models/popular`);
      
      if (!response.ok) {
        throw new Error('Failed to fetch popular models');
      }
      
      const data = await response.json();
      setPopularModels(data.models);
    } catch (error) {
      console.error("Error fetching popular models:", error);
    } finally {
      setIsLoadingPopular(false);
    }
  };

  // Load popular models on component mount
  useEffect(() => {
    fetchPopularModels();
  }, []);

  // Handle model URL submission
  const handleModelUrlSubmit = async (url: string) => {
    const response = await validateModelUrl(url, true);
    if (response.valid) {
      await fetchModelConfig(url);
      setCurrentStep(2);
    }
    // If it's a gated model, the validateModelUrl function will handle showing the config input
  };

  // Handle configuration and calculation
  const handleCalculate = async () => {
    await calculateMemory();
  };

  // Add to comparison
  const addToComparison = () => {
    if (results && modelConfig) {
      const newComparison: ComparisonItem = {
        id: Date.now(),
        model_id: modelConfig.model_id,
        model_name: modelConfig.model_id.split('/')[1],
        logo: modelConfig.logo,
        config: userConfig,
        results: results,
      };
      setComparisonResults(prev => [...prev, newComparison]);
    }
  };

  // Hardware navigation handlers
  const handleHardwareNavigation = (hardwareName: string) => {
    if (hardwareName) {
      setSelectedHardwareName(hardwareName);
      setCurrentScreen('hardware-detail');
    } else {
      setCurrentScreen('hardware');
    }
  };

  // Fetch hardware recommendations when results are available
  const fetchHardwareRecommendations = async () => {
    if (!results) return;
    
    try {
      const recommendations = await hardwareAPI.recommend(
        results.total_memory_gb,
        results.parameter_count / 1e9
      );
      setHardwareRecommendations(recommendations);
    } catch (error) {
      console.error('Error fetching hardware recommendations:', error);
    }
  };

  useEffect(() => {
    if (results) {
      fetchHardwareRecommendations();
    }
  }, [results]);

  // Navigation component
  const Navigation = () => (
    <div className="fixed top-0 left-0 right-0 z-50 bg-black/90 backdrop-blur-sm border-b border-purple-500/20">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-2">
              <div className="w-8 h-8 bg-gradient-to-br from-purple-500 to-pink-500 rounded-lg flex items-center justify-center">
                <Cpu className="w-4 h-4 text-white" />
              </div>
              <span className="text-xl font-bold text-white">AI Memory Calculator</span>
            </div>
          </div>
          
          <nav className="hidden md:flex space-x-1">
            {[
              { id: 'home', label: 'Home', icon: Home },
              { id: 'models', label: 'Models', icon: Database },
              { id: 'calculator', label: 'Calculator', icon: Calculator },
              { id: 'hardware', label: 'Hardware', icon: Cpu },
              { id: 'comparison', label: 'Compare', icon: GitCompare },
              { id: 'analysis', label: 'Analysis', icon: BarChart3 },
            ].map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                onClick={() => {
                  setCurrentScreen(id);
                  if (id === 'calculator') {
                    setCurrentStep(1);
                    setModelConfig(null);
                    setModelUrl('');
                  }
                }}
                className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all duration-200 ${
                  currentScreen === id
                    ? 'bg-purple-600 text-white shadow-lg shadow-purple-500/25'
                    : 'text-gray-300 hover:text-white hover:bg-purple-600/20'
                }`}
              >
                <Icon className="w-4 h-4" />
                <span className="text-sm font-medium">{label}</span>
              </button>
            ))}
          </nav>
        </div>
      </div>
    </div>
  );

  // Home Screen
  const HomeScreen = () => (
    <div className="min-h-screen bg-black text-white pt-16">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-12">
        {/* Hero Section */}
        <div className="text-center mb-16">
          <div className="mb-8">
            <div className="w-20 h-20 bg-gradient-to-br from-purple-500 to-pink-500 rounded-2xl flex items-center justify-center mx-auto mb-6">
              <Sparkles className="w-10 h-10 text-white" />
            </div>
            <h1 className="text-5xl font-bold mb-4 bg-gradient-to-r from-purple-400 to-pink-400 bg-clip-text text-transparent">
              AI Memory Calculator
            </h1>
            <p className="text-xl text-gray-300 max-w-3xl mx-auto">
              Calculate precise memory requirements for AI models from HuggingFace. 
              Optimize your GPU usage and deployment costs with accurate predictions.
            </p>
          </div>
          
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <button
              onClick={() => setCurrentScreen('calculator')}
              className="bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 text-white px-8 py-4 rounded-xl font-semibold transition-all duration-200 transform hover:scale-105 shadow-lg shadow-purple-500/25"
            >
              <div className="flex items-center space-x-2">
                <Calculator className="w-5 h-5" />
                <span>Start Calculating</span>
                <ArrowRight className="w-5 h-5" />
              </div>
            </button>
            
            <button
              onClick={() => setCurrentScreen('comparison')}
              className="bg-gray-800 hover:bg-gray-700 text-white px-8 py-4 rounded-xl font-semibold transition-all duration-200 border border-gray-600"
            >
              <div className="flex items-center space-x-2">
                <GitCompare className="w-5 h-5" />
                <span>Compare Models</span>
              </div>
            </button>
          </div>
        </div>

        {/* Features Grid */}
        <div className="grid md:grid-cols-3 gap-8 mb-16">
          {[
            {
              icon: Database,
              title: 'Precise Memory Calculation',
              description: 'Get exact memory requirements for weights, KV cache, activations, and more.',
              color: 'from-purple-500 to-blue-500'
            },
            {
              icon: TrendingUp,
              title: 'Multi-Architecture Support',
              description: 'Support for Transformers, Mamba, MoE, Diffusion, and multimodal models.',
              color: 'from-pink-500 to-purple-500'
            },
            {
              icon: BarChart3,
              title: 'Advanced Analysis',
              description: 'Analyze attention efficiency, sequence length scaling, and deployment options.',
              color: 'from-blue-500 to-cyan-500'
            }
          ].map((feature, index) => (
            <div key={index} className="bg-gray-900/50 p-6 rounded-2xl border border-gray-800 hover:border-purple-500/50 transition-all duration-200">
              <div className={`w-12 h-12 bg-gradient-to-br ${feature.color} rounded-xl flex items-center justify-center mb-4`}>
                <feature.icon className="w-6 h-6 text-white" />
              </div>
              <h3 className="text-xl font-semibold mb-2">{feature.title}</h3>
              <p className="text-gray-400">{feature.description}</p>
            </div>
          ))}
        </div>

        {/* Featured Hardware */}
        <FeaturedHardware onHardwareClick={handleHardwareNavigation} />

        {/* Popular Models */}
        <div className="bg-gray-900/30 rounded-2xl p-8 border border-gray-800">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-2xl font-bold">Popular Models</h2>
            <button
              onClick={fetchPopularModels}
              className="text-purple-400 hover:text-purple-300 transition-colors"
              disabled={isLoadingPopular}
            >
              <RefreshCw className={`w-5 h-5 ${isLoadingPopular ? 'animate-spin' : ''}`} />
            </button>
          </div>
          
          {isLoadingPopular ? (
            <div className="text-center py-8">
              <Loader2 className="w-8 h-8 animate-spin text-purple-500 mx-auto mb-4" />
              <p className="text-gray-400">Loading popular models...</p>
            </div>
          ) : (
            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
              {popularModels.map((model, index) => (
                <button
                  key={index}
                  onClick={() => {
                    setModelUrl(model.model_id);
                    setCurrentScreen('calculator');
                    handleModelUrlSubmit(model.model_id);
                  }}
                  className="bg-gray-800/50 p-4 rounded-xl border border-gray-700 hover:border-purple-500/50 transition-all duration-200 text-left group"
                >
                  <div className="flex items-start space-x-3 mb-3">
                    <ModelLogoWithFallback logo={model.logo} modelId={model.model_id} size="md" />
                    <div className="flex-1">
                      <div className="flex items-center justify-between mb-1">
                        <h3 className="font-semibold text-white group-hover:text-purple-300 transition-colors">{model.name}</h3>
                        <span className="text-sm text-purple-400">{model.parameters}</span>
                      </div>
                      <p className="text-sm text-gray-400">{model.model_id}</p>
                    </div>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-xs text-gray-500">{model.model_type}</span>
                    <div className="flex items-center space-x-2 text-xs text-gray-500">
                      <span>{model.downloads?.toLocaleString()} downloads</span>
                      <span>•</span>
                      <span>{model.likes?.toLocaleString()} likes</span>
                    </div>
                  </div>
                </button>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );

  // Calculator Screen
  const CalculatorScreen = () => (
    <div className="min-h-screen bg-black text-white pt-16">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="flex items-center space-x-4 mb-8">
          <button
            onClick={() => setCurrentScreen('home')}
            className="text-gray-400 hover:text-white transition-colors"
          >
            <ArrowLeft className="w-5 h-5" />
          </button>
          <h1 className="text-3xl font-bold">Memory Calculator</h1>
        </div>

        {/* Progress Steps */}
        <div className="flex items-center space-x-4 mb-8">
          <div className={`flex items-center space-x-2 ${currentStep >= 1 ? 'text-purple-400' : 'text-gray-500'}`}>
            <div className={`w-8 h-8 rounded-full flex items-center justify-center ${currentStep >= 1 ? 'bg-purple-600' : 'bg-gray-600'}`}>
              <span className="text-sm font-medium">1</span>
            </div>
            <span className="text-sm font-medium">Model URL</span>
          </div>
          <div className={`flex-1 h-0.5 ${currentStep >= 2 ? 'bg-purple-600' : 'bg-gray-600'}`}></div>
          <div className={`flex items-center space-x-2 ${currentStep >= 2 ? 'text-purple-400' : 'text-gray-500'}`}>
            <div className={`w-8 h-8 rounded-full flex items-center justify-center ${currentStep >= 2 ? 'bg-purple-600' : 'bg-gray-600'}`}>
              <span className="text-sm font-medium">2</span>
            </div>
            <span className="text-sm font-medium">Configuration</span>
          </div>
          <div className={`flex-1 h-0.5 ${currentStep >= 3 ? 'bg-purple-600' : 'bg-gray-600'}`}></div>
          <div className={`flex items-center space-x-2 ${currentStep >= 3 ? 'text-purple-400' : 'text-gray-500'}`}>
            <div className={`w-8 h-8 rounded-full flex items-center justify-center ${currentStep >= 3 ? 'bg-purple-600' : 'bg-gray-600'}`}>
              <span className="text-sm font-medium">3</span>
            </div>
            <span className="text-sm font-medium">Results</span>
          </div>
        </div>

        {/* Step 1: Model URL Input */}
        {currentStep === 1 && (
          <div className="bg-gray-900/50 rounded-2xl p-8 border border-gray-800">
            <div className="text-center mb-8">
              <ExternalLink className="w-12 h-12 text-purple-500 mx-auto mb-4" />
              <h2 className="text-2xl font-semibold mb-2">Enter HuggingFace Model URL</h2>
              <p className="text-gray-400">We'll fetch the model configuration automatically</p>
            </div>

            <div className="max-w-2xl mx-auto">
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-2">
                    Model URL or ID
                  </label>
                  <input
                    type="text"
                    value={modelUrl}
                    onChange={(e) => setModelUrl(e.target.value)}
                    placeholder="e.g., meta-llama/Llama-2-7b-hf or https://huggingface.co/meta-llama/Llama-2-7b-hf"
                    className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-3 text-white placeholder-gray-500 focus:border-purple-500 focus:outline-none"
                    onKeyPress={(e) => {
                      if (e.key === 'Enter' && modelUrl.trim()) {
                        handleModelUrlSubmit(modelUrl);
                      }
                    }}
                  />
                </div>

                {validationError && (
                  <div className="bg-red-900/20 border border-red-500/20 rounded-lg p-4">
                    <div className="flex items-center space-x-2">
                      <AlertCircle className="w-5 h-5 text-red-500" />
                      <span className="text-red-300">{validationError}</span>
                    </div>
                  </div>
                )}
                
                {/* Gated Model Config Input */}
                {showConfigInput && (
                  <div className="bg-amber-900/20 border border-amber-500/20 rounded-lg p-6 space-y-4">
                    <div className="flex items-start space-x-3">
                      <AlertCircle className="w-5 h-5 text-amber-500 mt-0.5" />
                      <div className="flex-1">
                        <h3 className="font-semibold text-amber-300 mb-2">Gated Model Detected</h3>
                        <p className="text-gray-300 text-sm mb-4">
                          This model is gated and requires authentication to access. Please provide the model's config.json manually:
                        </p>
                        
                        <div className="space-y-4">
                          <div>
                            <label className="block text-sm font-medium text-gray-300 mb-2">
                              Paste config.json contents
                            </label>
                            <textarea
                              value={configJson}
                              onChange={(e) => setConfigJson(e.target.value)}
                              placeholder={`{
  "model_type": "llama",
  "hidden_size": 4096,
  "num_hidden_layers": 32,
  "num_attention_heads": 32,
  ...
}`}
                              className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-3 text-white placeholder-gray-500 focus:border-purple-500 focus:outline-none font-mono text-sm min-h-[200px]"
                            />
                          </div>
                          
                          {configSubmissionError && (
                            <div className="bg-red-900/20 border border-red-500/20 rounded-lg p-3">
                              <div className="flex items-center space-x-2">
                                <AlertCircle className="w-4 h-4 text-red-500" />
                                <span className="text-red-300 text-sm">{configSubmissionError}</span>
                              </div>
                            </div>
                          )}
                          
                          <div className="flex space-x-3">
                            <button
                              onClick={submitConfig}
                              disabled={!configJson.trim() || isSubmittingConfig}
                              className="flex-1 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 disabled:from-gray-600 disabled:to-gray-600 disabled:cursor-not-allowed text-white px-4 py-2 rounded-lg font-medium transition-all duration-200"
                            >
                              <div className="flex items-center justify-center space-x-2">
                                {isSubmittingConfig ? (
                                  <>
                                    <Loader2 className="w-4 h-4 animate-spin" />
                                    <span>Submitting...</span>
                                  </>
                                ) : (
                                  <>
                                    <CheckCircle className="w-4 h-4" />
                                    <span>Submit Config</span>
                                  </>
                                )}
                              </div>
                            </button>
                            
                            <button
                              onClick={() => {
                                setShowConfigInput(false);
                                setGatedModelId(null);
                                setConfigJson('');
                                setConfigSubmissionError('');
                                setPendingAction(null);
                              }}
                              className="px-4 py-2 bg-gray-800 hover:bg-gray-700 text-white rounded-lg font-medium transition-all duration-200 border border-gray-600"
                            >
                              Cancel
                            </button>
                          </div>
                          
                          <div className="text-xs text-gray-400 space-y-1">
                            <p>You can find config.json on the model's HuggingFace page under "Files and versions".</p>
                            <p>Make sure you have access to the model before downloading the config.</p>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                <button
                  onClick={() => handleModelUrlSubmit(modelUrl)}
                  disabled={!modelUrl.trim() || isValidating}
                  className="w-full bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 disabled:from-gray-600 disabled:to-gray-600 disabled:cursor-not-allowed text-white px-6 py-3 rounded-xl font-semibold transition-all duration-200"
                >
                  <div className="flex items-center justify-center space-x-2">
                    {isValidating ? (
                      <>
                        <Loader2 className="w-5 h-5 animate-spin" />
                        <span>Validating...</span>
                      </>
                    ) : (
                      <>
                        <span>Load Model Configuration</span>
                        <ArrowRight className="w-5 h-5" />
                      </>
                    )}
                  </div>
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Step 2: Configuration */}
        {currentStep === 2 && modelConfig && (
          <div className="space-y-8">
            {/* Model Info */}
            <div className="bg-gray-900/50 rounded-2xl p-6 border border-gray-800">
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center space-x-4">
                  <ModelLogoWithFallback logo={modelConfig.logo} modelId={modelConfig.model_id} />
                  <div>
                    <h3 className="text-xl font-semibold">{modelConfig.model_id}</h3>
                    <p className="text-gray-400">{modelConfig.model_type}</p>
                  </div>
                </div>
                <button
                  onClick={() => {
                    setSelectedModelId(modelConfig.model_id);
                    setCurrentScreen('details');
                  }}
                  className="text-purple-400 hover:text-purple-300 transition-colors flex items-center space-x-1"
                >
                  <span>See More</span>
                  <ArrowRight className="w-4 h-4" />
                </button>
              </div>
              
              <div className="grid md:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <div className="flex justify-between items-center py-2 border-b border-gray-700/50">
                    <span className="text-gray-400">Model ID</span>
                    <span className="text-white font-medium">{modelConfig.model_id}</span>
                  </div>
                  <div className="flex justify-between items-center py-2 border-b border-gray-700/50">
                    <span className="text-gray-400">Type</span>
                    <span className="text-white font-medium">{modelConfig.model_type}</span>
                  </div>
                  <div className="flex justify-between items-center py-2 border-b border-gray-700/50">
                    <span className="text-gray-400">Architecture</span>
                    <span className="text-white font-medium">{modelConfig.architecture}</span>
                  </div>
                  <div className="flex justify-between items-center py-2 border-b border-gray-700/50">
                    <span className="text-gray-400">Attention</span>
                    <span className="text-white font-medium">{modelConfig.attention_type?.toUpperCase()}</span>
                  </div>
                </div>
                
                <div className="space-y-4">
                  <div className="flex justify-between items-center py-2 border-b border-gray-700/50">
                    <span className="text-gray-400">Parameters</span>
                    <span className="text-white font-medium">{(modelConfig.parameter_count / 1e9).toFixed(1)}B</span>
                  </div>
                  <div className="flex justify-between items-center py-2 border-b border-gray-700/50">
                    <span className="text-gray-400">Model Size</span>
                    <span className="text-white font-medium">{modelConfig.metadata?.size_gb?.toFixed(1)} GB</span>
                  </div>
                  <div className="flex justify-between items-center py-2 border-b border-gray-700/50">
                    <span className="text-gray-400">Downloads</span>
                    <span className="text-white font-medium">{modelConfig.metadata?.downloads?.toLocaleString()}</span>
                  </div>
                  <div className="flex justify-between items-center py-2 border-b border-gray-700/50">
                    <span className="text-gray-400">Likes</span>
                    <span className="text-white font-medium">{modelConfig.metadata?.likes?.toLocaleString()}</span>
                  </div>
                </div>
              </div>
            </div>

            {/* User Configuration */}
            <div className="bg-gray-900/50 rounded-2xl p-6 border border-gray-800">
              <h2 className="text-xl font-semibold mb-6">Inference Configuration</h2>
              
              <div className="grid md:grid-cols-2 gap-6">
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">Precision</label>
                    <select
                      value={userConfig.precision}
                      onChange={(e) => setUserConfig(prev => ({ ...prev, precision: e.target.value }))}
                      className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-3 text-white focus:border-purple-500 focus:outline-none"
                    >
                      <option value="fp32">FP32 (32-bit float)</option>
                      <option value="fp16">FP16 (16-bit float)</option>
                      <option value="bf16">BF16 (bfloat16)</option>
                      <option value="int8">INT8 (8-bit integer)</option>
                      <option value="int4">INT4 (4-bit integer)</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Batch Size (Concurrent Requests)
                    </label>
                    <input
                      type="number"
                      min="1"
                      max="128"
                      value={userConfig.batchSize}
                      onChange={(e) => setUserConfig(prev => ({ ...prev, batchSize: parseInt(e.target.value) || 1 }))}
                      className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-3 text-white focus:border-purple-500 focus:outline-none"
                    />
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Sequence Length (Context Window)
                    </label>
                    <input
                      type="number"
                      min="1"
                      max="100000"
                      value={userConfig.seqLength}
                      onChange={(e) => setUserConfig(prev => ({ ...prev, seqLength: parseInt(e.target.value) || 2048 }))}
                      className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-3 text-white focus:border-purple-500 focus:outline-none"
                    />
                  </div>
                </div>

                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Decode Length (Generation)
                    </label>
                    <input
                      type="number"
                      min="0"
                      max="10000"
                      value={userConfig.decodeLength}
                      onChange={(e) => setUserConfig(prev => ({ ...prev, decodeLength: parseInt(e.target.value) || 0 }))}
                      className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-3 text-white focus:border-purple-500 focus:outline-none"
                    />
                    <p className="text-xs text-gray-500 mt-1">Set to 0 for inference-only</p>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Number of Images (Multimodal)
                    </label>
                    <input
                      type="number"
                      min="0"
                      max="100"
                      value={userConfig.numImages}
                      onChange={(e) => setUserConfig(prev => ({ ...prev, numImages: parseInt(e.target.value) || 0 }))}
                      className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-3 text-white focus:border-purple-500 focus:outline-none"
                    />
                  </div>

                  <div className="flex items-center space-x-3">
                    <input
                      type="checkbox"
                      id="includeGradients"
                      checked={userConfig.includeGradients}
                      onChange={(e) => setUserConfig(prev => ({ ...prev, includeGradients: e.target.checked }))}
                      className="w-4 h-4 text-purple-600 bg-gray-800 border-gray-600 rounded focus:ring-purple-500"
                    />
                    <label htmlFor="includeGradients" className="text-sm text-gray-300">
                      Include Gradients (Training Mode)
                    </label>
                  </div>
                </div>
              </div>

              {calculationError && (
                <div className="mt-4 bg-red-900/20 border border-red-500/20 rounded-lg p-4">
                  <div className="flex items-center space-x-2">
                    <AlertCircle className="w-5 h-5 text-red-500" />
                    <span className="text-red-300">{calculationError}</span>
                  </div>
                </div>
              )}

              {/* Gated Model Config Input for Calculate/Analyze */}
              {showConfigInput && (
                <div className="mt-4 bg-amber-900/20 border border-amber-500/20 rounded-lg p-6 space-y-4">
                  <div className="flex items-start space-x-3">
                    <AlertCircle className="w-5 h-5 text-amber-500 mt-0.5" />
                    <div className="flex-1">
                      <h3 className="font-semibold text-amber-300 mb-2">Gated Model Detected</h3>
                      <p className="text-gray-300 text-sm mb-4">
                        This model is gated and requires authentication to access. Please provide the model's config.json manually:
                      </p>
                      
                      <div className="space-y-4">
                        <div>
                          <label className="block text-sm font-medium text-gray-300 mb-2">
                            Paste config.json contents
                          </label>
                          <textarea
                            value={configJson}
                            onChange={(e) => setConfigJson(e.target.value)}
                            placeholder={`{
  "model_type": "llama",
  "hidden_size": 4096,
  "num_hidden_layers": 32,
  "num_attention_heads": 32,
  ...
}`}
                            className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-3 text-white placeholder-gray-500 focus:border-purple-500 focus:outline-none font-mono text-sm min-h-[200px]"
                          />
                        </div>
                        
                        {configSubmissionError && (
                          <div className="bg-red-900/20 border border-red-500/20 rounded-lg p-3">
                            <div className="flex items-center space-x-2">
                              <AlertCircle className="w-4 h-4 text-red-500" />
                              <span className="text-red-300 text-sm">{configSubmissionError}</span>
                            </div>
                          </div>
                        )}
                        
                        <div className="flex space-x-3">
                          <button
                            onClick={submitConfig}
                            disabled={!configJson.trim() || isSubmittingConfig}
                            className="flex-1 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 disabled:from-gray-600 disabled:to-gray-600 disabled:cursor-not-allowed text-white px-4 py-2 rounded-lg font-medium transition-all duration-200"
                          >
                            <div className="flex items-center justify-center space-x-2">
                              {isSubmittingConfig ? (
                                <>
                                  <Loader2 className="w-4 h-4 animate-spin" />
                                  <span>Submitting...</span>
                                </>
                              ) : (
                                <>
                                  <CheckCircle className="w-4 h-4" />
                                  <span>Submit Config & Continue</span>
                                </>
                              )}
                            </div>
                          </button>
                          
                          <button
                            onClick={() => {
                              setShowConfigInput(false);
                              setGatedModelId(null);
                              setConfigJson('');
                              setConfigSubmissionError('');
                              setPendingAction(null);
                            }}
                            className="px-4 py-2 bg-gray-800 hover:bg-gray-700 text-white rounded-lg font-medium transition-all duration-200 border border-gray-600"
                          >
                            Cancel
                          </button>
                        </div>
                        
                        <div className="text-xs text-gray-400 space-y-1">
                          <p>After submitting, we'll automatically continue with your calculation.</p>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              <div className="mt-8 flex space-x-4">
                <button
                  onClick={handleCalculate}
                  disabled={isCalculating}
                  className="flex-1 bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 disabled:from-gray-600 disabled:to-gray-600 disabled:cursor-not-allowed text-white px-6 py-3 rounded-xl font-semibold transition-all duration-200"
                >
                  <div className="flex items-center justify-center space-x-2">
                    {isCalculating ? (
                      <>
                        <Loader2 className="w-5 h-5 animate-spin" />
                        <span>Calculating...</span>
                      </>
                    ) : (
                      <>
                        <Calculator className="w-5 h-5" />
                        <span>Calculate Memory Requirements</span>
                      </>
                    )}
                  </div>
                </button>
                
                <button
                  onClick={analyzeModel}
                  disabled={isAnalyzing}
                  className="px-6 py-3 bg-gray-800 hover:bg-gray-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white rounded-xl font-semibold transition-all duration-200 border border-gray-600"
                >
                  <div className="flex items-center space-x-2">
                    {isAnalyzing ? (
                      <Loader2 className="w-5 h-5 animate-spin" />
                    ) : (
                      <BarChart3 className="w-5 h-5" />
                    )}
                    <span>Analyze</span>
                  </div>
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );

  // Results Screen
  const ResultsScreen = () => (
    <div className="min-h-screen bg-black text-white pt-16">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="flex items-center justify-between mb-8">
          <div className="flex items-center space-x-4">
            <button
              onClick={() => setCurrentScreen('calculator')}
              className="text-gray-400 hover:text-white transition-colors"
            >
              <ArrowLeft className="w-5 h-5" />
            </button>
            <h1 className="text-3xl font-bold">Memory Analysis Results</h1>
          </div>
          
          <div className="flex space-x-3">
            <button
              onClick={addToComparison}
              className="bg-purple-600 hover:bg-purple-700 text-white px-4 py-2 rounded-lg transition-colors"
            >
              Add to Comparison
            </button>
            <button
              onClick={() => setCurrentScreen('comparison')}
              className="bg-gray-800 hover:bg-gray-700 text-white px-4 py-2 rounded-lg transition-colors border border-gray-600"
            >
              View Comparisons
            </button>
          </div>
        </div>

        {results && (
          <div className="grid lg:grid-cols-3 gap-8">
            {/* Summary Cards */}
            <div className="lg:col-span-2 grid md:grid-cols-2 gap-6">
              {[
                {
                  title: 'Total Memory',
                  value: `${results.total_memory_gb?.toFixed(1)} GB`,
                  icon: Database,
                  color: 'from-purple-500 to-pink-500',
                  description: 'Complete memory requirement'
                },
                {
                  title: 'Model Weights',
                  value: `${results.memory_breakdown?.weight_memory_gb?.toFixed(1)} GB`,
                  icon: HardDrive,
                  color: 'from-blue-500 to-cyan-500',
                  description: 'Parameter storage'
                },
                {
                  title: 'KV Cache',
                  value: `${results.memory_breakdown?.kv_cache_gb?.toFixed(1)} GB`,
                  icon: Cpu,
                  color: 'from-green-500 to-emerald-500',
                  description: 'Attention cache'
                },
                {
                  title: 'Activations',
                  value: `${results.memory_breakdown?.activation_memory_gb?.toFixed(1)} GB`,
                  icon: Zap,
                  color: 'from-orange-500 to-red-500',
                  description: 'Intermediate computations'
                }
              ].map((card, index) => (
                <div key={index} className="bg-gray-900/50 rounded-2xl p-6 border border-gray-800">
                  <div className="flex items-center justify-between mb-4">
                    <div className={`w-12 h-12 bg-gradient-to-br ${card.color} rounded-xl flex items-center justify-center`}>
                      <card.icon className="w-6 h-6 text-white" />
                    </div>
                    <div className="text-right">
                      <div className="text-2xl font-bold text-white">{card.value}</div>
                      <div className="text-sm text-gray-400">{card.description}</div>
                    </div>
                  </div>
                  <h3 className="text-lg font-semibold text-white">{card.title}</h3>
                </div>
              ))}
            </div>

            {/* GPU Recommendations */}
            <div className="bg-gray-900/50 rounded-2xl p-6 border border-gray-800">
              <h2 className="text-xl font-semibold mb-6 flex items-center space-x-2">
                <CheckCircle className="w-5 h-5 text-green-500" />
                <span>GPU Recommendations</span>
              </h2>

              <div className="space-y-4">
                <div className="p-4 bg-green-900/20 rounded-lg border border-green-500/20">
                  <div className="flex items-center space-x-2 mb-2">
                    <CheckCircle className="w-5 h-5 text-green-500" />
                    <span className="font-medium text-green-300">Recommended</span>
                  </div>
                  <p className="text-white font-bold text-lg">{results.recommendations?.recommended_gpu_memory_gb} GB GPU</p>
                  <p className="text-sm text-gray-400">Optimal for this configuration</p>
                </div>

                <div className="space-y-3">
                  <div className={`p-3 rounded-lg border ${results.recommendations?.can_fit_24gb_gpu ? 'bg-green-900/10 border-green-500/20' : 'bg-red-900/10 border-red-500/20'}`}>
                    <div className="flex items-center justify-between">
                      <span className="text-white">24GB GPU</span>
                      {results.recommendations?.can_fit_24gb_gpu ? (
                        <CheckCircle className="w-5 h-5 text-green-500" />
                      ) : (
                        <AlertCircle className="w-5 h-5 text-red-500" />
                      )}
                    </div>
                    <p className={`text-xs ${results.recommendations?.can_fit_24gb_gpu ? 'text-green-400' : 'text-red-400'}`}>
                      {results.recommendations?.can_fit_24gb_gpu ? 'Will fit' : 'Too large'}
                    </p>
                  </div>

                  <div className={`p-3 rounded-lg border ${results.recommendations?.can_fit_80gb_gpu ? 'bg-green-900/10 border-green-500/20' : 'bg-red-900/10 border-red-500/20'}`}>
                    <div className="flex items-center justify-between">
                      <span className="text-white">80GB GPU</span>
                      {results.recommendations?.can_fit_80gb_gpu ? (
                        <CheckCircle className="w-5 h-5 text-green-500" />
                      ) : (
                        <AlertCircle className="w-5 h-5 text-red-500" />
                      )}
                    </div>
                    <p className={`text-xs ${results.recommendations?.can_fit_80gb_gpu ? 'text-green-400' : 'text-red-400'}`}>
                      {results.recommendations?.can_fit_80gb_gpu ? 'Will fit' : 'Too large'}
                    </p>
                  </div>
                </div>

                <div className="pt-4 border-t border-gray-700">
                  <h3 className="font-medium text-white mb-2">Model Details</h3>
                  <div className="space-y-2 text-sm">
                    <div className="flex justify-between">
                      <span className="text-gray-400">Type</span>
                      <span className="text-white">{results.model_type}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Attention</span>
                      <span className="text-white">{results.attention_type?.toUpperCase() || 'N/A'}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Parameters</span>
                      <span className="text-white">{(results.parameter_count / 1e9).toFixed(1)}B</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Precision</span>
                      <span className="text-white">{results.precision}</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Hardware Recommendations */}
        {hardwareRecommendations.length > 0 && (
          <div className="mt-8 bg-gray-900/50 rounded-2xl p-6 border border-gray-800">
            <h2 className="text-xl font-semibold mb-6 flex items-center space-x-2">
              <Cpu className="w-5 h-5 text-purple-500" />
              <span>Recommended Hardware</span>
              <span className="text-sm text-gray-400 ml-2">
                ({hardwareRecommendations.length} options)
              </span>
            </h2>

            <div className="space-y-3">
              {hardwareRecommendations.slice(0, 8).map((hw, index) => {
                // Get optimality styling
                const getOptimalityStyles = (optimality: string) => {
                  switch (optimality) {
                    case 'optimal':
                      return {
                        border: 'border-green-500/30',
                        bg: 'bg-green-500/5',
                        indicator: 'bg-green-500',
                        text: 'text-green-400'
                      };
                    case 'good':
                      return {
                        border: 'border-yellow-500/30',
                        bg: 'bg-yellow-500/5',
                        indicator: 'bg-yellow-500',
                        text: 'text-yellow-400'
                      };
                    default:
                      return {
                        border: 'border-orange-500/30',
                        bg: 'bg-orange-500/5',
                        indicator: 'bg-orange-500',
                        text: 'text-orange-400'
                      };
                  }
                };

                const styles = getOptimalityStyles(hw.optimality);

                return (
                  <button
                    key={index}
                    onClick={() => handleHardwareNavigation(hw.hardware_name)}
                    className={`w-full p-4 rounded-xl border ${styles.border} ${styles.bg} hover:border-purple-500/50 transition-all duration-200 text-left group relative`}
                  >
                    {/* Optimality Indicator */}
                    <div className={`absolute left-0 top-0 bottom-0 w-1 ${styles.indicator} rounded-l-xl`}></div>
                    
                    <div className="flex items-start justify-between">
                      <div className="flex-1 min-w-0 pr-4">
                        <div className="flex items-center space-x-2 mb-2">
                          <h3 className="font-semibold text-white group-hover:text-purple-300 transition-colors line-clamp-2">
                            {hw.hardware_name}
                          </h3>
                          <span className="text-xs bg-purple-900/50 text-purple-300 px-2 py-1 rounded-full whitespace-nowrap border border-purple-500/30">
                            {hw.type?.toUpperCase()}
                          </span>
                        </div>
                        
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm text-gray-300">
                          <div>
                            <span className="text-gray-500">Nodes:</span>
                            <span className="font-medium ml-1">{hw.nodes_required}</span>
                          </div>
                          <div>
                            <span className="text-gray-500">Memory:</span>
                            <span className="font-medium ml-1">{hw.memory_per_chip} GB</span>
                          </div>
                          <div>
                            <span className="text-gray-500">Utilization:</span>
                            <span className={`font-medium ml-1 ${styles.text}`}>{hw.utilization}%</span>
                          </div>
                          {hw.manufacturer && (
                            <div>
                              <span className="text-gray-500">Mfr:</span>
                              <span className="font-medium ml-1">{hw.manufacturer}</span>
                            </div>
                          )}
                        </div>
                      </div>
                      
                      {/* Optimality Badge */}
                      <div className={`flex items-center space-x-1 ${styles.text}`}>
                        <div className={`w-2 h-2 rounded-full ${styles.indicator}`}></div>
                        <span className="text-xs font-medium capitalize">{hw.optimality}</span>
                      </div>
                    </div>
                  </button>
                );
              })}
            </div>

            {/* Legend */}
            <div className="mt-4 pt-4 border-t border-gray-700">
              <div className="flex items-center justify-center space-x-6 text-xs text-gray-400">
                <div className="flex items-center space-x-1">
                  <div className="w-2 h-2 rounded-full bg-green-500"></div>
                  <span>Most Optimal</span>
                </div>
                <div className="flex items-center space-x-1">
                  <div className="w-2 h-2 rounded-full bg-yellow-500"></div>
                  <span>Good Choice</span>
                </div>
                <div className="flex items-center space-x-1">
                  <div className="w-2 h-2 rounded-full bg-orange-500"></div>
                  <span>Acceptable</span>
                </div>
              </div>
            </div>

            <div className="text-center mt-4">
              <button
                onClick={() => handleHardwareNavigation('')}
                className="text-purple-400 hover:text-purple-300 transition-colors text-sm"
              >
                View All Hardware →
              </button>
            </div>
          </div>
        )}

        {/* Detailed Breakdown */}
        {results && (
          <div className="mt-8 bg-gray-900/50 rounded-2xl p-6 border border-gray-800">
            <h2 className="text-xl font-semibold mb-6">Detailed Memory Breakdown</h2>
            
            <div className="grid md:grid-cols-2 gap-8">
              <div className="space-y-4">
                {[
                  { label: 'Model Weights', value: results.memory_breakdown?.weight_memory_gb || 0, color: 'bg-blue-500' },
                  { label: 'KV Cache', value: results.memory_breakdown?.kv_cache_gb || 0, color: 'bg-green-500' },
                  { label: 'Activations', value: results.memory_breakdown?.activation_memory_gb || 0, color: 'bg-orange-500' },
                  { label: 'State Memory', value: results.memory_breakdown?.state_memory_gb || 0, color: 'bg-purple-500' },
                  { label: 'Image Memory', value: results.memory_breakdown?.image_memory_gb || 0, color: 'bg-pink-500' },
                  { label: 'Extra Work', value: results.memory_breakdown?.extra_work_gb || 0, color: 'bg-gray-500' }
                ].map((item, index) => (
                  <div key={index} className="flex items-center space-x-4">
                    <div className={`w-4 h-4 rounded ${item.color}`}></div>
                    <div className="flex-1 flex justify-between">
                      <span className="text-gray-300">{item.label}</span>
                      <span className="text-white font-medium">{item.value.toFixed(2)} GB</span>
                    </div>
                    <div className="w-24 bg-gray-700 rounded-full h-2">
                      <div
                        className={`h-2 rounded-full ${item.color}`}
                        style={{
                          width: `${Math.min(100, (item.value / results.total_memory_gb) * 100)}%`
                        }}
                      ></div>
                    </div>
                  </div>
                ))}
              </div>
              
              <div className="bg-gray-800/50 rounded-xl p-4">
                <h3 className="font-medium text-white mb-4">Memory Distribution</h3>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Weights</span>
                    <span className="text-white">{((results.memory_breakdown?.weight_memory_gb / results.total_memory_gb) * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Runtime</span>
                    <span className="text-white">{(((results.total_memory_gb - results.memory_breakdown?.weight_memory_gb) / results.total_memory_gb) * 100).toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between pt-2 border-t border-gray-600">
                    <span className="text-gray-400">Total</span>
                    <span className="text-white font-bold">{results.total_memory_gb?.toFixed(2)} GB</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );

  // Comparison Screen
  const ComparisonScreen = () => (
    <div className="min-h-screen bg-black text-white pt-16">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="flex items-center justify-between mb-8">
          <div className="flex items-center space-x-4">
            <button
              onClick={() => setCurrentScreen('home')}
              className="text-gray-400 hover:text-white transition-colors"
            >
              <ArrowLeft className="w-5 h-5" />
            </button>
            <h1 className="text-3xl font-bold">Model Comparison</h1>
          </div>
          
          <button
            onClick={() => setCurrentScreen('calculator')}
            className="bg-purple-600 hover:bg-purple-700 text-white px-4 py-2 rounded-lg transition-colors"
          >
            Add More Models
          </button>
        </div>

        {comparisonResults.length === 0 ? (
          <div className="text-center py-16">
            <GitCompare className="w-16 h-16 text-gray-600 mx-auto mb-4" />
            <h2 className="text-2xl font-bold text-gray-300 mb-2">No Models to Compare</h2>
            <p className="text-gray-500 mb-6">Calculate memory for models and add them to comparison</p>
            <button
              onClick={() => setCurrentScreen('calculator')}
              className="bg-purple-600 hover:bg-purple-700 text-white px-6 py-3 rounded-lg transition-colors"
            >
              Start Calculating
            </button>
          </div>
        ) : (
          <div className="space-y-8">
            {/* Comparison Table */}
            <div className="bg-gray-900/50 rounded-2xl border border-gray-800 overflow-hidden">
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead className="bg-gray-800">
                    <tr>
                      <th className="px-6 py-4 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Model</th>
                      <th className="px-6 py-4 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Type</th>
                      <th className="px-6 py-4 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Parameters</th>
                      <th className="px-6 py-4 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Total Memory</th>
                      <th className="px-6 py-4 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">GPU Rec.</th>
                      <th className="px-6 py-4 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">24GB GPU</th>
                      <th className="px-6 py-4 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">Actions</th>
                    </tr>
                  </thead>
                  <tbody className="bg-gray-900/30 divide-y divide-gray-700">
                    {comparisonResults.map((item) => (
                      <tr key={item.id} className="hover:bg-gray-800/30">
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="flex items-center space-x-3">
                            <ModelLogoWithFallback logo={item.logo} modelId={item.model_id} size="sm" />
                            <div>
                              <div className="text-sm font-medium text-white">{item.model_name}</div>
                              <div className="text-sm text-gray-400">{item.config?.precision}</div>
                            </div>
                          </div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-300">
                          {item.results?.model_type}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-300">
                          {(item.results?.parameter_count / 1e9).toFixed(1)}B
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="text-sm font-medium text-white">{item.results?.total_memory_gb?.toFixed(1)} GB</div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-300">
                          {item.results?.recommendations?.recommended_gpu_memory_gb} GB
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          {item.results?.recommendations?.can_fit_24gb_gpu ? (
                            <CheckCircle className="w-5 h-5 text-green-500" />
                          ) : (
                            <AlertCircle className="w-5 h-5 text-red-500" />
                          )}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="flex items-center space-x-4">
                            <button
                              onClick={() => {
                                setSelectedModelId(item.model_id);
                                setCurrentScreen('details');
                              }}
                              className="text-purple-400 hover:text-purple-300 transition-colors text-sm flex items-center space-x-1"
                            >
                              <span>See More</span>
                              <ArrowRight className="w-4 h-4" />
                            </button>
                            <button
                              onClick={() => setComparisonResults(prev => prev.filter(r => r.id !== item.id))}
                              className="text-red-400 hover:text-red-300 transition-colors"
                            >
                              Remove
                            </button>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );

  // Analysis Screen
  const AnalysisScreen = () => (
    <div className="min-h-screen bg-black text-white pt-16">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="flex items-center space-x-4 mb-8">
          <button
            onClick={() => setCurrentScreen('calculator')}
            className="text-gray-400 hover:text-white transition-colors"
          >
            <ArrowLeft className="w-5 h-5" />
          </button>
          <h1 className="text-3xl font-bold">Advanced Analysis</h1>
          {analysisResults && (
            <button
              onClick={() => {
                setSelectedModelId(analysisResults.model_id);
                setCurrentScreen('details');
              }}
              className="text-purple-400 hover:text-purple-300 transition-colors text-sm flex items-center space-x-1"
            >
              <span>See Model Details</span>
              <ArrowRight className="w-4 h-4" />
            </button>
          )}
        </div>

        {analysisResults ? (
          <div className="space-y-8">
            <div className="bg-gray-900/50 rounded-2xl p-6 border border-gray-800">
              <h2 className="text-xl font-semibold mb-6">Attention Efficiency Analysis</h2>
              <div className="grid md:grid-cols-2 gap-8">
                <div>
                  <h3 className="font-medium text-white mb-4">Sequence Length Scaling</h3>
                  <div className="space-y-4">
                    {Object.entries(analysisResults.analysis || {}).map(([seqLen, data]) => (
                      <div key={seqLen} className="bg-gray-800/50 rounded-lg p-4 border border-gray-700">
                        <div className="flex justify-between items-center mb-2">
                          <span className="text-gray-300">{parseInt(seqLen).toLocaleString()} tokens</span>
                          <span className="text-white font-medium">{data.total_memory_gb?.toFixed(1)} GB</span>
                        </div>
                        <div className="text-sm text-gray-400">
                          KV Cache: {data.kv_cache_gb?.toFixed(1)} GB ({data.kv_cache_percent?.toFixed(1)}%)
                        </div>
                        <div className="mt-2 w-full bg-gray-700 rounded-full h-2">
                          <div
                            className="bg-gradient-to-r from-purple-500 to-pink-500 h-2 rounded-full"
                            style={{ width: `${Math.min(100, data.kv_cache_percent)}%` }}
                          ></div>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
                
                <div>
                  <h3 className="font-medium text-white mb-4">Efficiency Insights</h3>
                  <div className="space-y-4">
                    <div className="bg-purple-900/20 rounded-lg p-4 border border-purple-500/20">
                      <h4 className="font-medium text-purple-300 mb-2">Attention Type</h4>
                      <p className="text-white text-lg">{analysisResults.attention_type?.toUpperCase() || 'N/A'}</p>
                      <p className="text-sm text-gray-400 mt-1">
                        Efficiency Rating: {analysisResults.insights?.efficiency_rating || 'Unknown'}
                      </p>
                    </div>
                    
                    <div className="bg-gray-800/50 rounded-lg p-4 border border-gray-700">
                      <h4 className="font-medium text-white mb-2">Recommendations</h4>
                      <ul className="text-sm text-gray-300 space-y-1">
                        {analysisResults.insights?.recommendations?.map((rec, index) => (
                          <li key={index}>• {rec}</li>
                        )) || [
                          <li key="default">• No specific recommendations available</li>
                        ]}
                      </ul>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        ) : (
          <div className="text-center py-16">
            <BarChart3 className="w-16 h-16 text-gray-600 mx-auto mb-4" />
            <h2 className="text-2xl font-bold text-gray-300 mb-2">No Analysis Data</h2>
            <p className="text-gray-500 mb-6">Run an analysis from the calculator to see detailed insights</p>
            <button
              onClick={() => setCurrentScreen('calculator')}
              className="bg-purple-600 hover:bg-purple-700 text-white px-6 py-3 rounded-lg transition-colors"
            >
              Go to Calculator
            </button>
          </div>
        )}
      </div>
    </div>
  );

  // Models Screen
  const ModelsScreen = () => {
    const [models, setModels] = useState<ModelSummary[]>([]);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState('');

    useEffect(() => {
      fetchModels();
    }, []);

    const fetchModels = async () => {
      setIsLoading(true);
      setError('');
      try {
        const response = await fetch(`${API_BASE}/models/list`);
        if (!response.ok) {
          throw new Error('Failed to fetch models');
        }
        const data = await response.json();
        setModels(data.models);
      } catch (err) {
        setError('Failed to load models. Please try again later.');
        console.error('Error fetching models:', err);
      } finally {
        setIsLoading(false);
      }
    };

    const handleModelSelect = (modelId: string) => {
      setModelUrl(modelId);
      setCurrentScreen('calculator');
      handleModelUrlSubmit(modelId);
    };

    return (
      <div className="min-h-screen bg-black text-white pt-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          <div className="flex items-center justify-between mb-8">
            <h1 className="text-3xl font-bold">Available Models</h1>
            <button
              onClick={fetchModels}
              className="text-purple-400 hover:text-purple-300 transition-colors"
              disabled={isLoading}
            >
              <RefreshCw className={`w-5 h-5 ${isLoading ? 'animate-spin' : ''}`} />
            </button>
          </div>

          {isLoading ? (
            <div className="text-center py-16">
              <Loader2 className="w-8 h-8 animate-spin text-purple-500 mx-auto mb-4" />
              <p className="text-gray-400">Loading models...</p>
            </div>
          ) : error ? (
            <div className="text-center py-16">
              <AlertCircle className="w-8 h-8 text-red-500 mx-auto mb-4" />
              <p className="text-red-400">{error}</p>
              <button
                onClick={fetchModels}
                className="mt-4 text-purple-400 hover:text-purple-300 transition-colors"
              >
                Try Again
              </button>
            </div>
          ) : (
            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
              {models.map((model, index) => (
                <div
                  key={index}
                  className="bg-gray-900/50 p-6 rounded-xl border border-gray-800 hover:border-purple-500/50 transition-all duration-200 flex flex-col justify-between"
                >
                  <div>
                    <div className="flex items-center justify-between mb-4">
                      <ModelLogoWithFallback logo={model.logo} modelId={model.model_id} size="md" />
                      <span className="text-sm text-purple-400">
                        {model.parameter_count ? `${(model.parameter_count / 1e9).toFixed(1)}B params` : ''}
                      </span>
                    </div>
                    
                    <h3 className="text-xl font-semibold mb-2">{model.name}</h3>
                    <p className="text-gray-400 text-sm mb-4">{model.model_id}</p>
                    
                    <div className="space-y-2 mb-6">
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-400">Type</span>
                        <span className="text-white">{model.model_type}</span>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-400">Attention</span>
                        <span className="text-white">{model.attention_type || 'N/A'}</span>
                      </div>
                    </div>
                  </div>
                  
                  <div className="flex items-center justify-between mt-auto">
                    <button
                      onClick={() => {
                        setSelectedModelId(model.model_id);
                        setCurrentScreen('details');
                      }}
                      className="text-purple-400 hover:text-purple-300 transition-colors text-sm flex items-center space-x-1"
                    >
                      <span>See More</span>
                      <ArrowRight className="w-4 h-4" />
                    </button>
                    <button
                      onClick={() => handleModelSelect(model.model_id)}
                      className="bg-purple-600 hover:bg-purple-700 text-white px-4 py-2 rounded-lg text-sm font-medium transition-colors"
                    >
                      Analyze
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-black">
      <Navigation />
      
      {currentScreen === 'home' && <HomeScreen />}
      {currentScreen === 'models' && <ModelsScreen />}
      {currentScreen === 'calculator' && <CalculatorScreen />}
      {currentScreen === 'results' && <ResultsScreen />}
      {currentScreen === 'comparison' && <ComparisonScreen />}
      {currentScreen === 'analysis' && <AnalysisScreen />}
      {currentScreen === 'hardware' && (
        <HardwareList onHardwareSelect={handleHardwareNavigation} />
      )}
      {currentScreen === 'hardware-detail' && selectedHardwareName && (
        <HardwareDetail
          hardwareName={selectedHardwareName}
          onBack={() => setCurrentScreen('hardware')}
        />
      )}
      {currentScreen === 'details' && selectedModelId && (
        <ModelDetails
          modelId={selectedModelId}
          onBack={() => {
            setSelectedModelId(null);
            setCurrentScreen('models');
          }}
        />
      )}
    </div>
  );
};

export default AIMemoryCalculator;