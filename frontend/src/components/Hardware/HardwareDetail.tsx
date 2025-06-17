import React, { useState, useEffect } from 'react';
import { ArrowLeft, Zap, HardDrive, Activity, Cpu, DollarSign, Globe, Building, ExternalLink, AlertCircle, Loader2, CheckCircle, XCircle } from 'lucide-react';
import { HardwareDetail as HardwareDetailType, ModelCompatibility } from '../../types/hardware';
import { hardwareAPI } from '../../services/hardwareAPI';
import { SpecificationTooltip } from '../Common/SpecificationTooltip';

interface HardwareDetailProps {
  hardwareName: string;
  onBack: () => void;
}

export const HardwareDetail: React.FC<HardwareDetailProps> = ({ hardwareName, onBack }) => {
  const [hardware, setHardware] = useState<HardwareDetailType | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'overview' | 'pricing' | 'compatibility'>('overview');

  useEffect(() => {
    fetchHardwareDetails();
  }, [hardwareName]);

  const fetchHardwareDetails = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const data = await hardwareAPI.getDetails(hardwareName);
      setHardware(data);
    } catch (error) {
      console.error('Error fetching hardware details:', error);
      setError('Failed to load hardware details. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const formatNumber = (num: number) => {
    if (num >= 1000) {
      return (num / 1000).toFixed(1) + 'K';
    }
    return num.toLocaleString();
  };

  const getTypeColor = (type: string) => {
    switch (type) {
      case 'gpu':
        return 'bg-green-100 text-green-700 border-green-200';
      case 'cpu':
        return 'bg-blue-100 text-blue-700 border-blue-200';
      case 'accelerator':
        return 'bg-purple-100 text-purple-700 border-purple-200';
      case 'asic':
        return 'bg-orange-100 text-orange-700 border-orange-200';
      default:
        return 'bg-gray-100 text-gray-700 border-gray-200';
    }
  };

  const getCompatibilityStatus = (memoryGB: number, hardwareMemory: number) => {
    const ratio = memoryGB / hardwareMemory;
    if (ratio <= 0.8) return { status: 'green', label: 'Optimal' };
    if (ratio <= 1.0) return { status: 'yellow', label: 'Tight' };
    return { status: 'red', label: 'Insufficient' };
  };

  // Mock model compatibility data - in real implementation, this would come from API
  const mockCompatibility: ModelCompatibility[] = [
    {
      model_id: 'meta-llama/Llama-2-7b-hf',
      model_name: 'Llama 2 7B',
      parameters: 7000000000,
      compatibility: {
        seq2k: { memory_gb: 14.2, status: 'green' },
        seq4k: { memory_gb: 16.8, status: 'green' },
        seq8k: { memory_gb: 22.1, status: 'yellow' },
        seq16k: { memory_gb: 32.7, status: 'red' }
      },
      overallStatus: 'optimal'
    },
    {
      model_id: 'mistralai/Mistral-7B-v0.1',
      model_name: 'Mistral 7B',
      parameters: 7200000000,
      compatibility: {
        seq2k: { memory_gb: 13.8, status: 'green' },
        seq4k: { memory_gb: 16.2, status: 'green' },
        seq8k: { memory_gb: 21.4, status: 'yellow' },
        seq16k: { memory_gb: 31.8, status: 'red' }
      },
      overallStatus: 'optimal'
    }
  ];

  if (loading) {
    return (
      <div className="min-h-screen bg-black text-white pt-16">
        <div className="max-w-7xl mx-auto px-4 py-8">
          <div className="text-center py-16">
            <Loader2 className="w-12 h-12 animate-spin text-purple-500 mx-auto mb-4" />
            <p className="text-gray-400">Loading hardware details...</p>
          </div>
        </div>
      </div>
    );
  }

  if (error || !hardware) {
    return (
      <div className="min-h-screen bg-black text-white pt-16">
        <div className="max-w-7xl mx-auto px-4 py-8">
          <div className="flex items-center space-x-4 mb-8">
            <button
              onClick={onBack}
              className="text-gray-400 hover:text-white transition-colors"
            >
              <ArrowLeft className="w-5 h-5" />
            </button>
            <h1 className="text-3xl font-bold">Hardware Details</h1>
          </div>
          <div className="text-center py-16">
            <AlertCircle className="w-12 h-12 text-red-500 mx-auto mb-4" />
            <p className="text-gray-400 mb-4">{error}</p>
            <button
              onClick={fetchHardwareDetails}
              className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700"
            >
              Try Again
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-black text-white pt-16">
      <div className="max-w-7xl mx-auto px-4 py-8">
        {/* Header */}
        <div className="flex items-center space-x-4 mb-8">
          <button
            onClick={onBack}
            className="text-gray-400 hover:text-white transition-colors"
          >
            <ArrowLeft className="w-5 h-5" />
          </button>
          <div className="flex-1">
            <div className="flex items-center space-x-4 mb-2">
              <h1 className="text-3xl font-bold text-white">{hardware.name}</h1>
              <span className={`text-xs font-medium px-3 py-1 rounded-full border ${getTypeColor(hardware.type)}`}>
                {hardware.type.toUpperCase()}
              </span>
              {hardware.real_values && (
                <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-green-900/50 text-green-300 border border-green-500/30">
                  <CheckCircle className="w-3 h-3 mr-1" />
                  Verified
                </span>
              )}
            </div>
            {hardware.manufacturer && (
              <p className="text-gray-400">{hardware.manufacturer}</p>
            )}
          </div>
          {hardware.url && (
            <a
              href={hardware.url}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center space-x-2 px-4 py-2 bg-purple-600 hover:bg-purple-700 text-white rounded-lg transition-colors"
            >
              <ExternalLink className="w-4 h-4" />
              <span>Learn More</span>
            </a>
          )}
        </div>

        {/* Description */}
        {hardware.description && (
          <div className="bg-gray-900/50 rounded-lg p-6 mb-8 border border-gray-800">
            <p className="text-gray-300 leading-relaxed">{hardware.description}</p>
          </div>
        )}

        {/* Tabs */}
        <div className="mb-8">
          <div className="border-b border-gray-800">
            <nav className="-mb-px flex space-x-8">
              {[
                { id: 'overview', label: 'Overview', icon: Cpu },
                { id: 'pricing', label: 'Pricing', icon: DollarSign },
                { id: 'compatibility', label: 'Model Compatibility', icon: CheckCircle }
              ].map(({ id, label, icon: Icon }) => (
                <button
                  key={id}
                  onClick={() => setActiveTab(id as any)}
                  className={`flex items-center space-x-2 py-4 px-1 border-b-2 font-medium text-sm ${
                    activeTab === id
                      ? 'border-purple-500 text-purple-400'
                      : 'border-transparent text-gray-400 hover:text-gray-300 hover:border-gray-700'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  <span>{label}</span>
                </button>
              ))}
            </nav>
          </div>
        </div>

        {/* Tab Content */}
        {activeTab === 'overview' && (
          <div className="space-y-8">
            {/* Key Specifications */}
            <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
              <div className="bg-gray-900/50 rounded-lg p-6 border border-gray-800">
                <div className="flex items-center space-x-3 mb-4">
                  <div className="w-10 h-10 bg-gradient-to-br from-yellow-500 to-orange-500 rounded-lg flex items-center justify-center">
                    <Zap className="w-5 h-5 text-white" />
                  </div>
                  <div>
                    <h3 className="font-semibold text-white">Performance</h3>
                    <p className="text-2xl font-bold text-white">{formatNumber(hardware.flops)} TFLOPS</p>
                  </div>
                </div>
                <p className="text-sm text-gray-400">Peak computational performance</p>
              </div>

              <div className="bg-gray-900/50 rounded-lg p-6 border border-gray-800">
                <div className="flex items-center space-x-3 mb-4">
                  <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-cyan-500 rounded-lg flex items-center justify-center">
                    <HardDrive className="w-5 h-5 text-white" />
                  </div>
                  <div>
                    <h3 className="font-semibold text-white">Memory</h3>
                    <p className="text-2xl font-bold text-white">{hardware.memory_size} GB</p>
                  </div>
                </div>
                <p className="text-sm text-gray-400">Total memory capacity</p>
              </div>

              <div className="bg-gray-900/50 rounded-lg p-6 border border-gray-800">
                <div className="flex items-center space-x-3 mb-4">
                  <div className="w-10 h-10 bg-gradient-to-br from-green-500 to-emerald-500 rounded-lg flex items-center justify-center">
                    <Activity className="w-5 h-5 text-white" />
                  </div>
                  <div>
                    <h3 className="font-semibold text-white">Bandwidth</h3>
                    <p className="text-2xl font-bold text-white">{formatNumber(hardware.memory_bw)} GB/s</p>
                  </div>
                </div>
                <p className="text-sm text-gray-400">Memory bandwidth</p>
              </div>

              {hardware.power && (
                <div className="bg-gray-900/50 rounded-lg p-6 border border-gray-800">
                  <div className="flex items-center space-x-3 mb-4">
                    <div className="w-10 h-10 bg-gradient-to-br from-red-500 to-pink-500 rounded-lg flex items-center justify-center">
                      <Cpu className="w-5 h-5 text-white" />
                    </div>
                    <div>
                      <h3 className="font-semibold text-white">Power</h3>
                      <p className="text-2xl font-bold text-white">{hardware.power} W</p>
                    </div>
                  </div>
                  <p className="text-sm text-gray-400">Power consumption</p>
                </div>
              )}
            </div>

            {/* Detailed Specifications */}
            <div className="bg-gray-900/50 rounded-lg p-6 border border-gray-800">
              <h2 className="text-xl font-semibold mb-6 text-white">Technical Specifications</h2>
              <div className="grid md:grid-cols-2 gap-8">
                <div className="space-y-4">
                  <div className="flex justify-between items-center py-2 border-b border-gray-700/50">
                    <span className="text-gray-400 flex items-center">
                      Hardware Type
                      <SpecificationTooltip content="The category of hardware (GPU, CPU, Accelerator, or ASIC)" />
                    </span>
                    <span className="text-white font-medium">{hardware.type.toUpperCase()}</span>
                  </div>
                  <div className="flex justify-between items-center py-2 border-b border-gray-700/50">
                    <span className="text-gray-400 flex items-center">
                      Manufacturer
                      <SpecificationTooltip content="The company that manufactures this hardware" />
                    </span>
                    <span className="text-white font-medium">{hardware.manufacturer || 'N/A'}</span>
                  </div>
                  <div className="flex justify-between items-center py-2 border-b border-gray-700/50">
                    <span className="text-gray-400 flex items-center">
                      Performance
                      <SpecificationTooltip content="Peak floating-point operations per second in TFLOPS" />
                    </span>
                    <span className="text-white font-medium">{hardware.flops.toLocaleString()} TFLOPS</span>
                  </div>
                  <div className="flex justify-between items-center py-2 border-b border-gray-700/50">
                    <span className="text-gray-400 flex items-center">
                      Memory Size
                      <SpecificationTooltip content="Total memory capacity available for computations" />
                    </span>
                    <span className="text-white font-medium">{hardware.memory_size} GB</span>
                  </div>
                </div>
                
                <div className="space-y-4">
                  <div className="flex justify-between items-center py-2 border-b border-gray-700/50">
                    <span className="text-gray-400 flex items-center">
                      Memory Bandwidth
                      <SpecificationTooltip content="Rate at which data can be read from or written to memory" />
                    </span>
                    <span className="text-white font-medium">{hardware.memory_bw.toLocaleString()} GB/s</span>
                  </div>
                  {hardware.icn && (
                    <div className="flex justify-between items-center py-2 border-b border-gray-700/50">
                      <span className="text-gray-400 flex items-center">
                        Interconnect
                        <SpecificationTooltip content="Inter-chip network bandwidth for multi-chip configurations" />
                      </span>
                      <span className="text-white font-medium">{hardware.icn} GB/s</span>
                    </div>
                  )}
                  {hardware.icn_ll && (
                    <div className="flex justify-between items-center py-2 border-b border-gray-700/50">
                      <span className="text-gray-400 flex items-center">
                        Interconnect LL
                        <SpecificationTooltip content="Low-latency interconnect bandwidth" />
                      </span>
                      <span className="text-white font-medium">{hardware.icn_ll} GB/s</span>
                    </div>
                  )}
                  {hardware.power && (
                    <div className="flex justify-between items-center py-2 border-b border-gray-700/50">
                      <span className="text-gray-400 flex items-center">
                        Power Consumption
                        <SpecificationTooltip content="Maximum power consumption under full load" />
                      </span>
                      <span className="text-white font-medium">{hardware.power} W</span>
                    </div>
                  )}
                  <div className="flex justify-between items-center py-2 border-b border-gray-700/50">
                    <span className="text-gray-400 flex items-center">
                      Data Source
                      <SpecificationTooltip content="Source of the hardware specifications" />
                    </span>
                    <span className="text-white font-medium">{hardware.source}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'pricing' && (
          <div className="space-y-8">
            {/* On-Premise Vendors */}
            {hardware.vendors && hardware.vendors.length > 0 && (
              <div className="bg-gray-900/50 rounded-lg p-6 border border-gray-800">
                <h2 className="text-xl font-semibold mb-6 text-white flex items-center">
                  <Building className="w-5 h-5 mr-2 text-blue-500" />
                  On-Premise Vendors
                </h2>
                <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
                  {hardware.vendors.map((vendor, index) => (
                    <div key={index} className="bg-gray-800/50 rounded-lg p-4 border border-gray-700">
                      <h3 className="font-semibold text-white mb-3">{vendor.vendor_name}</h3>
                      <div className="space-y-2">
                        {vendor.price_lower && vendor.price_upper ? (
                          <div className="flex justify-between">
                            <span className="text-gray-400">Price Range:</span>
                            <span className="text-green-400 font-medium">
                              ${vendor.price_lower.toLocaleString()} - ${vendor.price_upper.toLocaleString()}
                            </span>
                          </div>
                        ) : vendor.price_lower ? (
                          <div className="flex justify-between">
                            <span className="text-gray-400">Starting at:</span>
                            <span className="text-green-400 font-medium">
                              ${vendor.price_lower.toLocaleString()}
                            </span>
                          </div>
                        ) : (
                          <div className="text-gray-500 text-sm">Contact for pricing</div>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Cloud Providers */}
            {hardware.clouds && hardware.clouds.length > 0 && (
              <div className="bg-gray-900/50 rounded-lg p-6 border border-gray-800">
                <h2 className="text-xl font-semibold mb-6 text-white flex items-center">
                  <Globe className="w-5 h-5 mr-2 text-purple-500" />
                  Cloud Providers
                </h2>
                <div className="space-y-4">
                  {hardware.clouds.map((cloud, index) => (
                    <div key={index} className="bg-gray-800/50 rounded-lg p-4 border border-gray-700">
                      <div className="flex items-start justify-between mb-3">
                        <div>
                          <h3 className="font-semibold text-white">{cloud.cloud_name}</h3>
                          <p className="text-sm text-gray-400">{cloud.instance_name}</p>
                        </div>
                        {cloud.price_lower && cloud.price_upper ? (
                          <div className="text-right">
                            <div className="text-green-400 font-medium">
                              ${cloud.price_lower.toFixed(2)} - ${cloud.price_upper.toFixed(2)}
                            </div>
                            <div className="text-xs text-gray-500">per hour</div>
                          </div>
                        ) : cloud.price_lower ? (
                          <div className="text-right">
                            <div className="text-green-400 font-medium">
                              ${cloud.price_lower.toFixed(2)}
                            </div>
                            <div className="text-xs text-gray-500">per hour</div>
                          </div>
                        ) : (
                          <div className="text-gray-500 text-sm">Contact for pricing</div>
                        )}
                      </div>
                      {cloud.regions && cloud.regions.length > 0 && (
                        <div>
                          <span className="text-xs text-gray-500">Available in: </span>
                          <span className="text-xs text-gray-400">
                            {cloud.regions.join(', ')}
                          </span>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}

            {(!hardware.vendors || hardware.vendors.length === 0) && 
             (!hardware.clouds || hardware.clouds.length === 0) && (
              <div className="text-center py-12">
                <DollarSign className="w-16 h-16 text-gray-600 mx-auto mb-4" />
                <h3 className="text-lg font-medium text-white mb-2">No Pricing Information</h3>
                <p className="text-gray-400">Pricing details are not available for this hardware.</p>
              </div>
            )}
          </div>
        )}

        {activeTab === 'compatibility' && (
          <div className="space-y-8">
            <div className="bg-gray-900/50 rounded-lg p-6 border border-gray-800">
              <h2 className="text-xl font-semibold mb-6 text-white">Model Compatibility Matrix</h2>
              <p className="text-gray-400 mb-6">
                Memory requirements for popular AI models at different sequence lengths (batch size: 10)
              </p>
              
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className="border-b border-gray-700">
                      <th className="text-left py-3 px-4 text-gray-300 font-medium">Model</th>
                      <th className="text-center py-3 px-4 text-gray-300 font-medium">Parameters</th>
                      <th className="text-center py-3 px-4 text-gray-300 font-medium">2K Tokens</th>
                      <th className="text-center py-3 px-4 text-gray-300 font-medium">4K Tokens</th>
                      <th className="text-center py-3 px-4 text-gray-300 font-medium">8K Tokens</th>
                      <th className="text-center py-3 px-4 text-gray-300 font-medium">16K Tokens</th>
                      <th className="text-center py-3 px-4 text-gray-300 font-medium">Overall</th>
                    </tr>
                  </thead>
                  <tbody>
                    {mockCompatibility.map((model, index) => (
                      <tr key={index} className="border-b border-gray-800 hover:bg-gray-800/30">
                        <td className="py-4 px-4">
                          <div>
                            <div className="font-medium text-white">{model.model_name}</div>
                            <div className="text-sm text-gray-400">{model.model_id}</div>
                          </div>
                        </td>
                        <td className="text-center py-4 px-4 text-gray-300">
                          {(model.parameters / 1e9).toFixed(1)}B
                        </td>
                        {Object.entries(model.compatibility).map(([key, req]) => {
                          const compat = getCompatibilityStatus(req.memory_gb, hardware.memory_size);
                          return (
                            <td key={key} className="text-center py-4 px-4">
                              <div className="flex flex-col items-center space-y-1">
                                <span className="text-white font-medium">{req.memory_gb.toFixed(1)} GB</span>
                                <span className={`text-xs px-2 py-1 rounded-full ${
                                  compat.status === 'green' ? 'bg-green-900/50 text-green-300' :
                                  compat.status === 'yellow' ? 'bg-yellow-900/50 text-yellow-300' :
                                  'bg-red-900/50 text-red-300'
                                }`}>
                                  {compat.label}
                                </span>
                              </div>
                            </td>
                          );
                        })}
                        <td className="text-center py-4 px-4">
                          {model.overallStatus === 'optimal' ? (
                            <CheckCircle className="w-5 h-5 text-green-500 mx-auto" />
                          ) : model.overallStatus === 'partial' ? (
                            <AlertCircle className="w-5 h-5 text-yellow-500 mx-auto" />
                          ) : (
                            <XCircle className="w-5 h-5 text-red-500 mx-auto" />
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              
              <div className="mt-6 flex items-center space-x-6 text-sm">
                <div className="flex items-center space-x-2">
                  <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                  <span className="text-gray-400">Optimal (&lt;80% memory)</span>
                </div>
                <div className="flex items-center space-x-2">
                  <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
                  <span className="text-gray-400">Tight (80-100% memory)</span>
                </div>
                <div className="flex items-center space-x-2">
                  <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                  <span className="text-gray-400">Insufficient (&gt;100% memory)</span>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}; 