# Hardware Frontend Implementation Plan - BudSimulator

## Overview
This document outlines the implementation plan for integrating hardware functionality into the existing BudSimulator React/TypeScript frontend. The plan builds upon the current AIMemoryCalculator component and extends it with hardware browsing, comparison, and recommendation features.

## 1. Architecture Integration

### 1.1 Current Structure Analysis
The existing frontend uses:
- React 18.2 with TypeScript
- Tailwind CSS for styling
- Lucide React for icons
- Single-page application with internal navigation states
- API proxy configured to `http://localhost:8000`

### 1.2 Proposed Navigation Enhancement
```typescript
// Update AIMemoryCalculator navigation states
const [currentScreen, setCurrentScreen] = useState<
  'home' | 'calculator' | 'results' | 'comparison' | 'analysis' | 'models' | 'hardware' | 'hardware-detail'
>('home');
```

## 2. Component Structure

### 2.1 New Components to Create

```
src/
├── components/
│   ├── Hardware/
│   │   ├── HardwareList.tsx
│   │   ├── HardwareCard.tsx
│   │   ├── HardwareDetail.tsx
│   │   ├── HardwareFilters.tsx
│   │   ├── HardwareRecommendations.tsx
│   │   ├── ModelCompatibilityTable.tsx
│   │   └── FeaturedHardware.tsx
│   └── Common/
│       ├── Pagination.tsx
│       └── SpecificationTooltip.tsx
├── types/
│   └── hardware.ts
├── hooks/
│   ├── useHardware.ts
│   └── useHardwareFilters.ts
└── utils/
    └── hardwareCalculations.ts
```

### 2.2 Type Definitions

```typescript
// src/types/hardware.ts
export interface Hardware {
  name: string;
  type: 'gpu' | 'cpu' | 'accelerator' | 'asic';
  manufacturer: string | null;
  flops: number;
  memory_size: number;
  memory_bw: number;
  icn: number | null;
  icn_ll: number | null;
  power: number | null;
  real_values: boolean;
  url: string | null;
  description: string | null;
  on_prem_vendors: string[];
  clouds: string[];
  min_on_prem_price: number | null;
  max_on_prem_price: number | null;
  source: string;
}

export interface HardwareDetail extends Hardware {
  vendors: VendorPricing[];
  clouds: CloudPricing[];
}

export interface VendorPricing {
  vendor_name: string;
  price_lower: number | null;
  price_upper: number | null;
}

export interface CloudPricing {
  cloud_name: string;
  instance_name: string;
  price_lower: number | null;
  price_upper: number | null;
  regions: string[];
}

export interface HardwareRecommendation {
  hardware_name: string;
  nodes_required: number;
  memory_per_chip: number;
  manufacturer: string | null;
  type: string;
}

export interface ModelCompatibility {
  model_id: string;
  model_name: string;
  parameters: number;
  compatibility: {
    seq2k: MemoryRequirement;
    seq4k: MemoryRequirement;
    seq8k: MemoryRequirement;
    seq16k: MemoryRequirement;
  };
  overallStatus: 'optimal' | 'partial' | 'incompatible';
}

export interface MemoryRequirement {
  memory_gb: number;
  status: 'green' | 'yellow' | 'red';
}
```

## 3. Implementation Details

### 3.1 Update Navigation Component

```typescript
// In AIMemoryCalculator.tsx - Update Navigation component
const Navigation = () => (
  <nav className="bg-gray-900 text-white p-4">
    <div className="max-w-7xl mx-auto flex items-center justify-between">
      <div className="flex items-center space-x-8">
        <button
          onClick={() => setCurrentScreen('home')}
          className="flex items-center space-x-2 hover:text-purple-400 transition-colors"
        >
          <Home className="w-5 h-5" />
          <span className="font-semibold">BudSimulator</span>
        </button>
        
        <button
          onClick={() => setCurrentScreen('calculator')}
          className={`hover:text-purple-400 transition-colors ${
            currentScreen === 'calculator' ? 'text-purple-400' : ''
          }`}
        >
          Calculator
        </button>
        
        <button
          onClick={() => setCurrentScreen('models')}
          className={`hover:text-purple-400 transition-colors ${
            currentScreen === 'models' ? 'text-purple-400' : ''
          }`}
        >
          Models
        </button>
        
        <button
          onClick={() => setCurrentScreen('hardware')}
          className={`hover:text-purple-400 transition-colors ${
            currentScreen === 'hardware' || currentScreen === 'hardware-detail' ? 'text-purple-400' : ''
          }`}
        >
          Hardware
        </button>
        
        <button
          onClick={() => setCurrentScreen('comparison')}
          className={`hover:text-purple-400 transition-colors ${
            currentScreen === 'comparison' ? 'text-purple-400' : ''
          }`}
        >
          Compare
        </button>
        
        <button
          onClick={() => setCurrentScreen('analysis')}
          className={`hover:text-purple-400 transition-colors ${
            currentScreen === 'analysis' ? 'text-purple-400' : ''
          }`}
        >
          Analysis
        </button>
      </div>
    </div>
  </nav>
);
```

### 3.2 Featured Hardware Component for Home Screen

```typescript
// src/components/Hardware/FeaturedHardware.tsx
import React, { useEffect, useState } from 'react';
import { Cpu, TrendingUp, Zap } from 'lucide-react';
import { Hardware } from '../../types/hardware';

interface FeaturedHardwareProps {
  onHardwareClick: (hardwareName: string) => void;
}

export const FeaturedHardware: React.FC<FeaturedHardwareProps> = ({ onHardwareClick }) => {
  const [hardware, setHardware] = useState<Hardware[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchFeaturedHardware();
  }, []);

  const fetchFeaturedHardware = async () => {
    try {
      const response = await fetch('/api/hardware/filter?type=gpu&type=accelerator&sort_by=flops&sort_order=desc&limit=6');
      const data = await response.json();
      setHardware(data);
    } catch (error) {
      console.error('Error fetching featured hardware:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="animate-pulse">
        <div className="h-8 bg-gray-200 rounded w-48 mb-4"></div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {[...Array(6)].map((_, i) => (
            <div key={i} className="h-32 bg-gray-200 rounded-lg"></div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="mb-12">
      <h2 className="text-2xl font-bold mb-6 flex items-center">
        <Cpu className="w-6 h-6 mr-2 text-purple-600" />
        Popular AI Hardware
      </h2>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {hardware.map((hw) => (
          <button
            key={hw.name}
            onClick={() => onHardwareClick(hw.name)}
            className="bg-white rounded-lg shadow-md p-4 hover:shadow-lg transition-shadow text-left"
          >
            <div className="flex items-start justify-between mb-2">
              <h3 className="font-semibold text-lg">{hw.name}</h3>
              <span className="text-xs bg-purple-100 text-purple-700 px-2 py-1 rounded">
                {hw.type.toUpperCase()}
              </span>
            </div>
            
            <div className="space-y-1 text-sm text-gray-600">
              <div className="flex items-center">
                <Zap className="w-4 h-4 mr-1" />
                {hw.flops.toLocaleString()} TFLOPS
              </div>
              <div className="flex items-center">
                <TrendingUp className="w-4 h-4 mr-1" />
                {hw.memory_size} GB Memory
              </div>
              {hw.manufacturer && (
                <div className="text-xs text-gray-500">{hw.manufacturer}</div>
              )}
            </div>
          </button>
        ))}
      </div>
      
      <div className="text-center mt-6">
        <button
          onClick={() => onHardwareClick('')}
          className="text-purple-600 hover:text-purple-700 font-medium"
        >
          View All Hardware →
        </button>
      </div>
    </div>
  );
};
```

### 3.3 Hardware List Screen

```typescript
// src/components/Hardware/HardwareList.tsx
import React, { useState, useEffect } from 'react';
import { Search, Filter, ChevronDown } from 'lucide-react';
import { Hardware } from '../../types/hardware';
import { HardwareCard } from './HardwareCard';
import { HardwareFilters } from './HardwareFilters';
import { Pagination } from '../Common/Pagination';

interface HardwareListProps {
  onHardwareSelect: (hardwareName: string) => void;
}

export const HardwareList: React.FC<HardwareListProps> = ({ onHardwareSelect }) => {
  const [hardware, setHardware] = useState<Hardware[]>([]);
  const [loading, setLoading] = useState(true);
  const [filters, setFilters] = useState({
    search: '',
    type: null as string | null,
    manufacturer: null as string | null,
    minMemory: null as number | null,
    maxMemory: null as number | null,
    sortBy: 'name',
    sortOrder: 'asc' as 'asc' | 'desc',
  });
  const [pagination, setPagination] = useState({
    page: 1,
    limit: 12,
    total: 0,
  });
  const [showFilters, setShowFilters] = useState(false);

  useEffect(() => {
    fetchHardware();
  }, [filters, pagination.page]);

  const fetchHardware = async () => {
    setLoading(true);
    try {
      const params = new URLSearchParams();
      if (filters.type) params.append('type', filters.type);
      if (filters.manufacturer) params.append('manufacturer', filters.manufacturer);
      if (filters.minMemory) params.append('min_memory', filters.minMemory.toString());
      if (filters.maxMemory) params.append('max_memory', filters.maxMemory.toString());
      params.append('sort_by', filters.sortBy);
      params.append('sort_order', filters.sortOrder);
      params.append('limit', pagination.limit.toString());
      params.append('offset', ((pagination.page - 1) * pagination.limit).toString());

      const response = await fetch(`/api/hardware?${params}`);
      const data = await response.json();
      
      setHardware(data);
      // Note: API should return total count for proper pagination
      setPagination(prev => ({ ...prev, total: data.length }));
    } catch (error) {
      console.error('Error fetching hardware:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold mb-2">AI Hardware Catalog</h1>
        <p className="text-gray-600">Explore GPUs, TPUs, and accelerators for AI workloads</p>
      </div>

      {/* Search and Filter Bar */}
      <div className="bg-white rounded-lg shadow-md p-4 mb-6">
        <div className="flex flex-col md:flex-row gap-4">
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
            <input
              type="text"
              placeholder="Search hardware..."
              className="w-full pl-10 pr-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
              value={filters.search}
              onChange={(e) => setFilters({ ...filters, search: e.target.value })}
            />
          </div>
          
          <button
            onClick={() => setShowFilters(!showFilters)}
            className="flex items-center px-4 py-2 bg-gray-100 rounded-lg hover:bg-gray-200 transition-colors"
          >
            <Filter className="w-5 h-5 mr-2" />
            Filters
            <ChevronDown className={`w-4 h-4 ml-2 transform transition-transform ${showFilters ? 'rotate-180' : ''}`} />
          </button>
        </div>

        {showFilters && (
          <HardwareFilters
            filters={filters}
            onFiltersChange={setFilters}
            className="mt-4 pt-4 border-t"
          />
        )}
      </div>

      {/* Hardware Grid */}
      {loading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {[...Array(6)].map((_, i) => (
            <div key={i} className="animate-pulse">
              <div className="bg-gray-200 rounded-lg h-64"></div>
            </div>
          ))}
        </div>
      ) : (
        <>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {hardware.map((hw) => (
              <HardwareCard
                key={hw.name}
                hardware={hw}
                onSelect={() => onHardwareSelect(hw.name)}
              />
            ))}
          </div>

          {hardware.length === 0 && (
            <div className="text-center py-12">
              <p className="text-gray-500">No hardware found matching your criteria</p>
            </div>
          )}

          {hardware.length > 0 && (
            <Pagination
              currentPage={pagination.page}
              totalPages={Math.ceil(pagination.total / pagination.limit)}
              onPageChange={(page) => setPagination({ ...pagination, page })}
              className="mt-8"
            />
          )}
        </>
      )}
    </div>
  );
};
```

### 3.4 Hardware Detail Screen

```typescript
// src/components/Hardware/HardwareDetail.tsx
import React, { useState, useEffect } from 'react';
import { ArrowLeft, Cpu, Zap, HardDrive, Activity, DollarSign, Cloud, Server, Info } from 'lucide-react';
import { HardwareDetail as HardwareDetailType } from '../../types/hardware';
import { ModelCompatibilityTable } from './ModelCompatibilityTable';
import { SpecificationTooltip } from '../Common/SpecificationTooltip';

interface HardwareDetailProps {
  hardwareName: string;
  onBack: () => void;
}

export const HardwareDetail: React.FC<HardwareDetailProps> = ({ hardwareName, onBack }) => {
  const [hardware, setHardware] = useState<HardwareDetailType | null>(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<'specs' | 'pricing' | 'models'>('specs');

  useEffect(() => {
    fetchHardwareDetail();
  }, [hardwareName]);

  const fetchHardwareDetail = async () => {
    setLoading(true);
    try {
      const response = await fetch(`/api/hardware/${encodeURIComponent(hardwareName)}`);
      const data = await response.json();
      setHardware(data);
    } catch (error) {
      console.error('Error fetching hardware details:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="max-w-7xl mx-auto px-4 py-8">
        <div className="animate-pulse">
          <div className="h-8 bg-gray-200 rounded w-64 mb-4"></div>
          <div className="h-64 bg-gray-200 rounded-lg"></div>
        </div>
      </div>
    );
  }

  if (!hardware) {
    return (
      <div className="max-w-7xl mx-auto px-4 py-8">
        <button onClick={onBack} className="flex items-center text-purple-600 hover:text-purple-700 mb-4">
          <ArrowLeft className="w-5 h-5 mr-2" />
          Back to Hardware
        </button>
        <p className="text-center text-gray-500">Hardware not found</p>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto px-4 py-8">
      {/* Header */}
      <button onClick={onBack} className="flex items-center text-purple-600 hover:text-purple-700 mb-6">
        <ArrowLeft className="w-5 h-5 mr-2" />
        Back to Hardware
      </button>

      <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
        <div className="flex items-start justify-between mb-4">
          <div>
            <h1 className="text-3xl font-bold mb-2">{hardware.name}</h1>
            {hardware.manufacturer && (
              <p className="text-lg text-gray-600">{hardware.manufacturer}</p>
            )}
            {hardware.description && (
              <p className="mt-3 text-gray-700">{hardware.description}</p>
            )}
          </div>
          <span className="bg-purple-100 text-purple-700 px-3 py-1 rounded-full text-sm font-medium">
            {hardware.type.toUpperCase()}
          </span>
        </div>

        {/* Quick Stats */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-6">
          <div className="bg-gray-50 rounded-lg p-4">
            <div className="flex items-center text-gray-600 mb-1">
              <Zap className="w-4 h-4 mr-1" />
              <span className="text-sm">Performance</span>
            </div>
            <p className="text-xl font-semibold">{hardware.flops.toLocaleString()} TFLOPS</p>
          </div>
          
          <div className="bg-gray-50 rounded-lg p-4">
            <div className="flex items-center text-gray-600 mb-1">
              <HardDrive className="w-4 h-4 mr-1" />
              <span className="text-sm">Memory</span>
            </div>
            <p className="text-xl font-semibold">{hardware.memory_size} GB</p>
          </div>
          
          <div className="bg-gray-50 rounded-lg p-4">
            <div className="flex items-center text-gray-600 mb-1">
              <Activity className="w-4 h-4 mr-1" />
              <span className="text-sm">Bandwidth</span>
            </div>
            <p className="text-xl font-semibold">{hardware.memory_bw.toLocaleString()} GB/s</p>
          </div>
          
          {hardware.power && (
            <div className="bg-gray-50 rounded-lg p-4">
              <div className="flex items-center text-gray-600 mb-1">
                <Zap className="w-4 h-4 mr-1" />
                <span className="text-sm">Power</span>
              </div>
              <p className="text-xl font-semibold">{hardware.power} W</p>
            </div>
          )}
        </div>
      </div>

      {/* Tabs */}
      <div className="bg-white rounded-lg shadow-lg">
        <div className="border-b">
          <div className="flex">
            <button
              onClick={() => setActiveTab('specs')}
              className={`px-6 py-3 font-medium transition-colors ${
                activeTab === 'specs'
                  ? 'text-purple-600 border-b-2 border-purple-600'
                  : 'text-gray-600 hover:text-gray-800'
              }`}
            >
              Specifications
            </button>
            <button
              onClick={() => setActiveTab('pricing')}
              className={`px-6 py-3 font-medium transition-colors ${
                activeTab === 'pricing'
                  ? 'text-purple-600 border-b-2 border-purple-600'
                  : 'text-gray-600 hover:text-gray-800'
              }`}
            >
              Pricing
            </button>
            <button
              onClick={() => setActiveTab('models')}
              className={`px-6 py-3 font-medium transition-colors ${
                activeTab === 'models'
                  ? 'text-purple-600 border-b-2 border-purple-600'
                  : 'text-gray-600 hover:text-gray-800'
              }`}
            >
              Compatible Models
            </button>
          </div>
        </div>

        <div className="p-6">
          {activeTab === 'specs' && (
            <div className="space-y-6">
              <div>
                <h3 className="text-lg font-semibold mb-4">Performance Specifications</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <SpecificationRow
                    label="Compute Performance"
                    value={`${hardware.flops.toLocaleString()} TFLOPS`}
                    tooltip="Floating-point operations per second measuring raw compute power"
                  />
                  <SpecificationRow
                    label="Memory Size"
                    value={`${hardware.memory_size} GB`}
                    tooltip="Total available memory for model weights and activations"
                  />
                  <SpecificationRow
                    label="Memory Bandwidth"
                    value={`${hardware.memory_bw.toLocaleString()} GB/s`}
                    tooltip="Data transfer rate between processor and memory"
                  />
                  {hardware.icn && (
                    <SpecificationRow
                      label="Interconnect Bandwidth"
                      value={`${hardware.icn.toLocaleString()} GB/s`}
                      tooltip="Communication speed between multiple devices"
                    />
                  )}
                  {hardware.power && (
                    <SpecificationRow
                      label="Power Consumption"
                      value={`${hardware.power} W`}
                      tooltip="Maximum power draw under full load"
                    />
                  )}
                  {hardware.power && (
                    <SpecificationRow
                      label="Performance per Watt"
                      value={`${(hardware.flops / hardware.power).toFixed(2)} TFLOPS/W`}
                      tooltip="Energy efficiency metric for compute operations"
                    />
                  )}
                </div>
              </div>
            </div>
          )}

          {activeTab === 'pricing' && (
            <div className="space-y-6">
              {/* On-Premise Vendors */}
              <div>
                <h3 className="text-lg font-semibold mb-4 flex items-center">
                  <Server className="w-5 h-5 mr-2" />
                  On-Premise Vendors
                </h3>
                {hardware.vendors.length > 0 ? (
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    {hardware.vendors.map((vendor) => (
                      <div key={vendor.vendor_name} className="border rounded-lg p-4">
                        <h4 className="font-medium mb-2">{vendor.vendor_name}</h4>
                        <p className="text-sm text-gray-600">
                          {vendor.price_lower && vendor.price_upper ? (
                            <>
                              ${vendor.price_lower.toLocaleString()} - ${vendor.price_upper.toLocaleString()}
                            </>
                          ) : (
                            'Contact for pricing'
                          )}
                        </p>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-gray-500">No vendor pricing information available</p>
                )}
              </div>

              {/* Cloud Providers */}
              <div>
                <h3 className="text-lg font-semibold mb-4 flex items-center">
                  <Cloud className="w-5 h-5 mr-2" />
                  Cloud Providers
                </h3>
                {hardware.clouds.length > 0 ? (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    {hardware.clouds.map((cloud) => (
                      <div key={cloud.cloud_name} className="border rounded-lg p-4">
                        <h4 className="font-medium mb-1">{cloud.cloud_name}</h4>
                        <p className="text-sm text-gray-600 mb-2">Instance: {cloud.instance_name}</p>
                        <p className="text-sm font-medium">
                          ${cloud.price_lower} - ${cloud.price_upper}/hour
                        </p>
                        {cloud.regions.length > 0 && (
                          <p className="text-xs text-gray-500 mt-2">
                            Regions: {cloud.regions.join(', ')}
                          </p>
                        )}
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-gray-500">Not available on cloud platforms</p>
                )}
              </div>
            </div>
          )}

          {activeTab === 'models' && (
            <ModelCompatibilityTable hardware={hardware} />
          )}
        </div>
      </div>
    </div>
  );
};

const SpecificationRow: React.FC<{
  label: string;
  value: string;
  tooltip: string;
}> = ({ label, value, tooltip }) => (
  <div className="flex items-center justify-between py-2 border-b">
    <div className="flex items-center">
      <span className="text-gray-600">{label}</span>
      <SpecificationTooltip content={tooltip} />
    </div>
    <span className="font-medium">{value}</span>
  </div>
);
```

### 3.5 Hardware Recommendations Integration

```typescript
// Update ResultsScreen in AIMemoryCalculator.tsx
const ResultsScreen = () => {
  const [recommendations, setRecommendations] = useState<HardwareRecommendation[]>([]);
  const [loadingRecs, setLoadingRecs] = useState(false);

  useEffect(() => {
    if (results) {
      fetchRecommendations();
    }
  }, [results]);

  const fetchRecommendations = async () => {
    if (!results) return;
    
    setLoadingRecs(true);
    try {
      const response = await fetch('/api/hardware/recommend', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          total_memory_gb: results.total_memory_gb,
          model_params_b: results.parameter_count / 1e9
        })
      });
      
      const data = await response.json();
      setRecommendations(data);
    } catch (error) {
      console.error('Error fetching recommendations:', error);
    } finally {
      setLoadingRecs(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto">
      {/* Existing results display */}
      
      {/* Hardware Recommendations Section */}
      <div className="mt-8 bg-white rounded-lg shadow-md p-6">
        <h3 className="text-xl font-semibold mb-4 flex items-center">
          <Cpu className="w-5 h-5 mr-2 text-purple-600" />
          Recommended Hardware
        </h3>
        
        {loadingRecs ? (
          <div className="animate-pulse">
            <div className="h-24 bg-gray-200 rounded mb-3"></div>
            <div className="h-24 bg-gray-200 rounded"></div>
          </div>
        ) : recommendations.length > 0 ? (
          <div className="space-y-3">
            {recommendations.map((rec) => (
              <div
                key={rec.hardware_name}
                className="border rounded-lg p-4 hover:border-purple-300 transition-colors cursor-pointer"
                onClick={() => {
                  setSelectedHardwareName(rec.hardware_name);
                  setCurrentScreen('hardware-detail');
                }}
              >
                <div className="flex justify-between items-start">
                  <div>
                    <h4 className="font-medium">{rec.hardware_name}</h4>
                    <p className="text-sm text-gray-600 mt-1">
                      {rec.nodes_required} node{rec.nodes_required > 1 ? 's' : ''} required
                    </p>
                    <p className="text-sm text-gray-600">
                      {rec.memory_per_chip} GB per chip • {rec.type}
                    </p>
                  </div>
                  <div className="text-right">
                    <div className="text-sm font-medium text-purple-600">
                      {((results.total_memory_gb / (rec.memory_per_chip * rec.nodes_required)) * 100).toFixed(0)}%
                    </div>
                    <div className="text-xs text-gray-500">utilization</div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-gray-500">No recommendations available</p>
        )}
      </div>
    </div>
  );
};
```

## 4. API Integration Summary

### 4.1 Hardware Endpoints

```typescript
// API service functions
const API_BASE = '/api';

export const hardwareAPI = {
  // List hardware with filters
  list: async (params: URLSearchParams) => {
    const response = await fetch(`${API_BASE}/hardware?${params}`);
    return response.json();
  },

  // Get hardware details
  getDetails: async (hardwareName: string) => {
    const response = await fetch(`${API_BASE}/hardware/${encodeURIComponent(hardwareName)}`);
    return response.json();
  },

  // Get hardware recommendations
  recommend: async (totalMemoryGB: number, modelParamsB: number) => {
    const response = await fetch(`${API_BASE}/hardware/recommend`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        total_memory_gb: totalMemoryGB,
        model_params_b: modelParamsB
      })
    });
    return response.json();
  },

  // Advanced filtering
  filter: async (params: URLSearchParams) => {
    const response = await fetch(`${API_BASE}/hardware/filter?${params}`);
    return response.json();
  }
};
```

### 4.2 Integration Points

1. **Home Screen**: Featured hardware section calls `/api/hardware/filter` with top GPUs/accelerators
2. **Hardware List**: Calls `/api/hardware` with pagination and filters
3. **Hardware Detail**: Calls `/api/hardware/{name}` for full details including pricing
4. **Model Results**: Calls `/api/hardware/recommend` to suggest suitable hardware
5. **Model Compatibility**: Client-side calculation based on hardware specs and model requirements

## 5. State Management

```typescript
// Add to AIMemoryCalculator component state
const [selectedHardwareName, setSelectedHardwareName] = useState<string | null>(null);
const [hardwareList, setHardwareList] = useState<Hardware[]>([]);
const [hardwareDetail, setHardwareDetail] = useState<HardwareDetail | null>(null);

// Navigation handler
const handleHardwareNavigation = (hardwareName: string) => {
  if (hardwareName) {
    setSelectedHardwareName(hardwareName);
    setCurrentScreen('hardware-detail');
  } else {
    setCurrentScreen('hardware');
  }
};
```

## 6. Styling Approach

- Use existing Tailwind CSS classes for consistency
- Maintain the purple accent color scheme
- Use shadow-md for cards, shadow-lg for modals
- Consistent spacing: p-4 for cards, p-6 for sections
- Responsive grid layouts with proper breakpoints

## 7. Performance Optimizations

1. **Lazy Loading**: Load hardware images only when visible
2. **Pagination**: Limit results to 12 items per page
3. **Debounced Search**: Add 300ms debounce on search input
4. **Caching**: Cache hardware details for 5 minutes
5. **Skeleton Loaders**: Show loading states for better UX

## 8. Error Handling

```typescript
// Generic error component
const ErrorState: React.FC<{ message: string; onRetry?: () => void }> = ({ message, onRetry }) => (
  <div className="text-center py-8">
    <AlertCircle className="w-12 h-12 text-red-500 mx-auto mb-4" />
    <p className="text-gray-600 mb-4">{message}</p>
    {onRetry && (
      <button
        onClick={onRetry}
        className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700"
      >
        Try Again
      </button>
    )}
  </div>
);
```

## 9. Testing Strategy

1. **Unit Tests**: Test calculation utilities and API functions
2. **Component Tests**: Test individual components with React Testing Library
3. **Integration Tests**: Test full user flows
4. **API Mocking**: Use MSW for consistent test data

## 10. Implementation Timeline

### Week 1
- Create type definitions and API service
- Implement HardwareList and HardwareCard components
- Add hardware navigation to main app

### Week 2
- Implement HardwareDetail component with tabs
- Add ModelCompatibilityTable
- Integrate hardware recommendations in results

### Week 3
- Add FeaturedHardware to home screen
- Implement filters and search
- Add pagination

### Week 4
- Polish UI and animations
- Add error handling and loading states
- Complete testing and documentation

## 11. Future Enhancements

1. **Comparison Tool**: Compare multiple hardware side-by-side
2. **Cost Calculator**: Calculate deployment costs for different configurations
3. **Performance Charts**: Visualize hardware performance metrics
4. **Bookmarks**: Save favorite hardware for quick access
5. **Export**: Download hardware specifications as PDF/CSV 