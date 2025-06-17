import React from 'react';
import { HardwareFilters as HardwareFiltersType } from '../../types/hardware';

interface HardwareFiltersProps {
  filters: HardwareFiltersType;
  onFiltersChange: (filters: HardwareFiltersType) => void;
  className?: string;
}

export const HardwareFilters: React.FC<HardwareFiltersProps> = ({
  filters,
  onFiltersChange,
  className = ''
}) => {
  const updateFilter = (key: keyof HardwareFiltersType, value: any) => {
    onFiltersChange({
      ...filters,
      [key]: value
    });
  };

  const resetFilters = () => {
    onFiltersChange({
      search: '',
      type: null,
      manufacturer: null,
      minMemory: null,
      maxMemory: null,
      minFlops: null,
      maxFlops: null,
      sortBy: 'name',
      sortOrder: 'asc'
    });
  };

  return (
    <div className={`space-y-4 ${className}`}>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Type Filter */}
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-1">
            Type
          </label>
          <select
            value={filters.type || ''}
            onChange={(e) => updateFilter('type', e.target.value || null)}
            className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent"
          >
            <option value="">All Types</option>
            <option value="gpu">GPU</option>
            <option value="cpu">CPU</option>
            <option value="accelerator">Accelerator</option>
            <option value="asic">ASIC</option>
          </select>
        </div>

        {/* Manufacturer Filter */}
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-1">
            Manufacturer
          </label>
          <select
            value={filters.manufacturer || ''}
            onChange={(e) => updateFilter('manufacturer', e.target.value || null)}
            className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent"
          >
            <option value="">All Manufacturers</option>
            <option value="NVIDIA">NVIDIA</option>
            <option value="AMD">AMD</option>
            <option value="Intel">Intel</option>
            <option value="Google">Google</option>
            <option value="AWS">AWS</option>
            <option value="Cerebras">Cerebras</option>
            <option value="Graphcore">Graphcore</option>
            <option value="Groq">Groq</option>
          </select>
        </div>

        {/* Sort By */}
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-1">
            Sort By
          </label>
          <select
            value={filters.sortBy}
            onChange={(e) => updateFilter('sortBy', e.target.value)}
            className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent"
          >
            <option value="name">Name</option>
            <option value="flops">Performance</option>
            <option value="memory_size">Memory Size</option>
            <option value="memory_bw">Memory Bandwidth</option>
            <option value="power">Power</option>
          </select>
        </div>

        {/* Sort Order */}
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-1">
            Order
          </label>
          <select
            value={filters.sortOrder}
            onChange={(e) => updateFilter('sortOrder', e.target.value as 'asc' | 'desc')}
            className="w-full px-3 py-2 bg-gray-800 border border-gray-700 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent"
          >
            <option value="asc">Ascending</option>
            <option value="desc">Descending</option>
          </select>
        </div>
      </div>

      {/* Range Filters */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {/* Memory Range */}
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">
            Memory Size (GB)
          </label>
          <div className="flex space-x-2">
            <input
              type="number"
              placeholder="Min"
              value={filters.minMemory || ''}
              onChange={(e) => updateFilter('minMemory', e.target.value ? Number(e.target.value) : null)}
              className="flex-1 px-3 py-2 bg-gray-800 border border-gray-700 rounded-md text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent"
            />
            <span className="flex items-center text-gray-400">to</span>
            <input
              type="number"
              placeholder="Max"
              value={filters.maxMemory || ''}
              onChange={(e) => updateFilter('maxMemory', e.target.value ? Number(e.target.value) : null)}
              className="flex-1 px-3 py-2 bg-gray-800 border border-gray-700 rounded-md text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent"
            />
          </div>
        </div>

        {/* Performance Range */}
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">
            Performance (TFLOPS)
          </label>
          <div className="flex space-x-2">
            <input
              type="number"
              placeholder="Min"
              value={filters.minFlops || ''}
              onChange={(e) => updateFilter('minFlops', e.target.value ? Number(e.target.value) : null)}
              className="flex-1 px-3 py-2 bg-gray-800 border border-gray-700 rounded-md text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent"
            />
            <span className="flex items-center text-gray-400">to</span>
            <input
              type="number"
              placeholder="Max"
              value={filters.maxFlops || ''}
              onChange={(e) => updateFilter('maxFlops', e.target.value ? Number(e.target.value) : null)}
              className="flex-1 px-3 py-2 bg-gray-800 border border-gray-700 rounded-md text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent"
            />
          </div>
        </div>
      </div>

      {/* Reset Button */}
      <div className="flex justify-end">
        <button
          onClick={resetFilters}
          className="px-4 py-2 text-sm text-gray-400 hover:text-white bg-gray-800 border border-gray-700 rounded-md hover:bg-gray-700 transition-colors"
        >
          Reset Filters
        </button>
      </div>
    </div>
  );
}; 