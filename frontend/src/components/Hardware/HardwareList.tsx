import React, { useState, useEffect, useCallback } from 'react';
import { Search, Filter, ChevronDown, AlertCircle } from 'lucide-react';
import { Hardware, HardwareFilters as HardwareFiltersType, PaginationState } from '../../types/hardware';
import { HardwareCard } from './HardwareCard';
import { HardwareFilters } from './HardwareFilters';
import { Pagination } from '../Common/Pagination';
import { PriceDisclaimer } from '../Common/PriceDisclaimer';
import { hardwareAPI, buildHardwareParams } from '../../services/hardwareAPI';

interface HardwareListProps {
  onHardwareSelect: (hardwareName: string) => void;
}

export const HardwareList: React.FC<HardwareListProps> = ({ onHardwareSelect }) => {
  const [hardware, setHardware] = useState<Hardware[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [filters, setFilters] = useState<HardwareFiltersType>({
    search: '',
    type: null,
    manufacturer: null,
    minMemory: null,
    maxMemory: null,
    minFlops: null,
    maxFlops: null,
    sortBy: 'name',
    sortOrder: 'asc',
  });
  const [pagination, setPagination] = useState<PaginationState>({
    page: 1,
    limit: 12,
    total: 0,
  });
  const [showFilters, setShowFilters] = useState(false);

  // Debounced search
  const [searchTerm, setSearchTerm] = useState('');
  useEffect(() => {
    const timer = setTimeout(() => {
      setFilters(prev => ({ ...prev, search: searchTerm }));
    }, 300);
    return () => clearTimeout(timer);
  }, [searchTerm]);

  const fetchHardware = useCallback(async () => {
    setLoading(true);
    setError(null);
    
    try {
      const params = buildHardwareParams({
        ...filters,
        limit: pagination.limit,
        offset: (pagination.page - 1) * pagination.limit
      });

      const data = await hardwareAPI.list(params);
      setHardware(data);
      
      // Note: In a real implementation, the API should return total count
      // For now, we'll estimate based on the returned data
      setPagination(prev => ({ 
        ...prev, 
        total: data.length < pagination.limit ? 
          (pagination.page - 1) * pagination.limit + data.length :
          pagination.page * pagination.limit + 1 // Estimate there's at least one more page
      }));
    } catch (error) {
      console.error('Error fetching hardware:', error);
      setError('Failed to load hardware. Please try again.');
    } finally {
      setLoading(false);
    }
  }, [filters, pagination.page, pagination.limit]);

  useEffect(() => {
    fetchHardware();
  }, [fetchHardware]);

  // Reset to first page when filters change
  useEffect(() => {
    if (pagination.page !== 1) {
      setPagination(prev => ({ ...prev, page: 1 }));
    }
  }, [filters]); // eslint-disable-line react-hooks/exhaustive-deps

  const handlePageChange = (page: number) => {
    setPagination(prev => ({ ...prev, page }));
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  const handleFiltersChange = (newFilters: HardwareFiltersType) => {
    setFilters(newFilters);
  };

  if (error) {
    return (
      <div className="min-h-screen bg-black text-white pt-16">
        <div className="max-w-7xl mx-auto px-4 py-8">
          <div className="text-center py-12">
            <AlertCircle className="w-12 h-12 text-red-500 mx-auto mb-4" />
            <p className="text-gray-400 mb-4">{error}</p>
            <button
              onClick={fetchHardware}
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
        <div className="mb-8">
          <h1 className="text-3xl font-bold mb-2 text-white">AI Hardware Catalog</h1>
          <p className="text-gray-400">Explore GPUs, TPUs, and accelerators for AI workloads</p>
        </div>

        {/* Search and Filter Bar */}
        <div className="bg-gray-900/50 rounded-lg shadow-md p-4 mb-6 border border-gray-800">
          <div className="flex flex-col md:flex-row gap-4">
            <div className="flex-1 relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
              <input
                type="text"
                placeholder="Search hardware..."
                className="w-full pl-10 pr-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
              />
            </div>
            
            <button
              onClick={() => setShowFilters(!showFilters)}
              className="flex items-center px-4 py-2 bg-gray-800 rounded-lg hover:bg-gray-700 transition-colors text-white border border-gray-700"
            >
              <Filter className="w-5 h-5 mr-2" />
              Filters
              <ChevronDown className={`w-4 h-4 ml-2 transform transition-transform ${showFilters ? 'rotate-180' : ''}`} />
            </button>
          </div>

          {showFilters && (
            <HardwareFilters
              filters={filters}
              onFiltersChange={handleFiltersChange}
              className="mt-4 pt-4 border-t border-gray-700"
            />
          )}
        </div>

        {/* Results Summary */}
        <div className="mb-4 flex items-center justify-between">
          <p className="text-sm text-gray-400">
            {loading ? 'Loading...' : `Showing ${hardware.length} hardware items`}
          </p>
          
          {/* Quick filter chips */}
          <div className="flex items-center space-x-2">
            {filters.type && (
              <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-purple-900/50 text-purple-300 border border-purple-500/30">
                {filters.type.toUpperCase()}
                <button
                  onClick={() => setFilters({ ...filters, type: null })}
                  className="ml-1 text-purple-400 hover:text-purple-200"
                >
                  ×
                </button>
              </span>
            )}
            {filters.manufacturer && (
              <span className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-blue-900/50 text-blue-300 border border-blue-500/30">
                {filters.manufacturer}
                <button
                  onClick={() => setFilters({ ...filters, manufacturer: null })}
                  className="ml-1 text-blue-400 hover:text-blue-200"
                >
                  ×
                </button>
              </span>
            )}
          </div>
        </div>

        {/* Hardware Grid */}
        {loading ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {[...Array(6)].map((_, i) => (
              <div key={i} className="animate-pulse">
                <div className="bg-gray-800 rounded-lg h-80"></div>
              </div>
            ))}
          </div>
        ) : (
          <>
            {hardware.length > 0 ? (
              <>
                {/* Show price disclaimer if any hardware has pricing */}
                {hardware.some(hw => hw.price_approx !== undefined && hw.price_approx !== null) && (
                  <PriceDisclaimer />
                )}
                
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                  {hardware.map((hw) => (
                    <HardwareCard
                      key={hw.name}
                      hardware={hw}
                      onSelect={() => onHardwareSelect(hw.name)}
                    />
                  ))}
                </div>

                {/* Pagination */}
                <Pagination
                  currentPage={pagination.page}
                  totalPages={Math.ceil(pagination.total / pagination.limit)}
                  onPageChange={handlePageChange}
                  className="mt-8"
                />
              </>
            ) : (
              <div className="text-center py-12">
                <div className="text-gray-400 mb-4">
                  <Search className="w-16 h-16 mx-auto" />
                </div>
                <h3 className="text-lg font-medium text-white mb-2">No hardware found</h3>
                <p className="text-gray-400 mb-4">
                  Try adjusting your search criteria or filters
                </p>
                <button
                  onClick={() => {
                    setFilters({
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
                    setSearchTerm('');
                  }}
                  className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700"
                >
                  Clear All Filters
                </button>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}; 