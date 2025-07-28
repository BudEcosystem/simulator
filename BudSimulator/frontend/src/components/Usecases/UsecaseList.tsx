import React, { useState, useEffect } from 'react';
import { Briefcase, Loader2, AlertCircle, ChevronLeft, ChevronRight } from 'lucide-react';
import { Usecase, UsecaseFilters as UsecaseFiltersType, UsecaseListResponse } from '../../types/usecase';
import { usecaseAPI } from '../../services/usecaseAPI';
import { UsecaseCard } from './UsecaseCard';
import { UsecaseFilters } from './UsecaseFilters';

interface UsecaseListProps {
  onUsecaseSelect: (usecaseId: string) => void;
}

export const UsecaseList: React.FC<UsecaseListProps> = ({ onUsecaseSelect }) => {
  const [usecases, setUsecases] = useState<Usecase[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [totalPages, setTotalPages] = useState(1);
  const [currentPage, setCurrentPage] = useState(1);
  const [totalUsecases, setTotalUsecases] = useState(0);
  
  const [filters, setFilters] = useState<UsecaseFiltersType>({
    search: '',
    industries: [],
    tags: [],
    latency_profile: undefined,
    is_active: true,
    page: 1,
    page_size: 12
  });

  const [availableIndustries, setAvailableIndustries] = useState<string[]>([]);
  const [availableTags, setAvailableTags] = useState<string[]>([]);

  // Fetch available filters
  useEffect(() => {
    const fetchFilterOptions = async () => {
      try {
        const [industries, tags] = await Promise.all([
          usecaseAPI.getIndustries(),
          usecaseAPI.getTags()
        ]);
        setAvailableIndustries(industries || []);
        setAvailableTags(tags || []);
      } catch (err) {
        console.error('Failed to fetch filter options:', err);
        // Set empty arrays as fallback
        setAvailableIndustries([]);
        setAvailableTags([]);
      }
    };
    fetchFilterOptions();
  }, []);

  // Fetch usecases
  useEffect(() => {
    const fetchUsecases = async () => {
      setLoading(true);
      setError(null);
      
      try {
        const response: UsecaseListResponse = await usecaseAPI.getUsecases(filters);
        setUsecases(response.usecases || []);
        setTotalPages(response.total_pages || 1);
        setCurrentPage(response.page || 1);
        setTotalUsecases(response.total || 0);
      } catch (err) {
        setError('Failed to load usecases. Please try again later.');
        console.error('Error fetching usecases:', err);
        // Set empty arrays as fallback
        setUsecases([]);
        setTotalPages(1);
        setCurrentPage(1);
        setTotalUsecases(0);
      } finally {
        setLoading(false);
      }
    };

    fetchUsecases();
  }, [filters]);

  const handleFiltersChange = (newFilters: UsecaseFiltersType) => {
    setFilters({ ...newFilters, page: 1 }); // Reset to first page on filter change
  };

  const handlePageChange = (page: number) => {
    setFilters({ ...filters, page });
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  if (loading && usecases.length === 0) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <Loader2 className="w-8 h-8 animate-spin text-purple-600 mx-auto mb-4" />
          <p className="text-gray-600">Loading usecases...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-black text-white pt-16">
      <div className="max-w-7xl mx-auto px-4 py-8">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center gap-3 mb-4">
            <Briefcase className="w-8 h-8 text-purple-500" />
            <h1 className="text-3xl font-bold text-white">AI Usecases</h1>
          </div>
          <p className="text-gray-400">
            Explore various AI implementation scenarios across different industries and performance requirements.
          </p>
        </div>

        {/* Filters */}
        <div className="mb-8">
          <UsecaseFilters
            filters={filters}
            onFiltersChange={handleFiltersChange}
            availableIndustries={availableIndustries}
            availableTags={availableTags}
          />
        </div>

        {/* Results Count */}
        {!loading && (
          <div className="mb-4 text-sm text-gray-400">
            Showing {usecases?.length || 0} of {totalUsecases} usecases
            {filters.search && ` matching "${filters.search}"`}
          </div>
        )}

        {/* Error State */}
        {error && (
          <div className="bg-red-900/20 border border-red-500/20 rounded-lg p-4 mb-6 flex items-center gap-3">
            <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0" />
            <p className="text-red-300">{error}</p>
          </div>
        )}

        {/* Usecases Grid */}
        {!loading && (!usecases || usecases.length === 0) ? (
          <div className="text-center py-12">
            <Briefcase className="w-12 h-12 text-gray-600 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-white mb-2">No usecases found</h3>
            <p className="text-gray-400">Try adjusting your filters or search criteria.</p>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
            {(usecases || []).map((usecase) => (
              <UsecaseCard
                key={usecase.id}
                usecase={usecase}
                                      onClick={() => onUsecaseSelect(usecase.unique_id)}
              />
            ))}
          </div>
        )}

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="flex items-center justify-center gap-2">
            <button
              onClick={() => handlePageChange(currentPage - 1)}
              disabled={currentPage === 1}
              className="p-2 rounded-lg border border-gray-700 hover:bg-gray-800 disabled:opacity-50 disabled:cursor-not-allowed text-white"
            >
              <ChevronLeft className="w-5 h-5" />
            </button>
            
            <div className="flex items-center gap-1">
              {Array.from({ length: totalPages }, (_, i) => i + 1)
                .filter(page => {
                  // Show first page, last page, current page, and pages around current
                  return page === 1 || 
                         page === totalPages || 
                         Math.abs(page - currentPage) <= 1;
                })
                .map((page, index, array) => (
                  <React.Fragment key={page}>
                    {index > 0 && array[index - 1] !== page - 1 && (
                      <span className="px-2 text-gray-500">...</span>
                    )}
                    <button
                      onClick={() => handlePageChange(page)}
                      className={`px-3 py-1 rounded-lg text-sm font-medium transition-colors ${
                        page === currentPage
                          ? 'bg-purple-600 text-white'
                          : 'text-gray-300 hover:bg-gray-800'
                      }`}
                    >
                      {page}
                    </button>
                  </React.Fragment>
                ))}
            </div>
            
            <button
              onClick={() => handlePageChange(currentPage + 1)}
              disabled={currentPage === totalPages}
              className="p-2 rounded-lg border border-gray-700 hover:bg-gray-800 disabled:opacity-50 disabled:cursor-not-allowed text-white"
            >
              <ChevronRight className="w-5 h-5" />
            </button>
          </div>
        )}
      </div>
    </div>
  );
}; 