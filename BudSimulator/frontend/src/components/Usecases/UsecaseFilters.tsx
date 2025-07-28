import React, { useState, useEffect } from 'react';
import { Search, Filter, X, ChevronDown, ChevronUp } from 'lucide-react';
import { UsecaseFilters as UsecaseFiltersType } from '../../types/usecase';

interface UsecaseFiltersProps {
  filters: UsecaseFiltersType;
  onFiltersChange: (filters: UsecaseFiltersType) => void;
  availableIndustries: string[];
  availableTags: string[];
}

export const UsecaseFilters: React.FC<UsecaseFiltersProps> = ({
  filters,
  onFiltersChange,
  availableIndustries,
  availableTags
}) => {
  const [searchValue, setSearchValue] = useState(filters.search || '');
  const [showIndustryDropdown, setShowIndustryDropdown] = useState(false);
  const [showTagsDropdown, setShowTagsDropdown] = useState(false);
  const [showFilters, setShowFilters] = useState(false);

  // Debounce search input
  useEffect(() => {
    const timer = setTimeout(() => {
      if (searchValue !== filters.search) {
        onFiltersChange({ ...filters, search: searchValue });
      }
    }, 300);

    return () => clearTimeout(timer);
  }, [searchValue]);

  const handleIndustryToggle = (industry: string) => {
    const currentIndustries = filters.industries || [];
    const newIndustries = currentIndustries.includes(industry)
      ? currentIndustries.filter(i => i !== industry)
      : [...currentIndustries, industry];
    
    onFiltersChange({ ...filters, industries: newIndustries });
  };

  const handleTagToggle = (tag: string) => {
    const currentTags = filters.tags || [];
    const newTags = currentTags.includes(tag)
      ? currentTags.filter(t => t !== tag)
      : [...currentTags, tag];
    
    onFiltersChange({ ...filters, tags: newTags });
  };

  const handleLatencyProfileChange = (profile: string) => {
    onFiltersChange({ 
      ...filters, 
      latency_profile: filters.latency_profile === profile ? undefined : profile 
    });
  };

  const clearAllFilters = () => {
    setSearchValue('');
    onFiltersChange({
      search: '',
      industries: [],
      tags: [],
      latency_profile: undefined,
      is_active: undefined
    });
  };

  const activeFilterCount = 
    (filters.industries?.length || 0) + 
    (filters.tags?.length || 0) + 
    (filters.latency_profile ? 1 : 0);

  return (
    <div className="space-y-4">
      {/* Search Bar */}
      <div className="relative">
        <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-500 w-5 h-5" />
        <input
          type="text"
          value={searchValue}
          onChange={(e) => setSearchValue(e.target.value)}
          placeholder="Search usecases by name or description..."
          className="w-full pl-10 pr-4 py-3 bg-gray-800 border border-gray-700 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500 focus:border-transparent text-white placeholder-gray-500"
        />
      </div>

      {/* Filter Toggle Button */}
      <div className="flex items-center justify-between">
        <button
          onClick={() => setShowFilters(!showFilters)}
          className="flex items-center gap-2 px-4 py-2 text-gray-300 hover:bg-gray-800 rounded-lg transition-colors"
        >
          <Filter className="w-4 h-4" />
          <span>Filters</span>
          {activeFilterCount > 0 && (
            <span className="ml-1 px-2 py-0.5 bg-purple-600 text-white rounded-full text-xs font-medium">
              {activeFilterCount}
            </span>
          )}
          {showFilters ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
        </button>

        {activeFilterCount > 0 && (
          <button
            onClick={clearAllFilters}
            className="text-sm text-gray-500 hover:text-gray-300 flex items-center gap-1"
          >
            <X className="w-4 h-4" />
            Clear all filters
          </button>
        )}
      </div>

      {/* Expandable Filters Section */}
      {showFilters && (
        <div className="border border-gray-700 rounded-lg p-4 space-y-4 bg-gray-900/30">
          {/* Industry Filter */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="text-sm font-medium text-gray-300">Industries</label>
              <button
                onClick={() => setShowIndustryDropdown(!showIndustryDropdown)}
                className="text-sm text-purple-400 hover:text-purple-300"
              >
                {showIndustryDropdown ? 'Hide' : 'Show'} all
              </button>
            </div>
            <div className="flex flex-wrap gap-2">
              {(availableIndustries || []).slice(0, showIndustryDropdown ? undefined : 5).map((industry) => (
                <button
                  key={industry}
                  onClick={() => handleIndustryToggle(industry)}
                  className={`px-3 py-1.5 rounded-full text-sm font-medium transition-colors ${
                    filters.industries?.includes(industry)
                      ? 'bg-purple-600 text-white'
                      : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
                  }`}
                >
                  {industry}
                </button>
              ))}
              {!showIndustryDropdown && (availableIndustries || []).length > 5 && (
                <span className="px-3 py-1.5 text-sm text-gray-500">
                  +{(availableIndustries || []).length - 5} more
                </span>
              )}
            </div>
          </div>

          {/* Latency Profile Filter */}
          <div>
            <label className="text-sm font-medium text-gray-300 block mb-2">Latency Profile</label>
            <div className="flex flex-wrap gap-2">
              {['real-time', 'interactive', 'responsive', 'batch'].map((profile) => (
                <button
                  key={profile}
                  onClick={() => handleLatencyProfileChange(profile)}
                  className={`px-3 py-1.5 rounded-full text-sm font-medium transition-colors capitalize ${
                    filters.latency_profile === profile
                      ? 'bg-purple-600 text-white'
                      : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
                  }`}
                >
                  {profile}
                </button>
              ))}
            </div>
          </div>

          {/* Tags Filter */}
          <div>
            <div className="flex items-center justify-between mb-2">
              <label className="text-sm font-medium text-gray-300">Tags</label>
              <button
                onClick={() => setShowTagsDropdown(!showTagsDropdown)}
                className="text-sm text-purple-400 hover:text-purple-300"
              >
                {showTagsDropdown ? 'Hide' : 'Show'} all
              </button>
            </div>
            <div className="flex flex-wrap gap-2">
              {(availableTags || []).slice(0, showTagsDropdown ? undefined : 8).map((tag) => (
                <button
                  key={tag}
                  onClick={() => handleTagToggle(tag)}
                  className={`px-3 py-1.5 rounded-full text-sm transition-colors ${
                    filters.tags?.includes(tag)
                      ? 'bg-purple-600 text-white'
                      : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
                  }`}
                >
                  {tag}
                </button>
              ))}
              {!showTagsDropdown && (availableTags || []).length > 8 && (
                <span className="px-3 py-1.5 text-sm text-gray-500">
                  +{(availableTags || []).length - 8} more
                </span>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}; 