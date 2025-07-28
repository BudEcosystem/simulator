import { Usecase, UsecaseListResponse, UsecaseFilters, UsecaseStats } from '../types/usecase';
import { API_BASE } from '../config/api';

export const usecaseAPI = {
  // Get list of usecases with filters
  async getUsecases(filters: UsecaseFilters = {}): Promise<UsecaseListResponse> {
    try {
      const response = await fetch(`${API_BASE}/usecases`, {
        signal: AbortSignal.timeout(5000) // 5 second timeout
      });
      
      if (!response.ok) {
        throw new Error('Failed to fetch usecases');
      }
      
      const data = await response.json();
      
      // The backend returns an array directly, not an object with usecases property
      if (Array.isArray(data)) {
        const page = filters.page || 1;
        const pageSize = filters.page_size || 12;
        const startIndex = (page - 1) * pageSize;
        const endIndex = startIndex + pageSize;
        
        // Apply client-side filtering
        let filteredData = [...data];
        
        if (filters.search) {
          const search = filters.search.toLowerCase();
          filteredData = filteredData.filter(usecase => {
            const name = (usecase.name || '').toLowerCase();
            const desc = (usecase.description || '').toLowerCase();
            const industry = (usecase.industry || '').toLowerCase();
            return name.includes(search) || desc.includes(search) || industry.includes(search);
          });
        }
        
        if (filters.industries && filters.industries.length > 0) {
          filteredData = filteredData.filter(usecase => 
            filters.industries!.includes(usecase.industry)
          );
        }
        
        if (filters.tags && filters.tags.length > 0) {
          filteredData = filteredData.filter(usecase => {
            const tagsArr: string[] = (usecase.tags || []) as string[];
            return filters.tags!.some(tag => tagsArr.includes(tag));
          });
        }
        
        if (filters.latency_profile) {
          filteredData = filteredData.filter(usecase => 
            usecase.latency_profile === filters.latency_profile
          );
        }
        
        if (filters.is_active !== undefined) {
          filteredData = filteredData.filter(usecase => {
            const activeFlag = Boolean(usecase.is_active);
            return activeFlag === filters.is_active;
          });
        }
        
        const paginatedResults = filteredData.slice(startIndex, endIndex);
        
        return {
          usecases: paginatedResults,
          total: filteredData.length,
          page: page,
          page_size: pageSize,
          total_pages: Math.ceil(filteredData.length / pageSize)
        };
      }
      
      return data;
    } catch (err) {
      console.error('Failed to fetch usecases:', err);
      // Return empty response structure on error
      return {
        usecases: [],
        total: 0,
        page: 1,
        page_size: 12,
        total_pages: 0
      };
    }
  },

  // Get single usecase by ID
  async getUsecase(id: string): Promise<Usecase | null> {
    try {
      const response = await fetch(`${API_BASE}/usecases/${id}`, {
        signal: AbortSignal.timeout(5000)
      });
      
      if (!response.ok) {
        throw new Error('Failed to fetch usecase');
      }
      return response.json();
    } catch (err) {
      console.error('Failed to fetch usecase:', err);
      return null;
    }
  },

  // Get usecase statistics
  async getUsecaseStats(): Promise<UsecaseStats> {
    try {
      const response = await fetch(`${API_BASE}/usecases/stats`, {
        signal: AbortSignal.timeout(5000)
      });
      
      if (!response.ok) {
        throw new Error('Failed to fetch usecase statistics');
      }
      return response.json();
    } catch (err) {
      console.error('Failed to fetch usecase statistics:', err);
      return {
        total_usecases: 0,
        active_usecases: 0,
        by_industry: {},
        by_latency_profile: {},
        most_common_tags: []
      };
    }
  },

  // Get list of all industries
  async getIndustries(): Promise<string[]> {
    try {
      // Extract from all usecases since dedicated endpoint doesn't exist
      const usecasesResponse = await fetch(`${API_BASE}/usecases`, {
        signal: AbortSignal.timeout(5000)
      });
      
      if (usecasesResponse.ok) {
        const usecases = await usecasesResponse.json();
        if (Array.isArray(usecases)) {
          const industries = Array.from(new Set(usecases.map(u => u.industry))).filter(Boolean);
          return industries.sort();
        }
      }
      
      throw new Error('Failed to fetch industries');
    } catch (err) {
      console.error('Failed to fetch industries:', err);
      return [];
    }
  },

  // Get list of all tags
  async getTags(): Promise<string[]> {
    try {
      // Extract from all usecases since dedicated endpoint doesn't exist
      const usecasesResponse = await fetch(`${API_BASE}/usecases`, {
        signal: AbortSignal.timeout(5000)
      });
      
      if (usecasesResponse.ok) {
        const usecases = await usecasesResponse.json();
        if (Array.isArray(usecases)) {
          const allTags = usecases.flatMap(u => u.tags || []);
          const uniqueTags = Array.from(new Set(allTags)).filter(Boolean);
          return uniqueTags.sort();
        }
      }
      
      throw new Error('Failed to fetch tags');
    } catch (err) {
      console.error('Failed to fetch tags:', err);
      return [];
    }
  }
}; 