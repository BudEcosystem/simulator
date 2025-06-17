// Hardware API service functions
import { Hardware, HardwareDetail, HardwareRecommendation } from '../types/hardware';

const API_BASE = '/api';

export const hardwareAPI = {
  // List hardware with filters and pagination
  list: async (params: URLSearchParams): Promise<Hardware[]> => {
    const response = await fetch(`${API_BASE}/hardware?${params}`);
    if (!response.ok) {
      throw new Error(`Failed to fetch hardware: ${response.statusText}`);
    }
    return response.json();
  },

  // Get hardware details with vendor and cloud pricing
  getDetails: async (hardwareName: string): Promise<HardwareDetail> => {
    const response = await fetch(`${API_BASE}/hardware/${encodeURIComponent(hardwareName)}`);
    if (!response.ok) {
      throw new Error(`Failed to fetch hardware details: ${response.statusText}`);
    }
    return response.json();
  },

  // Get hardware recommendations based on memory requirements
  recommend: async (totalMemoryGB: number, modelParamsB: number): Promise<HardwareRecommendation[]> => {
    const response = await fetch(`${API_BASE}/hardware/recommend`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        total_memory_gb: totalMemoryGB,
        model_params_b: modelParamsB
      })
    });
    
    if (!response.ok) {
      throw new Error(`Failed to get recommendations: ${response.statusText}`);
    }
    return response.json();
  },

  // Advanced filtering for featured hardware
  filter: async (params: URLSearchParams): Promise<Hardware[]> => {
    const response = await fetch(`${API_BASE}/hardware/filter?${params}`);
    if (!response.ok) {
      throw new Error(`Failed to filter hardware: ${response.statusText}`);
    }
    return response.json();
  },

  // Get vendor pricing for specific hardware
  getVendors: async (hardwareName: string) => {
    const response = await fetch(`${API_BASE}/hardware/vendors/${encodeURIComponent(hardwareName)}`);
    if (!response.ok) {
      throw new Error(`Failed to fetch vendor info: ${response.statusText}`);
    }
    return response.json();
  },

  // Get cloud pricing for specific hardware
  getClouds: async (hardwareName: string) => {
    const response = await fetch(`${API_BASE}/hardware/clouds/${encodeURIComponent(hardwareName)}`);
    if (!response.ok) {
      throw new Error(`Failed to fetch cloud info: ${response.statusText}`);
    }
    return response.json();
  }
};

// Helper function to build URL parameters for hardware filtering
export const buildHardwareParams = (filters: {
  search?: string;
  type?: string | null;
  manufacturer?: string | null;
  minMemory?: number | null;
  maxMemory?: number | null;
  minFlops?: number | null;
  maxFlops?: number | null;
  sortBy?: string;
  sortOrder?: 'asc' | 'desc';
  limit?: number;
  offset?: number;
}): URLSearchParams => {
  const params = new URLSearchParams();
  
  if (filters.search) params.append('query', filters.search);
  if (filters.type) params.append('type', filters.type);
  if (filters.manufacturer) params.append('manufacturer', filters.manufacturer);
  if (filters.minMemory !== null && filters.minMemory !== undefined) {
    params.append('min_memory', filters.minMemory.toString());
  }
  if (filters.maxMemory !== null && filters.maxMemory !== undefined) {
    params.append('max_memory', filters.maxMemory.toString());
  }
  if (filters.minFlops !== null && filters.minFlops !== undefined) {
    params.append('min_flops', filters.minFlops.toString());
  }
  if (filters.maxFlops !== null && filters.maxFlops !== undefined) {
    params.append('max_flops', filters.maxFlops.toString());
  }
  if (filters.sortBy) params.append('sort_by', filters.sortBy);
  if (filters.sortOrder) params.append('sort_order', filters.sortOrder);
  if (filters.limit) params.append('limit', filters.limit.toString());
  if (filters.offset) params.append('offset', filters.offset.toString());
  
  return params;
}; 