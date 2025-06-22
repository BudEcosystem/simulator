// Hardware type definitions for BudSimulator

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

export interface HardwareDetail extends Omit<Hardware, 'vendors' | 'clouds'> {
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
  optimality: 'optimal' | 'good' | 'ok';
  utilization: number;
  total_memory_available?: number;
  batch_recommendations?: BatchRecommendation[];
}

export interface BatchRecommendation {
  nodes: number;
  total_memory: number;
  recommended_batch_size: number;
  utilization_at_batch: number;
}

export interface HardwareRecommendationResponse {
  cpu_recommendations: HardwareRecommendation[];
  gpu_recommendations: HardwareRecommendation[];
  model_info: {
    is_small_model: boolean;
    cpu_compatible: boolean;
    total_memory_gb: number;
    model_params_b: number;
  };
  total_recommendations?: number;
}

export interface HardwareFilters {
  search: string;
  type: string | null;
  manufacturer: string | null;
  minMemory: number | null;
  maxMemory: number | null;
  minFlops: number | null;
  maxFlops: number | null;
  sortBy: string;
  sortOrder: 'asc' | 'desc';
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

export interface PaginationState {
  page: number;
  limit: number;
  total: number;
} 