export interface Usecase {
  id: number;
  unique_id: string;
  name: string;
  description: string;
  industry: string;
  tags: string[];
  // Performance/latency characteristics
  batch_size: number;
  beam_size: number;
  input_tokens_min: number;
  input_tokens_max: number;
  output_tokens_min: number;
  output_tokens_max: number;
  ttft_min: number;
  ttft_max: number;
  e2e_min: number;
  e2e_max: number;
  inter_token_min: number;
  inter_token_max: number;
  // Metadata
  source: string;
  created_at: string;
  updated_at: string;
  is_active: boolean;
  latency_profile: string;
  input_length_profile: string;
}

export interface UsecaseListResponse {
  usecases?: Usecase[];
  total?: number;
  page?: number;
  page_size?: number;
  total_pages?: number;
}

export interface UsecaseFilters {
  search?: string;
  industries?: string[];
  tags?: string[];
  latency_profile?: string;
  is_active?: boolean;
  page?: number;
  page_size?: number;
}

export interface UsecaseStats {
  total_usecases: number;
  active_usecases: number;
  by_industry: { [key: string]: number };
  by_latency_profile: { [key: string]: number };
  most_common_tags: Array<{ tag: string; count: number }>;
} 