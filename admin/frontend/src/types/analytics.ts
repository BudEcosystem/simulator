export interface AnalyticsSummary {
  total_users: number;
  active_users: number;
  users_by_country: Array<{ country: string; count: number }>;
  popular_models: Array<{ id: string; count: number }>;
  popular_hardware: Array<{ id: string; count: number }>;
  popular_usecases: Array<{ id: string; count: number }>;
  device_breakdown: Array<{ device: string; count: number }>;
  browser_breakdown: Array<{ browser: string; count: number }>;
  api_usage_timeline: Array<{ date: string; count: number }>;
  custom_slos_usage: number;
}

export interface SystemHealth {
  metrics: {
    [key: string]: {
      value: number;
      unit: string;
      timestamp: string;
    };
  };
  avg_response_time: number;
  error_count: number;
  status: string;
}

export interface AuditLog {
  id: number;
  admin_id: number;
  action: string;
  resource_type: string;
  resource_id: string;
  old_value: any;
  new_value: any;
  ip_address: string | null;
  timestamp: string;
}

export interface Feedback {
  id: number;
  session_id: string | null;
  feedback_type: string;
  category: string;
  rating: number | null;
  title: string;
  message: string;
  email: string | null;
  screenshot_url: string | null;
  status: string;
  priority: string;
  created_at: string;
  updated_at: string;
  responses: FeedbackResponse[];
}

export interface FeedbackResponse {
  id: number;
  feedback_id: number;
  admin_id: number;
  message: string;
  is_internal: boolean;
  created_at: string;
}