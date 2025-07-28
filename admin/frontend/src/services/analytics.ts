import api from './api';
import { AnalyticsSummary, SystemHealth, AuditLog, Feedback } from '../types/analytics';

export const analyticsService = {
  async getSummary(days: number = 30): Promise<AnalyticsSummary> {
    const response = await api.get<AnalyticsSummary>('/analytics/summary', {
      params: { days },
    });
    return response.data;
  },

  async getSystemHealth(): Promise<SystemHealth> {
    const response = await api.get<SystemHealth>('/analytics/health');
    return response.data;
  },

  async getAuditLogs(params?: {
    admin_id?: number;
    resource_type?: string;
    action?: string;
    limit?: number;
    offset?: number;
  }): Promise<{ total: number; logs: AuditLog[]; limit: number; offset: number }> {
    const response = await api.get('/analytics/audit-logs', { params });
    return response.data;
  },

  async getUsageTimeline(days: number = 30, feature_type?: string): Promise<Array<{ date: string; count: number }>> {
    const response = await api.get('/analytics/usage/timeline', {
      params: { days, feature_type },
    });
    return response.data;
  },

  async getTopEndpoints(limit: number = 10): Promise<Array<{
    endpoint: string;
    method: string;
    count: number;
    avg_response_time: number;
  }>> {
    const response = await api.get('/analytics/usage/top-endpoints', {
      params: { limit },
    });
    return response.data;
  },

  async getActiveUsers(hours: number = 24): Promise<Array<{
    session_id: string;
    country: string;
    city: string;
    device_type: string;
    browser: string;
    os: string;
    start_time: string;
    page_views: number;
  }>> {
    const response = await api.get('/analytics/users/active', {
      params: { hours },
    });
    return response.data;
  },
};