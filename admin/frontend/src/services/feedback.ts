import api from './api';
import { Feedback } from '../types/analytics';

export const feedbackService = {
  async getFeedback(params?: {
    status?: string;
    priority?: string;
    category?: string;
    feedback_type?: string;
    limit?: number;
    offset?: number;
  }): Promise<Feedback[]> {
    const response = await api.get<Feedback[]>('/feedback/', { params });
    return response.data;
  },

  async getFeedbackItem(id: number): Promise<Feedback> {
    const response = await api.get<Feedback>(`/feedback/${id}`);
    return response.data;
  },

  async updateFeedback(id: number, data: { status?: string; priority?: string }): Promise<Feedback> {
    const response = await api.put<Feedback>(`/feedback/${id}`, data);
    return response.data;
  },

  async respondToFeedback(id: number, data: { message: string; is_internal: boolean }): Promise<void> {
    await api.post(`/feedback/${id}/respond`, data);
  },

  async getFeedbackStats(): Promise<{
    status_breakdown: { [key: string]: number };
    priority_breakdown: { [key: string]: number };
    category_breakdown: { [key: string]: number };
    average_rating: number | null;
    recent_feedback_count: number;
  }> {
    const response = await api.get('/feedback/stats/summary');
    return response.data;
  },
};