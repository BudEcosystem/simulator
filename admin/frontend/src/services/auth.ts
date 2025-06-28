import api from './api';
import { AdminUser, LoginResponse } from '../types/auth';

export const authService = {
  async login(username: string, password: string): Promise<LoginResponse> {
    const response = await api.post<LoginResponse>('/auth/login', {
      username,
      password,
    });
    return response.data;
  },

  async logout(): Promise<void> {
    await api.post('/auth/logout');
  },

  async getCurrentUser(): Promise<AdminUser> {
    const response = await api.get<AdminUser>('/auth/me');
    return response.data;
  },
};