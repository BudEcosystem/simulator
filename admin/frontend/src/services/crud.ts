import api from './api';

export const crudService = {
  // Models
  async getModels(params?: { search?: string; limit?: number; offset?: number }) {
    const response = await api.get('/crud/models', { params });
    return response.data;
  },

  async getModel(modelId: string) {
    const response = await api.get(`/crud/models/${modelId}`);
    return response.data;
  },

  async createModel(data: any) {
    const response = await api.post('/crud/models', data);
    return response.data;
  },

  async updateModel(modelId: string, data: any) {
    const response = await api.put(`/crud/models/${modelId}`, data);
    return response.data;
  },

  async deleteModel(modelId: string) {
    const response = await api.delete(`/crud/models/${modelId}`);
    return response.data;
  },

  // Hardware
  async getHardware(params?: { 
    search?: string; 
    type?: string; 
    manufacturer?: string; 
    limit?: number; 
    offset?: number 
  }) {
    const response = await api.get('/crud/hardware', { params });
    return response.data;
  },

  async getHardwareItem(name: string) {
    const response = await api.get(`/crud/hardware/${name}`);
    return response.data;
  },

  async createHardware(data: any) {
    const response = await api.post('/crud/hardware', data);
    return response.data;
  },

  async updateHardware(name: string, data: any) {
    const response = await api.put(`/crud/hardware/${name}`, data);
    return response.data;
  },

  async deleteHardware(name: string) {
    const response = await api.delete(`/crud/hardware/${name}`);
    return response.data;
  },

  // Usecases
  async getUsecases(params?: { 
    industry?: string; 
    search?: string; 
    limit?: number; 
    offset?: number 
  }) {
    const response = await api.get('/crud/usecases', { params });
    return response.data;
  },

  async getUsecase(id: string) {
    const response = await api.get(`/crud/usecases/${id}`);
    return response.data;
  },

  async createUsecase(data: any) {
    const response = await api.post('/crud/usecases', data);
    return response.data;
  },

  async updateUsecase(id: string, data: any) {
    const response = await api.put(`/crud/usecases/${id}`, data);
    return response.data;
  },

  async deleteUsecase(id: string) {
    const response = await api.delete(`/crud/usecases/${id}`);
    return response.data;
  },
};