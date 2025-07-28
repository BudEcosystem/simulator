// API Configuration
// This module provides a centralized way to manage API URLs

// Get the API base URL from environment variable or use default
const getApiUrl = (): string => {
  // If REACT_APP_API_URL is set and not empty, use it
  if (process.env.REACT_APP_API_URL) {
    // Ensure the URL doesn't end with a slash
    return process.env.REACT_APP_API_URL.replace(/\/$/, '');
  }
  
  // Default to relative path - this will use the proxy in development
  // and will work with the same domain in production
  return '';
};

// Export the API base URL
export const API_BASE = `${getApiUrl()}/api`;

// Helper function to build full API URLs
export const buildApiUrl = (path: string): string => {
  // Ensure path starts with /
  const normalizedPath = path.startsWith('/') ? path : `/${path}`;
  return `${API_BASE}${normalizedPath}`;
};

// Export configuration for debugging
export const apiConfig = {
  baseUrl: getApiUrl(),
  apiBase: API_BASE,
  isDevelopment: process.env.NODE_ENV === 'development',
  isProduction: process.env.NODE_ENV === 'production',
};