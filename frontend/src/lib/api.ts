import axios from 'axios';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000/api/v1';

export const apiClient = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Interceptor to attach token
apiClient.interceptors.request.use((config) => {
  if (typeof window !== 'undefined') {
    const token = localStorage.getItem('token');
    if (token && config.headers) {
      config.headers.Authorization = `Bearer ${token}`;
    }
  }
  return config;
});

// Auth Methods
export const authApi = {
  login: (formData: FormData) => apiClient.post('/auth/login', formData, {
    headers: { 'Content-Type': 'application/x-www-form-urlencoded' }
  }),
  register: (userData: any) => apiClient.post('/auth/register', userData),
  getMe: () => apiClient.get('/auth/me'),
};

// Market Methods
export const marketApi = {
  getPrices: (symbol: string, interval = '1h') => apiClient.get(`/market/prices/${symbol}?interval=${interval}`),
  getRegime: (symbol: string) => apiClient.get(`/market/regime/${symbol}`),
  getIndicators: (symbol: string) => apiClient.get(`/market/indicators/${symbol}`),
};

// Analysis Methods
export const analysisApi = {
  getIctSetups: (symbol: string, timeframe = '15m') => apiClient.get(`/analysis/ict/${symbol}?timeframe=${timeframe}`),
  getSmcAnalysis: (symbol: string, timeframe = '1h') => apiClient.get(`/analysis/smc/${symbol}?timeframe=${timeframe}`),
  getWyckoffAnalysis: (symbol: string, timeframe = '1d') => apiClient.get(`/analysis/wyckoff/${symbol}?timeframe=${timeframe}`),
};

// Trading Methods
export const tradingApi = {
  getAccount: () => apiClient.get('/trading/account'),
  openTrade: (tradeData: any) => apiClient.post('/trading/open', tradeData),
  closeTrade: (tradeId: number) => apiClient.post(`/trading/close/${tradeId}`),
  getPositions: () => apiClient.get('/trading/positions'),
};
