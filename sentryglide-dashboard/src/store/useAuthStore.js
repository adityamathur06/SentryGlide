// src/store/useAuthStore.js
import { create } from 'zustand';

export const useAuthStore = create((set) => ({
  hospital: JSON.parse(localStorage.getItem('hospital')) || null,
  token: localStorage.getItem('token') || null,
  
  login: (hospital, token) => {
    localStorage.setItem('hospital', JSON.stringify(hospital));
    localStorage.setItem('token', token);
    set({ hospital, token });
  },
  
  logout: () => {
    localStorage.removeItem('hospital');
    localStorage.removeItem('token');
    set({ hospital: null, token: null });
  }
}));