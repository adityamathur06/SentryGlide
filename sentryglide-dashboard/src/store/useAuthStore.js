import { create } from 'zustand';

const useAuthStore = create((set) => ({
  hospital: JSON.parse(localStorage.getItem('hospital')) || null,
  token: localStorage.getItem('token') || null,
  
  login: (hospitalData, token) => {
    localStorage.setItem('hospital', JSON.stringify(hospitalData));
    localStorage.setItem('token', token);
    set({ hospital: hospitalData, token });
  },
  
  logout: () => {
    localStorage.removeItem('hospital');
    localStorage.removeItem('token');
    set({ hospital: null, token: null });
  }
}));

export default useAuthStore;