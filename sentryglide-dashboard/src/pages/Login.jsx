// src/pages/Login.jsx
import { useState } from 'react';
import axios from 'axios';
import { useAuthStore } from '../store/useAuthStore';
import { useNavigate } from 'react-router-dom';
import { ShieldPlus, Lock, Building2 } from 'lucide-react';

export default function Login() {
  const [isLogin, setIsLogin] = useState(true);
  const [name, setName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [errorMsg, setErrorMsg] = useState('');
  const [successMsg, setSuccessMsg] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  
  const login = useAuthStore((state) => state.login);
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setErrorMsg('');
    setSuccessMsg('');
    setIsLoading(true);

    try {
      if (isLogin) {
        // Sign In
        const response = await axios.post('http://localhost:5000/api/auth/login', { 
          email, 
          password 
        });
        login(response.data.hospital, response.data.token);
        navigate('/dashboard');
      } else {
        // Sign Up
        await axios.post('http://localhost:5000/api/auth/register', { 
          name, 
          email, 
          password 
        });
        setSuccessMsg('Facility registered successfully! You can now log in.');
        setIsLogin(true);
        setPassword('');
      }
    } catch (error) {
      console.error('Auth Error:', error);
      setErrorMsg(
        error.response?.data?.message || 'Authentication failed. Please verify your details.'
      );
    } finally {
      setIsLoading(false);
    }
  };

  const toggleMode = () => {
    setIsLogin(!isLogin);
    setErrorMsg('');
    setSuccessMsg('');
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-gradient-to-br from-slate-100 to-blue-50 p-4 font-sans">
      <div className="w-full max-w-md overflow-hidden rounded-2xl bg-white shadow-2xl ring-1 ring-slate-200">
        
        {/* Header Section */}
        <div className="bg-blue-600 p-8 text-center">
          <ShieldPlus className="mx-auto mb-3 h-12 w-12 text-white" strokeWidth={1.5} />
          <h2 className="text-3xl font-bold tracking-tight text-white">SentryGlide</h2>
          <p className="mt-2 text-sm font-medium text-blue-100">
            {isLogin ? 'Biomedical Waste Command Center' : 'Register New Hospital Facility'}
          </p>
        </div>
        
        {/* Form Section */}
        <div className="p-8">
          <form onSubmit={handleSubmit} className="space-y-4">
            
            {/* Status Messages */}
            {errorMsg && (
              <div className="rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-600">
                {errorMsg}
              </div>
            )}
            {successMsg && (
              <div className="rounded-lg border border-emerald-200 bg-emerald-50 p-3 text-sm text-emerald-700">
                {successMsg}
              </div>
            )}

            {/* Hospital Name (Sign Up only) */}
            {!isLogin && (
              <div>
                <label className="mb-1.5 block text-sm font-semibold text-slate-700">Facility / Hospital Name</label>
                <div className="relative">
                  <input
                    type="text"
                    placeholder="e.g. Apex Multi-Speciality Hospital"
                    className="w-full rounded-lg border border-slate-300 px-4 py-3 text-slate-700 placeholder-slate-400 transition-all focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/20"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    required
                  />
                  <Building2 className="absolute right-3.5 top-3.5 h-5 w-5 text-slate-400" />
                </div>
              </div>
            )}
            
            <div>
              <label className="mb-1.5 block text-sm font-semibold text-slate-700">Hospital Email</label>
              <input
                type="email"
                placeholder="admin@hospital.com"
                className="w-full rounded-lg border border-slate-300 px-4 py-3 text-slate-700 placeholder-slate-400 transition-all focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/20"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
              />
            </div>
            
            <div>
              <label className="mb-1.5 block text-sm font-semibold text-slate-700">Security Password</label>
              <input
                type="password"
                placeholder="••••••••"
                className="w-full rounded-lg border border-slate-300 px-4 py-3 text-slate-700 placeholder-slate-400 transition-all focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/20"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
              />
            </div>

            <button 
              type="submit" 
              disabled={isLoading}
              className="mt-2 flex w-full items-center justify-center gap-2 rounded-lg bg-blue-600 px-4 py-3.5 text-sm font-bold text-white transition-all hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-70"
            >
              <Lock className="h-4 w-4" />
              {isLoading 
                ? 'Processing...' 
                : isLogin 
                  ? 'Secure Login' 
                  : 'Register Facility'}
            </button>
          </form>

          {/* Mode Switcher */}
          <div className="mt-6 border-t border-slate-100 pt-4 text-center">
            <p className="text-sm text-slate-600">
              {isLogin ? "Need to onboard a new hospital facility?" : "Already registered your facility?"}{' '}
              <button
                type="button"
                onClick={toggleMode}
                className="font-semibold text-blue-600 transition-colors hover:text-blue-800 focus:underline focus:outline-none"
              >
                {isLogin ? 'Sign Up' : 'Sign In'}
              </button>
            </p>
          </div>
        </div>
        
      </div>
    </div>
  );
}