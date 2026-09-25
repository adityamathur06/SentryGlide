// src/pages/Dashboard.jsx
import { useEffect, useState } from 'react';
import axios from 'axios';
import { useAuthStore } from '../store/useAuthStore';
import { useNavigate } from 'react-router-dom';
import { LogOut, Plus, Activity, BatteryCharging, X } from 'lucide-react';

export default function Dashboard() {
  const { hospital, token, logout } = useAuthStore();
  const navigate = useNavigate();
  const [dustbins, setDustbins] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  
  // Modal State
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [serialNumber, setSerialNumber] = useState('');
  const [modalError, setModalError] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);

  useEffect(() => {
    const fetchFleet = async () => {
      try {
        const response = await axios.get('http://localhost:5000/api/dustbins', {
          headers: { Authorization: `Bearer ${token}` }
        });
        setDustbins(response.data);
      } catch (error) {
        console.error('Error fetching fleet:', error);
      } finally {
        setIsLoading(false);
      }
    };
    fetchFleet();
  }, [token]);

  const handleLogout = () => {
    logout();
    navigate('/');
  };

  const handleAddCart = async (e) => {
    e.preventDefault();
    setModalError('');
    setIsSubmitting(true);

    try {
      const response = await axios.post('http://localhost:5000/api/dustbins', 
        { serialNumber },
        { headers: { Authorization: `Bearer ${token}` }}
      );
      
      // Add new cart to UI instantly
      setDustbins([...dustbins, response.data]);
      setIsModalOpen(false);
      setSerialNumber('');
    } catch (error) {
      setModalError(error.response?.data?.message || 'Failed to add dustbin');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 font-sans">
      {/* Navigation */}
      <nav className="flex items-center justify-between bg-white px-8 py-4 shadow-sm">
        <div>
          <h1 className="text-xl font-bold text-blue-700">SentryGlide Fleet Command</h1>
          <p className="text-sm font-medium text-slate-500">{hospital?.name}</p>
        </div>
        <div className="flex items-center gap-4">
          <button 
            onClick={() => setIsModalOpen(true)}
            className="flex items-center gap-2 rounded-lg bg-blue-600 px-4 py-2 text-sm font-semibold text-white transition-colors hover:bg-blue-700"
          >
            <Plus className="h-4 w-4" /> Add Cart
          </button>
          <button 
            onClick={handleLogout}
            className="flex items-center gap-2 rounded-lg border border-slate-200 bg-white px-4 py-2 text-sm font-semibold text-slate-700 transition-colors hover:bg-slate-50"
          >
            <LogOut className="h-4 w-4" /> Logout
          </button>
        </div>
      </nav>

      {/* Main Content */}
      <main className="p-8">
        <h2 className="mb-6 text-lg font-semibold text-slate-800">Active Units ({dustbins.length})</h2>
        
        {isLoading ? (
          <div className="text-slate-500">Loading fleet data...</div>
        ) : dustbins.length === 0 ? (
          <div className="rounded-xl border border-dashed border-slate-300 p-12 text-center text-slate-500">
            No SentryGlide units registered. Click "Add Cart" to onboard your first device.
          </div>
        ) : (
          <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
            {dustbins.map((cart) => (
              <div key={cart._id} className="rounded-xl bg-white p-5 shadow-sm ring-1 ring-slate-200 transition-shadow hover:shadow-md">
                <div className="mb-4 flex items-start justify-between">
                  <div>
                    <h3 className="font-bold text-slate-800">Unit: {cart.serialNumber}</h3>
                  </div>
                  <span className={`rounded-full px-2.5 py-0.5 text-xs font-bold ${
                    cart.status === 'Standby' ? 'bg-emerald-100 text-emerald-700' : 'bg-amber-100 text-amber-700'
                  }`}>
                    {cart.status}
                  </span>
                </div>
                
                <div className="flex items-center justify-between border-t border-slate-100 pt-4 text-sm font-medium text-slate-600">
                  <span className="flex items-center gap-1.5">
                    <BatteryCharging className="h-4 w-4 text-emerald-500" />
                    {cart.batteryLevel}%
                  </span>
                  <span className="flex items-center gap-1.5">
                    <Activity className="h-4 w-4 text-blue-500" /> Active
                  </span>
                </div>
              </div>
            ))}
          </div>
        )}
      </main>

      {/* Add Cart Modal Overlay */}
      {isModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/50 backdrop-blur-sm">
          <div className="w-full max-w-md rounded-2xl bg-white p-6 shadow-xl">
            <div className="mb-4 flex items-center justify-between">
              <h3 className="text-lg font-bold text-slate-800">Onboard New Cart</h3>
              <button onClick={() => setIsModalOpen(false)} className="text-slate-400 hover:text-slate-600">
                <X className="h-5 w-5" />
              </button>
            </div>
            
            <form onSubmit={handleAddCart} className="space-y-4">
              {modalError && (
                <div className="rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-600">
                  {modalError}
                </div>
              )}
              
              <div>
                <label className="mb-1.5 block text-sm font-semibold text-slate-700">Hardware Serial Number</label>
                <input
                  type="text"
                  placeholder="e.g. SG-001"
                  className="w-full rounded-lg border border-slate-300 px-4 py-2.5 text-slate-700 focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/20"
                  value={serialNumber}
                  onChange={(e) => setSerialNumber(e.target.value)}
                  required
                />
              </div>

              <div className="flex justify-end gap-3 pt-4">
                <button 
                  type="button" 
                  onClick={() => setIsModalOpen(false)}
                  className="rounded-lg px-4 py-2 text-sm font-semibold text-slate-600 hover:bg-slate-100"
                >
                  Cancel
                </button>
                <button 
                  type="submit" 
                  disabled={isSubmitting}
                  className="rounded-lg bg-blue-600 px-4 py-2 text-sm font-semibold text-white hover:bg-blue-700 disabled:opacity-70"
                >
                  {isSubmitting ? 'Verifying...' : 'Add Cart'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}