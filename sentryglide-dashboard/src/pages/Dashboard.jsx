// src/pages/Dashboard.jsx
import { useEffect, useState } from 'react';
import axios from 'axios';
import { useAuthStore } from '../store/useAuthStore';
import { useNavigate } from 'react-router-dom';
import { LogOut, Plus, Activity, MapPin, BatteryCharging } from 'lucide-react';

export default function Dashboard() {
  const { hospital, token, logout } = useAuthStore();
  const navigate = useNavigate();
  const [dustbins, setDustbins] = useState([]);
  const [isLoading, setIsLoading] = useState(true);

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

  return (
    <div className="min-h-screen bg-slate-50 font-sans">
      
      {/* Top Navigation Bar */}
      <nav className="flex items-center justify-between bg-white px-8 py-4 shadow-sm">
        <div>
          <h1 className="text-xl font-bold text-blue-700">SentryGlide Fleet Command</h1>
          <p className="text-sm font-medium text-slate-500">{hospital?.name}</p>
        </div>
        <div className="flex items-center gap-4">
          <button 
            className="flex items-center gap-2 rounded-lg bg-blue-600 px-4 py-2 text-sm font-semibold text-white transition-colors hover:bg-blue-700"
            onClick={() => alert("Modal logic coming next!")}
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

      {/* Main Content Area */}
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
              <div key={cart._id} className="cursor-pointer rounded-xl bg-white p-5 shadow-sm ring-1 ring-slate-200 transition-shadow hover:shadow-md">
                <div className="mb-4 flex items-start justify-between">
                  <div>
                    <h3 className="font-bold text-slate-800">{cart.customName}</h3>
                    <span className="mt-1 flex items-center gap-1 text-xs font-medium text-slate-500">
                      <MapPin className="h-3 w-3" /> {cart.location}
                    </span>
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
                    <Activity className="h-4 w-4 text-blue-500" />
                    ID: {cart.hardwareId.slice(-4)}
                  </span>
                </div>
              </div>
            ))}
          </div>
        )}
      </main>
    </div>
  );
}