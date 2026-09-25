// src/pages/Dashboard.jsx
import { useEffect, useState, useRef } from 'react';
import axios from 'axios';
import { useAuthStore } from '../store/useAuthStore';
import { useNavigate } from 'react-router-dom';
import { LogOut, Plus, Activity, BatteryCharging, X, AlertTriangle, MapPin } from 'lucide-react';
import CartDetailsModal from '../components/CartDetailsModal';
import { io } from 'socket.io-client';
import { ToastContainer, toast } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';

const binColors = {
  yellow: 'bg-yellow-400',
  red: 'bg-red-500',
  white: 'bg-slate-300 border border-slate-400', 
  blue: 'bg-blue-500'
};

export default function Dashboard() {
  const { hospital, token, logout } = useAuthStore();
  const navigate = useNavigate();
  const [dustbins, setDustbins] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [serialNumber, setSerialNumber] = useState('');
  const [location, setLocation] = useState('');
  const [modalError, setModalError] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [selectedCart, setSelectedCart] = useState(null);

  const alertHistory = useRef({});

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
    
    const socket = io('http://localhost:5000');

    socket.on('cart-updated', (updatedCart) => {
      setDustbins((currentDustbins) => {
        const ownsCart = currentDustbins.some(cart => cart._id === updatedCart._id);
        if (!ownsCart) return currentDustbins;

        return currentDustbins.map(cart => 
          cart._id === updatedCart._id ? updatedCart : cart
        );
      });
      
      setSelectedCart((currentSelected) => {
        if (currentSelected && currentSelected._id === updatedCart._id) {
          return updatedCart;
        }
        return currentSelected;
      });

      const { _id, serialNumber, batteryLevel, capacity, location: cartLocation } = updatedCart;
      
      if (!alertHistory.current[_id]) {
        alertHistory.current[_id] = { batteryAlerted: false, capacityAlerted: false };
      }

      if (batteryLevel < 10 && !alertHistory.current[_id].batteryAlerted) {
        toast.error(`Critical Battery: ${serialNumber} at ${cartLocation} is at ${batteryLevel}%!`);
        alertHistory.current[_id].batteryAlerted = true;
      } else if (batteryLevel >= 10) {
        alertHistory.current[_id].batteryAlerted = false;
      }

      const isCapacityCritical = Object.values(capacity).some(val => val >= 90);
      if (isCapacityCritical && !alertHistory.current[_id].capacityAlerted) {
        toast.warning(`Capacity Alert: ${serialNumber} at ${cartLocation} is nearing maximum limit!`);
        alertHistory.current[_id].capacityAlerted = true;
      } else if (!isCapacityCritical) {
        alertHistory.current[_id].capacityAlerted = false;
      }
    });

    return () => socket.disconnect();
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
        { serialNumber, location },
        { headers: { Authorization: `Bearer ${token}` }}
      );
      
      setDustbins([...dustbins, response.data]);
      setIsModalOpen(false);
      setSerialNumber('');
      setLocation('');
    } catch (error) {
      setModalError(error.response?.data?.message || 'Failed to add dustbin');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 font-sans">
      
      <ToastContainer position="top-right" autoClose={5000} theme="colored" />

      {/* Navigation - Updated Layout */}
      <nav className="relative flex items-center justify-between bg-white px-8 py-4 shadow-sm">
        
        {/* Left: Hospital Name */}
        <div className="flex flex-1 items-center">
          <p className="text-lg font-bold text-slate-700">{hospital?.name}</p>
        </div>

        {/* Center: Title (Absolutely positioned to guarantee true center) */}
        <div className="absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-1/2">
          <h1 className="text-2xl font-extrabold tracking-tight text-blue-700">SentryGlide Hub</h1>
        </div>

        {/* Right: Actions */}
        <div className="flex flex-1 items-center justify-end gap-4">
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

      <main className="p-8">
        <h2 className="mb-6 text-lg font-semibold text-slate-800">Active Units ({dustbins.length})</h2>
        
        {isLoading && dustbins.length === 0 ? (
          <div className="text-slate-500">Loading fleet data...</div>
        ) : dustbins.length === 0 ? (
          <div className="rounded-xl border border-dashed border-slate-300 p-12 text-center text-slate-500">
            No SentryGlide units registered. Click "Add Cart" to onboard your first device.
          </div>
        ) : (
          <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4">
            {dustbins.map((cart) => {
              const isBatteryCritical = cart.batteryLevel < 10;
              const isCapacityCritical = Object.values(cart.capacity).some(val => val >= 90);
              const hasAlert = isBatteryCritical || isCapacityCritical;

              return (
                <div 
                  key={cart._id} 
                  onClick={() => setSelectedCart(cart)}
                  className={`cursor-pointer flex flex-col justify-between rounded-xl p-5 shadow-sm ring-1 transition-shadow hover:shadow-md ${
                    hasAlert 
                      ? 'bg-red-50/50 ring-red-300 hover:ring-red-400' 
                      : 'bg-white ring-slate-200 hover:ring-blue-200'
                  }`}
                >
                  
                  <div>
                    <div className="mb-4 flex items-start justify-between">
                      <div>
                        <h3 className={`font-bold flex items-center gap-2 ${hasAlert ? 'text-red-700' : 'text-slate-800'}`}>
                          {hasAlert && <AlertTriangle className="h-4 w-4" />}
                          Unit: {cart.serialNumber}
                        </h3>
                        <span className={`mt-1 flex items-center gap-1 text-xs font-medium ${hasAlert ? 'text-red-500' : 'text-slate-500'}`}>
                          <MapPin className="h-3 w-3" /> {cart.location}
                        </span>
                      </div>
                      <span className={`rounded-full px-2.5 py-0.5 text-xs font-bold ${
                        cart.status === 'Standby' ? 'bg-emerald-100 text-emerald-700' : 'bg-amber-100 text-amber-700'
                      }`}>
                        {cart.status}
                      </span>
                    </div>
                    
                    <div className="mb-4 flex items-center justify-between text-sm font-medium text-slate-600">
                      <span className={`flex items-center gap-1.5 ${isBatteryCritical ? 'text-red-600 font-bold' : ''}`}>
                        <BatteryCharging className={`h-4 w-4 ${isBatteryCritical ? 'text-red-600' : (cart.batteryLevel > 20 ? 'text-emerald-500' : 'text-amber-500')}`} />
                        {cart.batteryLevel}%
                      </span>
                      <span className="flex items-center gap-1.5">
                        <Activity className="h-4 w-4 text-blue-500" /> Active
                      </span>
                    </div>
                  </div>

                  <div className="border-t border-slate-100/80 pt-4">
                    <p className={`mb-2 text-xs font-semibold uppercase tracking-wider ${isCapacityCritical ? 'text-red-500' : 'text-slate-500'}`}>
                      Internal Capacity
                    </p>
                    <div className="grid grid-cols-4 gap-2">
                      {['yellow', 'red', 'white', 'blue'].map((color) => {
                        const isBinFull = cart.capacity[color] >= 90;
                        return (
                          <div key={color} className="flex flex-col gap-1">
                            <div className="h-2 w-full overflow-hidden rounded-full bg-slate-200">
                              <div 
                                className={`h-full rounded-full ${binColors[color]}`} 
                                style={{ width: `${cart.capacity[color]}%` }} 
                              />
                            </div>
                            <span className={`text-center text-[10px] font-bold uppercase ${isBinFull ? 'text-red-600' : 'text-slate-400'}`}>
                              {cart.capacity[color]}%
                            </span>
                          </div>
                        );
                      })}
                    </div>
                  </div>

                </div>
              );
            })}
          </div>
        )}
      </main>

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

              <div>
                <label className="mb-1.5 block text-sm font-semibold text-slate-700">Cart Assignment Location</label>
                <input
                  type="text"
                  placeholder="e.g. Ward 3, North Wing"
                  className="w-full rounded-lg border border-slate-300 px-4 py-2.5 text-slate-700 focus:border-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500/20"
                  value={location}
                  onChange={(e) => setLocation(e.target.value)}
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

      <CartDetailsModal 
        cart={selectedCart} 
        onClose={() => setSelectedCart(null)} 
      />
      
    </div>
  );
}