import { X, BatteryCharging, Activity, BarChart3, AlertCircle, AlertTriangle, MapPin } from 'lucide-react';

const binColors = {
  yellow: 'bg-yellow-400 text-yellow-900',
  red: 'bg-red-500 text-red-50',
  white: 'bg-slate-100 border-2 border-slate-300 text-slate-700',
  blue: 'bg-blue-500 text-blue-50'
};

export default function CartDetailsModal({ cart, onClose }) {
  if (!cart) return null;

  const isBatteryCritical = cart.batteryLevel < 10;
  const isCapacityCritical = Object.values(cart.capacity).some(val => val >= 90);
  const hasAlert = isBatteryCritical || isCapacityCritical;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-slate-900/50 p-4 backdrop-blur-sm">
      <div className={`w-full max-w-2xl overflow-hidden rounded-2xl bg-white shadow-2xl ring-1 ${hasAlert ? 'ring-red-200' : 'ring-slate-100'}`}>
        
        {/* Header */}
        <div className={`flex items-center justify-between border-b px-6 py-4 ${hasAlert ? 'border-red-100 bg-red-50/50' : 'border-slate-100 bg-slate-50'}`}>
          <div>
            <h2 className={`flex items-center gap-2 text-xl font-bold ${hasAlert ? 'text-red-700' : 'text-slate-800'}`}>
              {hasAlert && <AlertTriangle className="h-5 w-5" />}
              Hardware Profile
            </h2>
            <div className={`mt-1 flex items-center gap-3 text-sm font-medium ${hasAlert ? 'text-red-500' : 'text-slate-500'}`}>
              <span>Unit: {cart.serialNumber}</span>
              <span className="flex items-center gap-1">
                <MapPin className="h-3.5 w-3.5" />
                {cart.location}
              </span>
            </div>
          </div>
          <button 
            onClick={onClose} 
            className="rounded-full p-2 text-slate-400 transition-colors hover:bg-slate-200 hover:text-slate-700"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        {/* Content */}
        <div className="p-6">
          
          {/* Quick Stats Row */}
          <div className="mb-8 grid grid-cols-3 gap-4">
            
            <div className="rounded-xl border border-slate-100 bg-slate-50 p-4">
              <div className="mb-1 flex items-center gap-2 text-sm font-semibold text-slate-500">
                <Activity className="h-4 w-4" /> Status
              </div>
              <div className={`text-lg font-bold ${cart.status === 'Standby' ? 'text-emerald-600' : 'text-amber-600'}`}>
                {cart.status}
              </div>
            </div>

            <div className={`rounded-xl border p-4 ${isBatteryCritical ? 'border-red-200 bg-red-50' : 'border-slate-100 bg-slate-50'}`}>
              <div className={`mb-1 flex items-center gap-2 text-sm font-semibold ${isBatteryCritical ? 'text-red-600' : 'text-slate-500'}`}>
                <BatteryCharging className="h-4 w-4" /> Battery Life
              </div>
              <div className={`text-lg font-bold ${isBatteryCritical ? 'text-red-600' : 'text-slate-800'}`}>
                {cart.batteryLevel}%
              </div>
            </div>

            <div className={`rounded-xl border p-4 ${hasAlert ? 'border-red-200 bg-red-50' : 'border-slate-100 bg-slate-50'}`}>
              <div className={`mb-1 flex items-center gap-2 text-sm font-semibold ${hasAlert ? 'text-red-600' : 'text-slate-500'}`}>
                <AlertCircle className="h-4 w-4" /> Hardware Health
              </div>
              <div className={`text-lg font-bold ${hasAlert ? 'text-red-600' : 'text-emerald-600'}`}>
                {hasAlert ? 'Action Required' : 'Optimal'}
              </div>
            </div>

          </div>

          {/* Detailed Capacity Section */}
          <div className="mb-4 flex items-center gap-2">
            <BarChart3 className={`h-5 w-5 ${isCapacityCritical ? 'text-red-500' : 'text-slate-400'}`} />
            <h3 className={`text-lg font-bold ${isCapacityCritical ? 'text-red-700' : 'text-slate-800'}`}>
              Live Bin Capacities
            </h3>
          </div>
          
          <div className="grid gap-4 sm:grid-cols-2">
            {['yellow', 'red', 'white', 'blue'].map((color) => {
              const isBinFull = cart.capacity[color] >= 90;
              
              return (
                <div key={color} className={`flex flex-col gap-2 rounded-xl border p-4 ${isBinFull ? 'border-red-200 bg-red-50/50' : 'border-slate-100'}`}>
                  <div className={`flex justify-between text-sm font-bold uppercase tracking-wider ${isBinFull ? 'text-red-700' : 'text-slate-600'}`}>
                    <span>{color} Bin</span>
                    <span>{cart.capacity[color]}%</span>
                  </div>
                  <div className="h-3 w-full overflow-hidden rounded-full bg-slate-200">
                    <div 
                      className={`h-full rounded-full ${binColors[color].split(' ')[0]} ${color === 'white' ? 'border border-slate-300' : ''}`} 
                      style={{ width: `${cart.capacity[color]}%` }} 
                    />
                  </div>
                  {isBinFull && (
                    <p className="text-xs font-semibold text-red-600">Approaching maximum capacity</p>
                  )}
                </div>
              );
            })}
          </div>

        </div>
        
        {/* Footer Actions */}
        <div className={`px-6 py-4 text-right border-t ${hasAlert ? 'border-red-100 bg-red-50/30' : 'border-slate-100 bg-slate-50'}`}>
          <button 
            onClick={onClose}
            className="rounded-lg bg-white px-5 py-2 text-sm font-semibold text-slate-700 shadow-sm ring-1 ring-slate-300 transition-all hover:bg-slate-50"
          >
            Close Profile
          </button>
        </div>

      </div>
    </div>
  );
}