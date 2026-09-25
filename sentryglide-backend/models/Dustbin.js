const mongoose = require('mongoose');

const dustbinSchema = new mongoose.Schema({
  hospitalId: { 
    type: mongoose.Schema.Types.ObjectId, 
    ref: 'Hospital', 
    required: true 
  },
  hardwareId: { type: String, required: true, unique: true },
  customName: { type: String, required: true },
  location: { type: String, required: true, default: 'Utility Room' },
  batteryLevel: { type: Number, default: 100, min: 0, max: 100 },
  status: { 
    type: String, 
    enum: ['Standby', 'In Transit', 'Discharging', 'Maintenance'], 
    default: 'Standby' 
  },
  capacity: {
    yellow: { type: Number, default: 0, max: 100 },
    red: { type: Number, default: 0, max: 100 },
    white: { type: Number, default: 0, max: 100 },
    blue: { type: Number, default: 0, max: 100 }
  }
}, { timestamps: true });

module.exports = mongoose.model('Dustbin', dustbinSchema);