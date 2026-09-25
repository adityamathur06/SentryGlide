const mongoose = require('mongoose');

const dustbinSchema = new mongoose.Schema({
  serialNumber: { type: String, required: true, unique: true },
  hospitalId: { 
    type: mongoose.Schema.Types.ObjectId, 
    ref: 'Hospital', 
    default: null 
  },
  location: { type: String, default: 'Unassigned' }, // New field added
  batteryLevel: { type: Number, default: 100, min: 0, max: 100 },
  status: { type: String, default: 'Standby' },
  capacity: {
    yellow: { type: Number, default: 0 },
    red: { type: Number, default: 0 },
    white: { type: Number, default: 0 },
    blue: { type: Number, default: 0 }
  }
}, { timestamps: true });

module.exports = mongoose.model('Dustbin', dustbinSchema);