const mongoose = require('mongoose');

const hospitalSchema = new mongoose.Schema({
  name: { type: String, required: true },
  email: { type: String, required: true, unique: true },
  password: { type: String, required: true },
  dustbins: [{ type: String }] // Array of claimed serial numbers
}, { timestamps: true });

module.exports = mongoose.model('Hospital', hospitalSchema);