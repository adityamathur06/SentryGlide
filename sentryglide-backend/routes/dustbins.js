// routes/dustbins.js
const express = require('express');
const router = express.Router();
const Dustbin = require('../models/Dustbin');
const protect = require('../middleware/authMiddleware');

// Get all dustbins for the logged-in hospital
router.get('/', protect, async (req, res) => {
  try {
    const dustbins = await Dustbin.find({ hospitalId: req.hospital.id }).sort({ createdAt: -1 });
    res.json(dustbins);
  } catch (error) {
    res.status(500).json({ message: 'Server Error' });
  }
});

// Register a new SentryGlide cart
router.post('/', protect, async (req, res) => {
  try {
    const { hardwareId, customName, location } = req.body;
    
    // Check if hardware ID is already registered globally
    const existing = await Dustbin.findOne({ hardwareId });
    if (existing) return res.status(400).json({ message: 'Hardware ID already in use' });

    const newDustbin = new Dustbin({
      hospitalId: req.hospital.id,
      hardwareId,
      customName,
      location,
      batteryLevel: 100, // Starts fully charged
      status: 'Standby'
    });

    const savedDustbin = await newDustbin.save();
    res.status(201).json(savedDustbin);
  } catch (error) {
    res.status(500).json({ message: 'Server Error' });
  }
});

module.exports = router;