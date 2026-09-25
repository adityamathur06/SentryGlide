// routes/dustbins.js
const express = require('express');
const router = express.Router();
const Dustbin = require('../models/Dustbin');
const Hospital = require('../models/Hospital');
const protect = require('../middleware/authMiddleware');

// GET: Fetch only the dustbins owned by this hospital
router.get('/', protect, async (req, res) => {
  try {
    const hospital = await Hospital.findById(req.hospital.id);
    // Find all global dustbins where the serialNumber is in this hospital's array
    const fleet = await Dustbin.find({ serialNumber: { $in: hospital.dustbins } });
    res.json(fleet);
  } catch (error) {
    res.status(500).json({ message: 'Server Error' });
  }
});

// POST: Claim a dustbin
router.post('/', protect, async (req, res) => {
  try {
    const { serialNumber } = req.body;
    
    // Check 1: Does it exist in the factory database?
    const existingDustbin = await Dustbin.findOne({ serialNumber });
    if (!existingDustbin) {
      return res.status(404).json({ message: 'Dustbin does not exist' });
    }

    const currentHospital = await Hospital.findById(req.hospital.id);

    // Check 2: Is it already owned by the current hospital?
    if (currentHospital.dustbins.includes(serialNumber)) {
      return res.status(400).json({ message: 'Dustbin already owned' });
    }

    // Check 3: Is it owned by a DIFFERENT hospital?
    const otherHospital = await Hospital.findOne({ dustbins: serialNumber });
    if (otherHospital) {
      return res.status(400).json({ message: 'Dustbin owned by another hospital' });
    }

    // Success: Add serial number to the hospital's array
    currentHospital.dustbins.push(serialNumber);
    await currentHospital.save();

    res.status(200).json(existingDustbin);
  } catch (error) {
    res.status(500).json({ message: 'Server Error' });
  }
});

module.exports = router;