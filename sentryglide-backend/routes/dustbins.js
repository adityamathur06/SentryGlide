const express = require('express');
const router = express.Router();
const Dustbin = require('../models/Dustbin');
const protect = require('../middleware/authMiddleware');

// GET: Fetch only the dustbins claimed by this specific hospital
router.get('/', protect, async (req, res) => {
  try {
    const fleet = await Dustbin.find({ hospitalId: req.hospital.id });
    res.json(fleet);
  } catch (error) {
    res.status(500).json({ message: 'Server Error' });
  }
});

// POST: Claim a dustbin via Serial Number and assign Location
router.post('/', protect, async (req, res) => {
  try {
    // Now accepting location from the request body
    const { serialNumber, location } = req.body;
    
    const cart = await Dustbin.findOne({ serialNumber });

    if (!cart) {
      return res.status(404).json({ message: 'Dustbin does not exist' });
    }

    if (cart.hospitalId && cart.hospitalId.toString() === req.hospital.id) {
      return res.status(400).json({ message: 'Dustbin already owned' });
    }

    if (cart.hospitalId) {
      return res.status(400).json({ message: 'Dustbin owned by another hospital' });
    }

    // Success: Claim it and set the specific location
    cart.hospitalId = req.hospital.id;
    cart.location = location || 'Unassigned';
    await cart.save();

    res.status(200).json(cart);
  } catch (error) {
    console.error(error);
    res.status(500).json({ message: 'Server Error' });
  }
});

module.exports = router;