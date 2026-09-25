const express = require('express');
const bcrypt = require('bcrypt');
const jwt = require('jsonwebtoken');
const Hospital = require('../models/Hospital');

const router = express.Router();

// 1. REGISTER a new Hospital Account
router.post('/register', async (req, res) => {
  try {
    const { name, email, password } = req.body;

    // Check if hospital already exists
    const existingHospital = await Hospital.findOne({ email });
    if (existingHospital) {
      return res.status(400).json({ message: 'Hospital already registered' });
    }

    // Hash the password securely
    const salt = await bcrypt.genSalt(10);
    const hashedPassword = await bcrypt.hash(password, salt);

    // Save to MongoDB
    const newHospital = new Hospital({
      name,
      email,
      password: hashedPassword
    });
    
    await newHospital.save();
    res.status(201).json({ message: 'Hospital registered successfully' });

  } catch (error) {
    res.status(500).json({ message: 'Server error during registration' });
  }
});

// 2. LOGIN to Hospital Account
router.post('/login', async (req, res) => {
  try {
    const { email, password } = req.body;

    // Find the hospital in DB
    const hospital = await Hospital.findOne({ email });
    if (!hospital) {
      return res.status(400).json({ message: 'Invalid credentials' });
    }

    // Compare passwords
    const isMatch = await bcrypt.compare(password, hospital.password);
    if (!isMatch) {
      return res.status(400).json({ message: 'Invalid credentials' });
    }

    // Generate JWT Token (Secure Session Key)
    const token = jwt.sign(
      { id: hospital._id, name: hospital.name }, 
      process.env.JWT_SECRET, 
      { expiresIn: '1d' } // Token expires in 1 day
    );

    res.json({
      token,
      hospital: {
        id: hospital._id,
        name: hospital.name,
        email: hospital.email
      }
    });

  } catch (error) {
    res.status(500).json({ message: 'Server error during login' });
  }
});

module.exports = router;