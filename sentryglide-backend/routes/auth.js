// routes/auth.js
const express = require('express');
const bcrypt = require('bcrypt');
const jwt = require('jsonwebtoken');
const Hospital = require('../models/Hospital');

const router = express.Router();
const JWT_SECRET = process.env.JWT_SECRET || 'sentryglide_super_secret';

// Register a new Hospital Account
router.post('/register', async (req, res) => {
  try {
    const { name, email, password } = req.body;
    const existingHospital = await Hospital.findOne({ email });
    if (existingHospital) return res.status(400).json({ message: 'Hospital already registered' });

    const hashedPassword = await bcrypt.hash(password, 10);
    const hospital = new Hospital({ name, email, password: hashedPassword });
    await hospital.save();

    res.status(201).json({ message: 'Hospital registered successfully' });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Login and receive JWT
router.post('/login', async (req, res) => {
  try {
    const { email, password } = req.body;
    const hospital = await Hospital.findOne({ email });
    if (!hospital) return res.status(404).json({ message: 'Hospital not found' });

    const isMatch = await bcrypt.compare(password, hospital.password);
    if (!isMatch) return res.status(401).json({ message: 'Invalid credentials' });

    const token = jwt.sign({ id: hospital._id }, JWT_SECRET, { expiresIn: '1d' });
    res.json({ token, hospital: { id: hospital._id, name: hospital.name, email: hospital.email } });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

module.exports = router;