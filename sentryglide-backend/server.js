require('dotenv').config();
const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');

// Import the auth routes
const authRoutes = require('./routes/auth');

const app = express();
app.use(cors());
app.use(express.json());

mongoose.connect(process.env.MONGO_URI)
  .then(() => console.log('MongoDB Connected to SentryGlide Cluster'))
  .catch(err => console.error(err));

// Connect the auth routes to the /api/auth endpoint
app.use('/api/auth', authRoutes);

app.listen(5000, () => console.log('Server running on port 5000'));