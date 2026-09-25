require('dotenv').config();
const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');

const app = express();
app.use(cors());
app.use(express.json());

mongoose.connect(process.env.MONGO_URI)
  .then(() => console.log('MongoDB Connected to SentryGlide Cluster'))
  .catch(err => console.error(err));

const authRoutes = require('./routes/auth');
app.use('/api/auth', authRoutes);

const dustbinRoutes = require('./routes/dustbins');
app.use('/api/dustbins', dustbinRoutes);

app.listen(5000, () => console.log('Server running on port 5000'));