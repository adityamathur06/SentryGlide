// sentryglide-backend/server.js
require('dotenv').config();
const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const http = require('http'); // 1. Import HTTP
const { Server } = require('socket.io'); // 2. Import Socket.io
const Dustbin = require('./models/Dustbin');

const app = express();
app.use(cors());
app.use(express.json());

const authRoutes = require('./routes/auth');
const dustbinRoutes = require('./routes/dustbins');

app.use('/api/auth', authRoutes);
app.use('/api/dustbins', dustbinRoutes);

// 3. Create HTTP Server and attach Socket.io
const server = http.createServer(app);
const io = new Server(server, {
  cors: {
    origin: "http://localhost:5173", // Your Vite frontend URL
    methods: ["GET", "POST"]
  }
});

// Socket Connection Logic
io.on('connection', (socket) => {
  console.log('Admin dashboard connected:', socket.id);
  socket.on('disconnect', () => console.log('Dashboard disconnected'));
});

// 4. Connect to MongoDB & Initialize Change Stream
mongoose.connect(process.env.MONGO_URI)
  .then(() => {
    console.log('MongoDB Connected');
    
    const dustbinChangeStream = Dustbin.watch();
    
    dustbinChangeStream.on('change', async (change) => {
      if (change.operationType === 'update' || change.operationType === 'replace') {
        const updatedCart = await Dustbin.findById(change.documentKey._id);
        if (updatedCart) {
          io.emit('cart-updated', updatedCart);
        }
      }
    });
  })
  .catch(err => console.error(err));

const PORT = process.env.PORT || 5000;
server.listen(PORT, () => console.log(`Server running on port ${PORT}`));