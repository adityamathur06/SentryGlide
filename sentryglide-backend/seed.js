// sentryglide-backend/seed.js
require('dotenv').config();
const mongoose = require('mongoose');
const Dustbin = require('./models/Dustbin');

mongoose.connect(process.env.MONGO_URI)
  .then(async () => {
    console.log('Connected to MongoDB. Preparing to generate factory dustbins...');
    
    try {
      // Clear the collection first to prevent E11000 duplicate key errors on rerun
      await Dustbin.deleteMany({});
      console.log('Cleared previous dustbin data.');

      const factoryDustbins = [];
      
      for (let i = 1; i <= 50; i++) {
        const formattedNumber = String(i).padStart(3, '0');
        
        factoryDustbins.push({
          serialNumber: `SG-${formattedNumber}`,
          hospitalId: null, // Unowned, factory-fresh
          location: 'Unassigned', // Added to match the new schema
          batteryLevel: 100,
          status: 'Standby',
          capacity: { yellow: 0, red: 0, white: 0, blue: 0 }
        });
      }

      await Dustbin.insertMany(factoryDustbins);
      console.log('Successfully inserted 50 fresh dustbins into the factory cluster!');
    } catch (error) {
      console.error('Error during seeding:', error.message);
    } finally {
      process.exit();
    }
  })
  .catch(err => console.error(err));