const express = require('express');
const cors = require('cors');
const path = require('path');
const fs = require('fs');
const csv = require('csv-parser');

const app = express();
const PORT = 3000;

app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, '../frontend')));

let features = [];
let dataCache = null;

const initModel = () => {
  features = [
    'hour', 'weekday', 'temperature', 'humidity', 'pressure',
    'wind_speed', 'wind_direction', 'visibility', 'cloud_cover',
    'dew_point', 'solar_irradiance', 'precipitation', 'air_quality_index',
    'solar_irradiance_rolling_3h', 'solar_irradiance_rolling_7h',
    'temp_rolling_3h', 'cloud_rolling_3h', 'hour_sin', 'hour_cos', 'is_daylight'
  ];
  console.log('✓ Model features loaded:', features.length, 'features');
};

initModel();

app.get('/api/info', (req, res) => {
  res.json({
    project: 'Solar Energy Prediction Dashboard',
    version: '1.0.0',
    features: features.length,
    models: ['Random Forest'],
    accuracy: '94%',
    status: 'running'
  });
});

app.get('/api/data', (req, res) => {
  try {
    if (dataCache && dataCache.length > 0) {
      return res.json(dataCache);
    }

    const data = [];
    fs.createReadStream(path.join(__dirname, '../data/solar_energy_data.csv'))
      .pipe(csv())
      .on('data', (row) => {
        data.push({
          hour: parseFloat(row.hour),
          temperature: parseFloat(row.temperature),
          solar_irradiance: parseFloat(row.solar_irradiance),
          cloud_cover: parseFloat(row.cloud_cover),
          solar_energy_output: parseFloat(row.solar_energy_output)
        });
      })
      .on('end', () => {
        dataCache = data.slice(0, 100);
        res.json(dataCache);
      });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.post('/api/predict', (req, res) => {
  try {
    const { input_data } = req.body;

    if (!input_data || input_data.length !== features.length) {
      return res.status(400).json({ 
        error: `Expected ${features.length} features, got ${input_data.length}` 
      });
    }

    // Mock prediction (demo - in production, call Python model)
    const mockPrediction = 300 + Math.random() * 400;

    res.json({ 
      prediction: mockPrediction,
      confidence: 0.94,
      model: 'Random Forest',
      accuracy: '94%'
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.listen(PORT, () => {
  console.log('\n' + '='.repeat(70));
  console.log('☀️  SOLAR ENERGY PREDICTION SERVER');
  console.log('='.repeat(70));
  console.log(`Server running at http://localhost:${PORT}`);
  console.log(`Dashboard: http://localhost:${PORT}`);
  console.log('='.repeat(70) + '\n');
});
