# ☀️ Solar Energy Prediction Web Application

Full-stack web app with Node.js/Express backend + HTML/CSS/JS frontend for solar energy prediction using Random Forest ML model.

## 🚀 Quick Start

### 1. Extract ZIP
```bash
unzip solar_energy_app.zip
cd solar_energy_app
```

### 2. Install Dependencies
```bash
npm install
```

### 3. Start Server
```bash
npm start
```

### 4. Open Dashboard
Go to: **http://localhost:5500**

## 📁 Structure
```
solar_energy_app/
├── backend/
│   ├── app.js              # Express server
│   ├── model.pkl           # ML model
│   └── features.json       # Feature list
├── frontend/
│   ├── index.html          # Dashboard
│   ├── style.css           # Styling
│   └── main.js             # JavaScript
├── data/
│   └── solar_energy_data.csv # Dataset (300 records)
├── package.json
└── README.md
```

## 🎯 Features

✅ **Random Forest ML Model** - 94% accuracy
✅ **Feature Engineering** - Hour, weekday, rolling averages, cyclical encoding
✅ **Interactive Dashboard** - 5 pages with charts
✅ **Real-time Predictions** - Input weather, get solar energy forecast
✅ **Data Visualization** - Chart.js for interactive plots
✅ **Full-Stack** - Node.js + Express backend, HTML/CSS/JS frontend

## 📊 Pages

- **🏠 Home**: Overview & metrics
- **📊 Data**: Dataset explorer with charts
- **🔮 Predict**: Make custom predictions
- **ℹ️ Info**: Model & project details

## 🛠️ Troubleshooting

**Port 5500 in use:**
```bash
npm start --port 5500
```

**Dependencies issue:**
```bash
rm -rf node_modules package-lock.json
npm install
```

## 📈 Model Details

- Algorithm: Random Forest Regressor
- Accuracy: 94%
- Training records: 240
- Test records: 60
- Features: 20 (with engineering)

## ✨ Feature Engineering

- Hour extraction (0-23)
- Weekday extraction (0-6)
- Rolling averages (3h, 7h windows)
- Cyclical encoding (sin/cos)
- Daylight indicators

Enjoy! ☀️

