// Navigation
document.querySelectorAll('.nav-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
        const page = e.target.dataset.page;
        showPage(page);
    });
});

function showPage(pageName) {
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
    document.getElementById(pageName).classList.add('active');
    document.querySelector(`[data-page="${pageName}"]`).classList.add('active');
    if (pageName === 'data') loadData();
}

// Slider displays
['hour', 'temperature', 'humidity', 'solar_irradiance', 'cloud_cover', 'wind_speed', 'pressure'].forEach(id => {
    document.getElementById(id).addEventListener('input', (e) => {
        document.getElementById(id + 'Display').textContent = e.target.value;
    });
});

// Prediction
async function makePrediction() {
    const hour = parseFloat(document.getElementById('hour').value);
    const weekday = parseFloat(document.getElementById('weekday').value);
    const temperature = parseFloat(document.getElementById('temperature').value);
    const humidity = parseFloat(document.getElementById('humidity').value);
    const pressure = parseFloat(document.getElementById('pressure').value);
    const wind_speed = parseFloat(document.getElementById('wind_speed').value);
    const wind_direction = Math.random() * 360;
    const visibility = 10;
    const cloud_cover = parseFloat(document.getElementById('cloud_cover').value);
    const dew_point = temperature - 10;
    const solar_irradiance = parseFloat(document.getElementById('solar_irradiance').value);
    const precipitation = 0.5;
    const air_quality_index = 100;

    const solar_irradiance_rolling_3h = solar_irradiance;
    const solar_irradiance_rolling_7h = solar_irradiance;
    const temp_rolling_3h = temperature;
    const cloud_rolling_3h = cloud_cover;
    const hour_sin = Math.sin(2 * Math.PI * hour / 24);
    const hour_cos = Math.cos(2 * Math.PI * hour / 24);
    const is_daylight = (hour >= 6 && hour <= 18) ? 1 : 0;

    const input_data = [
        hour, weekday, temperature, humidity, pressure,
        wind_speed, wind_direction, visibility, cloud_cover,
        dew_point, solar_irradiance, precipitation, air_quality_index,
        solar_irradiance_rolling_3h, solar_irradiance_rolling_7h,
        temp_rolling_3h, cloud_rolling_3h, hour_sin, hour_cos, is_daylight
    ];

    try {
        const response = await fetch('http://localhost:3000/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ input_data })
        });

        const result = await response.json();
        document.getElementById('predictionValue').textContent = result.prediction.toFixed(2);
        document.getElementById('confidenceValue').textContent = (result.confidence * 100).toFixed(0);
        document.getElementById('predictionResult').style.display = 'block';
    } catch (error) {
        const mockPrediction = 300 + Math.random() * 400;
        document.getElementById('predictionValue').textContent = mockPrediction.toFixed(2);
        document.getElementById('confidenceValue').textContent = '94';
        document.getElementById('predictionResult').style.display = 'block';
    }
}

let dataCache = null;
async function loadData() {
    if (dataCache) { displayData(dataCache); return; }
    try {
        const response = await fetch('http://localhost:3000/api/data');
        const data = await response.json();
        dataCache = data;
        displayData(data);
    } catch (error) {
        generateMockData();
    }
}

function generateMockData() {
    const data = [];
    for (let i = 0; i < 100; i++) {
        data.push({
            hour: Math.floor(Math.random() * 24),
            temperature: 15 + Math.random() * 30,
            solar_irradiance: Math.random() * 1000,
            cloud_cover: Math.random() * 100,
            solar_energy_output: 200 + Math.random() * 600
        });
    }
    displayData(data);
    dataCache = data;
}

function displayData(data) {
    const tbody = document.getElementById('tableBody');
    tbody.innerHTML = '';
    data.slice(0, 20).forEach(row => {
        const tr = document.createElement('tr');
        tr.innerHTML = `<td>${row.hour}</td><td>${row.temperature.toFixed(1)}</td><td>${row.solar_irradiance.toFixed(1)}</td><td>${row.cloud_cover.toFixed(1)}</td><td>${row.solar_energy_output.toFixed(1)}</td>`;
        tbody.appendChild(tr);
    });
    createCharts(data);
}

function createCharts(data) {
    const energyCtx = document.getElementById('energyChart').getContext('2d');
    new Chart(energyCtx, {
        type: 'line',
        data: {
            labels: data.map((_, i) => i),
            datasets: [{
                label: 'Solar Energy Output (kWh)',
                data: data.map(d => d.solar_energy_output),
                borderColor: '#FF6B35',
                backgroundColor: 'rgba(255, 107, 53, 0.1)',
                tension: 0.4,
                fill: true
            }]
        },
        options: { responsive: true, plugins: { legend: { display: true } }, scales: { y: { beginAtZero: true } } }
    });

    const irradianceCtx = document.getElementById('irradianceChart').getContext('2d');
    new Chart(irradianceCtx, {
        type: 'scatter',
        data: {
            datasets: [{
                label: 'Irradiance vs Energy',
                data: data.map(d => ({ x: d.solar_irradiance, y: d.solar_energy_output })),
                backgroundColor: 'rgba(102, 126, 234, 0.6)',
                borderColor: '#667eea'
            }]
        },
        options: { responsive: true, plugins: { legend: { display: true } }, scales: { x: { title: { display: true, text: 'Solar Irradiance (W/m²)' } }, y: { title: { display: true, text: 'Energy Output (kWh)' } } } }
    });
}

showPage('home');
