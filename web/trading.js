// MaleCNS Trading Terminal - Premium 3D Dashboard

// --- State ---
let priceHistory = [];
let equityHistory = [];
let tradeLog = [];
let lastPrice = 0;
let lastEquity = 0;

// --- Three.js Brain Visualization ---
let scene, camera, renderer, brainPoints, brainMaterial;
let brainRotation = { x: 0, y: 0 };
let isDragging = false;
let lastMouse = { x: 0, y: 0 };

function initBrain() {
    const container = document.getElementById('brain-canvas-container');
    const canvas = document.getElementById('brain-canvas');
    const width = container.clientWidth;
    const height = container.clientHeight;

    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0a0f);

    camera = new THREE.PerspectiveCamera(60, width / height, 0.1, 1000);
    camera.position.z = 3;

    renderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(window.devicePixelRatio);

    // Create brain point cloud (166,700 neurons)
    const nNeurons = 166700;
    const positions = new Float32Array(nNeurons * 3);
    const colors = new Float32Array(nNeurons * 3);

    // Generate fly brain shape (ellipsoid with lobes)
    for (let i = 0; i < nNeurons; i++) {
        // Main brain ellipsoid
        const theta = Math.random() * Math.PI * 2;
        const phi = Math.acos(2 * Math.random() - 1);
        const r = 0.8 + Math.random() * 0.2;

        let x = r * Math.sin(phi) * Math.cos(theta);
        let y = r * Math.sin(phi) * Math.sin(theta) * 0.7;
        let z = r * Math.cos(phi) * 0.9;

        // Add visual lobes (larger)
        if (Math.random() < 0.3) {
            const lobeR = 0.3 + Math.random() * 0.15;
            const lobeTheta = Math.random() * Math.PI * 2;
            const lobePhi = Math.acos(2 * Math.random() - 1);
            x = (i < nNeurons / 2 ? -1 : 1) * (0.9 + lobeR * Math.sin(lobePhi) * Math.cos(lobeTheta));
            y = lobeR * Math.sin(lobePhi) * Math.sin(lobeTheta);
            z = lobeR * Math.cos(lobePhi);
        }

        positions[i * 3] = x;
        positions[i * 3 + 1] = y;
        positions[i * 3 + 2] = z;

        // Initial color (dark)
        colors[i * 3] = 0.1;
        colors[i * 3 + 1] = 0.1;
        colors[i * 3 + 2] = 0.15;
    }

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

    brainMaterial = new THREE.PointsMaterial({
        size: 0.008,
        vertexColors: true,
        transparent: true,
        opacity: 0.8,
        sizeAttenuation: true,
    });

    brainPoints = new THREE.Points(geometry, brainMaterial);
    scene.add(brainPoints);

    // Add ambient light
    const ambientLight = new THREE.AmbientLight(0x404040);
    scene.add(ambientLight);

    // Mouse interaction
    canvas.addEventListener('mousedown', (e) => {
        isDragging = true;
        lastMouse = { x: e.clientX, y: e.clientY };
    });

    canvas.addEventListener('mousemove', (e) => {
        if (isDragging) {
            const dx = e.clientX - lastMouse.x;
            const dy = e.clientY - lastMouse.y;
            brainRotation.y += dx * 0.005;
            brainRotation.x += dy * 0.005;
            lastMouse = { x: e.clientX, y: e.clientY };
        }
    });

    canvas.addEventListener('mouseup', () => { isDragging = false; });
    canvas.addEventListener('mouseleave', () => { isDragging = false; });

    // Auto-rotate when not dragging
    function animate() {
        requestAnimationFrame(animate);
        if (!isDragging) {
            brainRotation.y += 0.002;
        }
        brainPoints.rotation.x = brainRotation.x;
        brainPoints.rotation.y = brainRotation.y;
        renderer.render(scene, camera);
    }
    animate();

    // Handle resize
    window.addEventListener('resize', () => {
        const w = container.clientWidth;
        const h = container.clientHeight;
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
        renderer.setSize(w, h);
    });
}

function updateBrainActivity(rates) {
    if (!brainPoints || !rates) return;

    const colors = brainPoints.geometry.attributes.color;
    const n = colors.count;

    // Map rates to colors
    for (let i = 0; i < n; i++) {
        // Use activity pattern based on rates
        const activity = rates ? rates[i % rates.length] : 0;
        const intensity = Math.min(1, Math.max(0, activity));

        // Color gradient: dark -> blue -> cyan -> white
        if (intensity < 0.3) {
            colors.array[i * 3] = 0.1 + intensity * 0.5;
            colors.array[i * 3 + 1] = 0.1 + intensity * 0.5;
            colors.array[i * 3 + 2] = 0.15 + intensity;
        } else if (intensity < 0.7) {
            const t = (intensity - 0.3) / 0.4;
            colors.array[i * 3] = 0.25 + t * 0.2;
            colors.array[i * 3 + 1] = 0.25 + t * 0.75;
            colors.array[i * 3 + 2] = 0.45 + t * 0.55;
        } else {
            const t = (intensity - 0.7) / 0.3;
            colors.array[i * 3] = 0.45 + t * 0.55;
            colors.array[i * 3 + 1] = 1.0;
            colors.array[i * 3 + 2] = 1.0;
        }
    }
    colors.needsUpdate = true;
}

// --- Charts ---
let priceChart, equityChart;

function initCharts() {
    // Price Chart
    const priceCtx = document.getElementById('price-chart').getContext('2d');
    priceChart = new Chart(priceCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Price',
                data: [],
                borderColor: '#4a9eff',
                backgroundColor: 'rgba(74, 158, 255, 0.1)',
                borderWidth: 1.5,
                fill: true,
                tension: 0.1,
                pointRadius: 0,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            plugins: {
                legend: { display: false },
                tooltip: { enabled: false }
            },
            scales: {
                x: { display: false },
                y: {
                    display: true,
                    grid: { color: 'rgba(255,255,255,0.03)' },
                    ticks: {
                        color: '#555570',
                        font: { size: 9, family: 'monospace' },
                        maxTicksLimit: 5
                    }
                }
            }
        }
    });

    // Equity Chart
    const equityCtx = document.getElementById('equity-chart').getContext('2d');
    equityChart = new Chart(equityCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Equity',
                data: [],
                borderColor: '#ffd700',
                backgroundColor: 'rgba(255, 215, 0, 0.1)',
                borderWidth: 1.5,
                fill: true,
                tension: 0.1,
                pointRadius: 0,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            plugins: {
                legend: { display: false },
                tooltip: { enabled: false }
            },
            scales: {
                x: { display: false },
                y: {
                    display: true,
                    grid: { color: 'rgba(255,255,255,0.03)' },
                    ticks: {
                        color: '#555570',
                        font: { size: 9, family: 'monospace' },
                        maxTicksLimit: 4
                    }
                }
            }
        }
    });
}

function updateCharts(frame) {
    // Price chart
    priceHistory.push(frame.price);
    if (priceHistory.length > 200) priceHistory.shift();

    priceChart.data.labels = priceHistory.map((_, i) => i);
    priceChart.data.datasets[0].data = priceHistory;
    priceChart.update('none');

    // Equity chart
    equityHistory.push(frame.equity);
    if (equityHistory.length > 200) equityHistory.shift();

    equityChart.data.labels = equityHistory.map((_, i) => i);
    equityChart.data.datasets[0].data = equityHistory;
    equityChart.update('none');
}

// --- Order Book ---
function updateOrderBook(frame) {
    const asksEl = document.getElementById('orderbook-asks');
    const bidsEl = document.getElementById('orderbook-bids');

    // Generate simulated order book around current price
    let asksHtml = '';
    let bidsHtml = '';

    for (let i = 5; i >= 1; i--) {
        const askPrice = frame.ask + i * 0.05;
        const askSize = Math.floor(Math.random() * 500) + 100;
        asksHtml += `<div class="order-row ask"><span>${askPrice.toFixed(2)}</span><span>${askSize}</span></div>`;
    }

    for (let i = 1; i <= 5; i++) {
        const bidPrice = frame.bid - i * 0.05;
        const bidSize = Math.floor(Math.random() * 500) + 100;
        bidsHtml += `<div class="order-row bid"><span>${bidPrice.toFixed(2)}</span><span>${bidSize}</span></div>`;
    }

    asksEl.innerHTML = asksHtml;
    bidsEl.innerHTML = bidsHtml;

    document.getElementById('bid-value').textContent = frame.bid.toFixed(2);
    document.getElementById('ask-value').textContent = frame.ask.toFixed(2);
    document.getElementById('spread-value').textContent = (frame.spread * 10000).toFixed(1) + ' bps';
}

// --- UI Updates ---
function updateUI(frame) {
    // Price
    const priceEl = document.getElementById('price-value');
    priceEl.textContent = frame.price.toFixed(2);

    const changeEl = document.getElementById('price-change');
    const changePct = frame.price_change * 100;
    changeEl.textContent = (changePct >= 0 ? '+' : '') + changePct.toFixed(3) + '%';
    changeEl.className = 'price-change ' + (changePct >= 0 ? 'positive' : 'negative');

    // Portfolio
    document.getElementById('equity-value').textContent = '$' + frame.equity.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    document.getElementById('position-value').textContent = frame.position.toFixed(0) + ' shares';
    document.getElementById('cash-value').textContent = '$' + frame.cash.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 });

    const ddEl = document.getElementById('drawdown-value');
    ddEl.textContent = (frame.drawdown * 100).toFixed(2) + '%';

    const maxDdEl = document.getElementById('max-dd-value');
    maxDdEl.textContent = (frame.max_drawdown * 100).toFixed(2) + '%';

    document.getElementById('trades-value').textContent = frame.total_trades;

    // Regime
    document.getElementById('regime-value').textContent = frame.regime;

    // Indicators
    const rsiVal = (frame.rsi + 1) / 2 * 100; // Convert from [-1,1] to [0,100]
    document.getElementById('rsi-value').textContent = rsiVal.toFixed(1);
    document.getElementById('rsi-gauge').style.width = rsiVal + '%';

    document.getElementById('macd-value').textContent = (frame.macd * 100).toFixed(4);

    const bbVal = (frame.bb_position + 0.5) * 100;
    document.getElementById('bb-value').textContent = bbVal.toFixed(1);
    document.getElementById('bb-gauge').style.width = bbVal + '%';

    document.getElementById('atr-value').textContent = (frame.volatility * 100).toFixed(2) + '%';

    // Clock
    const now = new Date();
    document.getElementById('clock').textContent = now.toTimeString().split(' ')[0];
}

// --- Trade Log ---
function updateTradeLog(frame) {
    const logEl = document.getElementById('trade-log');

    // Detect trades (simplified - check for position changes)
    // In a real implementation, this would come from the server

    // Update neuron count
    if (frame.brain_rates) {
        document.getElementById('neuron-count').textContent = frame.brain_rates.length + ' neurons';
    }
}

// --- SSE Connection ---
function connectStream() {
    const source = new EventSource('/stream');

    source.onmessage = (event) => {
        try {
            const frame = JSON.parse(event.data);

            updateUI(frame);
            updateCharts(frame);
            updateOrderBook(frame);
            updateBrainActivity(frame.brain_rates);
            updateTradeLog(frame);

        } catch (e) {
            console.error('Failed to parse frame:', e);
        }
    };

    source.onerror = (err) => {
        console.error('SSE error:', err);
    };
}

// --- Initialize ---
window.addEventListener('DOMContentLoaded', () => {
    initBrain();
    initCharts();
    connectStream();
});
