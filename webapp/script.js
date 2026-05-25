// ------------------------------------
// Floating Lines Background
// ------------------------------------
class FloatingLines {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) return;
        this.ctx = this.canvas.getContext('2d');
        
        // Props from user
        this.colors = ["#E945F5", "#2F4BC0", "#E945F5"];
        this.animationSpeed = 1;
        this.bendRadius = 300; // mapped to pixel space
        this.bendStrength = -0.5;
        this.mouseDamping = 0.05;
        this.parallaxStrength = 0.2;
        
        this.linesCount = 6;
        this.pointsPerLine = 50;
        this.lines = [];
        this.time = 0;
        
        const parent = this.canvas.parentElement;
        this.mouse = { x: parent.offsetWidth / 2, y: parent.offsetHeight / 2, targetX: parent.offsetWidth / 2, targetY: parent.offsetHeight / 2 };
        
        this.resize();
        window.addEventListener('resize', () => this.resize());
        
        // Track mouse local to banner
        window.addEventListener('mousemove', (e) => {
            const rect = this.canvas.getBoundingClientRect();
            this.mouse.targetX = e.clientX - rect.left;
            this.mouse.targetY = e.clientY - rect.top;
        });
        
        this.initLines();
        this.animate();
    }
    
    resize() {
        const parent = this.canvas.parentElement;
        if (!parent) return;
        this.canvas.width = parent.offsetWidth;
        this.canvas.height = parent.offsetHeight;
        this.initLines(); // Recalculate aspect curves
    }
    
    initLines() {
        this.lines = [];
        for (let i = 0; i < this.linesCount; i++) {
            this.lines.push({
                offset: (i / this.linesCount) * Math.PI * 2,
                yBase: this.canvas.height * (0.3 + (i / this.linesCount) * 0.4)
            });
        }
    }
    
    animate() {
        this.time += 0.005 * this.animationSpeed;
        
        // Mouse damping (easing)
        this.mouse.x += (this.mouse.targetX - this.mouse.x) * this.mouseDamping;
        this.mouse.y += (this.mouse.targetY - this.mouse.y) * this.mouseDamping;
        
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        // Setup Gradient
        let gradient = this.ctx.createLinearGradient(0, 0, this.canvas.width, 0);
        gradient.addColorStop(0, this.colors[0]);
        gradient.addColorStop(0.5, this.colors[1]);
        gradient.addColorStop(1, this.colors[2]);
        
        this.ctx.lineWidth = 2;
        this.ctx.strokeStyle = gradient;
        
        this.lines.forEach((line, index) => {
            this.ctx.beginPath();
            
            // Parallax shift
            const parallaxX = (this.mouse.x - this.canvas.width / 2) * this.parallaxStrength * ((index + 1) / this.linesCount);
            const parallaxY = (this.mouse.y - this.canvas.height / 2) * this.parallaxStrength * ((index + 1) / this.linesCount);
            
            for (let i = 0; i <= this.pointsPerLine; i++) {
                let x = (i / this.pointsPerLine) * this.canvas.width;
                // Base sine wave
                let y = line.yBase + Math.sin(x * 0.005 + this.time + line.offset) * 100;
                
                // Add Parallax
                x += parallaxX;
                y += parallaxY;
                
                // Bend interaction
                let dx = this.mouse.x - x;
                let dy = this.mouse.y - y;
                let dist = Math.sqrt(dx * dx + dy * dy);
                
                if (dist < this.bendRadius) {
                    let force = (this.bendRadius - dist) / this.bendRadius;
                    let bendDist = force * 100 * this.bendStrength;
                    y += (dy / dist) * bendDist;
                }
                
                if (i === 0) {
                    this.ctx.moveTo(x, y);
                } else {
                    // Smooth curve approximation
                    this.ctx.lineTo(x, y);
                }
            }
            this.ctx.stroke();
        });
        
        requestAnimationFrame(() => this.animate());
    }
}

// Instantiate Background
new FloatingLines('floatingLinesBackground');


// ------------------------------------
// Navigation Logic
// ------------------------------------
function switchTab(tabId, btnElement) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Remove active state from all buttons
    document.querySelectorAll('.nav-btn').forEach(btn => {
        btn.classList.remove('active');
    });

    // Show active tab
    document.getElementById(tabId).classList.add('active');
    
    // Add active state to clicked button
    if(btnElement) btnElement.classList.add('active');
}

Chart.defaults.color = '#a0a0b0';
Chart.defaults.font.family = "'Inter', sans-serif";

// Initialize charts directly
document.addEventListener("DOMContentLoaded", () => {
    
    // Viewership Chart
    const ctxViewer = document.getElementById('viewershipChart');
    if (ctxViewer) {
        new Chart(ctxViewer.getContext('2d'), {
            type: 'bar',
            data: {
                labels: ['CSK vs MI', 'RCB vs KKR', 'RR vs PBKS', 'DC vs LSG', 'GT vs SRH'],
                datasets: [
                    { type: 'line', label: 'Projected Model Trend', data: [70, 65, 45, 55, 50], borderColor: '#00C9FF', borderWidth: 2, fill: false, tension: 0.3 },
                    { type: 'bar', label: 'Actual Historical Average', data: [65, 60, 42, 51, 48], backgroundColor: 'rgba(255, 65, 108, 0.7)', borderRadius: 5 }
                ]
            },
            options: { responsive: true, maintainAspectRatio: false, scales: { y: { grid: { color: 'rgba(255, 255, 255, 0.05)' } }, x: { grid: { display: false } } } }
        });
    }

    // Ticket Radar Chart
    const ctxTicket = document.getElementById('ticketChart');
    if (ctxTicket) {
        new Chart(ctxTicket.getContext('2d'), {
            type: 'radar',
            data: {
                labels: ['Wankhede', 'Chinnaswamy', 'Eden Gardens', 'Chepauk', 'Modi Stadium'],
                datasets: [{ label: 'Weekend Match Occupancy', data: [98, 95, 88, 96, 75], backgroundColor: 'rgba(248, 181, 0, 0.2)', borderColor: '#F8B500', pointBackgroundColor: '#F8B500' }, 
                           { label: 'Weekday Match Occupancy', data: [82, 85, 75, 88, 60], backgroundColor: 'rgba(0, 201, 255, 0.2)', borderColor: '#00C9FF', pointBackgroundColor: '#00C9FF' }]
            },
            options: { responsive: true, maintainAspectRatio: false, scales: { r: { grid: { color: 'rgba(255, 255, 255, 0.1)' }, angleLines: { color: 'rgba(255, 255, 255, 0.1)' }, ticks: { display: false } } } }
        });
    }

    // Sponsor Chart
    const ctxSponsor = document.getElementById('sponsorChart');
    if (ctxSponsor) {
        new Chart(ctxSponsor.getContext('2d'), {
            type: 'scatter',
            data: { datasets: [{ label: 'Sponsorship Contracts', data: [{x: 5, y: 12}, {x: 10, y: 22}, {x: 15, y: 28}, {x: 20, y: 45}, {x: 25, y: 55}, {x: 35, y: 70}, {x: 45, y: 110}], backgroundColor: '#1E9600', pointRadius: 6 }] },
            options: { responsive: true, maintainAspectRatio: false, scales: { x: { title: { display: true, text: 'Spend (Cr)', color: '#fff' }, grid: { color: 'rgba(255, 255, 255, 0.05)' } }, y: { title: { display: true, text: 'Estimated Value (Cr)', color: '#fff' }, grid: { color: 'rgba(255, 255, 255, 0.05)' } } } }
        });
    }

    // Merch Line
    const ctxMerch = document.getElementById('merchChart1');
    if(ctxMerch) {
        new Chart(ctxMerch.getContext('2d'), {
            type: 'line',
            data: {
                labels: ['Match 1', 'Match 2', 'Match 3', 'Match 4', 'Match 5', 'Match 6'],
                datasets: [{ label: 'Social Buzz Volume', data: [1000, 15000, 20000, 18000, 50000, 80000], borderColor: '#FF4B2B', yAxisID: 'y' }, 
                           { label: 'Sales Revenue (₹)', data: [50000, 60000, 65000, 63000, 120000, 180000], borderColor: '#00C9FF', yAxisID: 'y1' }]
            },
            options: { responsive: true, maintainAspectRatio: false, scales: { y: { type: 'linear', display: true, position: 'left', grid: {color: 'rgba(255,255,255,0.05)'} }, y1: { type: 'linear', display: true, position: 'right', grid: {drawOnChartArea: false} } } }
        });
    }

    const predictBtn = document.getElementById('predictBtn');
    const resultDiv = document.getElementById('predictionResult');

    if (predictBtn) {
        predictBtn.addEventListener('click', () => {
            predictBtn.innerHTML = 'Analyzing...';
            predictBtn.style.opacity = '0.7';
            predictBtn.style.pointerEvents = 'none';
            resultDiv.style.display = 'none';

            setTimeout(() => {
                predictBtn.innerHTML = 'Generate Prediction';
                predictBtn.style.opacity = '1';
                predictBtn.style.pointerEvents = 'auto';

                resultDiv.innerHTML = `✅ <strong>Success!</strong> Model Forecasts ~68.4M Peak Viewership (± 1.2M error bound) for <em>RCB vs MI (Weekend)</em>`;
                resultDiv.style.display = 'block';
            }, 1500);
        });
    }

    // Initialize Lucide Icons
    lucide.createIcons();
});
