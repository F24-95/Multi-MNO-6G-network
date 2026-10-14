# 📡 6G Multi-MNO Dynamic Spectrum Sharing (DSS)

An AI-driven dynamic spectrum sharing engine that pools **1,000 MHz** of RF spectrum across **16 channels** (62.5 MHz each) and dynamically allocates bandwidth slices among **4 Mobile Network Operators (Airtel 6G, Jio 6G, Vi 6G, BSNL 6G)** while guaranteeing zero interference to incumbent Primary Users (PU).

---

## 1. 📌 Project Overview

- **Problem**: In traditional static slicing, operators are locked to fixed spectrum bands. During off-peak hours, spectrum sits idle (<35% utilization), while surging operators face congestion and dropped calls.
- **Solution**: Dynamic Spectrum Sharing (DSS) pools the entire 1,000 MHz spectrum. An AI decision loop senses channel availability, predicts operator traffic demand, and reallocates bandwidth slices in real time (every 3 seconds).
- **Core Entities**:
  - **Primary Users (PU)**: High-priority incumbents (Radar/Satellite) with absolute transmission rights.
  - **4 Secondary MNOs**: Airtel 6G, Jio 6G, Vi 6G, and BSNL 6G competing for available secondary bandwidth.
  - **Physical Spectrum**: 16 RF channels $\times$ 62.5 MHz = 1,000 MHz total licensed pool.

---

## 2. 🧠 Core Algorithms: What, Why & When

| Algorithm | WHAT is it? | WHY use it? | WHEN is it used? |
| :--- | :--- | :--- | :--- |
| **VAE (Variational Autoencoder)** | Deep generative neural network mapping 4 RF features $[P_{rx}, N_{est}, I, \text{SNR}]$ into a 2D Gaussian latent distribution $\mathcal{N}(\mu, \sigma^2)$. | Traditional energy detection fails under noise uncertainty ($\sigma_u \le 3.0\text{ dB}$). VAE learns noise-robust representations without manual labels. | **Spectrum Sensing Layer**: Senses RF channel conditions on every observation step. |
| **GMM (Gaussian Mixture Model)** | 2-component Bayesian clustering model ($\mathcal{N}_{\text{free}}, \mathcal{N}_{\text{busy}}$). | Provides a soft posterior probability $P_{\text{busy}} \in [0.0, 1.0]$ instead of rigid binary decisions. | **Occupancy Classifier**: Applied to VAE latent vectors to classify PU activity ($P_{\text{busy}} \ge 0.5 \implies$ Occupied). |
| **Bottleneck Autoencoder** | 4-layer neural network compressing $16 \times 4 = 64$ channel features into an 8-dim vector. | Reduces dimensional complexity and eliminates cross-channel correlation for fast edge dispatch. | **Feature Extraction Layer**: Compresses multi-channel snapshots before decision-making. |
| **LSTM / GRU Forecaster** | Recurrent neural networks modeling sequential time-series traffic patterns. | Prevents allocation lag. Forecasts traffic spikes 12 steps ahead using past 12 steps, enabling proactive slicing. | **Traffic Forecasting Layer**: Predicts near-future bandwidth demands for all 4 MNOs. |
| **Constrained Optimizer (C1–C4)** | Deterministic game-theoretic resource allocation algorithm. | Pure AI/RL can violate physical safety rules. The optimizer mathematically guarantees strict physical bounds. | **Decision Engine**: Calculates exact MHz slices for all 4 operators every 3 seconds. |

### The 4 Physical Constraints (C1–C4):
- **C1 (PU Protection)**: If $P_{\text{busy}} \ge 0.50$, allocated bandwidth = 0 MHz (zero interference to radar/satellite).
- **C2 (Total Pool Cap)**: $\sum_{m=1}^4 B_m \le B_{\text{available}} \le 1,000\text{ MHz}$.
- **C3 (Demand Fulfillment)**: Bandwidth allocated to an MNO cannot exceed its requested demand ($B_m \le D_m$).
- **C4 (Fairness Guarantee)**: Jain's Fairness Index $J \ge 0.85$ to ensure no operator (e.g., BSNL) is starved.

---

## 3. 🏗️ System Architecture

```mermaid
flowchart LR
    RF["📡 16 RF Channels\n(1,000 MHz Pool)"] --> SENSE["🧠 VAE + GMM\nSensing Engine"]
    SENSE -->|"P_busy < 0.5 (Free Channels)"| OPT["⚡ C1–C4 Constrained\nOptimizer"]
    MNO["👥 4 MNOs Traffic\n(Airtel, Jio, Vi, BSNL)"] --> FORECAST["📈 LSTM / GRU\nDemand Forecast"]
    FORECAST -->|"Predicted Demands"| OPT
    OPT ==>|"Dynamic MHz Slices"| MNO
    OPT --> STREAM["🗄️ Real-Time Stream Worker\n(Persists Every 3s)"]
    STREAM --> DB[("🐘 PostgreSQL\nMulti-MNO DB")]
    DB --> API["🌐 Flask API\n(Port 5000)"]
    API --> UI["📊 Live Chart.js\nDashboard"]
```

---

## 4. 🚀 Step-by-Step Setup & Run Guide

### Step 1: Clone & Enter Directory
```bash
cd "c:\Users\mohd faizan\Documents\college project\Multi-MNO 6G network"
```

### Step 2: Create & Activate Virtual Environment
- **Windows (PowerShell)**:
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  ```
- **Linux / macOS (Bash)**:
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Configure Database (`.env`)
Create or edit `.env` in the project root:
```ini
DATABASE_URL=postgresql://postgres:Faizan9517@localhost:5432/Multi-MNO
PORT=5000
FLASK_ENV=development
```
*(The system automatically creates required tables `channel_obs` and `mno_demand` upon startup).*

### Step 5: Run Automated Tests
Verify that all 10 unit and pipeline tests pass:
```bash
python -m pytest
```

### Step 6: Start Server & Real-Time Stream
```bash
python src/dss/api/app.py
```

### Step 7: Open Dashboard
Open your browser and navigate to:
👉 **`http://127.0.0.1:5000`**

The dashboard will stream real-time RF channel sensing, MNO slicing, AI convergence curves, and PostgreSQL persistence every 3 seconds.
