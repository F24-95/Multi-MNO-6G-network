# AI-Based Dynamic Spectrum Sharing for Multi-MNO 6G Networks

## Overview
A software simulation prototype implementing AI-driven dynamic spectrum sharing across multiple Mobile Network Operators (MNOs) in a simulated 6G environment. The system uses a pipeline of deep learning and classical ML models to sense spectrum occupancy, predict traffic demand, augment rare-event training data, and make fair, constraint-safe channel allocations — all without real 6G hardware or licensed spectrum.

## Project Type
- **Domain:** Machine Learning / AI System
- **Scope:** Software simulation prototype (no real 6G hardware, no licensed spectrum, no live operator network)
- **Stack:** Python, Flask, PostgreSQL, PyTorch + scikit-learn, NumPy, Pandas, HTML/CSS/JavaScript (Chart.js), Git/GitHub
- **Framework:** PyTorch (deep models), scikit-learn (GMM/scalers)
- **Database:** PostgreSQL (experiments/metrics/dashboard), Parquet/NumPy (ML data I/O)

## Team
- **Developer:** Solo (Mohd Faizan)
- **Supervisor:** College supervisor (to confirm interpretation A6 on GAN-based fairness)

## Timeline
- **Duration:** 1–2 months
- **Start:** October 2026
- **Target Completion:** November 2026

## Key Source Documents
- `PROJECT_ARCHITECTURE.pdf` → Full architecture, data contracts, module boundaries, DB schema, API endpoints, layer table
- `STEP_BY_STEP_IMPLEMENTATION_PLAN.pdf` → 29-step executable build plan with tests, checkpoints, decision log, and git strategy

## Core AI Pipeline (fixed algorithms — no substitutions without decision log entry)
```
Simulator → Validation → Preprocessing
    │
(train split) → VAE-GMM → p_busy per channel
    │
    Autoencoder (snapshot incl. p_busy) → latent z_ae
    │
    FeatureAssembler → forecaster windows (L×F) + targets (H×M)
    │
    GAN (minority train windows) → validated synthetic windows
    │
    LSTM/GRU training (real, real+synthetic) → model selection
    │
Decision Engine + Allocation → Metrics → PostgreSQL → Flask → Dashboard
```

## Architecture Constants (from ARCH §4.1)
| Parameter | Value |
|-----------|-------|
| MNOs (M) | 4 |
| Channels (C) | 16 |
| Channel BW | 20 MHz each → 320 MHz pool |
| Spectrum cap per MNO (D_cap) | 160 MHz |
| Time step interval | 5 minutes |
| Simulation duration | 14 days (4,032 steps) |
| Train/Val/Test split | 10/2/2 days |
| Forecast window L | 12 steps look-back |
| Forecast horizon H | 3 steps ahead |
| Feature vector F | 20 features |

## Key Architecture Decisions (from Decision Log)
| ID | Decision |
|----|----------|
| A1/D-01 | VAE-GMM: unsupervised, per (step, channel) vector; `pu_state` used only for evaluation |
| A2/D-02 | Autoencoder runs after VAE-GMM; latent z_ae goes to forecaster |
| A3/D-03 | GAN is offline, trains on minority windows, ablation-gated |
| A4/D-04 | Forecaster: H=3, jointly for all 4 MNOs; output (3,4) MHz |
| A5/D-05 | Decision Engine: deterministic constrained allocator (no RL) |
| A6/D-06 | "GAN-based fairness" = GAN augmentation + engine-level fairness (confirm with supervisor) |
| A7/D-13 | "Hybrid AE" = denoising bottleneck autoencoder |
| A8/D-08 | Decision at t applies to t+1 (no look-ahead) |
| A9/D-07 | PyTorch only; scikit-learn for GMM/scalers |
| A10/D-10 | Parquet/NumPy for ML data; PostgreSQL for results |
| A11 | "Real-time" = step-wise replay on a simulated clock |

## Global Rules (from Implementation Plan §0)
- Algorithms are **fixed**: VAE-GMM, Autoencoder, GAN, LSTM/GRU, AI Decision Engine
- `pu_state` (ground truth) is **never** a model input — evaluation only
- Test split is **never** used for fitting, tuning, or selection
- Results are **only** from actual runs — never hard-coded
- Each step: BUILD → RUN → TEST → VERIFY → SAVE → DOCUMENT → next step

## Technology Choices
- **Python:** 3.10–3.12 (target 3.11)
- **Deep Learning:** PyTorch (CPU build sufficient for a laptop)
- **Charts:** Chart.js (vendored, no CDN)
- **Async experiments:** Background thread (no Celery/Redis — D-11)
- **Environment:** Windows PowerShell, `.venv` virtual environment
- **Project root:** `C:\projects\dss-6g\`

## Milestones
- **M1:** Foundation (Phases 0–3): Environment, structure, data generation, preprocessing
- **M2:** AI Models (Phases 4–7): VAE-GMM, Autoencoder, GAN, LSTM/GRU
- **M3:** Allocation System (Phases 8–12): Metrics, baseline, decision engine, integration, experiment runner
- **M4:** Backend & UI (Phases 13–15): PostgreSQL, Flask API, Dashboard
- **M5:** Validation & Delivery (Phases 16–20): Integration, testing, optimization, final checklist

## Success Criteria
1. All 29 steps complete with their "DONE when" conditions satisfied
2. All checkpoints P0–P18 evidence verified and committed
3. Full test suite green (`pytest -q`)
4. End-to-end pipeline works on a fresh checkout
5. Comparison table for all 9 scenarios × 2 allocators × 2 seeds produced
6. No hard-coded results in code, DB, or dashboard
7. All constraint properties C1–C4 respected (property test: 1,000 random states)
