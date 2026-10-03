# Requirements — AI-Based Dynamic Spectrum Sharing for Multi-MNO 6G Networks

## 1. Functional Requirements

### 1.1 Data Generation (FR-DG)
- **FR-DG-01:** Simulate 14-day, 4,032-step spectrum observations for 16 channels and 4 MNOs using a two-state Markov PU model with diurnal demand patterns
- **FR-DG-02:** Support 9 configurable scenarios: `normal`, `low_traffic`, `high_traffic`, `traffic_spike`, `spectrum_shortage`, `uneven_mno_demand`, `noise_uncertainty`, `congestion`, `demand_shift`
- **FR-DG-03:** Produce `ChannelObs` and `MnoDemand` DataFrames with exact column names and dtypes specified in ARCH §2
- **FR-DG-04:** Guarantee reproducibility — same seed always produces identical output; different seeds produce different outputs
- **FR-DG-05:** `pu_state` must be present in simulator output but tagged as evaluation-only (never a model input)

### 1.2 Data Preprocessing (FR-PP)
- **FR-PP-01:** Validate data against quality checks (record counts, value ranges, calibration bands, time continuity, reproducibility)
- **FR-PP-02:** Clean data without leaking validation/test statistics into training (scalers fit on train only)
- **FR-PP-03:** Produce chronological train/val/test split (steps 0–2879 / 2880–3455 / 3456–4031)
- **FR-PP-04:** Build sliding forecast windows: X (B, L=12, F=20) and Y (B, H=3, M=4) without crossing scenario boundaries

### 1.3 Spectrum Sensing — VAE-GMM (FR-SN)
- **FR-SN-01:** Train a Variational Autoencoder on per-channel vectors `[rx_power_dbm, noise_est_dbm, snr_est_db, interference_dbm]` without using `pu_state`
- **FR-SN-02:** Fit a 2-component GMM on VAE latent means to produce `p_busy ∈ [0,1]` per channel per step
- **FR-SN-03:** Output `SensingResult` arrays: `p_busy`, `sensed_state`, `uncertainty`, `recon_error`, shape (T, 16)
- **FR-SN-04:** Sensing evaluation must report Pd, Pf, ROC-AUC, Brier score per scenario and per SNR bin (using `pu_state` as ground truth for evaluation only)

### 1.4 Feature Extraction — Autoencoder (FR-AE)
- **FR-AE-01:** Train a denoising bottleneck autoencoder on spectrum snapshots (64-dim → 8-dim latent → 64-dim reconstruction)
- **FR-AE-02:** AE input includes `p_busy` from VAE-GMM; trains after VAE-GMM completes
- **FR-AE-03:** Produce `z_ae` (T, 8) latent vectors used as features in the forecaster

### 1.5 Synthetic Data Augmentation — GAN (FR-GAN)
- **FR-GAN-01:** Train a vanilla GAN (MLP Generator/Discriminator) on minority-scenario train windows only (offline, training-time only)
- **FR-GAN-02:** Validate synthetic samples: physical range checks, KS test (< 0.15), correlation check (< 0.15), novelty check, rejection rate (< 50%)
- **FR-GAN-03:** Include synthetic windows in training set only if all set-level validation checks pass (ablation-gated)
- **FR-GAN-04:** Augmentation ratio: N_synth = |minority set|; combined `is_synthetic` mask preserved

### 1.6 Demand Forecasting — LSTM/GRU (FR-FC)
- **FR-FC-01:** Implement `Forecaster` class supporting `cell_type='lstm'` and `cell_type='gru'` via a single shared architecture
- **FR-FC-02:** Input: last L=12 `FeatureRow` vectors (12, 20); Output: per-MNO demand forecast for H=3 steps, shape (3, 4) MHz
- **FR-FC-03:** Train each model with 3 random seeds × {real only, real+synthetic} → model selection via val RMSE
- **FR-FC-04:** GAN ablation: use +synthetic only if minority-window val error improves and overall val error worsens ≤ 2%
- **FR-FC-05:** Beat persistence forecast baseline on validation RMSE

### 1.7 AI Decision Engine (FR-DE)
- **FR-DE-01:** Implement a deterministic constrained allocator consuming `p_busy`, predicted demand, quality signals, and fairness history
- **FR-DE-02:** Enforce constraints C1–C4 (no double-assignment, no busy/blocked channel assignment, capacity ≤ pool, min guarantee)
- **FR-DE-03:** Implement weighted max-min fairness with hysteresis and effective weight `w_eff` from satisfaction history
- **FR-DE-04:** Fall back gracefully (previous valid allocation → static split) when all channels are blocked
- **FR-DE-05:** Decision at step t applies to t+1 (no look-ahead cheating)

### 1.8 Allocation Baselines (FR-BL)
- **FR-BL-01:** Implement static equal-split baseline (B1): 4 channels per MNO regardless of sensing or demand
- **FR-BL-02:** (Optional) Reactive baseline (B2) and oracle baseline (B3)

### 1.9 Experiment Runner & Metrics (FR-EX)
- **FR-EX-01:** Compute and store: spectrum utilization, Jain fairness index, congestion rate, QoS rate, collision count, reassignment count, forecast MAE/RMSE/WAPE, sensing Pd/Pf/AUC, AE reconstruction mean/p99
- **FR-EX-02:** Reproducible experiments: same config + seed → identical `metrics.json`
- **FR-EX-03:** Run all 9 scenarios × 2 allocators × 2 seeds and store comparison JSON

### 1.10 PostgreSQL Database (FR-DB)
- **FR-DB-01:** Create 14 tables per ARCH §5 schema: `scenarios`, `mnos`, `channels`, `datasets`, `channel_observations`, `mno_demand`, `model_registry`, `experiments`, `sensing_results`, `predictions`, `allocation_decisions`, `channel_assignments`, `step_metrics`, `experiment_metrics`
- **FR-DB-02:** Loader must be idempotent (ON CONFLICT DO NOTHING)

### 1.11 Flask API (FR-API)
- **FR-API-01:** Implement all 12 endpoints from ARCH §15: `/api/health`, `/api/scenarios`, `/api/models`, `/api/experiments`, experiment detail endpoints, `/api/compare`, and `POST /api/experiments`
- **FR-API-02:** Uniform JSON envelope: `{"status": ..., "data": ... | "error": ...}`
- **FR-API-03:** Background thread for experiment runs with polling support

### 1.12 Dashboard (FR-UI)
- **FR-UI-01:** Implement 12 panels using Chart.js (vendored, no CDN): spectrum heatmap, MNO allocation bars, demand/forecast charts, utilization/fairness/QoS/congestion metrics, comparison table, experiment status
- **FR-UI-02:** No hard-coded numeric values in dashboard JS/HTML — all values come from API
- **FR-UI-03:** Flask serves dashboard at `/`

---

## 2. Non-Functional Requirements

### 2.1 Reproducibility
- **NFR-REP-01:** Same seed → identical outputs at every stage (generator, preprocessing, model training, allocation decisions)
- **NFR-REP-02:** Separate `seed_train=42` and `seed_unseen=2026` for independent evaluation

### 2.2 Modularity
- **NFR-MOD-01:** Each module: `fit`, `transform`/`infer`, `save`, `load` interface
- **NFR-MOD-02:** Modules exchange only typed DataFrames, arrays, or dataclasses — no module reads another's internals
- **NFR-MOD-03:** Training code and inference code are separate files; `FeatureAssembler` is the only shared code

### 2.3 Integrity
- **NFR-INT-01:** `pu_state` must never appear in any model input feature list
- **NFR-INT-02:** No numeric performance results may be hard-coded anywhere in code, DB, or dashboard
- **NFR-INT-03:** Results are reported only from actual runs

### 2.4 Testability
- **NFR-TEST-01:** Each step has a `pytest` test command that must pass before proceeding
- **NFR-TEST-02:** Constraint property test: 1,000 random CycleState inputs — C1–C4 never violated

### 2.5 Performance (for a prototype on a laptop)
- **NFR-PERF-01:** CPU-only PyTorch build (8–16 GB RAM laptop)
- **NFR-PERF-02:** Batch sensing/AE calls with `torch.no_grad()` for inference

### 2.6 Academic Integrity
- **NFR-ACAD-01:** Software simulation only — no claims of real 6G deployment
- **NFR-ACAD-02:** No algorithms beyond VAE-GMM, AE, GAN, LSTM/GRU, decision engine without a (optional) label and decision-log entry

---

## 3. Constraints
- Windows PowerShell as the development shell
- Python 3.10–3.12 (target 3.11)
- No Celery/Redis — background threads for async experiment runs
- Chart.js must be vendored (no CDN)
- Git branching: one branch per phase, merge after checkpoint

---

## 4. Out of Scope
- Real 6G hardware integration
- Licensed spectrum usage
- Live MNO network integration
- Reinforcement Learning allocation engine (listed as optional future work)
- Production deployment / scaling
