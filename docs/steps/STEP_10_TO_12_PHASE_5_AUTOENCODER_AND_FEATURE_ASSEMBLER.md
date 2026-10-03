# 📘 PHASE 5: Autoencoder Feature Extraction & Feature Assembler (Steps 10 - 12)

---

## 1. Step Overview & Objectives (Kya Hua & Kyun Zaroori Tha)
Dynamic spectrum sharing requires forecasting future traffic demand ($H=3$ steps ahead) across $M=4$ MNOs under changing spectrum availability. 
Feeding raw 64-dimensional channel parameters directly into LSTM/GRU forecasters causes severe overfitting and dimensional explosion. 
Phase 5 solves this via a two-stage feature engineering pipeline:
1. **Denoising Bottleneck Autoencoder (Step 10):** Compresses high-dimensional 64-dim spectrum snapshots into compact, robust 8-dimensional latent vectors $z_{ae} \in \mathbb{R}^8$ while filtering RF sensor noise.
2. **Feature Assembler (Step 11):** Integrates $z_{ae}$, normalized MNO demands, network-wide load pressure, spectrum occupancy metrics, and cyclical temporal encodings into an operational **20-dimensional feature vector ($F=20$)** per timestep.
3. **Sliding Window Construction (Step 12):** Prepares leak-proof forecast datasets:
   - Input: $X \in \mathbb{R}^{B \times L=12 \times F=20}$ (past 60 minutes)
   - Target: $Y \in \mathbb{R}^{B \times H=3 \times M=4}$ (future 15 minutes demand in MHz)
   - Splits: Train ($25,794$ windows), Val ($5,058$ windows), Test ($5,058$ windows).

---

## 2. Detailed File Breakdown (Kya, Kyun, Kaise, Kab & Kahan)

### 2.1 Model Architecture (`src/dss/feature_extraction/autoencoder.py`)
- **Kyun banaya:** Compresses the 64-dimensional physical spectrum state into 8 latent features while learning a noise-resistant manifold.
- **Kaise work karta hai:**
  - **Encoder:** $64 \to 32 \to 16 \to 8$ with ReLU activations.
  - **Denoising Mechanism:** During training, adds Gaussian perturbation $\mathcal{N}(0, 0.05^2)$ to inputs and clamps to $[0, 1]$. Loss is calculated against original uncorrupted inputs.
  - **Decoder:** $8 \to 16 \to 32 \to 64$ with final Sigmoid activation (strictly outputs bounded reconstructions in $[0, 1]$).
  - **Deterministic Inference:** In `eval()` mode, input is uncorrupted, and `encode()` produces strictly reproducible $z_{ae}$.
- **Kab/Kahan use hota hai:** Trained once offline on train-split snapshots; used by `encode.py` and `assemble_forecast_windows.py` to extract features.

### 2.2 Snapshot Builder (`src/dss/feature_extraction/snapshots.py`)
- **Kyun banaya:** Converts heterogeneous RF metrics and VAE-GMM sensing outputs into a unified, normalized 64-dim vector.
- **Kaise work karta hai:**
  - Extracts 4 metrics for all 16 channels at time $t$:
    1. $p_{busy}[t, 0..15]$: Occupancy probabilities from Phase 4 VAE-GMM (already $[0, 1]$).
    2. $snr\_est[t, 0..15]$: Normalized to $[0, 1]$ using train-fitted `MinMaxScaler`.
    3. $interference[t, 0..15]$: Normalized to $[0, 1]$ using train-fitted `MinMaxScaler`.
    4. $noise\_est[t, 0..15]$: Normalized to $[0, 1]$ using train-fitted `MinMaxScaler`.
  - Block concatenation: $[16\ p_{busy}, 16\ snr, 16\ int, 16\ noise] \to (T, 64)$.
- **Kab/Kahan use hota hai:** Used by training pipeline (`train.py`) and batch extraction (`encode.py`).

### 2.3 Training Pipeline (`src/dss/feature_extraction/train.py`)
- **Kyun banaya:** Orchestrates leak-proof model training across all 9 operational scenarios.
- **Kaise work karta hai:**
  - `fit_global_snapshot_scaler()` fits `MinMaxScaler` **strictly on Day 0–9 training split** ($25,920$ timesteps = $414,720$ channel observations). Zero validation/test leakage!
  - Trains Autoencoder using Adam ($\text{lr}=10^{-3}$, batch size 256) with early stopping patience of 10.
  - Saves artifacts: `models/ae/v001/autoencoder.pt`, `models/ae/v001/snapshot_scaler.joblib`, and `models/ae/v001/model_card.json`.
  - Computes predict-the-mean baseline MSE for objective benchmarking.

### 2.4 Inference & Latent Caching (`src/dss/feature_extraction/encode.py`)
- **Kyun banaya:** Offline batch inference and online real-time feature extraction.
- **Kaise work karta hai:**
  - `load_autoencoder(version)`: Loads PyTorch weights and fitted scaler.
  - `encode_scenario()`: Evaluates snapshots and returns $(T, 8)$ array $z_{ae}$.
  - `save_z_ae()`: Caches $z_{ae}$ arrays to `data/features/z_ae_<scenario>_s<seed>_<split>.npy`.

### 2.5 Feature Assembler (`src/dss/preprocessing/feature_assembler.py`)
- **Kyun banaya:** Single source of truth for constructing the 20-dimensional feature vector.
- **Kaise work karta hai:**
  Assembles the 20 features in exact order:
  - `z_ae_0` to `z_ae_7` (8 features): Autoencoder latent spectrum representation.
  - `mno_demand_norm_0` to `mno_demand_norm_3` (4 features): Current bandwidth demand per MNO normalized by $D_{cap} = 160$ MHz.
  - `total_demand_norm` (1 feature): Mean normalized demand across all 4 MNOs.
  - `mean_p_busy` (1 feature): Network-wide spectrum congestion index (average $p_{busy}$ across 16 channels).
  - `tod_sin`, `tod_cos`, `is_weekend` (3 features): Cyclical time-of-day encodings and weekend indicator.
  - `demand_mno_std` (1 feature): Standard deviation of demand across MNOs (inequality / skewness indicator).
  - `demand_delta_norm` (1 feature): Aggregate demand change from previous timestep ($D_t - D_{t-1}$).
  - `high_demand_flag` (1 feature): Binary overload indicator ($1.0$ if total demand $> 50\%$ capacity).
- **Streaming vs Batch Parity:**
  - `assemble_batch(...)`: Vectorized batch calculation over entire 14-day sequence.
  - `assemble_row(...)`: Real-time streaming evaluation for step-by-step decision replay.
  - Strictly identical outputs guaranteed by automated unit tests (`atol < 1e-6`).

### 2.6 CLI Scripts (`scripts/`)
- `scripts/train_autoencoder.py`: CLI to train Autoencoder and evaluate against baseline.
- `scripts/run_feature_extraction.py`: Batch runner caching $z_{ae}$ for all scenarios and splits.
- `scripts/assemble_forecast_windows.py`: Assembles and saves sliding windows to `data/processed/windows_{train,val,test}.npz`.

---

## 3. How to Run & Verify (Kaise Run Karna Hai)

1. **Train Denoising Autoencoder:**
   ```bash
   .\.venv\Scripts\python scripts/train_autoencoder.py --seed 42 --version v001 --epochs 100 --batch 256 --lr 0.001
   ```
   *Benchmark Result:*
   - **Baseline (Predict-the-Mean) Val MSE:** `0.057828`
   - **Autoencoder Best Val MSE:** **`0.014074`** (Over 75% error reduction! ✅)

2. **Extract & Cache $z_{ae}$ Latents (27 files):**
   ```bash
   .\.venv\Scripts\python scripts/run_feature_extraction.py --seed 42 --version v001
   ```
   *Outcome:* 27 latent arrays ($T \times 8$) generated in `data/features/`.

3. **Assemble Sliding Forecast Windows:**
   ```bash
   .\.venv\Scripts\python scripts/assemble_forecast_windows.py --seed 42
   ```
   *Output Datasets Generated:*
   - `data/processed/windows_train.npz`: $X \in (25794, 12, 20)$, $Y \in (25794, 3, 4)$
   - `data/processed/windows_val.npz`: $X \in (5058, 12, 20)$, $Y \in (5058, 3, 4)$
   - `data/processed/windows_test.npz`: $X \in (5058, 12, 20)$, $Y \in (5058, 3, 4)$

4. **Run Unit Tests:**
   ```bash
   .\.venv\Scripts\python -m pytest tests/unit/test_autoencoder.py tests/unit/test_feature_assembler.py -v
   ```
   *Result:* All 8 unit tests passed!

5. **Full Project Test Suite:**
   ```bash
   .\.venv\Scripts\python -m pytest tests/ -v
   ```
   *Result:* **47 / 47 tests passed (100%)!**
