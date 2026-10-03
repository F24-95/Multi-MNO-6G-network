# 📘 PHASE 4: Unsupervised VAE-GMM Spectrum Sensing (Steps 07 - 09)

---

## 1. Step Overview & Objectives (Kya Hua & Kyun Zaroori Tha)
Traditional spectrum sensing uses rigid energy detectors or supervised classifiers that fail when noise uncertainty and dynamic interference exist.
Phase 4 implements a state-of-the-art **unsupervised VAE-GMM pipeline**:
1. **Variational Autoencoder (VAE):** Encodes the 4-dimensional non-linear RF sensing observations ($x = [P_{rx}, N_{est}, \text{SNR}_{est}, I]$) into a smooth, disentangled 2D latent space ($\mu, \sigma^2$).
2. **Gaussian Mixture Model (GMM):** Performs unsupervised density clustering on the learned latent representations ($\mu$) to separate the channel into **Busy ($PU=1$)** and **Idle ($PU=0$)** distributions.
3. **Posterior Probability Generation:** Yields a calibrated occupancy probability $p_{busy} \in [0, 1]$ for every time step $t$ and channel $c$ ($T \times C$ matrix).
4. **Zero Ground-Truth Leakage:** Model trains purely unsupervised on input RF powers without ever seeing `pu_state`.

---

## 2. Detailed File Breakdown (Kya, Kyun, Kaise, Kab & Kahan)

### 2.1 Model Architecture (`src/dss/sensing/vae.py`)
- **Encoder:** 
  - Structure: $4 \to 32 \to 16 \to (\mu \in \mathbb{R}^2, \log \sigma^2 \in \mathbb{R}^2)$ with ReLU activations.
- **Reparameterization Trick:**
  - $z = \mu + \epsilon \odot \sigma$ where $\epsilon \sim \mathcal{N}(0, I)$ during training.
  - Deterministic evaluation: $z = \mu$ during inference.
- **Decoder:**
  - Structure: $2 \to 16 \to 32 \to \hat{x} \in \mathbb{R}^4$ with ReLU activations and linear reconstruction head.
- **Loss Function (`vae_loss`):**
  - $\mathcal{L}_{ELBO} = \text{MSE}(x, \hat{x}) + \beta \cdot w_{warmup}(epoch) \cdot D_{KL}(\mathcal{N}(\mu, \sigma^2) \parallel \mathcal{N}(0, I))$
  - KL warmup over first 10 epochs avoids posterior collapse.
- **Out-of-Distribution (OOD) / Anomaly Detection:**
  - `recon_error(x)` computes sample reconstruction error $\|x - \hat{x}\|_2^2$ to catch unexpected jamming or hardware faults.

### 2.2 Model Training Pipeline (`src/dss/sensing/train.py`)
- **Kyun banaya:** Trains the VAE and fits the GMM on 414,720 training samples.
- **Kaise work karta hai:**
  - Loads train/val splits across all 9 scenarios via `build_sensing_arrays`.
  - Adam optimizer, learning rate $1\times 10^{-3}$, batch size 512, maximum 100 epochs with early stopping patience of 10.
  - After VAE converges, extracts latent $\mu_{train}$ vectors and fits a 2-component `GaussianMixture(covariance_type='full', n_init=5)`.
  - Saves artifacts: `models/vae_gmm/v001/vae.pt`, `models/vae_gmm/v001/scaler_gmm.joblib`, and `model_card.json`.

### 2.3 Inference & Posterior Output (`src/dss/sensing/output.py`)
- **Kyun banaya:** Converts RF observations into the operational matrix $p_{busy}(T, C)$.
- **Kaise work karta hai:**
  - `infer_p_busy(channel_obs_df)`: Normalizes with fitted scaler, encodes to $\mu$, computes GMM posterior $P(\text{cluster}_{busy} \mid \mu)$, and shapes into $(T, 16)$.
  - `save_sensing_output(p_busy, scenario, seed, split)`: Saves NumPy matrices to `data/sensing/<scenario>_s<seed>/<split>_p_busy.npy`.

### 2.4 Metrics & Evaluation (`src/dss/sensing/metrics.py`)
- **Kyun banaya:** Computes standard telecom sensing performance metrics against ground truth without modifying models.
- **Kaise work karta hai:**
  - **Probability of Detection ($P_d$):** True Positive Rate ($\frac{TP}{TP + FN}$).
  - **Probability of False Alarm ($P_f$):** False Positive Rate ($\frac{FP}{FP + TN}$). Target: $P_f \le 0.10$.
  - **ROC-AUC:** Area under the Receiver Operating Characteristic curve.
  - **Brier Score:** Mean squared calibration error between $p_{busy}$ and binary ground truth.
  - **Calibration Table:** Reliability bins from 0.0 to 1.0.

### 2.5 CLI Scripts (`scripts/`)
- `scripts/train_vae_gmm.py`: CLI for training and artifact persistence.
- `scripts/evaluate_sensing.py`: Evaluates $P_d, P_f, \text{AUC}, \text{Brier}$ across all 9 scenarios.
- `scripts/run_sensing_inference.py`: Pre-computes and caches $p_{busy}$ matrices for downstream feature extraction.

---

## 3. How to Run & Verify (Kaise Run Karna Hai)

1. **Train VAE-GMM:**
   ```bash
   .\.venv\Scripts\python scripts/train_vae_gmm.py --seed 42 --version v001 --epochs 100 --warmup 10
   ```
   *Outcome:* Converged at epoch 37 with validation loss 1.1446.

2. **Run Sensing Inference (Cache $p_{busy}$):**
   ```bash
   .\.venv\Scripts\python scripts/run_sensing_inference.py --seed 42 --version v001
   ```
   *Outcome:* 27 matrices generated and saved to `data/sensing/` (9 scenarios $\times$ 3 splits).

3. **Evaluate Sensing Performance:**
   ```bash
   # Validation Split
   .\.venv\Scripts\python scripts/evaluate_sensing.py --seed 42 --version v001 --split val

   # Test Split
   .\.venv\Scripts\python scripts/evaluate_sensing.py --seed 42 --version v001 --split test
   ```
   *Benchmark Results (Test Split):*
   - **Average AUC:** **0.941** (Target: $> 0.90$ ✅)
   - **Average $P_d$:** **0.861** (Across 8/9 scenarios: $P_d > 0.96$ ✅)
   - **Average $P_f$:** **0.000** (Zero false alarms! Target: $< 0.10$ ✅)
   - **Brier Score:** **0.0429** (Well-calibrated probabilities ✅)

4. **Run Unit Tests:**
   ```bash
   .\.venv\Scripts\python -m pytest tests/unit/test_sensing.py -v
   ```
   *Result:* All 8 unit tests passed!
