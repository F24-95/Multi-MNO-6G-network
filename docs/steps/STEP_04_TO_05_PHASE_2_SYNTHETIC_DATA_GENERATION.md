# 📘 PHASE 2: Synthetic 6G RF & MNO Demand Data Generation (Steps 04 - 05)

---

## 1. Step Overview & Objectives (Kya Hua & Kyun Zaroori Tha)
Phase 2 generates physically realistic synthetic data for dynamic spectrum sharing without ground-truth leakage into sensing features.
Simulation details:
- **Duration:** 14 continuous days at 5-minute sampling interval ($T = 4,032$ timesteps).
- **RF Spectrum:** $C = 16$ licensed sub-bands, each 20 MHz. Total spectrum = 320 MHz, shared capacity $D_{cap} = 160$ MHz.
- **MNO Ecosystem:** $M = 4$ operators with distinct user densities and diurnal active traffic profiles.
- **Seeds:** Seed 42 (training/validation/test) and Seed 2026 (unseen deployment/robustness test).
- **Total Datasets:** 9 scenarios $\times$ 2 seeds = 18 Parquet datasets.

---

## 2. Detailed File Breakdown (Kya, Kyun, Kaise, Kab & Kahan)

### 2.1 Physics & Traffic Simulators (`src/dss/data_generation/`)
- **File:** `src/dss/data_generation/pu_model.py`
  - **Kyun banaya:** Primary Users (PUs) have incumbent rights. When a PU transmits, the channel is busy and secondary users (MNOs) must yield to avoid collision.
  - **Kaise work karta hai:** 
    - 2-state Discrete-Time Markov Chain ($S \in \{0, 1\}$).
    - Diurnal modulation: Transition probability $P_{01}(t) = P_{base} \cdot [1 + A \cdot \sin(2\pi (t - 6)/288)]$.
    - Returns binary occupancy array `pu_occupancy` of shape $(T, C)$.

- **File:** `src/dss/data_generation/noise_model.py`
  - **Kyun banaya:** Models realistic wireless propagation, thermal noise, shadowing, and interference.
  - **Kaise work karta hai:**
    - Thermal noise floor $N_0 \approx -100$ dBm + log-normal shadowing variation.
    - When $PU=1$: Received signal power $P_{rx} = P_{tx} - PL(d) + \mathcal{N}(0, \sigma^2)$, with $P_{rx} \in [-85, -60]$ dBm.
    - When $PU=0$: $P_{rx} = \text{noise floor} \in [-105, -95]$ dBm.
    - Estimates: $\text{noise\_est\_dbm} = \text{noise\_true} + \mathcal{N}(0, \sigma_{err}^2)$, $\text{snr\_est\_db} = P_{rx} - \text{noise\_est\_dbm}$.

- **File:** `src/dss/data_generation/demand_model.py`
  - **Kyun banaya:** Generates bandwidth demand (in MHz) for $M=4$ MNOs over time.
  - **Kaise work karta hai:**
    - Active users $U_m(t) \sim \text{Poisson}(\lambda_m(t))$ where $\lambda_m(t)$ follows diurnal curve + occasional Poisson traffic spikes.
    - Required throughput $R_m(t) = U_m(t) \times r_{user}$.
    - Bandwidth demand $D_m(t) = R_m(t) / (\text{SE}_{ref} \cdot \log_2(1 + \text{SNR}))$, clamped to maximum allowed capacity.
    - Temporal correlation injected via AR(1) autoregressive noise: $\epsilon_t = \phi \epsilon_{t-1} + \sqrt{1-\phi^2}\eta_t$.

- **File:** `src/dss/data_generation/scenarios.py`
  - **Kyun banaya:** Parameter sets for all 9 scenarios:
    1. `normal`: Standard duty cycle (30-40%) and standard Poisson traffic.
    2. `low_traffic`: Minimal PU presence (10-15%), light MNO demand.
    3. `high_traffic`: Heavy PU occupancy (60-75%), heavy MNO demand.
    4. `traffic_spike`: Sudden, intense burst of users for specific MNOs.
    5. `spectrum_shortage`: Extreme PU occupancy (>80%), scarce free channels.
    6. `uneven_mno_demand`: MNO 0 and 1 have 5x the demand of MNO 2 and 3.
    7. `noise_uncertainty`: High noise estimation variance ($\sigma = 3.5$ dB).
    8. `congestion`: Near-capacity aggregate demand across all MNOs simultaneously.
    9. `demand_shift`: Dynamic shift in peak traffic timing across MNOs.

- **File:** `src/dss/data_generation/simulator.py`
  - **Kyun banaya:** Master simulation runner combining PU, noise, and demand models.
  - **Kaise work karta hai:**
    - Generates 64,512 channel observation rows per scenario ($4,032 \times 16$).
    - Generates 16,128 MNO demand rows per scenario ($4,032 \times 4$).
    - Saves clean Parquet files to `data/raw/ds_<scenario>_s<seed>_<hash>/`.

### 2.2 Validation Gatekeeper
- **File:** `src/dss/preprocessing/validate.py`
  - **Kyun banaya:** 8 automated sanity checks before any data enters the ML pipeline.
  - **Kaise work karta hai:** Checks non-empty files, exact expected row counts, no unhandled NaNs, physical power ranges ($[-130, -40]$ dBm), demand $\ge 0$, and timestamp continuity.

---

## 3. How to Run & Verify (Kaise Run Karna Hai)

1. **Generate Raw Datasets:**
   ```bash
   # Generate for seed 42 (training/val/test)
   .\.venv\Scripts\python scripts/generate_data.py --seed 42

   # Generate for seed 2026 (unseen robustness evaluation)
   .\.venv\Scripts\python scripts/generate_data.py --seed 2026
   ```

2. **Run Validation Gate:**
   ```bash
   .\.venv\Scripts\python scripts/validate_data.py --seed 42
   .\.venv\Scripts\python scripts/validate_data.py --seed 2026
   ```
   *Result:* All 9 scenarios for both seeds passed all 8 gate checks!

3. **Run Unit Tests:**
   ```bash
   .\.venv\Scripts\python -m pytest tests/unit/test_data_generation.py tests/data/test_dataset_validation.py -v
   ```
   *Result:* 13 tests passed!
