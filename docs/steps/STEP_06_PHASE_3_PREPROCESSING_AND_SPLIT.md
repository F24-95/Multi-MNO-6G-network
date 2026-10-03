# 📘 PHASE 3: Preprocessing, Chronological Splitting & Scaling (Step 06)

---

## 1. Step Overview & Objectives (Kya Hua & Kyun Zaroori Tha)
Raw RF data cannot be fed directly into deep learning models without preprocessing. Phase 3 ensures:
1. **Strict Data Leakage Prevention:** Scalers must fit **only** on the training split (Day 0–9). Under no circumstances are validation (Day 10–11) or test (Day 12–13) sets included during fitting.
2. **Missing Value & Sensor Fault Handling:** Robust forward-fill and boundary clipping.
3. **Chronological Splitting:** Preserves the temporal time-series structure of wireless channels.
4. **Sliding Window Generation:** Prepares leak-proof historical look-back windows ($L=12$, i.e., 60 minutes) for forecast horizons ($H=3$, i.e., 15 minutes).

---

## 2. Detailed File Breakdown (Kya, Kyun, Kaise, Kab & Kahan)

### 2.1 Preprocessing Pipeline Modules (`src/dss/preprocessing/`)
- **File:** `src/dss/preprocessing/clean.py`
  - **Kyun banaya:** Cleans outliers, invalid readings, and sensor dropouts without leaking future values.
  - **Kaise work karta hai:**
    - `clean_channel_obs()`: Validates power ranges, applies forward-fill (`ffill`), and backward-fill (`bfill`) for edge cases. Drops any unrepairable nulls.
    - `clean_mno_demand()`: Clips negative values to 0 and caps demands exceeding 160 MHz.

- **File:** `src/dss/preprocessing/split.py`
  - **Kyun banaya:** Divides 14-day sequence chronologically:
    - **Train:** Steps 0 to 2,879 (Days 0–9, 10 days = 71.4%)
    - **Val:** Steps 2,880 to 3,455 (Days 10–11, 2 days = 14.3%)
    - **Test:** Steps 3,456 to 4,031 (Days 12–13, 2 days = 14.3%)
  - **Kaise work karta hai:** `tag_split(df)` appends an explicit `split` category column (`'train' | 'val' | 'test'`) based on `step_idx`.

- **File:** `src/dss/preprocessing/scale.py`
  - **Kyun banaya:** Deep neural networks require zero-mean unit-variance normalized inputs.
  - **Kaise work karta hai:**
    - `SENSING_COLS = ["rx_power_dbm", "noise_est_dbm", "snr_est_db", "interference_dbm"]`
    - `fit_sensing_scaler(train_df)`: Fits `sklearn.preprocessing.StandardScaler` strictly on training rows (`split == 'train'`).
    - `save_scalers()` / `load_scalers()`: Saves fitted scalers to `data/processed/scalers_s<seed>.joblib` using Joblib.

- **File:** `src/dss/preprocessing/windows.py`
  - **Kyun banaya:** Converts sequential series into $(N, L, F)$ sliding windows for LSTM/GRU forecasting.
  - **Kaise work karta hai:**
    - `build_sliding_windows(series, L=12, H=3)`: For each step $t \ge L$, window covers $[t-L, t)$ as input and $[t, t+H)$ as target.
    - `split_windows_by_target_step()`: Ensures window boundaries never cross split partitions (no look-ahead leak).

### 2.2 CLI Execution Script
- **File:** `scripts/preprocess.py`
  - **Kyun banaya:** Orchestrates cleaning, splitting, scaler fitting, and saves processed datasets.
  - **Kaise work karta hai:**
    - Loads raw datasets for all 9 scenarios.
    - Fits scaler on 414,720 training rows ($2,880 \text{ steps} \times 16 \text{ channels} \times 9 \text{ scenarios}$).
    - Saves Parquet files to `data/processed/<scenario>_s<seed>/channel_obs_processed.parquet`.

---

## 3. How to Run & Verify (Kaise Run Karna Hai)

1. **Run Preprocessing Pipeline:**
   ```bash
   # Preprocess seed 42
   .\.venv\Scripts\python scripts/preprocess.py --seed 42

   # Preprocess seed 2026
   .\.venv\Scripts\python scripts/preprocess.py --seed 2026
   ```

2. **Run Preprocessing Unit Tests:**
   ```bash
   .\.venv\Scripts\python -m pytest tests/unit/test_preprocessing.py -v
   ```
   *Result:* All 5 tests passed!
   - `test_cleaning_fault_injected_data`: PASSED
   - `test_no_leakage_in_sensing_columns`: PASSED
   - `test_train_split_and_scaler_no_leakage`: PASSED
   - `test_sliding_windows_count_and_no_leakage`: PASSED
   - `test_scaler_round_trip`: PASSED
