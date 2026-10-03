# 📘 PHASE 1: Project Setup, Architecture & Common Foundations (Steps 01 - 03)

---

## 1. Step Overview & Objectives (Kya Hua & Kyun Zaroori Tha)
Phase 1 establishes the rock-solid production foundation for the entire 6G Multi-MNO Dynamic Spectrum Sharing (DSS) AI system. 
In dynamic spectrum sharing with 4 Mobile Network Operators (MNOs) and 16 channels, reproducibility, leak-proof configurations, isolated paths, unified logging, and single-source-of-truth database connections are strictly mandatory.

---

## 2. Detailed File Breakdown (Kya, Kyun, Kaise, Kab & Kahan)

### 2.1 Configuration & Environment
- **File:** `configs/default.yaml`
  - **Kyun banaya:** Global simulation parameters define the 6G physical & MAC layer bounds (14 days simulation, 5-minute timestep = 4,032 steps, 16 channels of 20 MHz each, total pool $D_{cap} = 160$ MHz, train/val/test splits 10/2/2 days).
  - **Kaise work karta hai:** YAML structure with keys `sim`, `split`, and `windows` ($L=12$ look-back, $H=3$ forecast horizon).
  - **Kab/Kahan use hota hai:** `dss.common.config` is file ko load karke typed dataclasses inject karta hai pure pipeline mein.

- **File:** `configs/scenarios.yaml`
  - **Kyun banaya:** 9 distinct RF traffic & interference operating regimes (normal, low_traffic, high_traffic, traffic_spike, spectrum_shortage, uneven_mno_demand, noise_uncertainty, congestion, demand_shift).
  - **Kaise work karta hai:** Har scenario ke primary user (PU) Markov transition probabilities ($P_{01}, P_{10}$), diurnal multiplier, noise mean/variance, aur MNO traffic spikes define karta hai.

- **File:** `configs/models.yaml` & `configs/allocation.yaml`
  - **Kyun banaya:** Hyperparameters for VAE, GMM, Autoencoder, GAN, LSTM/GRU, and utility weights ($\alpha, \beta, \theta_c, \tau_{avail}$).
  - **Kaise work karta hai:** Strict parameters to guarantee convergence and fairness ($Jain's\ index \ge 0.85$).

### 2.2 Database & Single Source of Truth
- **File:** `src/dss/database/connection.py`
  - **Kyun banaya:** **Zero Code Duplication Principle**. Pure project mein PostgreSQL connection sirf aur sirf isi ek file se hoga. Koi bhi doosri file duplicate `psycopg2.connect` ya hardcoded connection string nahi likhegi.
  - **Kaise work karta hai:**
    - `load_dotenv()` sirf yahan call hota hai project startup pe.
    - `get_db_url()`: Environment variable `DATABASE_URL` read karta hai (`postgresql://postgres:Faizan9517@localhost:5432/Multi-MNO`). Agar missing ho to clear error throw karta hai.
    - `get_connection()`: Active psycopg2 connection return karta hai.
    - `db_session()`: Python `@contextmanager` jo auto-commit karta hai aur exception aane par rollback karke connection gracefully close karta hai.
  - **Kab/Kahan use hota hai:** Health check script, database schema migration, experiment registry, aur downstream API services mein.

### 2.3 Common Utilities (`src/dss/common/`)
- **File:** `src/dss/common/paths.py`
  - **Kyun banaya:** OS-agnostic absolute path resolution (Windows/Linux compatible).
  - **Kaise work karta hai:** `PROJECT_ROOT`, `data_dir("raw" | "processed" | "sensing")`, `model_dir()`, `log_dir()`. Auto directory creation (`mkdir(parents=True, exist_ok=True)`).
- **File:** `src/dss/common/seed.py`
  - **Kyun banaya:** Full algorithmic reproducibility across NumPy, PyTorch CPU, and Python random.
  - **Kaise work karta hai:** `set_all_seeds(seed)` sets global RNG. `rng_for(seed, scenario_index)` generates isolated scenario-specific independent NumPy Generator.
- **File:** `src/dss/common/ids.py`
  - **Kyun banaya:** Unique, traceable hashes for datasets and model versions.
  - **Kaise work karta hai:** `dataset_key(scenario, seed, config_dict)` -> `ds_<scenario>_s<seed>_<hash8>`.
- **File:** `src/dss/common/logging_setup.py`
  - **Kyun banaya:** Unified structured JSON logging to files and clean formatted output to console.
- **File:** `src/dss/common/config.py`
  - **Kyun banaya:** Loads YAML files into immutable `@dataclass(frozen=True)` (`SimConfig`, `SplitConfig`, `AllocationConfig`, `Config`).

---

## 3. How to Run & Verify (Kaise Run Karna Hai)

1. **Verify Environment & DB Connectivity:**
   ```bash
   .\.venv\Scripts\python scripts/health_check.py
   ```
   *Expected output:*
   ```text
   [NUMPY OK], [PANDAS OK], [SKLEARN OK], [TORCH OK], [FLASK OK]
   [DATABASE OK] Connected to: postgresql://postgres:Faizan9517@localhost:5432/Multi-MNO
   ```

2. **Run Phase 1 Unit Tests:**
   ```bash
   .\.venv\Scripts\python -m pytest tests/unit/test_config.py tests/unit/test_paths.py tests/unit/test_seed.py tests/unit/test_ids.py tests/unit/test_logging.py tests/unit/test_db_connection.py -v
   ```
   *Result:* All 13 tests passed!
