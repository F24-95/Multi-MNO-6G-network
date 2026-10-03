# Roadmap — AI-Based Dynamic Spectrum Sharing for Multi-MNO 6G Networks

> **Source documents:** `PROJECT_ARCHITECTURE.pdf` and `STEP_BY_STEP_IMPLEMENTATION_PLAN.pdf`
> **29 steps across 20 phases** mapped to 5 GSD milestones.
> Each phase ends with a CHECKPOINT — do not start the next phase before evidence exists.

---

## Milestone 1 — Foundation (Phases 0–3)
> **Goal:** A fully initialized, validated, and preprocessed dataset. Everything downstream depends on this.

### Phase 1: Project Initialization
**Steps:** 01 (Environment), 02 (Project Structure)
**Goal:** Working development environment + folder skeleton with passing imports
**Deliverables:**
- `C:\projects\dss-6g\` git repo, `.venv`, `requirements.txt`, PyTorch CPU build
- PostgreSQL DB (`dss_db`), `.env`, `.gitignore`
- `scripts/health_check.py` → all imports OK, DB OK
- Full package skeleton under `src/dss/` with `__init__.py` files
- `pyproject.toml` + `pip install -e .` working

**UAT Criteria:**
- `python scripts/health_check.py` → no FAIL lines
- `python -c "import dss.common, dss.sensing; print('imports ok')"` → passes
- `git commit` exists: `feat: initialize project`, `feat: add project structure`

**CHECKPOINT P0:** health check passes, imports work, DB exists, committed ✓

---

### Phase 2: Configuration & Common Utilities
**Steps:** 03 (Config, Logging, Seeds, Paths, IDs, DB Connection)
**Goal:** Shared infrastructure — every module uses the same seeds, paths, config, logging, IDs
**Deliverables:**
- `configs/default.yaml`, `configs/scenarios.yaml`, `configs/models.yaml`, `configs/allocation.yaml`
- `common/config.py`, `common/seed.py`, `common/paths.py`, `common/ids.py`, `common/logging_setup.py`
- `database/connection.py` with context manager

**UAT Criteria:**
- `pytest tests/unit/test_config.py tests/unit/test_seed.py tests/unit/test_ids.py tests/unit/test_paths.py tests/unit/test_logging.py tests/unit/test_db_connection.py -v` → all pass
- Same seed → same RNG stream; different scenario indices → different streams

**CHECKPOINT P1:** tests green, committed ✓

---

### Phase 3: Synthetic Data Generator
**Steps:** 04 (Simulator), 05 (Data Validation)
**Goal:** 18 valid Parquet dataset pairs (9 scenarios × 2 seeds), all quality checks green
**Deliverables:**
- `src/dss/data_generation/{simulator,scenarios,pu_model,demand_model,noise_model}.py`
- `scripts/generate_data.py`, `scripts/validate_data.py`
- Data: `data/raw/<dataset_key>/` — 64,512 channel rows + 16,128 demand rows per scenario
- `quality_report.json` per scenario (all checks pass)

**UAT Criteria:**
- `python scripts/generate_data.py --scenario all --seed 42` then `--seed 2026`
- `pytest tests/unit/test_data_generation.py -v` → all pass
- `python scripts/validate_data.py --seed 42` → all checks green

**CHECKPOINT P2:** 18 × 2 Parquet files, quality reports clean, committed ✓

---

### Phase 4: Data Preprocessing
**Steps:** 06 (Preprocessing Pipeline)
**Goal:** Clean, scaled, split, windowed data ready for each model consumer
**Deliverables:**
- `src/dss/preprocessing/{validate,clean,scale,split,windows,feature_assembler}.py`
- `data/processed/` — processed arrays + joblib scalers
- Scaler: train-fitted only; leakage tests pass

**UAT Criteria:**
- `python scripts/preprocess.py --seed 42`
- `pytest tests/unit/test_preprocessing.py -v` → all pass
- No leakage: scaler min/max equal the train min/max
- No `pu_state` or `noise_true_dbm` in any model input column list

**CHECKPOINT P3:** processed files + scalers, tests green, committed ✓

---

## Milestone 2 — AI Models (Phases 4–7)
> **Goal:** All four AI models trained, evaluated, and saved to the model registry.

### Phase 5: VAE-GMM Spectrum Sensing
**Steps:** 07 (VAE), 08 (GMM on Latents), 09 (Sensing Output + Evaluation)
**Goal:** Unsupervised spectrum sensing producing `p_busy ∈ [0,1]` per channel
**Deliverables:**
- `src/dss/sensing/{vae,train,metrics}.py`
- `models/vae_gmm/v001/{vae.pt, gmm.joblib, scaler.joblib, model_card.json}`
- Sensing metrics table: Pd, Pf, ROC-AUC, Brier per scenario and SNR bin

**UAT Criteria:**
- `pytest tests/unit/test_vae.py tests/unit/test_gmm.py tests/unit/test_sensing_model.py -v`
- `python scripts/eval_sensing.py --model v001 --split test --seed 42`
- ROC-AUC > 0.7 on `normal` scenario (sanity gate)
- `p_busy ∈ [0,1]`, output shape (T, 16)

**CHECKPOINT P4:** model files exist, metrics tabulated, sample predictions generated, committed ✓

---

### Phase 6: Autoencoder Feature Extraction
**Steps:** 10 (Autoencoder + FeatureAssembler stub completion)
**Goal:** 64-dim snapshots compressed to 8-dim latents `z_ae` for forecaster features
**Deliverables:**
- `src/dss/feature_extraction/{autoencoder,train,encode}.py`
- `models/ae/v001/`
- `data/features/z_ae_<key>.npy` — shape (T, 8)

**UAT Criteria:**
- `pytest tests/unit/test_autoencoder.py -v` → all pass
- Val MSE < predict-the-mean baseline
- Same snapshot → same `z_ae` (determinism in eval mode)
- AE model save/load round-trip equality

**CHECKPOINT P5:** AE artifacts, latents, reconstruction metrics, tests, committed ✓

---

### Phase 7: GAN Synthetic Data Augmentation
**Steps:** 11 (Forecast Dataset Assembly), 12 (GAN)
**Goal:** Validated synthetic minority-scenario windows added to training set (if ablation passes)
**Deliverables:**
- `src/dss/preprocessing/feature_assembler.py` (completed)
- `data/processed/windows_{train,val,test}.npz`
- `src/dss/augmentation/{gan,train,validate,combine}.py`
- `data/synthetic/gan_v001/{synthetic_windows.npz, quality_report.json}`
- `models/gan/v001/`

**UAT Criteria:**
- `pytest tests/unit/test_feature_assembler.py tests/unit/test_gan.py tests/unit/test_gan_validation.py -v`
- Streaming vs batch FeatureRow equality
- Synthetic set: KS < 0.15 for ≥ 90% features, correlation diff < 0.15, rejection rate < 50%
- Hand-made invalid sample (negative demand) → rejected by validator

**CHECKPOINT P6:** windows, GAN artifacts, quality report (pass/fail), committed ✓

---

### Phase 8: LSTM/GRU Traffic Demand Forecasting
**Steps:** 13 (LSTM), 14 (GRU), 15 (Evaluation + Selection)
**Goal:** Selected forecaster (LSTM or GRU) stored in model registry, beating persistence baseline
**Deliverables:**
- `src/dss/forecasting/{model,datasets,train,service}.py`
- `models/lstm/v001..v003`, `models/gru/v001..v003`
- `experiments/forecast_comparison.json` — mean ± std RMSE per model × data variant
- Selected `is_active` model in registry

**UAT Criteria:**
- `pytest tests/unit/test_forecaster.py tests/unit/test_forecast_service.py -v`
- Forward shape (B, 3, 4), no NaN
- Beats persistence RMSE on validation
- Perturbing data after the window → prediction unchanged (no leakage)
- Service runs in fresh Python process without training code imported

**CHECKPOINT P7:** comparison report, both model types loadable, sample forecasts, committed ✓

---

## Milestone 3 — Allocation System (Phases 8–12)
> **Goal:** Full AI-driven dynamic spectrum allocation pipeline, from metrics to integrated replay.

### Phase 9: Evaluation Metrics & Baseline Allocation
**Steps:** 16 (Metrics Core), 17 (Static Baseline B1)
**Goal:** Pure metric functions tested; static baseline measured as reference
**Deliverables:**
- `src/dss/evaluation/{metrics,evaluator}.py`
- `src/dss/allocation/baselines.py` (B1 static equal split)
- `experiments/<id>/` per scenario — baseline results stored

**UAT Criteria:**
- `pytest tests/unit/test_metrics.py tests/unit/test_baseline.py -v`
- Equal satisfaction → Jain = 1; one MNO gets all → Jain = 1/4
- Static baseline: exactly 4 channels per MNO, no double-assignment

**CHECKPOINT P8:** metric tests, baseline results stored, committed ✓

---

### Phase 10: AI Decision Engine
**Steps:** 18 (Decision Engine)
**Goal:** Deterministic constrained allocator satisfying C1–C4 for all inputs
**Deliverables:**
- `src/dss/decision_engine/{state,targets,fairness,cycle}.py`
- `src/dss/allocation/{availability,assign,validator}.py`
- AllocationDecision dataclass with all ARCH §12.4 fields

**UAT Criteria:**
- `pytest tests/unit/test_decision_engine.py -v` → all 8 controlled cases pass
- **Property test:** 1,000 random `CycleState` inputs → C1–C4 never violated
- "No spectrum" scenario → no crash, status `NO_SPECTRUM`

**CHECKPOINT P9:** engine tests (including property test) green, committed ✓

---

### Phase 11: Dynamic Allocation Integration
**Steps:** 19 (Full Decision Cycle), 20 (Aggregation + Baselines Comparison)
**Goal:** VAE-GMM + AE + LSTM/GRU + engine connected; comparison table produced
**Deliverables:**
- `src/dss/pipeline/inference_pipeline.py`
- `src/dss/evaluation/{aggregate,compare}.py`
- `experiments/comparison_<seed>.json`

**UAT Criteria:**
- `pytest tests/integration/test_cycle.py -v` → 30-step replay completes
- Two replays → identical decisions
- `python scripts/replay.py --scenario high_traffic --seed 42 --split test --allocator dynamic`

**CHECKPOINT P10:** integrated replay on all 9 scenarios, tests green, committed ✓

---

### Phase 12: Experiment Runner
**Steps:** 21 (One-Command Experiments)
**Goal:** Every result is traceable and reproducible via a single command
**Deliverables:**
- `src/dss/pipeline/experiment_runner.py`, `scripts/run_experiment.py`
- Experiment folders: `decisions.parquet`, `metrics.json`, `run.log`, `config.json`

**UAT Criteria:**
- `pytest tests/integration/test_runner.py -v`
- Same command twice → identical `metrics.json` (except timestamps)
- Failed model load → `failed` status file with reason

**CHECKPOINT P11-12:** comparison JSON + reproducible experiment folders, committed ✓

---

## Milestone 4 — Backend & Dashboard (Phases 13–15)
> **Goal:** A fully functional Flask API and real-data dashboard.

### Phase 13: PostgreSQL Database Integration
**Steps:** 22 (Schema, Loaders, Repositories)
**Goal:** All 14 ARCH §5 tables created; data and experiments loadable idempotently
**Deliverables:**
- `src/dss/database/{schema.sql,repositories.py,loader.py}`
- `scripts/init_db.py`, `scripts/load_to_db.py`
- 580,608 channel rows + 145,152 demand rows per seed in DB

**UAT Criteria:**
- `pytest tests/integration/test_db.py -v`
- Loading same experiment twice → no duplicates (idempotent)
- FK violations raise; PK rejects duplicates

**CHECKPOINT P13:** DB initialized, datasets + experiments loaded, tests green, committed ✓

---

### Phase 14: Flask REST API
**Steps:** 23 (API)
**Goal:** All 12 endpoints working and tested
**Deliverables:**
- `backend/app.py`, `backend/api/*.py`, `backend/services/*.py`
- All endpoints tested with Flask test client

**UAT Criteria:**
- `pytest tests/integration/test_api.py -v`
- Status codes: 200/400/404/409/503 in correct scenarios
- `Invoke-RestMethod http://127.0.0.1:5000/api/health` → `{"status":"ok",...}`
- DB down → 503; bad ID → 404; missing model → 409 on POST

**CHECKPOINT P14:** API tests green, manual curl/PowerShell calls work, committed ✓

---

### Phase 15: Interactive Dashboard
**Steps:** 24 (Dashboard)
**Goal:** 12 panels displaying real data from the API, no hard-coded values
**Deliverables:**
- `dashboard/index.html`, `dashboard/css/style.css`, `dashboard/js/{api,charts,app}.js`
- Chart.js vendored in `dashboard/js/vendor/`

**UAT Criteria:**
- All 12 panels render real data
- Switching scenario/experiment updates all panels
- Error state (DB down) and empty state both handled gracefully
- `Select-String -Path dashboard -Pattern "\d{2,}\.\d" -Recurse` → no suspicious literals

**CHECKPOINT P15:** screenshot per panel from real data, no-hardcoded-value check, committed ✓

---

## Milestone 5 — Validation & Delivery (Phases 16–20)
> **Goal:** A clean-checkout reproducible pipeline with a full green test suite and final checklist.

### Phase 16: End-to-End Integration
**Steps:** 25 (End-to-End Run)
**Goal:** Fresh-clone full pipeline run succeeds with no manual edits
**Deliverables:**
- `scripts/train_all.py` (orchestrates VAE-GMM → AE → windows → GAN → LSTM/GRU → select)
- `tests/e2e/` test suite

**UAT Criteria:**
- `pytest tests/e2e -v` → all pass
- DB counts == file counts
- `/compare` output == offline comparison JSON
- Dashboard values == API values

**CHECKPOINT P16:** clean-checkout run succeeds, committed ✓

---

### Phase 17: Full Test Suite Consolidation
**Steps:** 26 (Consolidate Test Suite)
**Goal:** All test types A–I green; `TEST_REPORT.md` generated
**Deliverables:**
- `tests/TEST_REPORT.md` (generated by `pytest --junitxml`)

**UAT Criteria:**
- `pytest -q` → all green
- Test types covered: Unit, Integration, Model, Data Validation, API, Database, Dashboard, E2E, Scenario

**CHECKPOINT P17:** test report saved, all green, committed ✓

---

### Phase 18: Scenario Evaluation
**Steps:** 27 (Scenario Evaluation)
**Goal:** All 9 scenarios × 2 allocators × 2 seeds run and compared
**UAT Criteria:**
- `pytest tests/e2e/test_all_scenarios.py -v`
- Every experiment has status `done`, all metrics non-null
- Every dynamic decision passes C1–C4
- Comparison table visible in dashboard

**CHECKPOINT P18:** scenario results for all combinations, comparison JSON committed ✓

---

### Phase 19: Optimization
**Steps:** 28 (Optimize After Correctness)
**Goal:** Profile-driven performance improvements; identical outputs after pure-perf changes
**Deliverables:**
- `docs/optimization_log.md` (what | why | before | after | measured how)

**UAT Criteria:**
- Full test suite still green after each optimization
- Same decisions on scenario reruns (if change is purely performance-related)

---

### Phase 20: Final Validation & Delivery
**Steps:** 29 (Final Checklist)
**Goal:** All 20 checklist boxes ticked; project ready for submission
**Final Checklist:**
- [ ] Data generation works
- [ ] Data validation works
- [ ] Preprocessing works
- [ ] VAE-GMM works
- [ ] Autoencoder works
- [ ] GAN works (accepted or honestly rejected by gate)
- [ ] LSTM works
- [ ] GRU works
- [ ] Forecasting evaluation works
- [ ] Baseline allocation works
- [ ] AI decision engine works
- [ ] Dynamic allocation works
- [ ] Constraints respected (property test green)
- [ ] Metrics work
- [ ] PostgreSQL works
- [ ] Flask API works
- [ ] Dashboard works
- [ ] End-to-end pipeline works
- [ ] Multiple scenarios work
- [ ] Results reproducible (same seed → same output)
- [ ] Tests pass
- [ ] No hard-coded final results exist

---

## Master Step Order (29 steps)
```
STEP 01 → Environment                         STEP 16 → Allocation metrics core
STEP 02 → Project structure                   STEP 17 → Static baseline
STEP 03 → Configuration + utilities           STEP 18 → Decision engine
STEP 04 → Data generator                      STEP 19 → Dynamic allocation
STEP 05 → Data validation                     STEP 20 → Aggregation + comparison
STEP 06 → Preprocessing                       STEP 21 → Experiment runner
STEP 07 → VAE                                 STEP 22 → PostgreSQL
STEP 08 → GMM                                 STEP 23 → Flask
STEP 09 → VAE-GMM sensing                     STEP 24 → Dashboard
STEP 10 → Autoencoder                         STEP 25 → Integration
STEP 11 → Forecast dataset assembly           STEP 26 → Testing
STEP 12 → GAN                                 STEP 27 → Scenario evaluation
STEP 13 → LSTM                                STEP 28 → Optimization
STEP 14 → GRU                                 STEP 29 → Final validation
STEP 15 → Forecasting evaluation + selection
```

## Git Strategy
- Branch per phase: `phase-1-init`, `phase-4-sensing`, etc.
- Merge after checkpoint evidence exists
- Never commit: `.env`, `data/`, model files
- Tags: `git tag cp-P<n>` at each checkpoint
