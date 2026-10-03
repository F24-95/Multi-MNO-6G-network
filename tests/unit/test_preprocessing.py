import numpy as np
import pandas as pd
from dss.data_generation.simulator import generate_dataset
from dss.preprocessing.clean import clean_channel_obs, clean_mno_demand
from dss.preprocessing.split import tag_split
from dss.preprocessing.scale import fit_sensing_scaler, save_scalers, load_scalers, SENSING_COLS
from dss.preprocessing.windows import build_sliding_windows, split_windows_by_target_step

def test_cleaning_fault_injected_data():
    # Inject faults
    c_df, m_df, _ = generate_dataset("normal", seed=42, save=False, fault_rate=0.05)
    assert c_df["noise_est_dbm"].isna().sum() > 0
    assert m_df["traffic_demand_mbps"].isna().sum() > 0

    c_clean, c_rep = clean_channel_obs(c_df)
    m_clean, m_rep = clean_mno_demand(m_df)

    assert c_clean.isna().sum().sum() == 0
    assert m_clean.isna().sum().sum() == 0

def test_no_leakage_in_sensing_columns():
    # Verify no truth columns are in model input features
    assert "pu_state" not in SENSING_COLS
    assert "noise_true_dbm" not in SENSING_COLS

def test_train_split_and_scaler_no_leakage():
    c_df, _, _ = generate_dataset("normal", seed=42, save=False)
    c_clean, _ = clean_channel_obs(c_df)
    c_tagged = tag_split(c_clean)

    train_c = c_tagged[c_tagged["split"] == "train"]
    # 2880 steps * 16 channels = 46,080 samples per scenario
    assert len(train_c) == 2880 * 16

    scaler = fit_sensing_scaler(train_c)
    # Check scaler means match the train mean exactly
    expected_means = train_c[SENSING_COLS].mean().values
    np.testing.assert_allclose(scaler.mean_, expected_means, rtol=1e-5)

def test_sliding_windows_count_and_no_leakage():
    # Dummy features (4032 steps, 20 features) and targets (4032 steps, 4 MNOs)
    T = 4032
    L = 12
    H = 3
    features = np.arange(T * 20, dtype=np.float32).reshape(T, 20)
    targets = np.arange(T * 4, dtype=np.float32).reshape(T, 4)

    X, Y, steps = build_sliding_windows(features, targets, seq_len=L, horizon=H)

    # Window count arithmetic: 4032 - 12 - 3 + 1 = 4018
    assert len(X) == 4018
    assert len(Y) == 4018
    assert X.shape == (4018, 12, 20)
    assert Y.shape == (4018, 3, 4)

    # Check target step logic: Y[i] must be strictly in the future of X[i]
    for i in range(5):
        t = steps[i]
        # Y contains steps t+1, t+2, t+3
        assert np.array_equal(Y[i], targets[t + 1 : t + 4])

    # Split assignment
    split_dict = split_windows_by_target_step(X, Y, steps, train_max_step=2879)
    train_X, train_Y, train_steps = split_dict["train"]

    # All targets in train split must end at or before step 2879
    assert np.all(train_steps + 3 <= 2879)

def test_scaler_round_trip(tmp_path):
    c_df, _, _ = generate_dataset("normal", seed=42, save=False)
    scaler = fit_sensing_scaler(c_df)
    scaler_dict = {"scaler": scaler}
    path = tmp_path / "test_scaler.joblib"

    save_scalers(scaler_dict, path)
    loaded = load_scalers(path)

    np.testing.assert_array_equal(scaler.mean_, loaded["scaler"].mean_)
