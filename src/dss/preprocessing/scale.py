from pathlib import Path
from typing import Dict, Any
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

SENSING_COLS = ["rx_power_dbm", "noise_est_dbm", "snr_est_db", "interference_dbm"]

def fit_sensing_scaler(train_df: pd.DataFrame) -> StandardScaler:
    """Fits StandardScaler on the 4 sensing input features (train split only)."""
    scaler = StandardScaler()
    scaler.fit(train_df[SENSING_COLS].values)
    return scaler

def fit_snapshot_scaler(train_snr: np.ndarray, train_int: np.ndarray, train_noise: np.ndarray) -> MinMaxScaler:
    """
    Fits MinMaxScaler [0, 1] for AE features (snr, interference, noise).
    p_busy is already in [0, 1] so it is not scaled.
    """
    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    # Stack features to fit range
    stacked = np.stack([train_snr.flatten(), train_int.flatten(), train_noise.flatten()], axis=1)
    scaler.fit(stacked)
    return scaler

def save_scalers(scalers: Dict[str, Any], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scalers, path)

def load_scalers(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Scaler file not found at {path}")
    return joblib.load(path)
