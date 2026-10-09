"""
Constructs 64-dimensional spectrum snapshots from processed channel observations
and cached VAE-GMM p_busy matrices.
Structure per time step t (64 values):
  - p_busy[:, 0..15]           (16 channels, already in [0, 1])
  - snr_scaled[:, 0..15]       (16 channels, normalized to [0, 1])
  - interference_scaled[:, 0..15] (16 channels, normalized to [0, 1])
  - noise_scaled[:, 0..15]     (16 channels, normalized to [0, 1])
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

from dss.common.paths import data_dir
from dss.preprocessing.scale import fit_snapshot_scaler


def load_raw_rf_matrices(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts (T, 16) matrices for snr_est_db, interference_dbm, and noise_est_dbm
    from a channel_obs dataframe.
    """
    steps = df["step_idx"].values
    channels = df["channel_id"].values
    T = int(steps.max() - steps.min() + 1)
    C = 16

    t_idx = steps - steps.min()
    c_idx = channels

    snr_mat = np.zeros((T, C), dtype=np.float32)
    int_mat = np.zeros((T, C), dtype=np.float32)
    noise_mat = np.zeros((T, C), dtype=np.float32)

    snr_mat[t_idx, c_idx] = df["snr_est_db"].values.astype(np.float32)
    int_mat[t_idx, c_idx] = df["interference_dbm"].values.astype(np.float32)
    noise_mat[t_idx, c_idx] = df["noise_est_dbm"].values.astype(np.float32)

    return snr_mat, int_mat, noise_mat


def transform_rf_matrices(
    snr_mat: np.ndarray,
    int_mat: np.ndarray,
    noise_mat: np.ndarray,
    scaler: MinMaxScaler,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Applies fitted MinMaxScaler to RF matrices, maintaining (T, 16) shape."""
    T, C = snr_mat.shape
    stacked = np.stack([snr_mat.flatten(), int_mat.flatten(), noise_mat.flatten()], axis=1)
    scaled = scaler.transform(stacked).astype(np.float32)

    snr_s = np.clip(scaled[:, 0].reshape(T, C), 0.0, 1.0)
    int_s = np.clip(scaled[:, 1].reshape(T, C), 0.0, 1.0)
    noise_s = np.clip(scaled[:, 2].reshape(T, C), 0.0, 1.0)

    return snr_s, int_s, noise_s


def build_scenario_snapshots(
    scenario: str,
    seed: int,
    split: str,
    scaler: MinMaxScaler | None = None,
) -> tuple[np.ndarray, MinMaxScaler]:
    """
    Loads channel_obs for given scenario and split, and pairs it with cached p_busy.
    Returns:
        snapshots: (T, 64) array bounded in [0, 1]
        scaler: fitted or provided MinMaxScaler
    """
    # 1. Load channel obs
    p = data_dir("processed") / f"{scenario}_s{seed}" / "channel_obs_processed.parquet"
    if not p.exists():
        raise FileNotFoundError(f"Processed channel obs not found at {p}")

    df = pd.read_parquet(p)
    df_split = df[df["split"] == split]
    if df_split.empty:
        raise ValueError(f"Split {split} is empty in {p}")

    snr_mat, int_mat, noise_mat = load_raw_rf_matrices(df_split)

    # 2. Fit or apply scaler
    if scaler is None:
        scaler = fit_snapshot_scaler(snr_mat, int_mat, noise_mat)

    snr_s, int_s, noise_s = transform_rf_matrices(snr_mat, int_mat, noise_mat, scaler)

    # 3. Load cached p_busy (T, 16)
    p_busy_path = data_dir("sensing") / f"{scenario}_s{seed}" / f"{split}_p_busy.npy"
    if not p_busy_path.exists():
        raise FileNotFoundError(
            f"Cached sensing p_busy not found at {p_busy_path}. Run scripts/run_sensing_inference.py first."
        )
    p_busy = np.load(p_busy_path).astype(np.float32)
    p_busy = np.clip(p_busy, 0.0, 1.0)

    if p_busy.shape != snr_s.shape:
        raise ValueError(f"Shape mismatch: p_busy {p_busy.shape} vs snr {snr_s.shape}")

    # 4. Concatenate into (T, 64) snapshot
    # [16 p_busy, 16 snr, 16 int, 16 noise]
    snapshots = np.concatenate([p_busy, snr_s, int_s, noise_s], axis=1).astype(np.float32)
    return snapshots, scaler


def build_snapshots_from_df(df: pd.DataFrame) -> tuple[np.ndarray, list]:
    """Builds (T, 64) snapshot matrix directly from channel observation dataframe."""
    snr_mat, int_mat, noise_mat = load_raw_rf_matrices(df)
    scaler = fit_snapshot_scaler(snr_mat, int_mat, noise_mat)
    snr_s, int_s, noise_s = transform_rf_matrices(snr_mat, int_mat, noise_mat, scaler)

    steps = df["step_idx"].values
    T = int(steps.max() - steps.min() + 1)
    C = 16
    t_idx = steps - steps.min()
    c_idx = df["channel_id"].values
    p_busy = np.zeros((T, C), dtype=np.float32)
    p_busy[t_idx, c_idx] = df["pu_state"].values.astype(np.float32)

    snapshots = np.concatenate([p_busy, snr_s, int_s, noise_s], axis=1).astype(np.float32)
    ts_list = df[["step_idx", "ts"]].drop_duplicates().sort_values("step_idx")["ts"].tolist()
    return snapshots, ts_list

