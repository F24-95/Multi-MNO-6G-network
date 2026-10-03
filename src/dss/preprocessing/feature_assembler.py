"""
Feature Assembler: Combines spectrum latent representations (z_ae),
normalized MNO demand, spectrum occupancy indicators, and temporal encodings
into a unified 20-dimensional feature vector per timestep.
Features (F=20):
  0-7:  z_ae (8-dim compressed spectrum snapshot from Autoencoder)
  8-11: mno_demand_norm (4 MNO demands normalized by D_cap = 160 MHz)
  12:   total_demand_norm (mean normalized demand across 4 MNOs)
  13:   mean_p_busy (average channel occupancy probability across 16 channels)
  14:   tod_sin (sin of time-of-day cyclical encoding)
  15:   tod_cos (cos of time-of-day cyclical encoding)
  16:   is_weekend (binary flag for Saturday/Sunday)
  17:   demand_mno_std (standard deviation across the 4 MNO demands)
  18:   demand_delta_norm (change in aggregate demand from previous step)
  19:   high_demand_flag (1.0 if total normalized demand > 0.50 else 0.0)
"""
import numpy as np
import pandas as pd
from typing import Tuple

FEATURE_NAMES = [
    "z_ae_0",
    "z_ae_1",
    "z_ae_2",
    "z_ae_3",
    "z_ae_4",
    "z_ae_5",
    "z_ae_6",
    "z_ae_7",
    "mno_demand_norm_0",
    "mno_demand_norm_1",
    "mno_demand_norm_2",
    "mno_demand_norm_3",
    "total_demand_norm",
    "mean_p_busy",
    "tod_sin",
    "tod_cos",
    "is_weekend",
    "demand_mno_std",
    "demand_delta_norm",
    "high_demand_flag",
]

N_FEATURES = 20
D_CAP_MHZ = 160.0


def compute_time_encodings(
    step_idx: np.ndarray, interval_min: int = 5
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes cyclical time-of-day encodings and weekend indicator:
    - tod_sin = sin(2 * pi * minute_of_day / 1440)
    - tod_cos = cos(2 * pi * minute_of_day / 1440)
    - is_weekend = 1 if Saturday/Sunday else 0 (assuming start day is Monday)
    """
    minute_of_day = (step_idx * interval_min) % 1440
    tod_sin = np.sin(2.0 * np.pi * minute_of_day / 1440.0).astype(np.float32)
    tod_cos = np.cos(2.0 * np.pi * minute_of_day / 1440.0).astype(np.float32)

    day_idx = (step_idx * interval_min) // 1440
    day_of_week = day_idx % 7  # 0=Monday ... 5=Saturday, 6=Sunday
    is_weekend = np.isin(day_of_week, [5, 6]).astype(np.float32)

    return tod_sin, tod_cos, is_weekend


class FeatureAssembler:
    """
    Assembles the 20-dimensional feature vector both in batch mode and streaming mode.
    Guarantees strict equality between batch and streaming outputs.
    """
    def __init__(self, d_cap_mhz: float = D_CAP_MHZ, n_mno: int = 4, n_channels: int = 16):
        self.d_cap_mhz = float(d_cap_mhz)
        self.n_mno = int(n_mno)
        self.n_channels = int(n_channels)
        self.feature_names = list(FEATURE_NAMES)

    def assemble_batch(
        self,
        z_ae: np.ndarray,
        mno_demand_mat: np.ndarray,
        p_busy_mat: np.ndarray,
        step_indices: np.ndarray,
    ) -> np.ndarray:
        """
        Batch assembly of feature vectors for T time steps.
        Args:
            z_ae: (T, 8)
            mno_demand_mat: (T, 4) in MHz
            p_busy_mat: (T, 16) in [0, 1]
            step_indices: (T,) integers
        Returns:
            features: (T, 20) float32 array
        """
        T = len(z_ae)
        if mno_demand_mat.shape != (T, self.n_mno):
            raise ValueError(f"Expected demand shape {(T, self.n_mno)}, got {mno_demand_mat.shape}")
        if p_busy_mat.shape != (T, self.n_channels):
            raise ValueError(f"Expected p_busy shape {(T, self.n_channels)}, got {p_busy_mat.shape}")

        # 0-7: z_ae
        f_z = z_ae.astype(np.float32)

        # 8-11: mno_demand_norm (T, 4)
        f_demand = np.clip(mno_demand_mat / self.d_cap_mhz, 0.0, 1.0).astype(np.float32)

        # 12: total_demand_norm (mean of 4 MNO normalized demands)
        f_total_demand = np.mean(f_demand, axis=1, keepdims=True)

        # 13: mean_p_busy (mean occupancy across 16 channels)
        f_mean_p_busy = np.mean(p_busy_mat, axis=1, keepdims=True).astype(np.float32)

        # 14-16: time encodings
        tod_sin, tod_cos, is_weekend = compute_time_encodings(step_indices)
        f_time = np.stack([tod_sin, tod_cos, is_weekend], axis=1)

        # 17: demand_mno_std (spread among MNOs)
        f_std = np.std(f_demand, axis=1, keepdims=True).astype(np.float32)

        # 18: demand_delta_norm (first difference, 0 at t=0)
        f_delta = np.zeros((T, 1), dtype=np.float32)
        if T > 1:
            f_delta[1:] = f_total_demand[1:] - f_total_demand[:-1]

        # 19: high_demand_flag (1.0 if total normalized demand > 0.50 else 0.0)
        f_high = (f_total_demand > 0.50).astype(np.float32)

        features = np.concatenate(
            [f_z, f_demand, f_total_demand, f_mean_p_busy, f_time, f_std, f_delta, f_high],
            axis=1,
        ).astype(np.float32)

        if features.shape[1] != N_FEATURES:
            raise ValueError(f"Expected {N_FEATURES} features, got {features.shape[1]}")

        return features

    def assemble_row(
        self,
        z_ae_row: np.ndarray,
        current_demand: np.ndarray,
        prev_demand: np.ndarray | None,
        p_busy_row: np.ndarray,
        step_idx: int,
    ) -> np.ndarray:
        """
        Streaming assembly of a single 20-dim FeatureRow for online inference / decision replay.
        Strictly matches the row computed by assemble_batch.
        """
        z = np.asarray(z_ae_row, dtype=np.float32).ravel()
        dem = np.clip(np.asarray(current_demand, dtype=np.float32).ravel() / self.d_cap_mhz, 0.0, 1.0)
        tot_dem = float(np.mean(dem))
        mean_pb = float(np.mean(p_busy_row))

        s, c, w = compute_time_encodings(np.array([step_idx]))
        tod_s, tod_c, is_w = float(s[0]), float(c[0]), float(w[0])

        dem_std = float(np.std(dem))

        if prev_demand is None:
            dem_delta = 0.0
        else:
            prev_dem = np.clip(np.asarray(prev_demand, dtype=np.float32).ravel() / self.d_cap_mhz, 0.0, 1.0)
            dem_delta = tot_dem - float(np.mean(prev_dem))

        high_flag = 1.0 if tot_dem > 0.50 else 0.0

        row = np.array(
            [
                z[0], z[1], z[2], z[3], z[4], z[5], z[6], z[7],
                dem[0], dem[1], dem[2], dem[3],
                tot_dem,
                mean_pb,
                tod_s, tod_c, is_w,
                dem_std,
                dem_delta,
                high_flag,
            ],
            dtype=np.float32,
        )
        return row
