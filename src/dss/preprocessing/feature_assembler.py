import numpy as np
import pandas as pd

def compute_time_encodings(step_idx: np.ndarray, interval_min: int = 5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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

class FeatureAssemblerStub:
    """
    Stub for FeatureAssembler, completed when VAE-GMM p_busy and AE z_ae exist.
    Produces base temporal and demand features.
    """
    def __init__(self, n_mno: int = 4):
        self.n_mno = n_mno

    def build_time_features(self, step_idx: np.ndarray) -> np.ndarray:
        s, c, w = compute_time_encodings(step_idx)
        return np.stack([s, c, w], axis=-1)
