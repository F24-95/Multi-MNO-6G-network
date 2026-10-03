import numpy as np
from typing import Tuple, Dict

def build_sliding_windows(
    feature_matrix: np.ndarray,
    target_matrix: np.ndarray,
    seq_len: int = 12,
    horizon: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Constructs sliding windows:
    - X[i]: feature_matrix[t - seq_len + 1 : t + 1] -> shape (B, seq_len, F)
    - Y[i]: target_matrix[t + 1 : t + 1 + horizon]  -> shape (B, horizon, M)
    - step_indices[i]: decision step t
    Valid decision steps t range from seq_len - 1 to T - horizon - 1.
    Total windows per scenario = T - seq_len - horizon + 1.
    For T=4032, L=12, H=3: 4032 - 12 - 3 + 1 = 4018 windows.
    """
    total_steps = len(feature_matrix)
    X_list = []
    Y_list = []
    t_list = []

    for t in range(seq_len - 1, total_steps - horizon):
        x_win = feature_matrix[t - seq_len + 1 : t + 1]
        y_win = target_matrix[t + 1 : t + 1 + horizon]
        X_list.append(x_win)
        Y_list.append(y_win)
        t_list.append(t)

    X = np.array(X_list, dtype=np.float32)
    Y = np.array(Y_list, dtype=np.float32)
    step_indices = np.array(t_list, dtype=np.int32)
    return X, Y, step_indices

def split_windows_by_target_step(
    X: np.ndarray,
    Y: np.ndarray,
    decision_steps: np.ndarray,
    train_max_step: int = 2879,
    val_max_step: int = 3455,
) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Assigns each window to train/val/test based on its target end step (decision_step + H):
    - All train window targets must be <= train_max_step (decision_step + 3 <= 2879)
    - All val window targets must be <= val_max_step
    - Remaining are test windows
    Prevents any lookahead leakage across split boundaries!
    """
    target_end_steps = decision_steps + 3  # H=3

    train_mask = target_end_steps <= train_max_step
    val_mask = (target_end_steps > train_max_step) & (target_end_steps <= val_max_step)
    test_mask = target_end_steps > val_max_step

    return {
        "train": (X[train_mask], Y[train_mask], decision_steps[train_mask]),
        "val": (X[val_mask], Y[val_mask], decision_steps[val_mask]),
        "test": (X[test_mask], Y[test_mask], decision_steps[test_mask]),
    }
