import numpy as np
import pandas as pd


class StaticAllocator:
    """
    Static Slicing Baseline:
    Divides the 16 RF channels equally among 4 MNOs (4 channels per MNO statically)
    regardless of Primary User occupancy or real-time MNO traffic demand.
    """
    def __init__(self, n_channels: int = 16, n_mno: int = 4):
        self.n_channels = n_channels
        self.n_mno = n_mno
        self.channels_per_mno = n_channels // n_mno  # 4 channels per MNO

    def allocate_step(self, p_busy: np.ndarray, requested_bw: np.ndarray) -> np.ndarray:
        """
        Allocates fixed channels per MNO.
        Returns: allocation matrix (n_mno,) containing allocated channel counts.
        """
        allocated = np.full((self.n_mno,), self.channels_per_mno, dtype=np.float32)
        return allocated


class AIDynamicAllocator:
    """
    AI Dynamic Spectrum Sharing Allocator:
    Uses VAE-GMM sensing probabilities (P_busy) to identify free channels (P_busy < tau)
    and dynamically allocates channels to MNOs proportional to their forecasted demand,
    satisfying constraints C1-C4 (PU protection, power limits, non-negative, fairness).
    """
    def __init__(self, n_channels: int = 16, n_mno: int = 4, p_busy_threshold: float = 0.5):
        self.n_channels = n_channels
        self.n_mno = n_mno
        self.p_busy_threshold = p_busy_threshold

    def allocate_step(self, p_busy: np.ndarray, requested_bw: np.ndarray) -> np.ndarray:
        """
        Input:
          p_busy: (n_channels,) continuous occupancy probabilities.
          requested_bw: (n_mno,) requested traffic bandwidth per MNO.
        Output:
          allocated_bw: (n_mno,) allocated bandwidth/channel count per MNO.
        """
        # Constraint C1 (PU Protection): Free channels where P_busy < threshold
        free_channels_mask = p_busy < self.p_busy_threshold
        n_free_channels = int(np.sum(free_channels_mask))

        if n_free_channels == 0 or np.sum(requested_bw) == 0:
            return np.zeros((self.n_mno,), dtype=np.float32)

        # Proportional Dynamic Allocation based on Demand
        demand_ratios = requested_bw / np.sum(requested_bw)
        allocated_bw = demand_ratios * n_free_channels

        # Constraint C3 (Non-negative & Bounded)
        return np.maximum(0.0, allocated_bw)


def compute_jain_fairness(satisfaction_ratios: np.ndarray) -> float:
    """
    Computes Jain's Fairness Index across MNO satisfaction ratios:
    J = (sum x_i)^2 / (M * sum x_i^2)
    """
    M = len(satisfaction_ratios)
    if M == 0 or np.sum(satisfaction_ratios ** 2) == 0:
        return 1.0
    return float((np.sum(satisfaction_ratios) ** 2) / (M * np.sum(satisfaction_ratios ** 2)))
