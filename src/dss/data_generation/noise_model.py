import numpy as np

def generate_noise_and_interference(
    n_steps: int,
    n_channels: int,
    sigma_u: float,
    p_burst: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates:
    - noise_true_dbm: AR(1) process centered around -96 dBm, sigma=0.5 dB, rho=0.98
    - noise_est_dbm: noise_true + N(0, sigma_u)
    - interference_dbm: base -108 + N(0, 1.5) dBm + bursts with prob p_burst adding U(10, 20) dB lasting 1-3 steps
    Returns (noise_true, noise_est, interference) each of shape (T, C).
    """
    base_noise = -96.0
    rho = 0.98
    sigma_drift = 0.5

    # 1. noise_true AR(1) per channel
    noise_true = np.zeros((n_steps, n_channels), dtype=np.float32)
    noise_true[0, :] = base_noise + rng.normal(0, sigma_drift, size=n_channels)
    for t in range(1, n_steps):
        innov = rng.normal(0, sigma_drift * np.sqrt(1 - rho**2), size=n_channels)
        noise_true[t, :] = base_noise + rho * (noise_true[t - 1, :] - base_noise) + innov

    # 2. noise_est = noise_true + N(0, sigma_u)
    noise_est = noise_true + rng.normal(0, sigma_u, size=(n_steps, n_channels)).astype(np.float32)

    # 3. interference: base -108 + N(0, 1.5)
    interference = -108.0 + rng.normal(0, 1.5, size=(n_steps, n_channels)).astype(np.float32)

    # Bursts per channel
    burst_remaining = np.zeros(n_channels, dtype=int)
    burst_add = np.zeros(n_channels, dtype=np.float32)

    for t in range(n_steps):
        for c in range(n_channels):
            if burst_remaining[c] > 0:
                interference[t, c] += burst_add[c]
                burst_remaining[c] -= 1
            elif rng.random() < p_burst:
                duration = rng.integers(1, 4)  # 1 to 3 steps
                add_db = rng.uniform(10.0, 20.0)
                interference[t, c] += add_db
                burst_remaining[c] = duration - 1
                burst_add[c] = add_db

    return noise_true, noise_est, interference

def compute_rx_power_and_snr(
    pu_state: np.ndarray,
    noise_true_dbm: np.ndarray,
    noise_est_dbm: np.ndarray,
    interference_dbm: np.ndarray,
    scenario_name: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes:
    rx_power_dbm = 10 * log10(10^(n_true/10) + pu_state * 10^(p_pu/10) + 10^(i/10)) + N(0, 0.5)
    where p_pu = n_true + snr_pu, snr_pu ~ N(mu_c, 2)
    mu_c ~ U(4, 10) dB (-3 dB if noise_uncertainty)
    snr_est_db = rx_power_dbm - noise_est_dbm
    """
    n_steps, n_channels = pu_state.shape
    if scenario_name == "noise_uncertainty":
        mu_c = np.full(n_channels, -3.0, dtype=np.float32)
    else:
        mu_c = rng.uniform(4.0, 10.0, size=n_channels).astype(np.float32)

    rx_power = np.zeros((n_steps, n_channels), dtype=np.float32)

    for t in range(n_steps):
        # sample snr_pu for each channel
        snr_pu = rng.normal(mu_c, 2.0).astype(np.float32)
        p_pu = noise_true_dbm[t, :] + snr_pu

        lin_noise = 10.0 ** (noise_true_dbm[t, :] / 10.0)
        lin_pu = (pu_state[t, :] * (10.0 ** (p_pu / 10.0))).astype(np.float32)
        lin_int = 10.0 ** (interference_dbm[t, :] / 10.0)

        total_lin = lin_noise + lin_pu + lin_int
        measurement_noise = rng.normal(0, 0.5, size=n_channels).astype(np.float32)
        rx_power[t, :] = 10.0 * np.log10(np.maximum(total_lin, 1e-13)) + measurement_noise

    snr_est = rx_power - noise_est_dbm
    return rx_power, snr_est
