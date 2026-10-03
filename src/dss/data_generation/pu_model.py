import numpy as np

def generate_pu_occupancy(
    n_steps: int,
    n_channels: int,
    duty_min: float,
    duty_max: float,
    rng: np.random.Generator,
    interval_min: int = 5,
) -> np.ndarray:
    """
    Generates PU occupancy (T, C) using a two-state Markov chain per channel.
    b = 1/6 (mean busy duration = 6 steps = 30 minutes).
    a = delta * b / (1 - delta), modulated by a diurnal factor in [0.8, 1.2].
    """
    # Sample static base duty cycle for each channel
    delta_c = rng.uniform(duty_min, duty_max, size=n_channels)
    b = 1.0 / 6.0  # Transition prob from busy -> idle

    # Calculate hours for each step for diurnal modulation
    hours = (np.arange(n_steps) * interval_min / 60.0) % 24.0
    # Diurnal factor in [0.8, 1.2], peak around 14:00
    diurnal = 1.0 + 0.2 * np.cos(2 * np.pi * (hours - 14.0) / 24.0)  # shape (T,)

    pu_state = np.zeros((n_steps, n_channels), dtype=np.int8)

    # Initial state sampled from stationary distribution
    for c in range(n_channels):
        pu_state[0, c] = 1 if rng.random() < delta_c[c] else 0

    # Simulation loop
    for t in range(1, n_steps):
        d_factor = diurnal[t]
        for c in range(n_channels):
            prev = pu_state[t - 1, c]
            if prev == 1:
                # In busy state: transition to idle with prob b
                pu_state[t, c] = 0 if rng.random() < b else 1
            else:
                # In idle state: transition to busy with prob a
                base_a = (delta_c[c] * b) / (1.0 - delta_c[c] + 1e-9)
                a = np.clip(base_a * d_factor, 0.0, 1.0)
                pu_state[t, c] = 1 if rng.random() < a else 0

    return pu_state
