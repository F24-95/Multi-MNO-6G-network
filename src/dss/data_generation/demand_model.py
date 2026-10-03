import numpy as np

def generate_mno_demand(
    n_steps: int,
    n_mno: int,
    scen_mult: float,
    spike_lambda_per_day: float,
    scenario_name: str,
    rng: np.random.Generator,
    u_peak: int = 250,
    r_user_mbps: float = 0.8,
    se_ref: float = 4.0,
    d_cap_mhz: float = 160.0,
    interval_min: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates MNO demand:
    Returns (active_users, traffic_demand_mbps, requested_bw_mhz) each of shape (T, M).
    """
    steps_per_day = int(1440 / interval_min)  # 288 steps per day
    n_days = int(np.ceil(n_steps / steps_per_day))

    # Base operator weights
    if scenario_name == "uneven_mno_demand":
        w_m = np.array([1.8, 1.2, 0.6, 0.4], dtype=np.float32)
    else:
        w_m = np.ones(n_mno, dtype=np.float32)

    # Phase shifts in hours for MNOs
    phi_m = np.array([0.0, 1.0, -1.0, 2.0], dtype=np.float32)

    # Daily drift per MNO
    daily_drift = np.zeros((n_days, n_mno), dtype=np.float32)
    for m in range(n_mno):
        drift_init = 1.0 + rng.normal(0, 0.05)
        for d in range(n_days):
            daily_drift[d, m] = 1.0 + 0.08 * np.sin(2 * np.pi * d / 14.0 + m)

    # AR(1) lognormal noise per MNO
    rho_ar = 0.90
    sigma_ar = 0.08
    log_ar = np.zeros((n_steps, n_mno), dtype=np.float32)
    log_ar[0, :] = rng.normal(0, sigma_ar, size=n_mno)
    for t in range(1, n_steps):
        innov = rng.normal(0, sigma_ar * np.sqrt(1 - rho_ar**2), size=n_mno)
        log_ar[t, :] = rho_ar * log_ar[t - 1, :] + innov
    ar_noise = np.exp(log_ar)

    # Spike model
    p_spike_step = (spike_lambda_per_day / steps_per_day)
    spike_mult = np.ones((n_steps, n_mno), dtype=np.float32)
    active_spikes = [[] for _ in range(n_mno)]  # list of [remaining_steps, amplitude, decay_rate]

    for t in range(n_steps):
        for m in range(n_mno):
            if rng.random() < p_spike_step:
                amp = rng.uniform(0.8, 2.0)
                dur = rng.integers(3, 13)  # 3 to 12 steps
                decay = amp / dur
                active_spikes[m].append([dur, amp, decay])

            # Apply active spikes
            curr_spike = 0.0
            still_active = []
            for s in active_spikes[m]:
                curr_spike += s[1]
                s[0] -= 1
                s[1] = max(0.0, s[1] - s[2])
                if s[0] > 0 and s[1] > 0.01:
                    still_active.append(s)
            active_spikes[m] = still_active
            spike_mult[t, m] = 1.0 + curr_spike

    # Compute demand step-by-step
    active_users = np.zeros((n_steps, n_mno), dtype=np.int32)
    traffic_demand_mbps = np.zeros((n_steps, n_mno), dtype=np.float32)
    requested_bw_mhz = np.zeros((n_steps, n_mno), dtype=np.float32)

    for t in range(n_steps):
        step_min = t * interval_min
        hour = (step_min / 60.0) % 24.0
        day = int(step_min / 1440.0)
        day_of_week = day % 7
        is_weekend = 1 if day_of_week in [5, 6] else 0
        weekend_mult = 0.85 if is_weekend else 1.0

        for m in range(n_mno):
            # Diurnal profile: two Gaussian peaks at 12:00 and 20:00 on 0.25 floor
            eff_hour = (hour + phi_m[m]) % 24.0
            p1 = np.exp(-0.5 * ((eff_hour - 12.0) / 2.5) ** 2)
            p2 = np.exp(-0.5 * ((eff_hour - 20.0) / 2.5) ** 2)
            raw_profile = 0.25 + 0.75 * np.maximum(p1, p2)
            profile_val = raw_profile / 1.0  # normalized max = 1.0

            lam = (
                u_peak
                * w_m[m]
                * profile_val
                * scen_mult
                * weekend_mult
                * spike_mult[t, m]
                * daily_drift[min(day, n_days - 1), m]
                * ar_noise[t, m]
            )
            lam = max(1.0, float(lam))

            users = rng.poisson(lam)
            active_users[t, m] = users

            # Traffic demand
            user_var = rng.lognormal(0.0, 0.1)
            traffic = users * r_user_mbps * user_var
            traffic_demand_mbps[t, m] = traffic

            # Requested BW in MHz, capped at d_cap_mhz
            req_bw = min(traffic / se_ref, d_cap_mhz)
            requested_bw_mhz[t, m] = req_bw

    return active_users, traffic_demand_mbps, requested_bw_mhz
