import json
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from dss.common.config import get_config
from dss.common.paths import data_dir
from dss.common.seed import rng_for
from dss.common.ids import dataset_key, compute_config_hash
from dss.data_generation.scenarios import get_scenario, list_scenarios
from dss.data_generation.pu_model import generate_pu_occupancy
from dss.data_generation.noise_model import generate_noise_and_interference, compute_rx_power_and_snr
from dss.data_generation.demand_model import generate_mno_demand
from dss.database.connection import save_dataset_to_db

def generate_dataset(
    scenario_name: str,
    seed: int = 42,
    save: bool = True,
    fault_rate: float = 0.0,
    save_to_db: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """
    Simulate channel observations and MNO demands for a scenario.
    Returns (channel_obs_df, mno_demand_df, dataset_key).
    """
    cfg = get_config()
    scen = get_scenario(scenario_name)

    scen_idx = list_scenarios().index(scenario_name) if scenario_name in list_scenarios() else 0
    rng = rng_for(seed, scen_idx)

    n_steps = cfg.sim.days * int(1440 / cfg.sim.interval_min)  # 14 * 288 = 4032
    n_channels = cfg.sim.n_channels  # 16
    n_mno = cfg.sim.n_mno  # 4
    interval_min = cfg.sim.interval_min  # 5

    start_dt = datetime.fromisoformat(cfg.sim.start_ts)
    timestamps = [start_dt + timedelta(minutes=t * interval_min) for t in range(n_steps)]

    # 1. Generate PU Occupancy
    pu_state = generate_pu_occupancy(
        n_steps=n_steps,
        n_channels=n_channels,
        duty_min=scen.pu_duty_min,
        duty_max=scen.pu_duty_max,
        rng=rng,
        interval_min=interval_min,
    )

    # 2. Generate Noise & Interference
    noise_true, noise_est, interference = generate_noise_and_interference(
        n_steps=n_steps,
        n_channels=n_channels,
        sigma_u=scen.sigma_u,
        p_burst=scen.p_burst,
        rng=rng,
    )

    # 3. Compute rx_power_dbm and snr_est_db
    rx_power, snr_est = compute_rx_power_and_snr(
        pu_state=pu_state,
        noise_true_dbm=noise_true,
        noise_est_dbm=noise_est,
        interference_dbm=interference,
        scenario_name=scenario_name,
        rng=rng,
    )

    # 4. Generate MNO Demand
    active_users, traffic_demand, requested_bw = generate_mno_demand(
        n_steps=n_steps,
        n_mno=n_mno,
        scen_mult=scen.scen_mult,
        spike_lambda_per_day=scen.spike_lambda,
        scenario_name=scenario_name,
        rng=rng,
        u_peak=cfg.sim.u_peak,
        r_user_mbps=cfg.sim.r_user_mbps,
        se_ref=cfg.sim.se_ref,
        d_cap_mhz=cfg.sim.d_cap_mhz,
        interval_min=interval_min,
    )

    # Build ChannelObs records: T * C rows = 4032 * 16 = 64,512 rows
    c_step_idx = np.repeat(np.arange(n_steps, dtype=np.int32), n_channels)
    c_ts = [timestamps[step] for step in c_step_idx]
    c_channel_id = np.tile(np.arange(n_channels, dtype=np.int16), n_steps)

    channel_obs_df = pd.DataFrame({
        "step_idx": c_step_idx,
        "ts": pd.to_datetime(c_ts),
        "channel_id": c_channel_id,
        "pu_state": pu_state.flatten().astype(np.int8),
        "noise_true_dbm": noise_true.flatten().astype(np.float32),
        "noise_est_dbm": noise_est.flatten().astype(np.float32),
        "interference_dbm": interference.flatten().astype(np.float32),
        "rx_power_dbm": rx_power.flatten().astype(np.float32),
        "snr_est_db": snr_est.flatten().astype(np.float32),
    })

    # Build MnoDemand records: T * M rows = 4032 * 4 = 16,128 rows
    m_step_idx = np.repeat(np.arange(n_steps, dtype=np.int32), n_mno)
    m_ts = [timestamps[step] for step in m_step_idx]
    m_mno_id = np.tile(np.arange(n_mno, dtype=np.int16), n_steps)

    mno_demand_df = pd.DataFrame({
        "step_idx": m_step_idx,
        "ts": pd.to_datetime(m_ts),
        "mno_id": m_mno_id,
        "active_users": active_users.flatten().astype(np.int32),
        "traffic_demand_mbps": traffic_demand.flatten().astype(np.float32),
        "requested_bw_mhz": requested_bw.flatten().astype(np.float32),
    })

    # Optional Fault Injection for cleaning test
    if fault_rate > 0.0:
        mask_c = rng.random(len(channel_obs_df)) < fault_rate
        channel_obs_df.loc[mask_c, "noise_est_dbm"] = np.nan
        mask_m = rng.random(len(mno_demand_df)) < fault_rate
        mno_demand_df.loc[mask_m, "traffic_demand_mbps"] = np.nan

    cfg_dict = {
        "scenario": scenario_name,
        "seed": seed,
        "n_steps": n_steps,
        "n_channels": n_channels,
        "n_mno": n_mno,
    }
    cfg_hash = compute_config_hash(cfg_dict)
    key = dataset_key(scenario_name, seed, cfg_hash)

    if save:
        out_dir = data_dir("raw") / key
        out_dir.mkdir(parents=True, exist_ok=True)
        channel_obs_df.to_parquet(out_dir / "channel_obs.parquet", index=False)
        mno_demand_df.to_parquet(out_dir / "mno_demand.parquet", index=False)

        meta = {
            "dataset_key": key,
            "scenario": scenario_name,
            "seed": seed,
            "config_hash": cfg_hash,
            "n_steps": n_steps,
            "channel_obs_rows": len(channel_obs_df),
            "mno_demand_rows": len(mno_demand_df),
            "generated_at": datetime.now().isoformat(),
        }
        with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    if save_to_db:
        save_dataset_to_db(channel_obs_df, mno_demand_df)

    return channel_obs_df, mno_demand_df, key

