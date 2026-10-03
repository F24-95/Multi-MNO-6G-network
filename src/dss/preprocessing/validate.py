import json
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd

from dss.data_generation.scenarios import get_scenario
from dss.common.config import get_config

def validate_dataset(
    channel_obs_df: pd.DataFrame,
    mno_demand_df: pd.DataFrame,
    scenario_name: str,
    output_path: Path | None = None,
) -> Dict[str, Any]:
    """
    Validates synthetic data against all gates in STEP 05.
    Returns a dict report with pass/fail per check.
    """
    cfg = get_config()
    scen = get_scenario(scenario_name)
    report = {"scenario": scenario_name, "checks": {}, "passed": True}

    def check(name: str, condition: bool, details: Any = None):
        report["checks"][name] = {"passed": bool(condition), "details": details}
        if not condition:
            report["passed"] = False

    # 1. Record counts
    expected_c_rows = cfg.sim.days * 288 * cfg.sim.n_channels  # 64,512
    expected_m_rows = cfg.sim.days * 288 * cfg.sim.n_mno       # 16,128
    check("record_count_channel", len(channel_obs_df) == expected_c_rows, f"{len(channel_obs_df)} == {expected_c_rows}")
    check("record_count_demand", len(mno_demand_df) == expected_m_rows, f"{len(mno_demand_df)} == {expected_m_rows}")

    # 2. Missing values (0 NaN)
    c_nans = int(channel_obs_df.isna().sum().sum())
    m_nans = int(mno_demand_df.isna().sum().sum())
    check("missing_values", (c_nans == 0) and (m_nans == 0), f"c_nans={c_nans}, m_nans={m_nans}")

    # 3. Invalid values
    pu_valid = bool(channel_obs_df["pu_state"].isin([0, 1]).all())
    check("pu_state_binary", pu_valid)

    dbm_cols = ["noise_true_dbm", "noise_est_dbm", "interference_dbm", "rx_power_dbm"]
    dbm_in_range = True
    for col in dbm_cols:
        vals = channel_obs_df[col].values
        if np.any(vals < -135.0) or np.any(vals > -20.0):
            dbm_in_range = False
    check("dbm_in_range", dbm_in_range)

    users_non_neg = bool((mno_demand_df["active_users"] >= 0).all())
    check("active_users_non_negative", users_non_neg)

    bw_valid = bool(((mno_demand_df["requested_bw_mhz"] >= 0) & (mno_demand_df["requested_bw_mhz"] <= 160.0)).all())
    check("requested_bw_range", bw_valid)

    # 4. Time continuity: step_idx 0..4031 without gaps, 5 min step
    c_steps = sorted(channel_obs_df["step_idx"].unique())
    m_steps = sorted(mno_demand_df["step_idx"].unique())
    expected_steps = list(range(4032))
    check("time_continuity", (c_steps == expected_steps) and (m_steps == expected_steps))

    # 5. Duty cycle range check
    mean_duty = float(channel_obs_df["pu_state"].mean())
    # Mean duty should be within scenario range with some tolerance
    duty_ok = (mean_duty >= scen.pu_duty_min * 0.7) and (mean_duty <= scen.pu_duty_max * 1.3)
    check("duty_cycle_range", duty_ok, f"mean_duty={mean_duty:.3f}, target=[{scen.pu_duty_min}, {scen.pu_duty_max}]")

    # 6. Noise estimation error
    noise_err_std = float((channel_obs_df["noise_est_dbm"] - channel_obs_df["noise_true_dbm"]).std())
    noise_ok = abs(noise_err_std - scen.sigma_u) < 0.4
    check("noise_estimation_std", noise_ok, f"noise_std={noise_err_std:.3f}, target={scen.sigma_u}")

    # 7. MNO balance check
    if scenario_name == "normal":
        means = mno_demand_df.groupby("mno_id")["requested_bw_mhz"].mean().values
        mean_all = np.mean(means)
        is_balanced = bool(np.all(np.abs(means - mean_all) / (mean_all + 1e-6) < 0.20))
        check("mno_balance", is_balanced, f"means={means.tolist()}")
    elif scenario_name == "uneven_mno_demand":
        means = mno_demand_df.groupby("mno_id")["requested_bw_mhz"].mean().values
        # MNO 0 should have higher mean demand than MNO 3
        check("uneven_mno_balance", bool(means[0] > means[3]))

    # 8. Calibration bands: pool = 320 MHz (16 channels * 20 MHz)
    pool_bw = cfg.sim.n_channels * cfg.sim.channel_bw_mhz
    step_total_demand = mno_demand_df.groupby("step_idx")["requested_bw_mhz"].sum().values
    peak_demand = float(np.percentile(step_total_demand, 95))
    peak_ratio = peak_demand / pool_bw

    if scenario_name == "normal":
        check("calibration_normal", 0.45 <= peak_ratio <= 0.75, f"peak_ratio={peak_ratio:.2f}")
    elif scenario_name == "low_traffic":
        check("calibration_low", 0.10 <= peak_ratio <= 0.35, f"peak_ratio={peak_ratio:.2f}")
    elif scenario_name == "high_traffic":
        check("calibration_high", 0.85 <= peak_ratio <= 1.35, f"peak_ratio={peak_ratio:.2f}")
    elif scenario_name == "spectrum_shortage":
        # Idle fraction < 0.35
        idle_frac = 1.0 - mean_duty
        check("calibration_shortage_idle", idle_frac < 0.40, f"idle_frac={idle_frac:.2f}")

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    return report
