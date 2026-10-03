import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple

def clean_channel_obs(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Cleans ChannelObs DataFrame:
    - Out of range values set to NaN
    - Forward fill <= 2 steps per channel
    - Drop any remaining NaNs
    """
    cleaned = df.copy()
    initial_nans = int(cleaned.isna().sum().sum())

    # Range validation: dbm in [-135, -20]
    dbm_cols = ["noise_est_dbm", "interference_dbm", "rx_power_dbm", "snr_est_db"]
    for col in dbm_cols:
        if col in cleaned.columns:
            mask = (cleaned[col] < -135.0) | (cleaned[col] > 40.0)
            cleaned.loc[mask, col] = np.nan

    range_nans = int(cleaned.isna().sum().sum())

    # Forward fill <= 2 steps grouped by channel_id
    cleaned = cleaned.sort_values(["channel_id", "step_idx"]).reset_index(drop=True)
    cleaned[dbm_cols] = cleaned.groupby("channel_id")[dbm_cols].ffill(limit=2)

    remaining_nans = int(cleaned.isna().sum().sum())
    cleaned = cleaned.dropna().reset_index(drop=True)

    report = {
        "initial_nans": initial_nans,
        "range_nans_injected": range_nans - initial_nans,
        "filled_nans": range_nans - remaining_nans,
        "dropped_rows": len(df) - len(cleaned),
        "final_rows": len(cleaned),
    }
    return cleaned, report

def clean_mno_demand(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Cleans MnoDemand DataFrame:
    - Out of range values set to NaN
    - Interpolate <= 2 steps per MNO
    - Drop remaining NaNs
    """
    cleaned = df.copy()
    initial_nans = int(cleaned.isna().sum().sum())

    # Range validation
    if "active_users" in cleaned.columns:
        cleaned.loc[cleaned["active_users"] < 0, "active_users"] = np.nan
    if "requested_bw_mhz" in cleaned.columns:
        mask = (cleaned["requested_bw_mhz"] < 0.0) | (cleaned["requested_bw_mhz"] > 160.0)
        cleaned.loc[mask, "requested_bw_mhz"] = np.nan
    if "traffic_demand_mbps" in cleaned.columns:
        cleaned.loc[cleaned["traffic_demand_mbps"] < 0.0, "traffic_demand_mbps"] = np.nan

    num_cols = ["active_users", "traffic_demand_mbps", "requested_bw_mhz"]
    range_nans = int(cleaned.isna().sum().sum())

    # Interpolate <= 2 steps grouped by mno_id
    cleaned = cleaned.sort_values(["mno_id", "step_idx"]).reset_index(drop=True)
    cleaned[num_cols] = cleaned.groupby("mno_id")[num_cols].transform(
        lambda s: s.interpolate(method="linear", limit=2)
    )

    remaining_nans = int(cleaned.isna().sum().sum())
    cleaned = cleaned.dropna().reset_index(drop=True)

    report = {
        "initial_nans": initial_nans,
        "range_nans_injected": range_nans - initial_nans,
        "filled_nans": range_nans - remaining_nans,
        "dropped_rows": len(df) - len(cleaned),
        "final_rows": len(cleaned),
    }
    return cleaned, report
