import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from dss.common.paths import data_dir
from dss.common.logging_setup import setup_logger
from dss.data_generation.scenarios import list_scenarios
from dss.preprocessing.clean import clean_channel_obs, clean_mno_demand
from dss.preprocessing.split import tag_split
from dss.preprocessing.scale import fit_sensing_scaler, save_scalers, SENSING_COLS

logger = setup_logger("preprocess")

def main():
    parser = argparse.ArgumentParser(description="Preprocess raw datasets into training/evaluation arrays.")
    parser.add_argument("--seed", type=int, default=42, help="Seed to preprocess (default 42)")
    args = parser.parse_args()

    scenarios = list_scenarios()
    logger.info(f"Preprocessing datasets for {len(scenarios)} scenarios with seed={args.seed}...")

    raw_base = data_dir("raw")
    proc_base = data_dir("processed")

    all_train_sensing = []

    for scen in scenarios:
        pattern = f"ds_{scen}_s{args.seed}_*"
        matching = list(raw_base.glob(pattern))
        if not matching:
            logger.error(f"Missing raw dataset for {scen} (seed {args.seed})")
            continue

        ds_dir = matching[0]
        c_df = pd.read_parquet(ds_dir / "channel_obs.parquet")
        m_df = pd.read_parquet(ds_dir / "mno_demand.parquet")

        # 1. Clean
        c_clean, c_rep = clean_channel_obs(c_df)
        m_clean, m_rep = clean_mno_demand(m_df)

        # 2. Tag Split
        c_tagged = tag_split(c_clean)
        m_tagged = tag_split(m_clean)

        # Collect train split for fitting global sensing scaler
        c_train = c_tagged[c_tagged["split"] == "train"]
        all_train_sensing.append(c_train)

        # Save processed data per scenario
        out_scen_dir = proc_base / f"{scen}_s{args.seed}"
        out_scen_dir.mkdir(parents=True, exist_ok=True)
        c_tagged.to_parquet(out_scen_dir / "channel_obs_processed.parquet", index=False)
        m_tagged.to_parquet(out_scen_dir / "mno_demand_processed.parquet", index=False)

    # 3. Fit scaler ONLY on train split (all scenarios combined)
    train_sensing_df = pd.concat(all_train_sensing, ignore_index=True)
    sensing_scaler = fit_sensing_scaler(train_sensing_df)

    # Save scalers
    scaler_dict = {"sensing_scaler": sensing_scaler, "feature_names": SENSING_COLS}
    save_scalers(scaler_dict, proc_base / f"scalers_s{args.seed}.joblib")

    # Save sensing arrays: train, val, test
    train_x = sensing_scaler.transform(train_sensing_df[SENSING_COLS].values).astype(np.float32)
    np.save(proc_base / f"sensing_train_x_s{args.seed}.npy", train_x)

    logger.info(f"Fitted sensing scaler on {len(train_sensing_df)} train samples.")
    logger.info(f"Saved processed datasets and scalers to {proc_base}!")

if __name__ == "__main__":
    main()
