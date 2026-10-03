"""
CLI: Assemble sliding forecast windows (X: B x 12 x 20, Y: B x 3 x 4)
for Train, Val, and Test splits using FeatureAssembler.
Saves to data/processed/windows_{train,val,test}.npz.
Usage:
    python scripts/assemble_forecast_windows.py --seed 42
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from dss.preprocessing.feature_assembler import FeatureAssembler
from dss.preprocessing.windows import build_sliding_windows
from dss.common.paths import data_dir
from dss.common.logging_setup import setup_logger
from dss.data_generation.scenarios import list_scenarios

log = setup_logger("assemble_windows")


def load_mno_demand_matrix(df_split: pd.DataFrame, n_mno: int = 4) -> np.ndarray:
    """Extracts (T, 4) bandwidth demand matrix (in MHz)."""
    steps = df_split["step_idx"].values
    mnos = df_split["mno_id"].values
    T = int(steps.max() - steps.min() + 1)

    mat = np.zeros((T, n_mno), dtype=np.float32)
    t_idx = steps - steps.min()
    m_idx = mnos.astype(int)
    mat[t_idx, m_idx] = df_split["requested_bw_mhz"].values.astype(np.float32)
    return mat


def assemble_windows_for_split(
    scenarios: list,
    seed: int,
    split: str,
    seq_len: int = 12,
    horizon: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    assembler = FeatureAssembler(d_cap_mhz=160.0, n_mno=4, n_channels=16)

    all_X, all_Y, all_scens = [], [], []

    for scen in scenarios:
        # 1. Load z_ae (T, 8)
        z_ae_path = data_dir("features") / f"z_ae_{scen}_s{seed}_{split}.npy"
        if not z_ae_path.exists():
            raise FileNotFoundError(f"Missing {z_ae_path}. Run scripts/run_feature_extraction.py first.")
        z_ae = np.load(z_ae_path)

        # 2. Load p_busy (T, 16)
        p_busy_path = data_dir("sensing") / f"{scen}_s{seed}" / f"{split}_p_busy.npy"
        if not p_busy_path.exists():
            raise FileNotFoundError(f"Missing {p_busy_path}. Run scripts/run_sensing_inference.py first.")
        p_busy = np.load(p_busy_path)

        # 3. Load demand (T, 4)
        demand_path = data_dir("processed") / f"{scen}_s{seed}" / "mno_demand_processed.parquet"
        df_dem = pd.read_parquet(demand_path)
        df_dem_split = df_dem[df_dem["split"] == split]
        demand_mat = load_mno_demand_matrix(df_dem_split, n_mno=4)

        # 4. Step indices
        steps = np.sort(df_dem_split["step_idx"].unique())

        # 5. Assemble 20 features (T, 20)
        features = assembler.assemble_batch(z_ae, demand_mat, p_busy, steps)

        # 6. Build sliding windows (B, 12, 20) and (B, 3, 4)
        X, Y, _ = build_sliding_windows(features, demand_mat, seq_len=seq_len, horizon=horizon)

        all_X.append(X)
        all_Y.append(Y)
        all_scens.extend([scen] * len(X))

    X_cat = np.concatenate(all_X, axis=0)
    Y_cat = np.concatenate(all_Y, axis=0)
    scens_arr = np.array(all_scens)

    return X_cat, Y_cat, scens_arr


def main():
    parser = argparse.ArgumentParser(description="Assemble forecast sliding windows")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seq-len", type=int, default=12, dest="seq_len")
    parser.add_argument("--horizon", type=int, default=3)
    args = parser.parse_args()

    scenarios = list_scenarios()
    splits = ["train", "val", "test"]

    for sp in splits:
        log.info(f"Assembling windows for split={sp}...")
        X, Y, scens = assemble_windows_for_split(
            scenarios=scenarios,
            seed=args.seed,
            split=sp,
            seq_len=args.seq_len,
            horizon=args.horizon,
        )
        out_path = data_dir("processed") / f"windows_{sp}.npz"
        np.savez_compressed(out_path, X=X, Y=Y, scenarios=scens)
        log.info(f"[SAVED] {out_path.name} | X: {X.shape} | Y: {Y.shape} | total windows: {len(X)}")


if __name__ == "__main__":
    main()
