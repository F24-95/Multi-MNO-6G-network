"""
CLI: Run offline batch inference for trained VAE-GMM sensing model.
Generates and caches p_busy matrices (shape: T x C) for all scenarios and splits.
Usage:
    python scripts/run_sensing_inference.py --seed 42 --version v001
    python scripts/run_sensing_inference.py --seed 2026 --version v001
"""
import argparse
import pandas as pd
from dss.sensing.output import load_vae_gmm, infer_p_busy, save_sensing_output
from dss.common.paths import data_dir
from dss.common.logging_setup import setup_logger
from dss.data_generation.scenarios import list_scenarios

log = setup_logger("run_sensing_inference")


def run_inference_for_seed(seed: int = 42, version: str = "v001"):
    log.info(f"Loading VAE-GMM model version={version}")
    artifacts = load_vae_gmm(version)
    scenarios = list_scenarios()
    splits = ["train", "val", "test"]

    for scen in scenarios:
        p = data_dir("processed") / f"{scen}_s{seed}" / "channel_obs_processed.parquet"
        if not p.exists():
            log.warning(f"File not found: {p}")
            continue

        full_df = pd.read_parquet(p)
        for sp in splits:
            split_df = full_df[full_df["split"] == sp]
            if split_df.empty:
                continue
            p_busy = infer_p_busy(split_df, artifacts=artifacts)
            save_sensing_output(p_busy, scen, seed, sp)
            log.info(f"Generated p_busy for {scen} | seed={seed} | split={sp} | shape={p_busy.shape}")


def main():
    parser = argparse.ArgumentParser(description="Run VAE-GMM sensing inference and save p_busy")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--version", type=str, default="v001")
    args = parser.parse_args()

    run_inference_for_seed(seed=args.seed, version=args.version)


if __name__ == "__main__":
    main()
