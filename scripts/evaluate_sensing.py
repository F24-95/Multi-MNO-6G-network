"""
CLI: Evaluate trained VAE-GMM sensing model.
Computes Pd, Pf, AUC, Brier, calibration table for each scenario × split.
Usage:
    python scripts/evaluate_sensing.py
    python scripts/evaluate_sensing.py --seed 42 --version v001 --split val
"""
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path

from dss.sensing.output import load_vae_gmm, infer_p_busy
from dss.sensing.metrics import detection_metrics, calibration_table
from dss.preprocessing.scale import SENSING_COLS
from dss.common.paths import data_dir
from dss.common.logging_setup import setup_logger
from dss.data_generation.scenarios import list_scenarios

log = setup_logger("evaluate_sensing")


def evaluate_scenario(scen: str, seed: int, split: str, artifacts: dict) -> dict:
    p = data_dir("processed") / f"{scen}_s{seed}" / "channel_obs_processed.parquet"
    if not p.exists():
        log.warning(f"Missing: {p}")
        return {}
    df = pd.read_parquet(p)
    df = df[df["split"] == split]
    if df.empty:
        log.warning(f"Empty split '{split}' for {scen}")
        return {}

    p_busy_mat = infer_p_busy(df, artifacts=artifacts)            # (T, C)
    p_busy_flat = p_busy_mat.ravel()

    # Reshape pu_state to (T, C) matching p_busy
    steps    = df["step_idx"].values
    channels = df["channel_id"].values
    T = int(steps.max() - steps.min() + 1)
    C = int(channels.max() + 1)
    pu_mat = np.zeros((T, C), dtype=np.int8)
    pu_mat[steps - steps.min(), channels] = df["pu_state"].values.astype(np.int8)
    pu_flat = pu_mat.ravel()

    metrics = detection_metrics(p_busy_flat, pu_flat)
    cal     = calibration_table(p_busy_flat, pu_flat)
    return {"scenario": scen, "split": split, **metrics, "calibration": cal}


def main():
    p = argparse.ArgumentParser(description="Evaluate VAE-GMM spectrum sensing")
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--version", type=str, default="v001")
    p.add_argument("--split",   type=str, default="val", choices=["train", "val", "test"])
    args = p.parse_args()

    log.info(f"Loading VAE-GMM v={args.version} seed={args.seed}")
    artifacts = load_vae_gmm(args.version)
    scenarios = list_scenarios()

    results = []
    for scen in scenarios:
        r = evaluate_scenario(scen, args.seed, args.split, artifacts)
        if r:
            results.append(r)
            log.info(f"[{scen}] Pd={r['Pd']:.3f} Pf={r['Pf']:.3f} AUC={r['AUC']:.3f} Brier={r['Brier']:.4f}")

    # Summary
    if results:
        avg_pd  = np.mean([r["Pd"]    for r in results])
        avg_pf  = np.mean([r["Pf"]    for r in results])
        avg_auc = np.mean([r["AUC"]   for r in results])
        avg_br  = np.mean([r["Brier"] for r in results])
        log.info(f"\n=== OVERALL ({args.split}) ===")
        log.info(f"Pd={avg_pd:.3f} | Pf={avg_pf:.3f} | AUC={avg_auc:.3f} | Brier={avg_br:.4f}")

        # Save results
        out_path = data_dir("sensing") / f"eval_{args.split}_s{args.seed}.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Remove calibration for summary JSON (too verbose)
        summary = [{k: v for k, v in r.items() if k != "calibration"} for r in results]
        out_path.write_text(json.dumps(summary, indent=2))
        log.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
