"""
CLI: Run offline batch feature extraction using trained Autoencoder.
Extracts and saves z_ae latent vectors of shape (T, 8) for all scenarios and splits.
Usage:
    python scripts/run_feature_extraction.py --seed 42 --version v001
"""
import argparse
from dss.feature_extraction.encode import load_autoencoder, encode_scenario, save_z_ae
from dss.common.logging_setup import setup_logger
from dss.data_generation.scenarios import list_scenarios

log = setup_logger("run_feature_extraction_cli")


def run_feature_extraction_for_seed(seed: int = 42, version: str = "v001"):
    log.info(f"Loading Autoencoder model version={version}...")
    artifacts = load_autoencoder(version)
    scenarios = list_scenarios()
    splits = ["train", "val", "test"]

    for scen in scenarios:
        for sp in splits:
            z_ae = encode_scenario(scen, seed, sp, artifacts=artifacts)
            out_path = save_z_ae(z_ae, scen, seed, sp)
            log.info(f"Extracted z_ae for {scen} | {sp} | shape={z_ae.shape} -> {out_path.name}")


def main():
    parser = argparse.ArgumentParser(description="Extract z_ae latent features across scenarios")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--version", type=str, default="v001")
    args = parser.parse_args()

    run_feature_extraction_for_seed(seed=args.seed, version=args.version)


if __name__ == "__main__":
    main()
