import argparse
import sys
import pandas as pd
from dss.common.paths import data_dir
from dss.data_generation.scenarios import list_scenarios
from dss.preprocessing.validate import validate_dataset
from dss.common.logging_setup import setup_logger

logger = setup_logger("validate_data")

def main():
    parser = argparse.ArgumentParser(description="Validate generated datasets against quality gates.")
    parser.add_argument("--seed", type=int, default=42, help="Seed to validate (default 42)")
    args = parser.parse_args()

    scenarios = list_scenarios()
    logger.info(f"Validating {len(scenarios)} datasets for seed={args.seed}...")

    all_passed = True
    for scen in scenarios:
        # Find directory matching scenario and seed
        raw_dir = data_dir("raw")
        pattern = f"ds_{scen}_s{args.seed}_*"
        matching = list(raw_dir.glob(pattern))

        if not matching:
            logger.error(f"Dataset for scenario '{scen}' with seed {args.seed} not found in {raw_dir}!")
            all_passed = False
            continue

        ds_path = matching[0]
        c_path = ds_path / "channel_obs.parquet"
        m_path = ds_path / "mno_demand.parquet"

        if not c_path.exists() or not m_path.exists():
            logger.error(f"Missing parquet files in {ds_path}!")
            all_passed = False
            continue

        c_df = pd.read_parquet(c_path)
        m_df = pd.read_parquet(m_path)

        report = validate_dataset(
            channel_obs_df=c_df,
            mno_demand_df=m_df,
            scenario_name=scen,
            output_path=ds_path / "quality_report.json",
        )

        status_str = "PASSED" if report["passed"] else "FAILED"
        logger.info(f"[{status_str}] Scenario '{scen}' (Key: {ds_path.name})")

        if not report["passed"]:
            all_passed = False
            for chk, details in report["checks"].items():
                if not details["passed"]:
                    logger.warning(f"   -> Failed check: {chk}: {details.get('details')}")

    if all_passed:
        logger.info("ALL DATASET VALIDATION CHECKS PASSED!")
    else:
        logger.error("SOME DATASET VALIDATION CHECKS FAILED.")
        sys.exit(1)

if __name__ == "__main__":
    main()
