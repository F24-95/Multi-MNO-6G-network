import argparse
from dss.data_generation.scenarios import list_scenarios
from dss.data_generation.simulator import generate_dataset
from dss.common.logging_setup import setup_logger

logger = setup_logger("generate_data")

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic dataset for 6G DSS.")
    parser.add_argument("--scenario", type=str, default="all", help="Scenario name or 'all'")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (e.g. 42 or 2026)")
    parser.add_argument("--fault-rate", type=float, default=0.0, help="Fault injection rate (default 0)")
    args = parser.parse_args()

    scenarios = list_scenarios() if args.scenario == "all" else [args.scenario]
    logger.info(f"Generating data for scenarios: {scenarios} with seed: {args.seed}")

    for scen in scenarios:
        logger.info(f"Generating scenario '{scen}' (seed={args.seed})...")
        c_df, m_df, key = generate_dataset(
            scenario_name=scen,
            seed=args.seed,
            save=True,
            fault_rate=args.fault_rate,
        )
        logger.info(f"Scenario '{scen}' finished -> key='{key}' (ChannelObs: {len(c_df)} rows, MnoDemand: {len(m_df)} rows)")

    logger.info("All requested datasets successfully generated!")

if __name__ == "__main__":
    main()
