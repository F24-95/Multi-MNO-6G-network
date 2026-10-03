import pytest
from dss.data_generation.simulator import generate_dataset
from dss.preprocessing.validate import validate_dataset

@pytest.mark.parametrize("scen", [
    "normal", "low_traffic", "high_traffic", "traffic_spike",
    "spectrum_shortage", "uneven_mno_demand", "noise_uncertainty",
    "congestion", "demand_shift"
])
def test_all_scenarios_validation_gate(scen):
    c_df, m_df, key = generate_dataset(scen, seed=42, save=False)
    report = validate_dataset(c_df, m_df, scen)
    assert report["passed"], f"Validation failed for scenario {scen}: {report['checks']}"
