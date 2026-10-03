import numpy as np
from dss.common.seed import rng_for
from dss.data_generation.pu_model import generate_pu_occupancy
from dss.data_generation.noise_model import generate_noise_and_interference
from dss.data_generation.demand_model import generate_mno_demand
from dss.data_generation.simulator import generate_dataset

def test_pu_model_duty_cycle():
    rng = rng_for(42, 0)
    duty_min = 0.20
    duty_max = 0.40
    n_steps = 10000
    n_channels = 16
    pu = generate_pu_occupancy(n_steps, n_channels, duty_min, duty_max, rng)

    assert pu.shape == (n_steps, n_channels)
    mean_duty = np.mean(pu)
    # Average duty cycle should be close to mid of [0.20, 0.40] = 0.30
    assert 0.15 < mean_duty < 0.45

def test_noise_model_statistics():
    rng = rng_for(42, 0)
    sigma_u = 0.5
    p_burst = 0.01
    n_steps = 5000
    n_channels = 16
    noise_true, noise_est, interference = generate_noise_and_interference(
        n_steps, n_channels, sigma_u, p_burst, rng
    )

    err = noise_est - noise_true
    std_err = np.std(err)
    # Estimation error std should be close to sigma_u
    assert abs(std_err - sigma_u) < 0.1

def test_demand_model_shapes_and_ranges():
    rng = rng_for(42, 0)
    n_steps = 288
    n_mno = 4
    users, traffic, req_bw = generate_mno_demand(
        n_steps=n_steps,
        n_mno=n_mno,
        scen_mult=1.0,
        spike_lambda_per_day=0.1,
        scenario_name="normal",
        rng=rng,
    )

    assert users.shape == (n_steps, n_mno)
    assert traffic.shape == (n_steps, n_mno)
    assert req_bw.shape == (n_steps, n_mno)
    assert np.all(users >= 0)
    assert np.all(traffic >= 0)
    assert np.all((req_bw >= 0) & (req_bw <= 160.0))

def test_simulator_row_counts_and_determinism():
    c_df1, m_df1, key1 = generate_dataset("normal", seed=42, save=False)
    c_df2, m_df2, key2 = generate_dataset("normal", seed=42, save=False)

    # Counts must match exactly ARCH §2: 4032 * 16 = 64,512 and 4032 * 4 = 16,128
    assert len(c_df1) == 64512
    assert len(m_df1) == 16128
    assert key1 == key2

    # Determinism assertion: same seed -> identical DataFrames
    pd_testing = np.testing
    pd_testing.assert_array_equal(c_df1["rx_power_dbm"].values, c_df2["rx_power_dbm"].values)
    pd_testing.assert_array_equal(m_df1["requested_bw_mhz"].values, m_df2["requested_bw_mhz"].values)
