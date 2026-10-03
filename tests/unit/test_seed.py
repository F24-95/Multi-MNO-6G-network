import numpy as np
from dss.common.seed import set_all_seeds, rng_for

def test_rng_reproducibility():
    rng1 = rng_for(42, 0)
    val1 = rng1.random(5)

    rng2 = rng_for(42, 0)
    val2 = rng2.random(5)

    np.testing.assert_array_equal(val1, val2)

def test_different_scenario_indices():
    rng1 = rng_for(42, 0)
    val1 = rng1.random(5)

    rng2 = rng_for(42, 1)
    val2 = rng2.random(5)

    assert not np.array_equal(val1, val2)

def test_global_seeds():
    set_all_seeds(100)
    val1 = np.random.rand(5)

    set_all_seeds(100)
    val2 = np.random.rand(5)

    np.testing.assert_array_equal(val1, val2)
