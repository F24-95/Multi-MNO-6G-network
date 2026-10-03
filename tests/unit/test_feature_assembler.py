"""
Unit tests for FeatureAssembler.
Tests 20-dim feature vector construction, bounds, and strict equality
between batch assembly and streaming (row-by-row) assembly.
"""
import numpy as np
import pytest
from dss.preprocessing.feature_assembler import FeatureAssembler, N_FEATURES, FEATURE_NAMES


def test_feature_assembler_batch_shape():
    assembler = FeatureAssembler()
    T = 100
    z_ae = np.random.uniform(0.0, 1.0, size=(T, 8)).astype(np.float32)
    demands = np.random.uniform(10.0, 120.0, size=(T, 4)).astype(np.float32)
    p_busy = np.random.uniform(0.0, 1.0, size=(T, 16)).astype(np.float32)
    steps = np.arange(T)

    features = assembler.assemble_batch(z_ae, demands, p_busy, steps)

    assert features.shape == (T, N_FEATURES), f"Expected shape ({T}, {N_FEATURES}), got {features.shape}"
    assert len(assembler.feature_names) == N_FEATURES


def test_streaming_vs_batch_feature_row_equality():
    assembler = FeatureAssembler()
    T = 25
    z_ae = np.random.uniform(0.0, 1.0, size=(T, 8)).astype(np.float32)
    demands = np.random.uniform(5.0, 150.0, size=(T, 4)).astype(np.float32)
    p_busy = np.random.uniform(0.0, 1.0, size=(T, 16)).astype(np.float32)
    steps = np.arange(T)

    # 1. Batch assembly
    batch_features = assembler.assemble_batch(z_ae, demands, p_busy, steps)

    # 2. Streaming assembly row-by-row
    streaming_rows = []
    for t in range(T):
        prev_dem = demands[t - 1] if t > 0 else None
        row = assembler.assemble_row(
            z_ae_row=z_ae[t],
            current_demand=demands[t],
            prev_demand=prev_dem,
            p_busy_row=p_busy[t],
            step_idx=steps[t],
        )
        streaming_rows.append(row)

    streaming_features = np.stack(streaming_rows, axis=0)

    # 3. Must be strictly equal within floating point precision
    np.testing.assert_allclose(
        batch_features,
        streaming_features,
        atol=1e-6,
        err_msg="Streaming row assembly must exactly match batch assembly output",
    )


def test_feature_normalization_bounds():
    assembler = FeatureAssembler(d_cap_mhz=160.0)
    T = 10
    z_ae = np.zeros((T, 8), dtype=np.float32)
    # Exceed capacity slightly to test clipping
    demands = np.array([[200.0, -10.0, 80.0, 160.0]] * T, dtype=np.float32)
    p_busy = np.full((T, 16), 0.5, dtype=np.float32)
    steps = np.arange(T)

    features = assembler.assemble_batch(z_ae, demands, p_busy, steps)

    # MNO normalized demands (features 8 to 11) must be clamped in [0, 1]
    assert np.all(features[:, 8:12] >= 0.0)
    assert np.all(features[:, 8:12] <= 1.0)
    # Check 200 clamped to 1.0 and -10 clamped to 0.0
    assert features[0, 8] == 1.0
    assert features[0, 9] == 0.0
