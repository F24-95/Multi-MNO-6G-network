import pytest
import torch
import numpy as np
import pandas as pd
from pathlib import Path

from dss.common.config import get_config
from dss.common.paths import PROJECT_ROOT, data_dir
from dss.database.connection import get_connection, get_db_url
from dss.data_generation.simulator import generate_dataset
from dss.preprocessing.clean import clean_channel_obs
from dss.preprocessing.scale import fit_sensing_scaler, SENSING_COLS
from dss.preprocessing.split import get_split_subsets
from dss.sensing.vae import VAE
from dss.feature_extraction.autoencoder import Autoencoder
from dss.feature_extraction.snapshots import build_snapshots_from_df
from dss.forecasting.models import LSTMForecaster, GRUForecaster
from dss.decision_engine import StaticAllocator, AIDynamicAllocator, compute_jain_fairness
from dss.api.app import app


def test_config_and_paths():
    cfg = get_config()
    assert cfg.sim.n_channels == 16
    assert cfg.sim.n_mno == 4
    assert PROJECT_ROOT.exists()


def test_db_connection():
    db_url = get_db_url()
    assert db_url is not None
    conn = get_connection()
    assert conn is not None
    cursor = conn.cursor()
    cursor.execute("SELECT 1;")
    result = cursor.fetchone()
    assert result[0] == 1
    conn.close()


def test_data_generation_and_cleaning():
    channel_obs_df, mno_demand_df, dataset_key = generate_dataset(
        scenario_name="normal",
        seed=42,
        save=False
    )
    assert len(channel_obs_df) > 0
    assert len(mno_demand_df) > 0
    assert dataset_key is not None

    cleaned_df, _ = clean_channel_obs(channel_obs_df)
    assert cleaned_df["noise_est_dbm"].isna().sum() == 0


def test_preprocessing_and_splitting():
    channel_obs_df, _, _ = generate_dataset("normal", seed=42, save=False)
    cleaned_df, _ = clean_channel_obs(channel_obs_df)
    train_df, val_df, test_df = get_split_subsets(cleaned_df)

    assert len(train_df) > 0
    assert len(val_df) > 0
    assert len(test_df) > 0

    scaler = fit_sensing_scaler(train_df)
    scaled_feats = scaler.transform(train_df[SENSING_COLS].values)
    assert scaled_feats.shape[1] == 4


def test_vae_model_forward():
    input_dim = 4
    latent_dim = 2
    model = VAE(input_dim=input_dim, latent_dim=latent_dim)

    x = torch.randn(10, input_dim)
    mu, logvar, z, recon = model(x)

    assert recon.shape == (10, input_dim)
    assert mu.shape == (10, latent_dim)
    assert logvar.shape == (10, latent_dim)


def test_autoencoder_forward():
    channel_obs_df, _, _ = generate_dataset("normal", seed=42, save=False)
    snapshots, _ = build_snapshots_from_df(channel_obs_df)

    input_dim = snapshots.shape[1]
    bottleneck_dim = 8
    model = Autoencoder(input_dim=input_dim, latent_dim=bottleneck_dim)

    x = torch.tensor(snapshots[:5], dtype=torch.float32)
    encoded = model.encode(x)
    z, recon = model(x)

    assert encoded.shape == (5, bottleneck_dim)
    assert recon.shape == (5, input_dim)


def test_lstm_forecaster_forward():
    batch_size, seq_len, input_dim, horizon, output_dim = 8, 12, 4, 12, 4
    model = LSTMForecaster(input_dim=input_dim, hidden_dim=32, num_layers=1, horizon=horizon, output_dim=output_dim)
    x = torch.randn(batch_size, seq_len, input_dim)
    out = model(x)
    assert out.shape == (batch_size, horizon, output_dim)


def test_gru_forecaster_forward():
    batch_size, seq_len, input_dim, horizon, output_dim = 8, 12, 4, 12, 4
    model = GRUForecaster(input_dim=input_dim, hidden_dim=32, num_layers=1, horizon=horizon, output_dim=output_dim)
    x = torch.randn(batch_size, seq_len, input_dim)
    out = model(x)
    assert out.shape == (batch_size, horizon, output_dim)


def test_allocators_and_jain_fairness():
    static_alloc = StaticAllocator(16, 4)
    dynamic_alloc = AIDynamicAllocator(16, 4, p_busy_threshold=0.5)

    p_busy = np.array([0.1, 0.2, 0.8, 0.9, 0.1, 0.1, 0.2, 0.1, 0.1, 0.2, 0.1, 0.1, 0.9, 0.9, 0.8, 0.9])
    req_bw = np.array([10.0, 20.0, 5.0, 15.0])

    s_out = static_alloc.allocate_step(p_busy, req_bw)
    d_out = dynamic_alloc.allocate_step(p_busy, req_bw)

    assert s_out.shape == (4,)
    assert d_out.shape == (4,)
    assert np.all(d_out >= 0.0)

    jain_score = compute_jain_fairness(np.array([0.8, 0.9, 0.85, 0.95]))
    assert 0.0 <= jain_score <= 1.0


def test_flask_api_endpoints():
    client = app.test_client()

    r_health = client.get("/api/health")
    assert r_health.status_code == 200
    assert r_health.json["status"] == "healthy"

    r_scenarios = client.get("/api/scenarios")
    assert r_scenarios.status_code == 200
    assert "scenarios" in r_scenarios.json

    r_metrics = client.get("/api/metrics?scenario=normal")
    assert r_metrics.status_code == 200
    assert "dynamic_satisfaction" in r_metrics.json

    r_alloc = client.get("/api/allocation?scenario=normal")
    assert r_alloc.status_code == 200
    assert "allocated_bw" in r_alloc.json
