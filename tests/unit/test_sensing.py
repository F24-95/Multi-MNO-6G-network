"""
Unit tests for Phase 4 sensing modules.
Tests VAE architecture, vae_loss, metrics, and inference pipeline.
"""
import numpy as np
import torch
import pytest
from dss.sensing.vae import VAE, VAEEncoder, VAEDecoder, vae_loss
from dss.sensing.metrics import detection_metrics, calibration_table


# ── VAE architecture tests ──────────────────────────────────────────────────


def test_vae_output_shapes():
    model = VAE(input_dim=4, encoder_dims=[32, 16], latent_dim=2, decoder_dims=[16, 32])
    x = torch.randn(64, 4)
    mu, log_var, z, x_hat = model(x)
    assert mu.shape     == (64, 2), f"mu shape: {mu.shape}"
    assert log_var.shape == (64, 2)
    assert z.shape       == (64, 2)
    assert x_hat.shape   == (64, 4)


def test_encoder_deterministic_at_eval():
    model = VAE()
    model.eval()
    x = torch.randn(32, 4)
    with torch.no_grad():
        mu1, lv1 = model.encoder(x)
        mu2, lv2 = model.encoder(x)
    assert torch.allclose(mu1, mu2)


def test_reparameterize_deterministic_at_eval():
    model = VAE()
    model.eval()
    x   = torch.randn(32, 4)
    mu  = model.encode(x)
    mu2 = model.encode(x)
    assert torch.allclose(mu, mu2), "encode() must be deterministic at eval"


def test_vae_loss_finite():
    model = VAE()
    x = torch.randn(128, 4)
    mu, log_var, z, x_hat = model(x)
    loss, recon, kl = vae_loss(x, x_hat, mu, log_var, beta=1.0, kl_weight=1.0)
    assert torch.isfinite(loss), f"Loss not finite: {loss}"
    assert recon >= 0
    assert kl    >= 0


def test_recon_error_shape():
    model = VAE()
    x = torch.randn(50, 4)
    err = model.recon_error(x)
    assert err.shape == (50,)
    assert (err >= 0).all()


# ── Metrics tests ────────────────────────────────────────────────────────────


def test_detection_metrics_perfect():
    pu_true = np.array([1, 1, 0, 0])
    p_busy  = np.array([0.9, 0.8, 0.1, 0.2])
    m = detection_metrics(p_busy, pu_true, threshold=0.5)
    assert m["Pd"] == pytest.approx(1.0)
    assert m["Pf"] == pytest.approx(0.0)
    assert m["AUC"] == pytest.approx(1.0)
    assert m["Brier"] < 0.1


def test_detection_metrics_random():
    rng     = np.random.default_rng(42)
    pu_true = rng.integers(0, 2, size=200)
    p_busy  = rng.uniform(0, 1, size=200)
    m = detection_metrics(p_busy, pu_true)
    for key in ("Pd", "Pf", "AUC", "Brier"):
        assert 0.0 <= m[key] <= 1.0, f"{key}={m[key]} out of range"


def test_calibration_table_sums():
    rng     = np.random.default_rng(0)
    pu_true = rng.integers(0, 2, size=500)
    p_busy  = rng.uniform(0, 1, size=500)
    table   = calibration_table(p_busy, pu_true, n_bins=10)
    total   = sum(row["count"] for row in table)
    assert total == 500
    for row in table:
        assert 0.0 <= row["frac_busy"] <= 1.0
