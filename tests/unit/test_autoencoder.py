"""
Unit tests for Denoising Bottleneck Autoencoder.
Tests architecture dimensions, deterministic evaluation, reconstruction loss,
and model persistence round-trip equality.
"""
import torch
import numpy as np
import pytest
from pathlib import Path

from dss.feature_extraction.autoencoder import Autoencoder, SnapshotEncoder, SnapshotDecoder
from dss.feature_extraction.encode import encode_snapshots


def test_autoencoder_shapes():
    model = Autoencoder(input_dim=64, encoder_dims=[32, 16], latent_dim=8, decoder_dims=[16, 32])
    x = torch.rand(32, 64)
    z, x_hat = model(x)

    assert z.shape == (32, 8), f"Expected latent shape (32, 8), got {z.shape}"
    assert x_hat.shape == (32, 64), f"Expected output shape (32, 64), got {x_hat.shape}"
    # Decoder output is Sigmoid -> strictly in [0, 1]
    assert (x_hat >= 0.0).all() and (x_hat <= 1.0).all()


def test_encoder_deterministic_in_eval():
    model = Autoencoder(input_dim=64, latent_dim=8)
    model.eval()
    x = torch.rand(16, 64)

    z1 = model.encode(x)
    z2 = model.encode(x)
    assert torch.allclose(z1, z2, atol=1e-7), "Autoencoder.encode must be strictly deterministic"


def test_denoising_corruption_only_in_training():
    model = Autoencoder(input_dim=64, latent_dim=8, corruption_noise_std=0.2)
    x = torch.full((10, 64), 0.5)

    # In eval mode, no corruption
    model.eval()
    with torch.no_grad():
        z_eval_1, _ = model(x)
        z_eval_2, _ = model(x)
    assert torch.allclose(z_eval_1, z_eval_2)

    # In training mode, stochastic corruption occurs
    model.train()
    torch.manual_seed(123)
    z_tr_1, _ = model(x)
    torch.manual_seed(456)
    z_tr_2, _ = model(x)
    assert not torch.allclose(z_tr_1, z_tr_2)


def test_reconstruction_error():
    model = Autoencoder(input_dim=64, latent_dim=8)
    model.eval()
    x = torch.rand(20, 64)
    errors = model.reconstruction_error(x)

    assert errors.shape == (20,)
    assert (errors >= 0.0).all()
    assert torch.isfinite(errors).all()


def test_model_save_load_roundtrip(tmp_path: Path):
    model = Autoencoder(input_dim=64, latent_dim=8)
    model.eval()
    x = torch.rand(8, 64)
    expected_z = model.encode(x)

    save_path = tmp_path / "test_ae.pt"
    torch.save(model.state_dict(), save_path)

    loaded_model = Autoencoder(input_dim=64, latent_dim=8)
    loaded_model.load_state_dict(torch.load(save_path, weights_only=True))
    loaded_model.eval()
    actual_z = loaded_model.encode(x)

    assert torch.allclose(expected_z, actual_z, atol=1e-7)
