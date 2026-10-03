"""
Inference & encoding module for trained Autoencoder.
Compresses 64-dim spectrum snapshots to 8-dim latent vectors z_ae.
Guarantees determinism in eval mode and round-trip artifact loading.
"""
import joblib
import numpy as np
import torch
from pathlib import Path

from dss.feature_extraction.autoencoder import Autoencoder
from dss.feature_extraction.snapshots import build_scenario_snapshots
from dss.common.paths import model_dir, data_dir


def load_autoencoder(version: str = "v001") -> dict:
    """Loads Autoencoder model weights and snapshot scaler from model_dir."""
    d = model_dir("ae", version)
    weights_path = d / "autoencoder.pt"
    scaler_path = d / "snapshot_scaler.joblib"

    if not weights_path.exists() or not scaler_path.exists():
        raise FileNotFoundError(f"Autoencoder artifacts missing in {d}")

    model = Autoencoder(
        input_dim=64,
        encoder_dims=[32, 16],
        latent_dim=8,
        decoder_dims=[16, 32],
        corruption_noise_std=0.0,
    )
    model.load_state_dict(torch.load(weights_path, weights_only=True))
    model.eval()

    scaler = joblib.load(scaler_path)
    return {"model": model, "scaler": scaler}


def encode_snapshots(snapshots: np.ndarray, model: Autoencoder) -> np.ndarray:
    """
    Deterministically transforms (N, 64) snapshots into (N, 8) latent representations z_ae.
    """
    model.eval()
    with torch.no_grad():
        tensor_x = torch.tensor(snapshots, dtype=torch.float32)
        z = model.encode(tensor_x)
        return z.cpu().numpy().astype(np.float32)


def encode_scenario(
    scenario: str,
    seed: int,
    split: str,
    version: str = "v001",
    artifacts: dict | None = None,
) -> np.ndarray:
    """
    Loads snapshots for given scenario and split, and encodes them to (T, 8) z_ae array.
    """
    if artifacts is None:
        artifacts = load_autoencoder(version)

    snapshots, _ = build_scenario_snapshots(scenario, seed, split, scaler=artifacts["scaler"])
    return encode_snapshots(snapshots, artifacts["model"])


def save_z_ae(z_ae: np.ndarray, scenario: str, seed: int, split: str) -> Path:
    """Saves z_ae array to data/features/z_ae_<scenario>_s<seed>_<split>.npy."""
    out_dir = data_dir("features")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"z_ae_{scenario}_s{seed}_{split}.npy"
    np.save(out_path, z_ae)
    return out_path
