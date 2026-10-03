"""
Sensing output: converts GMM posterior into p_busy(T, C) tensor.
Also handles the full inference pipeline: load model → encode → GMM → p_busy.
No duplicate scaler/path logic — uses existing modules.
"""
import numpy as np
import torch
import joblib
from pathlib import Path

from dss.sensing.vae import VAE
from dss.common.paths import model_dir, data_dir
from dss.preprocessing.scale import SENSING_COLS


def load_vae_gmm(version: str = "v001") -> dict:
    """
    Loads VAE weights + scaler + GMM from model_dir.
    Returns dict with keys: 'model', 'scaler', 'gmm'
    """
    d = model_dir("vae_gmm", version)
    model = VAE(input_dim=4, encoder_dims=[32, 16], latent_dim=2, decoder_dims=[16, 32])
    model.load_state_dict(torch.load(d / "vae.pt", weights_only=True))
    model.eval()
    artifacts = joblib.load(d / "scaler_gmm.joblib")
    return {"model": model, "scaler": artifacts["scaler"], "gmm": artifacts["gmm"]}


def _busy_component_idx(gmm, scaler_labels: np.ndarray = None) -> int:
    """
    Identify which GMM component index = 'channel busy'.
    Busy state has higher SNR/rx_power → higher mean in that latent dimension.
    We use the component with higher mean norm as the 'busy' one.
    """
    means = gmm.means_  # (n_components, latent_dim)
    norms = np.linalg.norm(means, axis=1)
    return int(np.argmax(norms))


def infer_p_busy(
    channel_obs_df,
    version: str = "v001",
    artifacts: dict = None,
) -> np.ndarray:
    """
    Given a channel_obs DataFrame (already split/filtered), returns
    p_busy array of shape (T, C) where T=timesteps, C=n_channels.

    - channel_obs_df must have columns: SENSING_COLS + ['step', 'channel_id']
    - p_busy[t, c] = posterior probability of busy for timestep t, channel c
    """
    if artifacts is None:
        artifacts = load_vae_gmm(version)

    model  = artifacts["model"]
    scaler = artifacts["scaler"]
    gmm    = artifacts["gmm"]

    # Scale input
    X = channel_obs_df[SENSING_COLS].values.astype(np.float32)
    X_s = scaler.transform(X).astype(np.float32)

    # Encode → latent mu
    mu = model.encode(torch.tensor(X_s)).numpy()  # (N, latent_dim)

    # GMM posterior P(component=k | z) — shape (N, n_components)
    posteriors = gmm.predict_proba(mu)            # (N, n_components)

    # Busy component index
    busy_idx = _busy_component_idx(gmm)
    p_busy_flat = posteriors[:, busy_idx]         # (N,)

    # Reshape to (T, C)
    steps      = channel_obs_df["step_idx"].values
    channels   = channel_obs_df["channel_id"].values
    T          = int(steps.max() - steps.min() + 1)
    C          = int(channels.max() + 1)

    p_busy = np.zeros((T, C), dtype=np.float32)
    t_idx  = (steps - steps.min()).astype(int)
    c_idx  = channels.astype(int)
    p_busy[t_idx, c_idx] = p_busy_flat

    return p_busy  # shape (T, C)


def save_sensing_output(p_busy: np.ndarray, scenario: str, seed: int, split: str):
    """Saves p_busy to data/sensing/<scenario>_s<seed>/<split>_p_busy.npy"""
    out_dir = data_dir("sensing") / f"{scenario}_s{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{split}_p_busy.npy"
    np.save(path, p_busy)
    print(f"[SENSING] Saved p_busy {p_busy.shape} -> {path}")
    return path
