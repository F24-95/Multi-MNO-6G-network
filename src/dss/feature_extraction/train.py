"""
Training pipeline for Denoising Bottleneck Autoencoder.
Trains on 64-dimensional spectrum snapshots from all 9 scenarios (train split only).
Guarantees strict train-only scaler fitting and zero data leakage.
"""
import json
import joblib
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

from dss.feature_extraction.autoencoder import Autoencoder
from dss.feature_extraction.snapshots import (
    load_raw_rf_matrices,
    transform_rf_matrices,
    build_scenario_snapshots,
)
from dss.common.paths import model_dir, data_dir
from dss.common.seed import set_all_seeds
from dss.common.logging_setup import setup_logger
from dss.data_generation.scenarios import list_scenarios
from dss.preprocessing.scale import fit_snapshot_scaler

log = setup_logger("train_autoencoder")


def fit_global_snapshot_scaler(seed: int = 42) -> MinMaxScaler:
    """
    Fits MinMaxScaler strictly on the train split across all 9 scenarios.
    No val or test data is ever seen during scaler fitting.
    """
    scenarios = list_scenarios()
    snr_list, int_list, noise_list = [], [], []

    for scen in scenarios:
        p = data_dir("processed") / f"{scen}_s{seed}" / "channel_obs_processed.parquet"
        import pandas as pd
        df = pd.read_parquet(p)
        df_train = df[df["split"] == "train"]
        s_m, i_m, n_m = load_raw_rf_matrices(df_train)
        snr_list.append(s_m)
        int_list.append(i_m)
        noise_list.append(n_m)

    all_snr = np.concatenate(snr_list, axis=0)
    all_int = np.concatenate(int_list, axis=0)
    all_noise = np.concatenate(noise_list, axis=0)

    scaler = fit_snapshot_scaler(all_snr, all_int, all_noise)
    log.info(f"Fitted global snapshot scaler on {len(all_snr)} train timesteps ({len(all_snr) * 16} channel samples).")
    return scaler


def load_dataset_snapshots(
    scenarios: list,
    seed: int,
    split: str,
    scaler: MinMaxScaler,
) -> np.ndarray:
    """Loads and stacks snapshots across scenarios for given split."""
    snapshots_list = []
    for scen in scenarios:
        snaps, _ = build_scenario_snapshots(scen, seed, split, scaler=scaler)
        snapshots_list.append(snaps)
    return np.concatenate(snapshots_list, axis=0)


def train_autoencoder(
    seed: int = 42,
    version: str = "v001",
    max_epochs: int = 100,
    patience: int = 10,
    batch_size: int = 256,
    lr: float = 1e-3,
    latent_dim: int = 8,
    corruption_noise_std: float = 0.05,
) -> dict:
    set_all_seeds(seed)
    scenarios = list_scenarios()

    # 1. Fit scaler on train split only
    scaler = fit_global_snapshot_scaler(seed=seed)

    # 2. Build train and validation datasets
    X_train = load_dataset_snapshots(scenarios, seed, "train", scaler)
    X_val = load_dataset_snapshots(scenarios, seed, "val", scaler)
    log.info(f"Train samples: {X_train.shape}, Val samples: {X_val.shape}")

    device = torch.device("cpu")
    model = Autoencoder(
        input_dim=64,
        encoder_dims=[32, 16],
        latent_dim=latent_dim,
        decoder_dims=[16, 32],
        corruption_noise_std=corruption_noise_std,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32, device=device)

    # Baseline: predict mean of training set
    train_mean = torch.mean(X_train_tensor, dim=0, keepdim=True)
    baseline_val_mse = criterion(train_mean.expand_as(X_val_tensor), X_val_tensor).item()
    log.info(f"Predict-the-mean baseline Val MSE: {baseline_val_mse:.6f}")

    best_val_loss = float("inf")
    best_weights = None
    no_improve = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        permutation = torch.randperm(len(X_train_tensor))
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(X_train_tensor), batch_size):
            indices = permutation[i : i + batch_size]
            batch = X_train_tensor[indices]

            optimizer.zero_grad()
            z, x_hat = model(batch)
            loss = criterion(x_hat, batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        train_loss = epoch_loss / max(1, n_batches)

        # Validation
        model.eval()
        with torch.no_grad():
            _, x_hat_val = model(X_val_tensor)
            val_loss = criterion(x_hat_val, X_val_tensor).item()

        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                log.info(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
                break

    # Restore best weights
    model.load_state_dict(best_weights)
    model.eval()

    # 3. Save artifacts
    out_dir = model_dir("ae", version)
    torch.save(model.state_dict(), out_dir / "autoencoder.pt")
    joblib.dump(scaler, out_dir / "snapshot_scaler.joblib")

    model_card = {
        "model_type": "denoising_bottleneck_autoencoder",
        "version": version,
        "seed": seed,
        "input_dim": 64,
        "latent_dim": latent_dim,
        "corruption_noise_std": corruption_noise_std,
        "best_val_mse": best_val_loss,
        "baseline_val_mse": baseline_val_mse,
        "epochs_trained": len(history),
        "train_samples": len(X_train),
        "val_samples": len(X_val),
    }
    (out_dir / "model_card.json").write_text(json.dumps(model_card, indent=2))
    log.info(f"[Autoencoder] Saved to {out_dir} | Val MSE: {best_val_loss:.6f} vs Baseline: {baseline_val_mse:.6f}")

    return {
        "model": model,
        "scaler": scaler,
        "best_val_loss": best_val_loss,
        "baseline_val_loss": baseline_val_mse,
        "history": history,
    }
