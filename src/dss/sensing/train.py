import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from dss.sensing.vae import VAE, vae_loss
from dss.preprocessing.scale import SENSING_COLS
from dss.common.config import get_config
from dss.common.paths import model_dir, data_dir
from dss.common.seed import set_all_seeds


def build_sensing_arrays(
    processed_dir: Path,
    scenarios: list,
    seed: int,
    split: str,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reads processed channel_obs parquet files for given scenarios/split.
    Returns (X, pu_labels). Parquet column is 'step_idx' (not 'step').
    """
    X_list, y_list = [], []
    for scen in scenarios:
        p = processed_dir / f"{scen}_s{seed}" / "channel_obs_processed.parquet"
        if not p.exists():
            print(f"[WARN] Missing: {p}")
            continue
        df = pd.read_parquet(p)
        df = df[df["split"] == split]
        if df.empty:
            print(f"[WARN] Empty split '{split}' for {scen}_s{seed}")
            continue
        X_list.append(df[SENSING_COLS].values.astype(np.float32))
        y_list.append(df["pu_state"].values.astype(np.int8))
    if not X_list:
        raise RuntimeError(f"No data found in {processed_dir} for seed={seed} split={split}")
    return np.concatenate(X_list), np.concatenate(y_list)


def train_vae(
    model: VAE,
    X_train: np.ndarray,
    X_val: np.ndarray = None,
    epochs: int = 25,
    batch_size: int = 256,
    lr: float = 1e-3,
    device: str = "cpu",
    beta: float = 1.0,
    kl_warmup_epochs: int = 10,
) -> tuple[VAE, dict]:
    """
    Trains a VAE model in-memory with mini-batching and ELBO loss tracking.
    Compatible with notebook usage and interactive experimentation.
    Returns:
        (model, history) where history is a dict with 'train_loss' and 'val_loss' lists.
    """
    dev = torch.device(device)
    model.to(dev)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    X_t = torch.tensor(X_train, dtype=torch.float32, device=dev)
    Xv_t = torch.tensor(X_val, dtype=torch.float32, device=dev) if X_val is not None else None

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        kl_w = min(1.0, epoch / max(1, kl_warmup_epochs))

        idx = torch.randperm(len(X_t))
        ep_loss = 0.0
        n_batches = 0

        for i in range(0, len(X_t), batch_size):
            batch = X_t[idx[i : i + batch_size]]
            optim.zero_grad()
            mu, log_var, z, x_hat = model(batch)
            loss, _, _ = vae_loss(batch, x_hat, mu, log_var, beta=beta, kl_weight=kl_w)
            loss.backward()
            optim.step()
            ep_loss += loss.item()
            n_batches += 1

        avg_train_loss = ep_loss / max(1, n_batches)
        history["train_loss"].append(avg_train_loss)

        if Xv_t is not None:
            model.eval()
            with torch.no_grad():
                mu_v, lv_v, _, xh_v = model(Xv_t)
                val_loss, _, _ = vae_loss(Xv_t, xh_v, mu_v, lv_v, beta=beta, kl_weight=1.0)
                history["val_loss"].append(val_loss.item())

    return model, history


def train_vae_gmm(
    seed: int = 42,
    version: str = "v001",
    max_epochs: int = 100,
    patience: int = 10,
    batch_size: int = 512,
    lr: float = 1e-3,
    latent_dim: int = 2,
    kl_warmup_epochs: int = 10,
    beta: float = 1.0,
    gmm_components: int = 2,
) -> dict:
    set_all_seeds(seed)
    cfg = get_config()
    from dss.data_generation.scenarios import list_scenarios
    scenarios = list_scenarios()

    proc_dir   = data_dir("processed")
    X_train, _ = build_sensing_arrays(proc_dir, scenarios, seed, "train")
    X_val,   _ = build_sensing_arrays(proc_dir, scenarios, seed, "val")

    # Fit StandardScaler on train only
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train).astype(np.float32)
    X_val_s   = scaler.transform(X_val).astype(np.float32)

    device = torch.device("cpu")
    model  = VAE(input_dim=4, encoder_dims=[32, 16], latent_dim=latent_dim, decoder_dims=[16, 32]).to(device)
    optim  = torch.optim.Adam(model.parameters(), lr=lr)

    X_t  = torch.tensor(X_train_s, device=device)
    Xv_t = torch.tensor(X_val_s,   device=device)

    best_val_loss = float("inf")
    best_state    = None
    no_improve    = 0
    history       = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        kl_w = min(1.0, epoch / kl_warmup_epochs)  # linear warmup

        # Mini-batch training
        idx      = torch.randperm(len(X_t))
        ep_loss  = 0.0
        n_batches = 0
        for i in range(0, len(X_t), batch_size):
            batch = X_t[idx[i : i + batch_size]]
            optim.zero_grad()
            mu, log_var, z, x_hat = model(batch)
            loss, _, _ = vae_loss(batch, x_hat, mu, log_var, beta=beta, kl_weight=kl_w)
            loss.backward()
            optim.step()
            ep_loss  += loss.item()
            n_batches += 1

        # Validation ELBO
        model.eval()
        with torch.no_grad():
            mu_v, lv_v, _, xh_v = model(Xv_t)
            val_loss, _, _ = vae_loss(Xv_t, xh_v, mu_v, lv_v, beta=beta, kl_weight=1.0)
        val_loss = val_loss.item()
        history.append({"epoch": epoch, "train_loss": ep_loss / n_batches, "val_loss": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve    = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Restore best weights
    model.load_state_dict(best_state)

    # --- Fit GMM on train mu ---
    model.eval()
    with torch.no_grad():
        mu_train = model.encoder(X_t)[0].numpy()

    gmm = GaussianMixture(
        n_components=gmm_components,
        covariance_type="full",
        n_init=5,
        random_state=seed,
    )
    gmm.fit(mu_train)

    # --- Save artifacts ---
    out_dir = model_dir("vae_gmm", version)
    torch.save(model.state_dict(), out_dir / "vae.pt")

    import joblib
    joblib.dump({"scaler": scaler, "gmm": gmm}, out_dir / "scaler_gmm.joblib")

    model_card = {
        "model_type": "vae_gmm",
        "version": version,
        "seed": seed,
        "latent_dim": latent_dim,
        "gmm_components": gmm_components,
        "best_val_loss": best_val_loss,
        "epochs_trained": len(history),
        "train_samples": len(X_train),
    }
    (out_dir / "model_card.json").write_text(json.dumps(model_card, indent=2))

    print(f"[VAE-GMM] Saved to {out_dir} | val_loss={best_val_loss:.4f} | epochs={len(history)}")
    return {"model": model, "gmm": gmm, "scaler": scaler, "history": history}
