"""
CLI: Train VAE + GMM for spectrum sensing.
Usage:
    python scripts/train_vae_gmm.py
    python scripts/train_vae_gmm.py --seed 2026 --version v002
"""
import argparse
import json
from dss.sensing.train import train_vae_gmm
from dss.common.logging_setup import setup_logger

log = setup_logger("train_vae_gmm")


def main():
    p = argparse.ArgumentParser(description="Train VAE + GMM for spectrum sensing")
    p.add_argument("--seed",    type=int,   default=42)
    p.add_argument("--version", type=str,   default="v001")
    p.add_argument("--epochs",  type=int,   default=100)
    p.add_argument("--batch",   type=int,   default=512)
    p.add_argument("--lr",      type=float, default=1e-3)
    p.add_argument("--latent",  type=int,   default=2)
    p.add_argument("--beta",    type=float, default=1.0)
    p.add_argument("--warmup",  type=int,   default=10)
    p.add_argument("--gmm-k",   type=int,   default=2, dest="gmm_k")
    args = p.parse_args()

    log.info(f"Starting VAE-GMM training | seed={args.seed} version={args.version}")
    result = train_vae_gmm(
        seed=args.seed,
        version=args.version,
        max_epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        latent_dim=args.latent,
        beta=args.beta,
        kl_warmup_epochs=args.warmup,
        gmm_components=args.gmm_k,
    )
    log.info(f"Training complete | epochs={len(result['history'])} | best_val_loss={result['history'][-1]['val_loss']:.4f}")


if __name__ == "__main__":
    main()
