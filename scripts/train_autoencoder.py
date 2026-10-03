"""
CLI: Train Denoising Bottleneck Autoencoder for spectrum snapshot compression.
Usage:
    python scripts/train_autoencoder.py --seed 42 --version v001
"""
import argparse
from dss.feature_extraction.train import train_autoencoder
from dss.common.logging_setup import setup_logger

log = setup_logger("train_autoencoder_cli")


def main():
    parser = argparse.ArgumentParser(description="Train Denoising Bottleneck Autoencoder")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--version", type=str, default="v001")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent", type=int, default=8)
    parser.add_argument("--noise-std", type=float, default=0.05, dest="noise_std")
    parser.add_argument("--patience", type=int, default=10)
    args = parser.parse_args()

    log.info(f"Starting Autoencoder training | seed={args.seed} version={args.version}")
    result = train_autoencoder(
        seed=args.seed,
        version=args.version,
        max_epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch,
        lr=args.lr,
        latent_dim=args.latent,
        corruption_noise_std=args.noise_std,
    )
    log.info(
        f"Autoencoder training finished successfully | "
        f"Best Val MSE: {result['best_val_loss']:.6f} vs Baseline: {result['baseline_val_loss']:.6f}"
    )


if __name__ == "__main__":
    main()
