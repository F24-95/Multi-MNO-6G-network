"""
Denoising Bottleneck Autoencoder for RF spectrum snapshot compression.
Compresses a 64-dimensional spectrum snapshot (16 channels x 4 metrics: p_busy, snr, interference, noise)
into an 8-dimensional latent representation z_ae.
"""
import torch
import torch.nn as nn


class SnapshotEncoder(nn.Module):
    """Encodes 64-dim snapshot -> 8-dim latent space."""
    def __init__(self, input_dim: int = 64, hidden_dims: list = None, latent_dim: int = 8):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [32, 16]

        layers = []
        in_d = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(in_d, h),
                nn.ReLU(),
            ])
            in_d = h
        layers.append(nn.Linear(in_d, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SnapshotDecoder(nn.Module):
    """Decodes 8-dim latent vector -> 64-dim reconstructed snapshot with Sigmoid."""
    def __init__(self, latent_dim: int = 8, hidden_dims: list = None, output_dim: int = 64):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [16, 32]

        layers = []
        in_d = latent_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(in_d, h),
                nn.ReLU(),
            ])
            in_d = h
        layers.extend([
            nn.Linear(in_d, output_dim),
            nn.Sigmoid(),  # Output values bounded in [0, 1]
        ])
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Autoencoder(nn.Module):
    """
    Denoising Bottleneck Autoencoder.
    Input x: (N, 64) bounded in [0, 1]
    Latent z: (N, 8)
    Output x_hat: (N, 64) bounded in [0, 1]
    """
    def __init__(
        self,
        input_dim: int = 64,
        encoder_dims: list = None,
        latent_dim: int = 8,
        decoder_dims: list = None,
        corruption_noise_std: float = 0.05,
    ):
        super().__init__()
        enc_dims = encoder_dims or [32, 16]
        dec_dims = decoder_dims or [16, 32]
        self.corruption_noise_std = corruption_noise_std

        self.encoder = SnapshotEncoder(input_dim, enc_dims, latent_dim)
        self.decoder = SnapshotDecoder(latent_dim, dec_dims, input_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.training and self.corruption_noise_std > 0.0:
            noise = torch.randn_like(x) * self.corruption_noise_std
            x_noisy = torch.clamp(x + noise, 0.0, 1.0)
            z = self.encoder(x_noisy)
        else:
            z = self.encoder(x)

        x_hat = self.decoder(z)
        return z, x_hat

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Deterministic inference: encodes uncorrupted x into latent z_ae (N, 8)."""
        self.eval()
        with torch.no_grad():
            return self.encoder(x)

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample MSE reconstruction error for anomaly detection / monitoring."""
        self.eval()
        with torch.no_grad():
            z = self.encoder(x)
            x_hat = self.decoder(z)
            return torch.mean((x - x_hat) ** 2, dim=1)
