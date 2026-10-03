import torch
import torch.nn as nn


class VAEEncoder(nn.Module):
    """Encoder: x(4) → [32 → 16] → (mu, log_var)(latent_dim)"""
    def __init__(self, input_dim: int = 4, hidden_dims: list = None, latent_dim: int = 2):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [32, 16]

        layers = []
        in_d = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_d, h), nn.ReLU()]
            in_d = h

        self.net    = nn.Sequential(*layers)
        self.fc_mu  = nn.Linear(in_d, latent_dim)
        self.fc_logvar = nn.Linear(in_d, latent_dim)

    def forward(self, x: torch.Tensor):
        h = self.net(x)
        return self.fc_mu(h), self.fc_logvar(h)


class VAEDecoder(nn.Module):
    """Decoder: z(latent_dim) → [16 → 32] → x_hat(input_dim)"""
    def __init__(self, latent_dim: int = 2, hidden_dims: list = None, output_dim: int = 4):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [16, 32]

        layers = []
        in_d = latent_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_d, h), nn.ReLU()]
            in_d = h
        layers.append(nn.Linear(in_d, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class VAE(nn.Module):
    """
    Variational Autoencoder for unsupervised spectrum sensing.
    Input: (N, 4) → [rx_power_dbm, noise_est_dbm, snr_est_db, interference_dbm]
    Output: mu(N, latent_dim), log_var(N, latent_dim), z(N, latent_dim), x_hat(N, 4)
    """
    def __init__(
        self,
        input_dim: int = 4,
        encoder_dims: list = None,
        latent_dim: int = 2,
        decoder_dims: list = None,
    ):
        super().__init__()
        enc_dims = encoder_dims or [32, 16]
        dec_dims = decoder_dims or [16, 32]

        self.encoder = VAEEncoder(input_dim, enc_dims, latent_dim)
        self.decoder = VAEDecoder(latent_dim, dec_dims, input_dim)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        if self.training:
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu  # deterministic at inference

    def forward(self, x: torch.Tensor):
        mu, log_var = self.encoder(x)
        z           = self.reparameterize(mu, log_var)
        x_hat       = self.decoder(z)
        return mu, log_var, z, x_hat

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Inference-only: returns mu (deterministic)."""
        self.eval()
        with torch.no_grad():
            mu, _ = self.encoder(x)
        return mu

    def recon_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample MSE reconstruction error — used for OOD detection."""
        self.eval()
        with torch.no_grad():
            mu, log_var, z, x_hat = self.forward(x)
        return ((x - x_hat) ** 2).mean(dim=1)


def vae_loss(x: torch.Tensor, x_hat: torch.Tensor, mu: torch.Tensor,
             log_var: torch.Tensor, beta: float = 1.0, kl_weight: float = 1.0) -> torch.Tensor:
    """
    ELBO loss = MSE reconstruction (sum over features) + beta * KL
    KL weight is increased linearly during warmup epochs (passed from trainer).
    """
    recon = nn.functional.mse_loss(x_hat, x, reduction="sum") / x.size(0)
    kl    = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
    return recon + beta * kl_weight * kl, recon.detach(), kl.detach()
