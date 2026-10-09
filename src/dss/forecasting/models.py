import torch
import torch.nn as nn


class LSTMForecaster(nn.Module):
    """
    LSTM-based Multi-Step Time-Series Forecaster.
    Input: (B, seq_len, input_dim)
    Output: (B, horizon, output_dim)
    """
    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 64,
        num_layers: int = 2,
        horizon: int = 12,
        output_dim: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.horizon = horizon
        self.output_dim = output_dim

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, horizon * output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (B, seq_len, input_dim)
        out, (hn, cn) = self.lstm(x)
        # Use last hidden state: (B, hidden_dim)
        last_hidden = out[:, -1, :]
        pred = self.fc(last_hidden)
        # Reshape to (B, horizon, output_dim)
        return pred.view(-1, self.horizon, self.output_dim)


class GRUForecaster(nn.Module):
    """
    GRU-based Multi-Step Time-Series Forecaster.
    Input: (B, seq_len, input_dim)
    Output: (B, horizon, output_dim)
    """
    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 64,
        num_layers: int = 2,
        horizon: int = 12,
        output_dim: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        self.horizon = horizon
        self.output_dim = output_dim

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Linear(32, horizon * output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, hn = self.gru(x)
        last_hidden = out[:, -1, :]
        pred = self.fc(last_hidden)
        return pred.view(-1, self.horizon, self.output_dim)
