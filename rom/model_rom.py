"""ROM coefficient regressor — maps (u0_flat, Cv, t) to POD coefficients."""
from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn


class ROMRegressor(nn.Module):
    """MLP that predicts POD modal coefficients from (u0, Cv, t)."""

    def __init__(
        self,
        u0_dim: int,
        n_modes: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
    ) -> None:
        super().__init__()

        # Encoder for the high-dimensional u0 input.
        encoder_layers = [nn.Linear(u0_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 2):
            encoder_layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        self.u0_encoder = nn.Sequential(*encoder_layers)

        # Head that merges encoded u0 with Cv and t, then outputs coefficients.
        head_layers = [nn.Linear(hidden_dim + 2, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 2):
            head_layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        head_layers.append(nn.Linear(hidden_dim, n_modes))
        self.head = nn.Sequential(*head_layers)

    def forward(
        self,
        u0_flat: torch.Tensor,
        cv: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        h = self.u0_encoder(u0_flat)
        h = torch.cat([h, cv, t], dim=-1)
        return self.head(h)


def build_rom_model(cfg: Mapping[str, object]) -> ROMRegressor:
    """Factory function matching the project's build_model pattern."""
    u0_dim = int(cfg["u0_dim"])
    n_modes = int(cfg["n_modes"])
    hidden_dim = int(cfg.get("hidden_dim", 256))  # type: ignore[arg-type]
    num_layers = int(cfg.get("num_layers", 4))  # type: ignore[arg-type]
    return ROMRegressor(
        u0_dim=u0_dim,
        n_modes=n_modes,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
    )
