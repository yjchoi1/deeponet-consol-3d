from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn


CONFIG: Dict[str, object] = {
    "branch": {
        "input_dim": 41 * 41,
        "hidden_dim": 256,
        "output_dim": 128,
        "num_blocks": 3,
    },
    "trunk": {
        "input_dim": 2 * 128,
        "hidden_dim": 256,
        "output_dim": 128,
        "num_blocks": 3,
    },
    "ff_features": 128,
    "ff_sigma": 1.0,
    "device": "cpu",
    "dtype": "float32",
}


def random_fourier_matrix(
    in_features: int,
    out_features: int,
    sigma: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    scale = math.sqrt(2.0) * float(sigma)
    matrix = torch.randn(in_features, out_features, device=device, dtype=dtype)
    return matrix * scale


class BranchNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_blocks: int) -> None:
        super().__init__()
        blocks = [ResidualBlock(hidden_dim) for _ in range(num_blocks)]
        self.initial = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.residual_blocks = nn.Sequential(*blocks)
        self.final = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial(x)
        x = self.residual_blocks(x)
        return self.final(x)


class TrunkNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_blocks: int) -> None:
        super().__init__()
        blocks = [ResidualBlock(hidden_dim) for _ in range(num_blocks)]
        self.initial = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.residual_blocks = nn.Sequential(*blocks)
        self.final = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.initial(x)
        x = self.residual_blocks(x)
        return self.final(x)


class DeepONet3D(nn.Module):
    def __init__(
        self,
        branch_config: Dict[str, int],
        trunk_config: Dict[str, int],
        ff_sigma: float,
        ff_features: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.branch = BranchNet(**branch_config).to(dtype=dtype, device=device)
        self.trunk = TrunkNet(**trunk_config).to(dtype=dtype, device=device)
        self.bias = nn.Parameter(torch.zeros(1, dtype=dtype, device=device))

        input_dim = 1 + 4  # Cv plus (t, x, y, z)
        self.register_buffer(
            "fourier_weight",
            random_fourier_matrix(input_dim, ff_features, ff_sigma, device, dtype),
            persistent=False,
        )

    def _random_features(self, trunk_input: torch.Tensor) -> torch.Tensor:
        projection = trunk_input @ self.fourier_weight
        sin_part = torch.sin(2.0 * math.pi * projection)
        cos_part = torch.cos(2.0 * math.pi * projection)
        return torch.cat([sin_part, cos_part], dim=-1)

    def forward(self, u: torch.Tensor, cv: torch.Tensor, coord: torch.Tensor) -> torch.Tensor:
        trunk_input = torch.cat([cv, coord], dim=-1)
        rff = self._random_features(trunk_input)

        branch_output = self.branch(u)
        trunk_output = self.trunk(rff)

        combined = torch.sum(branch_output * trunk_output, dim=-1, keepdim=True)
        return combined + self.bias


def build_model(overrides: Dict[str, object] | None = None) -> DeepONet3D:
    cfg = CONFIG.copy()
    if overrides:
        cfg.update(overrides)

    device = torch.device(cfg.get("device", "cpu"))
    dtype = getattr(torch, cfg.get("dtype", "float32"))

    branch_cfg = cfg["branch"].copy()  # type: ignore[assignment]
    trunk_cfg = cfg["trunk"].copy()  # type: ignore[assignment]

    ff_sigma = float(cfg["ff_sigma"])
    ff_features = int(cfg["ff_features"])

    # Trunk expects twice the random feature dimension after sin/cos concatenation.
    trunk_cfg["input_dim"] = ff_features * 2

    return DeepONet3D(
        branch_config=branch_cfg,
        trunk_config=trunk_cfg,
        ff_sigma=ff_sigma,
        ff_features=ff_features,
        device=device,
        dtype=dtype,
    )


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.activation = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.linear1(x)
        out = self.activation(out)
        out = self.linear2(out)
        out += residual
        out = self.activation(out)
        return out