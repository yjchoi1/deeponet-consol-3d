from __future__ import annotations

from typing import Dict, Mapping

import torch
import torch.nn as nn


class SimpleBranchNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 3) -> None:
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class SimpleTrunkNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 3) -> None:
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class VanillaDeepONetFF3D(nn.Module):
    """Vanilla MLP branch and trunk WITH Fourier features."""
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
        self.branch = SimpleBranchNet(**branch_config).to(dtype=dtype, device=device)
        self.trunk = SimpleTrunkNet(**trunk_config).to(dtype=dtype, device=device)
        self.bias = nn.Parameter(torch.zeros(1, dtype=dtype, device=device))
        
        # Fourier features
        input_dim = 1 + 4  # Cv plus (t, x, y, z)
        B = torch.randn(input_dim, ff_features, dtype=dtype, device=device) * torch.tensor(ff_sigma, dtype=dtype, device=device)
        self.B = nn.Parameter(B, requires_grad=False)

    def forward(self, u: torch.Tensor, cv: torch.Tensor, coord: torch.Tensor) -> torch.Tensor:
        trunk_input = torch.cat([cv, coord], dim=-1)
        
        # Apply Fourier features
        rff = torch.matmul(trunk_input, self.B)
        rff_input = torch.cat((torch.sin(2 * torch.pi * rff), torch.cos(2 * torch.pi * rff)), dim=-1)
        
        branch_output = self.branch(u)
        trunk_output = self.trunk(rff_input)
        
        combined = torch.sum(branch_output * trunk_output, dim=-1, keepdim=True)
        return combined + self.bias


def build_vanilla_ff_model(cfg: Mapping[str, object]) -> VanillaDeepONetFF3D:
    branch_cfg = dict(cfg["branch"])  # type: ignore[arg-type]
    trunk_cfg = dict(cfg["trunk"])  # type: ignore[arg-type]
    ff_sigma = float(cfg["ff_sigma"])
    ff_features = int(cfg["ff_features"])
    device = torch.device(cfg["device"])  # type: ignore[arg-type]
    dtype = getattr(torch, str(cfg["dtype"]))

    # Trunk expects twice the random feature dimension after sin/cos concatenation
    trunk_cfg["input_dim"] = ff_features * 2
    
    # For vanilla, num_blocks becomes num_layers
    if "num_blocks" in branch_cfg:
        branch_cfg["num_layers"] = branch_cfg.pop("num_blocks")
    if "num_blocks" in trunk_cfg:
        trunk_cfg["num_layers"] = trunk_cfg.pop("num_blocks")

    return VanillaDeepONetFF3D(
        branch_config=branch_cfg,
        trunk_config=trunk_cfg,
        ff_sigma=ff_sigma,
        ff_features=ff_features,
        device=device,
        dtype=dtype,
    )

