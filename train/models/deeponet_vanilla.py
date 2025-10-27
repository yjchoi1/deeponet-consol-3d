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


class VanillaDeepONet3D(nn.Module):
    def __init__(
        self,
        branch_config: Dict[str, int],
        trunk_config: Dict[str, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.branch = SimpleBranchNet(**branch_config).to(dtype=dtype, device=device)
        self.trunk = SimpleTrunkNet(**trunk_config).to(dtype=dtype, device=device)
        self.bias = nn.Parameter(torch.zeros(1, dtype=dtype, device=device))

    def forward(self, u: torch.Tensor, cv: torch.Tensor, coord: torch.Tensor) -> torch.Tensor:
        # Concatenate cv and coord directly (no Fourier features)
        trunk_input = torch.cat([cv, coord], dim=-1)
        
        branch_output = self.branch(u)
        trunk_output = self.trunk(trunk_input)
        
        combined = torch.sum(branch_output * trunk_output, dim=-1, keepdim=True)
        return combined + self.bias


def build_vanilla_model(cfg: Mapping[str, object]) -> VanillaDeepONet3D:
    branch_cfg = dict(cfg["branch"])  # type: ignore[arg-type]
    trunk_cfg = dict(cfg["trunk"])  # type: ignore[arg-type]
    device = torch.device(cfg["device"])  # type: ignore[arg-type]
    dtype = getattr(torch, str(cfg["dtype"]))

    # Trunk receives cv + coord directly (no Fourier features)
    trunk_cfg["input_dim"] = 1 + 4  # Cv plus (t, x, y, z)
    
    # For vanilla, num_blocks becomes num_layers
    if "num_blocks" in branch_cfg:
        branch_cfg["num_layers"] = branch_cfg.pop("num_blocks")
    if "num_blocks" in trunk_cfg:
        trunk_cfg["num_layers"] = trunk_cfg.pop("num_blocks")

    return VanillaDeepONet3D(
        branch_config=branch_cfg,
        trunk_config=trunk_cfg,
        device=device,
        dtype=dtype,
    )

