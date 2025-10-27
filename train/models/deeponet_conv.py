from __future__ import annotations

from typing import Dict, Mapping

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.linear1(x)
        out = self.activation(out)
        out = self.linear2(out)
        out += residual
        out = self.activation(out)
        return out


class ConvBranchNet(nn.Module):
    def __init__(self, output_dim: int) -> None:
        super().__init__()
        # Conv layers: 1→32→64→128 with 3x3 kernels, padding=1, MaxPool after each
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.activation = nn.ReLU()
        
        # Input: (batch, 1, 51, 51)
        # After conv1+pool1: (batch, 32, 25, 25)
        # After conv2+pool2: (batch, 64, 12, 12)
        # After conv3+pool3: (batch, 128, 6, 6)
        # Flattened: batch, 128*6*6 = 4608
        self.flatten_dim = 128 * 6 * 6
        self.fc = nn.Linear(self.flatten_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input has correct shape: (batch, 1, 51, 51)
        if x.ndim == 3:
            x = x.unsqueeze(1)  # Add channel dimension
        
        x = self.activation(self.conv1(x))
        x = self.pool1(x)
        x = self.activation(self.conv2(x))
        x = self.pool2(x)
        x = self.activation(self.conv3(x))
        x = self.pool3(x)
        
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x


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


class DeepONetConv3D(nn.Module):
    def __init__(
        self,
        branch_output_dim: int,
        trunk_config: Dict[str, int],
        ff_sigma: float,
        ff_features: int,
        device: torch.device,
        dtype: torch.dtype,
        use_fourier: bool = True,
    ) -> None:
        super().__init__()
        self.use_fourier = use_fourier
        self.branch = ConvBranchNet(output_dim=branch_output_dim).to(dtype=dtype, device=device)
        self.trunk = TrunkNet(**trunk_config).to(dtype=dtype, device=device)
        self.bias = nn.Parameter(torch.zeros(1, dtype=dtype, device=device))
        
        if use_fourier:
            input_dim = 1 + 4  # Cv plus (t, x, y, z)
            B = torch.randn(input_dim, ff_features, dtype=dtype, device=device) * torch.tensor(ff_sigma, dtype=dtype, device=device)
            self.B = nn.Parameter(B, requires_grad=False)

    def forward(self, u: torch.Tensor, cv: torch.Tensor, coord: torch.Tensor) -> torch.Tensor:
        trunk_input = torch.cat([cv, coord], dim=-1)
        
        if self.use_fourier:
            rff = torch.matmul(trunk_input, self.B)
            rff_input = torch.cat((torch.sin(2 * torch.pi * rff), torch.cos(2 * torch.pi * rff)), dim=-1)
            trunk_output = self.trunk(rff_input)
        else:
            trunk_output = self.trunk(trunk_input)

        branch_output = self.branch(u)
        combined = torch.sum(branch_output * trunk_output, dim=-1, keepdim=True)
        return combined + self.bias


def build_conv_model(cfg: Mapping[str, object], use_fourier: bool = True) -> DeepONetConv3D:
    branch_cfg = dict(cfg["branch"])  # type: ignore[arg-type]
    trunk_cfg = dict(cfg["trunk"])  # type: ignore[arg-type]
    ff_sigma = float(cfg["ff_sigma"])
    ff_features = int(cfg["ff_features"])
    device = torch.device(cfg["device"])  # type: ignore[arg-type]
    dtype = getattr(torch, str(cfg["dtype"]))

    branch_output_dim = int(branch_cfg["output_dim"])

    if use_fourier:
        # Trunk expects twice the random feature dimension after sin/cos concatenation.
        trunk_cfg["input_dim"] = ff_features * 2
    else:
        # Trunk receives cv + coord directly
        trunk_cfg["input_dim"] = 1 + 4  # Cv plus (t, x, y, z)

    return DeepONetConv3D(
        branch_output_dim=branch_output_dim,
        trunk_config=trunk_cfg,
        ff_sigma=ff_sigma,
        ff_features=ff_features,
        device=device,
        dtype=dtype,
        use_fourier=use_fourier,
    )

