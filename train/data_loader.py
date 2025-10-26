from __future__ import annotations

from pathlib import Path
from typing import Dict

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class DeepONetPointDataset(Dataset):
    """Point-wise DeepONet dataset that samples solution points across the HDF5 store."""

    def __init__(self, cfg: Dict[str, object]) -> None:
        super().__init__()
        data_path = Path(cfg["data_path"])  # type: ignore[arg-type]
        self.file = h5py.File(data_path, "r")

        self.u_data = self.file["u"]
        self.cv_data = self.file["Cv"]
        self.coord_data = self.file["y"]
        self.target_data = self.file["s"]

        stats_group = self.file.get("stats")
        if stats_group is None:
            raise RuntimeError(
                "Normalization statistics not found in dataset. "
                "Regenerate data with updated generator to include 'stats' group."
            )
        self.u_mean = np.asarray(stats_group["u_mean"], dtype=np.float32)
        self.u_std = np.asarray(stats_group["u_std"], dtype=np.float32)
        self.cv_mean = float(np.asarray(stats_group["cv_mean"], dtype=np.float32))
        self.cv_std = float(np.asarray(stats_group["cv_std"], dtype=np.float32))
        self.coord_mean = np.asarray(stats_group["coord_mean"], dtype=np.float32)
        self.coord_std = np.asarray(stats_group["coord_std"], dtype=np.float32)
        self.s_mean = float(np.asarray(stats_group["s_mean"], dtype=np.float32))
        self.s_std = float(np.asarray(stats_group["s_std"], dtype=np.float32))

        self.torch_dtype = getattr(torch, str(cfg["dtype"]))
        self.flatten_branch = bool(cfg["flatten_branch"])

        # Use solve the PDE for each u0, so we evaluate the solutions `u_data.shape[0]` times.
        self.num_solutions = int(self.u_data.shape[0])
        # For each solution, we evaluate the solution at `coord_data.shape[1]` points.
        self.points_per_solution = int(self.coord_data.shape[1])

        self.samples_per_epoch = self.num_solutions * self.points_per_solution

        seed_value = cfg["seed"]
        self.base_seed = int(seed_value) if seed_value is not None else None

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _prepare_sample(
        self,
        u0: np.ndarray,
        coord: np.ndarray,
        cv_scalar: float,
        target: float,
    ) -> Dict[str, torch.Tensor]:
        """Normalize sample components and convert them to tensors."""
        u_norm = (u0 - self.u_mean) / self.u_std
        if self.flatten_branch:
            branch_input = u_norm.reshape(-1)
        else:
            branch_input = u_norm

        cv_norm = (cv_scalar - self.cv_mean) / self.cv_std
        coord_norm = (coord - self.coord_mean) / self.coord_std
        s_norm = (target - self.s_mean) / self.s_std

        return {
            "u": torch.as_tensor(branch_input, dtype=self.torch_dtype),
            "cv": torch.as_tensor([cv_norm], dtype=self.torch_dtype),
            "coord": torch.as_tensor(coord_norm, dtype=self.torch_dtype),
            "s": torch.as_tensor([s_norm], dtype=self.torch_dtype),
        }

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        # We sample a random (solution_idx, point_idx) inside __getitem__.
        # DataLoader's `shuffle` only permutes dataset indices; it does not
        # control this internal random sampling. With a fixed seed, we derive
        # a deterministic RNG per index for reproducibility across epochs.
        if self.base_seed is None:
            rng = np.random.default_rng()
        else:
            rng = np.random.default_rng(self.base_seed + index)

        solution_idx = int(rng.integers(0, self.num_solutions))
        point_idx = int(rng.integers(0, self.points_per_solution))

        u0 = np.asarray(self.u_data[solution_idx], dtype=np.float32)
        coord = np.asarray(self.coord_data[solution_idx, point_idx], dtype=np.float32)
        target = float(self.target_data[solution_idx, point_idx])
        cv_scalar = float(self.cv_data[solution_idx])

        return self._prepare_sample(u0, coord, cv_scalar, target)

    def __del__(self) -> None:
        file_handle = getattr(self, "file", None)
        if file_handle:
            file_handle.close()
