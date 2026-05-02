"""ROM dataset — returns (u0_flat, cv, t) -> POD coefficient tuples."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class ROMDataset(Dataset):
    """Each sample is one (solution, time-step) pair."""

    def __init__(self, cfg: Dict[str, object]) -> None:
        super().__init__()
        fields_path = Path(cfg["fields_path"])  # type: ignore[arg-type]
        basis_path = Path(cfg["basis_path"])  # type: ignore[arg-type]
        self.torch_dtype = getattr(torch, str(cfg.get("dtype", "float32")))

        # Load POD coefficients and time grid from the basis file.
        basis = np.load(basis_path)
        self.coeffs = basis["coeffs"]  # (n_samples_total, nt, n_modes)
        self.times = basis["times"]  # (nt,)
        self.n_modes = int(basis["n_modes"])
        self.nt = int(basis["nt"])

        # Per-mode normalization for coefficients.
        self.coeffs_mean = basis["coeffs_mean"]  # (n_modes,)
        self.coeffs_std = basis["coeffs_std"]  # (n_modes,)

        # Load u0 and Cv from the fields HDF5 (keep handle open for lazy access).
        self.h5_file = h5py.File(fields_path, "r")
        self.u0_data = self.h5_file["u0"]  # (n_samples_total, nx, ny)
        self.cv_data = self.h5_file["Cv"]  # (n_samples_total,)

        # Determine which sample indices belong to this split.
        sample_indices = cfg.get("sample_indices")
        if sample_indices is not None:
            self.sample_indices = np.asarray(sample_indices, dtype=np.int64)
        else:
            self.sample_indices = np.arange(int(self.u0_data.shape[0]), dtype=np.int64)

        self.n_samples = len(self.sample_indices)

        # Load normalization stats from the HDF5 file.
        stats_group = self.h5_file["stats"]
        self.u_mean = float(np.asarray(stats_group["u_mean"], dtype=np.float32))
        self.u_std = float(np.asarray(stats_group["u_std"], dtype=np.float32))
        self.cv_mean = float(np.asarray(stats_group["cv_mean"], dtype=np.float32))
        self.cv_std = float(np.asarray(stats_group["cv_std"], dtype=np.float32))

        # Time normalization: use mean/std of the shared time grid.
        self.t_mean = float(np.mean(self.times))
        self.t_std = float(np.std(self.times))
        if self.t_std == 0.0:
            self.t_std = 1.0

    def __len__(self) -> int:
        return self.n_samples * self.nt

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        local_sample_idx = index // self.nt
        time_idx = index % self.nt

        global_sample_idx = int(self.sample_indices[local_sample_idx])

        u0 = np.asarray(self.u0_data[global_sample_idx], dtype=np.float32).reshape(-1)
        cv_scalar = float(self.cv_data[global_sample_idx])
        t_value = float(self.times[time_idx])
        target = self.coeffs[global_sample_idx, time_idx].astype(np.float32)

        # Normalize inputs.
        u0_norm = (u0 - self.u_mean) / self.u_std
        cv_norm = (cv_scalar - self.cv_mean) / self.cv_std
        t_norm = (t_value - self.t_mean) / self.t_std

        # Normalize target coefficients per mode.
        target_norm = (target - self.coeffs_mean) / self.coeffs_std

        return {
            "u0": torch.as_tensor(u0_norm, dtype=self.torch_dtype),
            "cv": torch.as_tensor([cv_norm], dtype=self.torch_dtype),
            "t": torch.as_tensor([t_norm], dtype=self.torch_dtype),
            "coeffs": torch.as_tensor(target_norm, dtype=self.torch_dtype),
        }

    def __del__(self) -> None:
        file_handle = getattr(self, "h5_file", None)
        if file_handle:
            file_handle.close()
