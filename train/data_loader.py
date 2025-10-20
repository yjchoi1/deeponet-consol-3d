from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


CONFIG: Dict[str, object] = {
    "data_path": Path("train/data/deeponet_terzaghi.h5"),
    "batch_size": 2048,
    "shuffle": False,
    "num_workers": 0,
    "pin_memory": True,
    "drop_last": False,
    "dtype": "float32",
    "flatten_branch": True,
    "samples_per_epoch": None,
    "seed": 42,
}


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

        self.torch_dtype = getattr(torch, str(cfg.get("dtype", "float32")))
        self.flatten_branch = bool(cfg.get("flatten_branch", True))

        # Use solve the PDE for each u0, so we evaluate the solutions `u_data.shape[0]` times.
        self.num_solutions = int(self.u_data.shape[0])
        # For each solution, we evaluate the solution at `coord_data.shape[1]` points.
        self.points_per_solution = int(self.coord_data.shape[1])

        # We can optionally sample a subset of the entire training examples.
        samples_per_epoch = cfg.get("samples_per_epoch")
        if samples_per_epoch is None:
            samples_per_epoch = self.num_solutions * self.points_per_solution
        self.samples_per_epoch = int(samples_per_epoch)

        seed_value = cfg.get("seed")
        self.base_seed = int(seed_value) if seed_value else None

    def __len__(self) -> int:
        return self.samples_per_epoch

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

        if self.flatten_branch:
            branch_input = u0.reshape(-1)
        else:
            branch_input = u0

        return {
            "u": torch.as_tensor(branch_input, dtype=self.torch_dtype),
            "cv": torch.as_tensor([cv_scalar], dtype=self.torch_dtype),
            "coord": torch.as_tensor(coord, dtype=self.torch_dtype),
            "s": torch.as_tensor([target], dtype=self.torch_dtype),
        }

    def __del__(self) -> None:
        file_handle = getattr(self, "file", None)
        if file_handle:
            file_handle.close()


def create_dataloader(overrides: Optional[Dict[str, object]] = None) -> DataLoader:
    """Create a DataLoader where batch_size counts individual solution points."""
    cfg = CONFIG.copy()
    if overrides:
        cfg.update(overrides)

    dataset = DeepONetPointDataset(cfg)
    return DataLoader(
        dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=bool(cfg["shuffle"]),
        num_workers=int(cfg["num_workers"]),
        pin_memory=bool(cfg["pin_memory"]),
        drop_last=bool(cfg["drop_last"]),
    )


if __name__ == "__main__":
    # Create dataloader
    dataloader = create_dataloader()
    
    print(f"Dataset size: {len(dataloader.dataset)}")
    print(f"Number of batches: {len(dataloader)}")
    print(f"Batch size: {dataloader.batch_size}")
    
    # Load a few batches and debug
    for i, batch in enumerate(dataloader):
        print(f"\nBatch {i}:")
        print(f"  u shape: {batch['u'].shape}")
        print(f"  cv shape: {batch['cv'].shape}")
        print(f"  coord shape: {batch['coord'].shape}")
        print(f"  s shape: {batch['s'].shape}")
        print(f"  u sample values: {batch['u'][0, :5]}")
        print(f"  cv sample values: {batch['cv'][:5]}")
        print(f"  coord sample values: {batch['coord'][0]}")
        print(f"  s sample values: {batch['s'][:5]}")
        
        # Only show first 3 batches for debugging
        if i >= 2:
            break
    
    print("\nData loading test completed successfully!")