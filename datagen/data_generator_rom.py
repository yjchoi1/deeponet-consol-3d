"""ROM data generator — stores full solver fields on the shared grid.

Unlike data_generator.py (which samples random space-time points for DeepONet),
this stores the complete (nt, nx, ny, nz) solution for each sample so that a
POD basis can be computed offline.
"""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
from typing import Dict

import h5py
import numpy as np
import torch

from solver.solver_batch import random_gaussian_pwp_batch, solve_terzaghi_3d_fdm_batch


# Configuration for ROM dataset generation.
CONFIG: Dict[str, object] = {
    "n_samples": 1,
    "batch_size": 1,
    "x_range": (0.0, 1.0),
    "y_range": (0.0, 1.0),
    "z_range": (0.0, 1.0),
    "nx": 51,
    "ny": 51,
    "nz": 51,
    "t_span": (0.0, 1.0),
    "n_time_points": 51,
    "cv_range": (0.1, 0.1),
    "gp_output_scale": 1000.0,
    "gp_length_scale_xy": 0.3,
    "u0_ranges": [(10_000.0, 20_000.0)],
    "bc": "drained_xy_top_nodrain_bottom",
    "output_path": Path("data/rom_fields.h5"),
    "seed": 42,
    "torch_dtype": "float32",
}


def enforce_drained_dirichlet_bc(u_batch: torch.Tensor) -> torch.Tensor:
    """Clamp horizontal surfaces to zero along the drained boundaries."""
    u_batch[:, 0, :] = 0.0
    u_batch[:, -1, :] = 0.0
    u_batch[:, :, 0] = 0.0
    u_batch[:, :, -1] = 0.0
    return u_batch


def generate_rom_data(cfg: Dict[str, object]) -> None:
    """Generate full-field ROM training data and store in an HDF5 file."""
    n_samples = int(cfg["n_samples"])
    batch_size = int(cfg["batch_size"])
    nx = int(cfg["nx"])
    ny = int(cfg["ny"])
    nz = int(cfg["nz"])
    t_span = tuple(float(x) for x in cfg["t_span"])  # type: ignore[index]
    n_time_points = int(cfg["n_time_points"])
    cv_min, cv_max = (float(x) for x in cfg["cv_range"])  # type: ignore[index]
    gp_output_scale = float(cfg["gp_output_scale"])
    gp_length_scale_xy = float(cfg["gp_length_scale_xy"])
    u0_ranges = tuple(tuple(float(v) for v in pair) for pair in cfg["u0_ranges"])  # type: ignore[index]
    bc = str(cfg.get("bc", "drained"))
    seed = int(cfg["seed"])
    dtype_str = str(cfg["torch_dtype"])
    torch_dtype = getattr(torch, dtype_str)
    output_path = Path(cfg["output_path"])  # type: ignore[arg-type]

    rng = np.random.default_rng(seed)

    if n_time_points < 2:
        raise ValueError("n_time_points must be at least 2.")
    if t_span[1] <= t_span[0]:
        raise ValueError("t_span must satisfy t_end > t_start.")

    time_samples = np.linspace(t_span[0], t_span[1], n_time_points, dtype=np.float64)
    x_range = tuple(float(v) for v in cfg["x_range"])  # type: ignore[index]
    y_range = tuple(float(v) for v in cfg["y_range"])  # type: ignore[index]
    z_range = tuple(float(v) for v in cfg["z_range"])  # type: ignore[index]

    xs = np.linspace(x_range[0], x_range[1], nx, dtype=np.float64)
    ys = np.linspace(y_range[0], y_range[1], ny, dtype=np.float64)
    zs = np.linspace(z_range[0], z_range[1], nz, dtype=np.float64)

    nt = n_time_points

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as h5_file:
        # Running totals for normalization statistics (u0, Cv).
        u_sum = torch.zeros((nx, ny), dtype=torch.float64)
        u_sq_sum = torch.zeros((nx, ny), dtype=torch.float64)
        cv_sum = 0.0
        cv_sq_sum = 0.0

        # Allocate HDF5 datasets.
        dset_u0 = h5_file.create_dataset("u0", (n_samples, nx, ny), dtype="float32")
        dset_cv = h5_file.create_dataset("Cv", (n_samples,), dtype="float32")
        dset_fields = h5_file.create_dataset(
            "fields",
            (n_samples, nt, nx, ny, nz),
            dtype="float32",
            chunks=(1, nt, nx, ny, nz),
        )
        dset_times = h5_file.create_dataset(
            "times", data=time_samples.astype(np.float32)
        )

        # Store grid coordinates as attributes for reproducibility.
        h5_file.attrs["x_coords"] = xs.astype(np.float32)
        h5_file.attrs["y_coords"] = ys.astype(np.float32)
        h5_file.attrs["z_coords"] = zs.astype(np.float32)
        h5_file.attrs["config"] = json.dumps(
            {
                key: (str(value) if isinstance(value, Path) else value)
                for key, value in cfg.items()
            }
        )

        # Precompute Cv values and sort for solver efficiency.
        cv_all_values = rng.uniform(cv_min, cv_max, size=n_samples).astype(np.float32)
        sorted_indices = np.argsort(cv_all_values)

        for offset in range(0, n_samples, batch_size):
            batch_indices = sorted_indices[offset : offset + batch_size]
            current_batch = int(batch_indices.shape[0])
            batch_seed = seed + offset
            start_idx = int(offset + 1)
            end_idx = int(offset + current_batch)
            print(f"Generating samples {start_idx}-{end_idx} / {n_samples} (Cv-sorted)")

            u0_batch = random_gaussian_pwp_batch(
                current_batch,
                nx,
                ny,
                x_range,
                y_range,
                {
                    "output_scale": gp_output_scale,
                    "length_scales": gp_length_scale_xy,
                },
                u0_ranges,
                seed=batch_seed,
                dtype=torch_dtype,
            )
            u0_batch = enforce_drained_dirichlet_bc(u0_batch)

            cv_values = cv_all_values[batch_indices]
            Cv_batch = torch.as_tensor(
                cv_values, dtype=torch_dtype, device=u0_batch.device
            )

            solver_result = solve_terzaghi_3d_fdm_batch(
                Cv_batch=Cv_batch,
                x_range=x_range,
                y_range=y_range,
                z_range=z_range,
                nx=nx,
                ny=ny,
                nz=nz,
                t_span=t_span,
                u0_xy_batch=u0_batch,
                t_eval=time_samples,
                bc=bc,
                dtype=torch_dtype,
            )

            # solver returns shape (batch, nt, nx, ny, nz)
            batch_fields = solver_result["u"].cpu().numpy()

            for local_idx in range(current_batch):
                sample_id = int(batch_indices[local_idx])
                cv_value = float(cv_values[local_idx])

                dset_u0[sample_id] = u0_batch[local_idx].cpu().numpy()
                dset_cv[sample_id] = cv_value
                dset_fields[sample_id] = batch_fields[local_idx]

                # Accumulate normalization statistics.
                u_tensor = u0_batch[local_idx].to(dtype=torch.float64)
                u_sum.add_(u_tensor)
                u_sq_sum.add_(u_tensor * u_tensor)
                cv_sum += cv_value
                cv_sq_sum += cv_value * cv_value

        # Compute and store normalization statistics.
        u_sum_np = u_sum.cpu().numpy()
        u_sq_sum_np = u_sq_sum.cpu().numpy()
        total_count = n_samples * nx * ny
        u_mean = float(u_sum_np.sum() / total_count)
        u_var = float(u_sq_sum_np.sum() / total_count) - u_mean * u_mean
        u_std = float(np.sqrt(u_var))

        cv_mean = cv_sum / float(n_samples)
        cv_var = (cv_sq_sum / float(n_samples)) - cv_mean * cv_mean
        cv_std = float(np.sqrt(cv_var))

        stats_group = h5_file.create_group("stats")
        stats_group.create_dataset(
            "u_mean", data=np.asarray(u_mean, dtype=np.float32)
        )
        stats_group.create_dataset(
            "u_std", data=np.asarray(u_std, dtype=np.float32)
        )
        stats_group.create_dataset(
            "cv_mean", data=np.asarray(cv_mean, dtype=np.float32)
        )
        stats_group.create_dataset(
            "cv_std", data=np.asarray(cv_std, dtype=np.float32)
        )

    print(f"ROM data saved to {output_path}")
    print(f"  n_samples={n_samples}, nt={nt}, nx={nx}, ny={ny}, nz={nz}")
    print(f"  u0 stats: mean={u_mean:.4f}, std={u_std:.4f}")
    print(f"  Cv stats: mean={cv_mean:.6f}, std={cv_std:.6f}")


if __name__ == "__main__":
    generate_rom_data(CONFIG)
