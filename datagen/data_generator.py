from __future__ import annotations

import sys
from pathlib import Path

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
from pathlib import Path
from typing import Dict, Tuple

import h5py
import numpy as np
import torch
from scipy.stats import qmc

from solver.solver_batch import random_gaussian_pwp_batch, solve_terzaghi_3d_fdm_batch


# Configuration for dataset generation parameters and solver setup.
CONFIG: Dict[str, object] = {
    "n_samples": 20,
    "batch_size": 20,
    "points_per_sample": 8192,
    "x_range": (0.0, 1.0),
    "y_range": (0.0, 1.0),
    "z_range": (0.0, 1.0),
    "nx": 51,
    "ny": 51,
    "nz": 51,
    "t_span": (0.0, 1.0),
    "n_time_points": 51,
    "cv_range": (0.02, 0.1),
    "gp_output_scale": 1000.0,
    "gp_length_scale_xy": 0.15,
    "u0_ranges": [(10_000.0, 20_000.0)],
    "output_path": Path("train/data/deeponet_terzaghi_val.h5"),
    "seed": 42,
    "torch_dtype": "float32",
}


def sample_solution_points(
    field: np.ndarray,
    times: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    n_points: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample grid points using Latin Hypercube Sampling (LHS).

    Args:
        field: Solution array of shape (nt, nx, ny, nz).
        times: Time coordinates of shape (nt,).
        xs: X coordinates of shape (nx,).
        ys: Y coordinates of shape (ny,).
        zs: Z coordinates of shape (nz,).
        n_points: Number of points to sample.
        rng: Random number generator.

    Returns:
        Tuple (points, values) where points has shape (n_points, 4)
        containing (t, x, y, z) and values has shape (n_points,).
    """
    nt, nx, ny, nz = field.shape

    sampler = qmc.LatinHypercube(d=4, seed=rng)
    unit_samples = sampler.random(n_points)

    ti = np.floor(unit_samples[:, 0] * nt).astype(int)
    xi = np.floor(unit_samples[:, 1] * nx).astype(int)
    yi = np.floor(unit_samples[:, 2] * ny).astype(int)
    zi = np.floor(unit_samples[:, 3] * nz).astype(int)

    # Ensure indices are within bounds due to possible 1.0 - eps rounding
    ti = np.clip(ti, 0, nt - 1)
    xi = np.clip(xi, 0, nx - 1)
    yi = np.clip(yi, 0, ny - 1)
    zi = np.clip(zi, 0, nz - 1)

    points = np.empty((n_points, 4), dtype=np.float32)
    points[:, 0] = times[ti].astype(np.float32)
    points[:, 1] = xs[xi].astype(np.float32)
    points[:, 2] = ys[yi].astype(np.float32)
    points[:, 3] = zs[zi].astype(np.float32)

    values = field[ti, xi, yi, zi].astype(np.float32)

    return points, values


def enforce_drained_dirichlet_bc(u_batch: torch.Tensor) -> torch.Tensor:
    """Clamp horizontal surfaces to zero along the drained boundaries."""
    if u_batch.ndim != 3:
        raise ValueError("u_batch must have shape (batch, nx, ny)")
    u_batch[:, 0, :] = 0.0
    u_batch[:, -1, :] = 0.0
    u_batch[:, :, 0] = 0.0
    u_batch[:, :, -1] = 0.0
    return u_batch


def generate_training_data(cfg: Dict[str, object]) -> None:
    """Generate DeepONet training samples and store them in an HDF5 file."""
    n_samples = int(cfg["n_samples"])
    batch_size = int(cfg["batch_size"])
    points_per_sample = int(cfg["points_per_sample"])
    nx = int(cfg["nx"])
    ny = int(cfg["ny"])
    nz = int(cfg["nz"])
    t_span = tuple(float(x) for x in cfg["t_span"])  # type: ignore[index]
    n_time_points = int(cfg["n_time_points"])
    cv_min, cv_max = (float(x) for x in cfg["cv_range"])  # type: ignore[index]
    gp_output_scale = float(cfg["gp_output_scale"])
    gp_length_scale_xy = float(cfg["gp_length_scale_xy"])
    u0_ranges = tuple(tuple(float(v) for v in pair) for pair in cfg["u0_ranges"])  # type: ignore[index]
    seed = int(cfg["seed"])
    dtype_str = str(cfg["torch_dtype"])
    torch_dtype = getattr(torch, dtype_str)
    output_path = Path(cfg["output_path"])  # type: ignore[arg-type]

    # Define random seed
    rng = np.random.default_rng(seed)
    
    # Check input parameter sanity
    if n_time_points < 2:
        raise ValueError("n_time_points must be at least 2.")
    if t_span[1] <= t_span[0]:
        raise ValueError("t_span must satisfy t_end > t_start.")
    
    # Generate equally spaced t, x, y, z grids
    time_samples = np.linspace(
        t_span[0],
        t_span[1],
        n_time_points,
        dtype=np.float64,
    )
    x_range = tuple(float(v) for v in cfg["x_range"])  # type: ignore[index]
    y_range = tuple(float(v) for v in cfg["y_range"])  # type: ignore[index]
    z_range = tuple(float(v) for v in cfg["z_range"])  # type: ignore[index]

    xs = np.linspace(x_range[0], x_range[1], nx, dtype=np.float64)
    ys = np.linspace(y_range[0], y_range[1], ny, dtype=np.float64)
    zs = np.linspace(z_range[0], z_range[1], nz, dtype=np.float64)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as h5_file:
        # Running totals for normalization statistics.
        u_sum = torch.zeros((nx, ny), dtype=torch.float64)
        u_sq_sum = torch.zeros((nx, ny), dtype=torch.float64)
        cv_sum = 0.0
        cv_sq_sum = 0.0
        coord_sum = np.zeros(4, dtype=np.float64)
        coord_sq_sum = np.zeros(4, dtype=np.float64)
        coord_count = 0
        s_sum = 0.0
        s_sq_sum = 0.0

        # Allocate datasets for DeepONet components: branch input, trunk input, outputs.
        dset_u = h5_file.create_dataset("u", (n_samples, nx, ny), dtype="float32")
        dset_cv = h5_file.create_dataset("Cv", (n_samples,), dtype="float32")
        dset_points = h5_file.create_dataset(
            "y", (n_samples, points_per_sample, 4), dtype="float32"
        )
        dset_values = h5_file.create_dataset(
            "s", (n_samples, points_per_sample), dtype="float32"
        )

        h5_file.attrs["x_coords"] = xs.astype(np.float32)
        h5_file.attrs["y_coords"] = ys.astype(np.float32)
        h5_file.attrs["z_coords"] = zs.astype(np.float32)
        # Persist grid coordinates and generator configuration for reproducibility.
        h5_file.attrs["time_samples"] = time_samples.astype(np.float32)
        h5_file.attrs["config"] = json.dumps(
            {
                key: (str(value) if isinstance(value, Path) else value)
                for key, value in cfg.items()
            }
        )

        # Precompute Cv values and sort indices for bucketing by similar Cv
        cv_all_values = rng.uniform(cv_min, cv_max, size=n_samples).astype(np.float32)
        sorted_indices = np.argsort(cv_all_values)

        # Iterate over batches in order of increasing Cv to reduce solver step shrinkage.
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

            # Ensure stored branch inputs match the drained boundaries seen by the solver.
            u0_batch = enforce_drained_dirichlet_bc(u0_batch)

            cv_values = cv_all_values[batch_indices]
            Cv_batch = torch.as_tensor(cv_values, dtype=torch_dtype, device=u0_batch.device)

            # Solve PDE for the sampled initial fields using per-sample Cv values.
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
                dtype=torch_dtype,
            )

            # fields are different for each sample in batch
            batch_fields = solver_result["u"].cpu().numpy()
            # times are shared across all samples in batch. This avoids creating a new time array for each sample.
            batch_times = solver_result["t"].cpu().numpy()

            # For each u0-solution in the batch, sample points and values (so-called target) from the solution.
            for local_idx in range(current_batch):
                sample_id = int(batch_indices[local_idx])
                field = batch_fields[local_idx]
                cv_value = float(cv_values[local_idx])

                # Draw random space-time query points and evaluate the solution there.
                points, values = sample_solution_points(
                    field,
                    batch_times,
                    xs,
                    ys,
                    zs,
                    points_per_sample,
                    rng,
                )

                # Save branch inputs, trunk inputs, and targets into HDF5 datasets.
                dset_u[sample_id] = u0_batch[local_idx].cpu().numpy()
                dset_cv[sample_id] = cv_value
                dset_points[sample_id] = points
                dset_values[sample_id] = values

                # Update normalization statistics using double precision to reduce drift.
                u_tensor = u0_batch[local_idx].to(dtype=torch.float64)
                u_sum.add_(u_tensor)
                u_sq_sum.add_(u_tensor * u_tensor)

                cv_sum += cv_value
                cv_sq_sum += cv_value * cv_value

                points64 = points.astype(np.float64, copy=False)
                coord_sum += points64.sum(axis=0)
                coord_sq_sum += np.square(points64).sum(axis=0)
                coord_count += points.shape[0]

                values64 = values.astype(np.float64, copy=False)
                s_sum += values64.sum()
                s_sq_sum += np.square(values64).sum()

        u_sum_np = u_sum.cpu().numpy()
        u_sq_sum_np = u_sq_sum.cpu().numpy()
        u_mean = u_sum_np / float(n_samples)
        u_var = (u_sq_sum_np / float(n_samples)) - np.square(u_mean)
        u_std = np.sqrt(u_var)

        cv_mean = cv_sum / float(n_samples)
        cv_var = (cv_sq_sum / float(n_samples)) - cv_mean * cv_mean
        cv_std = np.sqrt(cv_var)

        coord_mean = coord_sum / float(coord_count)
        coord_var = (coord_sq_sum / float(coord_count)) - np.square(coord_mean)
        coord_std = np.sqrt(coord_var)
        
        s_mean = s_sum / float(coord_count)
        s_var = (s_sq_sum / float(coord_count)) - s_mean * s_mean
        s_std = np.sqrt(s_var)

        stats_group = h5_file.create_group("stats")
        stats_group.create_dataset("u_mean", data=u_mean.astype(np.float32))
        stats_group.create_dataset("u_std", data=u_std.astype(np.float32))
        stats_group.create_dataset("cv_mean", data=np.asarray(cv_mean, dtype=np.float32))
        stats_group.create_dataset("cv_std", data=np.asarray(cv_std, dtype=np.float32))
        stats_group.create_dataset("coord_mean", data=coord_mean.astype(np.float32))
        stats_group.create_dataset("coord_std", data=coord_std.astype(np.float32))
        stats_group.create_dataset("s_mean", data=np.asarray(s_mean, dtype=np.float32))
        stats_group.create_dataset("s_std", data=np.asarray(s_std, dtype=np.float32))
        stats_group.attrs["u_count"] = int(n_samples)
        stats_group.attrs["coord_count"] = int(coord_count)


if __name__ == "__main__":
    generate_training_data(CONFIG)
