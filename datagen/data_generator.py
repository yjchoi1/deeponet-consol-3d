from __future__ import annotations

import sys
from pathlib import Path

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import h5py
import numpy as np
import torch

from solver.solver_batch import random_gaussian_pwp_batch, solve_terzaghi_3d_fdm_batch


# Configuration for dataset generation parameters and solver setup.
CONFIG: Dict[str, object] = {
    "n_samples": 4,
    "batch_size": 2,
    "points_per_sample": 2048,
    "x_range": (0.0, 1.0),
    "y_range": (0.0, 1.0),
    "z_range": (0.0, 1.0),
    "nx": 41,
    "ny": 41,
    "nz": 41,
    "t_span": (0.0, 1.0),
    "n_time_points": 41,
    "time_bias_exponent": 3.0,
    "cv_range": (0.2, 2.0),
    "gp_output_scale": 500.0,
    "gp_length_scale_xy": 0.15,
    "u0_ranges": [(10_000.0, 20_000.0)],
    "output_path": Path("train/data/deeponet_terzaghi.h5"),
    "seed": 42,
    "torch_dtype": "float32",
}


def biased_time_grid(
    t_span: Tuple[float, float], n_times: int, exponent: float
) -> np.ndarray:
    """Return monotonically increasing time samples concentrated near t0.

    Args:
        t_span: Tuple ``(t_start, t_end)`` with ``t_end > t_start``.
        n_times: Number of time samples (>= 2).
        exponent: Power-law exponent controlling early-time oversampling.

    Returns:
        Array of shape ``(n_times,)`` with ``times[0] == t_start`` and
        ``times[-1] == t_end``.
    """
    if n_times < 2:
        raise ValueError("n_times must be at least 2.")
    t_start, t_end = map(float, t_span)
    if t_end <= t_start:
        raise ValueError("t_span must satisfy t_end > t_start.")
    ramp = np.linspace(0.0, 1.0, n_times, dtype=np.float64) ** float(exponent)
    times = t_start + (t_end - t_start) * ramp
    times[0] = t_start
    times[-1] = t_end
    return times


def enforce_planar_dirichlet_bc(plane: torch.Tensor) -> torch.Tensor:
    """Set Dirichlet boundary values to zero on the outer rim of a 2D tensor."""
    plane = plane.clone()
    plane[0, :] = 0.0
    plane[-1, :] = 0.0
    plane[:, 0] = 0.0
    plane[:, -1] = 0.0
    return plane


def sample_solution_points(
    field: np.ndarray,
    times: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    n_points: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Sample solution values at random space-time points with boundary coverage.

    Args:
        field: Solution array of shape ``(nt, nx, ny, nz)``.
        times: Time coordinates of shape ``(nt,)``.
        xs: X coordinates of shape ``(nx,)``.
        ys: Y coordinates of shape ``(ny,)``.
        zs: Z coordinates of shape ``(nz,)``.
        n_points: Number of points to sample.
        rng: Random number generator.

    Returns:
        Tuple ``(points, values)`` where ``points`` has shape ``(n_points, 4)``
        containing ``(t, x, y, z)`` and ``values`` has shape ``(n_points,)``.
    """
    nt, nx, ny, nz = field.shape

    def random_index(limit: int) -> int:
        """Generate a random integer index in [0, limit) using the seeded `rng`."""
        return int(rng.integers(0, limit))

    selections: List[Tuple[int, int, int, int]] = []
    seen = set()

    # Helper to register unique index tuples so we avoid duplicates.
    def add_index(ti: int, xi: int, yi: int, zi: int) -> None:
        key = (ti, xi, yi, zi)
        if key in seen:
            return
        seen.add(key)
        selections.append(key)

    boundary_time_indices = (0, nt - 1)
    boundary_faces: Sequence[Tuple[str, Iterable[int]]] = (
        ("x", (0, nx - 1)),
        ("y", (0, ny - 1)),
        ("z", (0, nz - 1)),
    )

    # Sample early and late times to capture transient extremes.
    # Keep the spatial sampling (don't focusing on edge indices)
    for ti in boundary_time_indices:
        add_index(
            ti,
            random_index(nx),
            random_index(ny),
            random_index(nz),
        )

    # Sample each spatial boundary face to anchor boundary behaviour.
    for axis, boundary_indices in boundary_faces:
        for idx in boundary_indices:
            ti = random_index(nt)
            if axis == "x":
                add_index(ti, idx, random_index(ny), random_index(nz))
            elif axis == "y":
                add_index(ti, random_index(nx), idx, random_index(nz))
            else:
                add_index(ti, random_index(nx), random_index(ny), idx)

    # Fill the remainder with random interior samples.
    while len(selections) < n_points:
        add_index(
            random_index(nt),
            random_index(nx),
            random_index(ny),
            random_index(nz),
        )

    selections = selections[:n_points]

    points = np.empty((len(selections), 4), dtype=np.float32)
    values = np.empty((len(selections),), dtype=np.float32)

    # Convert sampled indices into physical coordinates and solution values.
    for idx, (ti, xi, yi, zi) in enumerate(selections):
        points[idx] = (
            float(times[ti]),
            float(xs[xi]),
            float(ys[yi]),
            float(zs[zi]),
        )
        values[idx] = float(field[ti, xi, yi, zi])

    return points, values


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
    time_bias = float(cfg["time_bias_exponent"])
    cv_min, cv_max = (float(x) for x in cfg["cv_range"])  # type: ignore[index]
    gp_output_scale = float(cfg["gp_output_scale"])
    gp_length_scale_xy = float(cfg["gp_length_scale_xy"])
    u0_ranges = tuple(tuple(float(v) for v in pair) for pair in cfg["u0_ranges"])  # type: ignore[index]
    seed = int(cfg["seed"])
    dtype_str = str(cfg["torch_dtype"])
    torch_dtype = getattr(torch, dtype_str)
    output_path = Path(cfg["output_path"])  # type: ignore[arg-type]

    rng = np.random.default_rng(seed)
    time_samples = biased_time_grid(t_span, n_time_points, time_bias)

    x_range = tuple(float(v) for v in cfg["x_range"])  # type: ignore[index]
    y_range = tuple(float(v) for v in cfg["y_range"])  # type: ignore[index]
    z_range = tuple(float(v) for v in cfg["z_range"])  # type: ignore[index]

    xs = np.linspace(x_range[0], x_range[1], nx, dtype=np.float64)
    ys = np.linspace(y_range[0], y_range[1], ny, dtype=np.float64)
    zs = np.linspace(z_range[0], z_range[1], nz, dtype=np.float64)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as h5_file:
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

        sample_idx = 0
        # Iterate over batches to limit solver memory footprint while covering all samples.
        while sample_idx < n_samples:
            current_batch = min(batch_size, n_samples - sample_idx)
            batch_seed = seed + sample_idx
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

            for local_idx in range(current_batch):
                cv_value = float(rng.uniform(cv_min, cv_max))
                u0_xy = enforce_planar_dirichlet_bc(u0_batch[local_idx])

                # Solve PDE for the sampled initial field and consolidation coefficient.
                solver_result = solve_terzaghi_3d_fdm_batch(
                    Cv=cv_value,
                    x_range=x_range,
                    y_range=y_range,
                    z_range=z_range,
                    nx=nx,
                    ny=ny,
                    nz=nz,
                    t_span=t_span,
                    u0_xy_batch=u0_xy[None],
                    t_eval=time_samples,
                    dtype=torch_dtype,
                )

                field = solver_result["u"][0].cpu().numpy()
                # Draw random space-time query points and evaluate the solution there.
                points, values = sample_solution_points(
                    field,
                    solver_result["t"].cpu().numpy(),
                    xs,
                    ys,
                    zs,
                    points_per_sample,
                    rng,
                )

                # Save branch inputs, trunk inputs, and targets into HDF5 datasets.
                dset_u[sample_idx] = u0_xy.cpu().numpy()
                dset_cv[sample_idx] = cv_value
                dset_points[sample_idx] = points
                dset_values[sample_idx] = values

                sample_idx += 1


if __name__ == "__main__":
    generate_training_data(CONFIG)
