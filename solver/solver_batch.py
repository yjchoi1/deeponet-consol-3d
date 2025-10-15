import math
import random as _random
from typing import Dict, List, Optional, Sequence, Tuple, Union

import gstools as gs
import numpy as np
import torch


TensorLike = Union[torch.Tensor, np.ndarray, Sequence[float]]


def random_gaussian_pwp_batch(
    n_samples: int,
    m: int,
    z_range: Tuple[float, float],
    gp_params: Dict[str, float],
    u0_ranges: Sequence[Tuple[float, float]],
    seed: int = 42,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate random Gaussian batch profiles u(z) on m depth points."""
    device = torch.device("cpu") if device is None else device

    np.random.seed(seed)
    _random.seed(seed)

    z = np.linspace(z_range[0], z_range[1], m)
    model = gs.Gaussian(
        dim=1,
        var=gp_params.get("output_scale", 1.0) ** 2,
        len_scale=gp_params.get("length_scales", 1.0),
    )

    profiles: List[np.ndarray] = []
    for _ in range(n_samples):
        lo, hi = _random.choice(list(u0_ranges))
        mean = np.random.uniform(lo, hi)
        srf = gs.SRF(model, mean=mean)
        profiles.append(np.asarray(srf(z), dtype=float))

    return torch.as_tensor(np.stack(profiles, axis=0), dtype=dtype, device=device)


def _apply_drained_dirichlet_bc_(U: torch.Tensor) -> None:
    """In-place Dirichlet (u=0) boundary condition across the last three dims."""
    U[..., 0, :, :] = 0.0
    U[..., -1, :, :] = 0.0
    U[..., :, 0, :] = 0.0
    U[..., :, -1, :] = 0.0
    U[..., :, :, 0] = 0.0
    U[..., :, :, -1] = 0.0


def build_initial_field_from_profiles(
    u0_batch: torch.Tensor, nx: int, ny: int, nz: int
) -> torch.Tensor:
    """Broadcast vertical profiles (batch, nz) into 3D grids with drained BC."""
    if u0_batch.ndim != 2 or u0_batch.shape[1] != nz:
        raise ValueError(f"u0_batch must have shape (batch, {nz})")

    field = u0_batch[:, None, None, :].expand(-1, nx, ny, -1).contiguous().clone()
    _apply_drained_dirichlet_bc_(field)
    return field


def solve_terzaghi_3d_fdm_batch(
    Cv: float,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    z_range: Tuple[float, float],
    nx: int,
    ny: int,
    nz: int,
    t_span: Tuple[float, float],
    batch_size: Optional[int] = None,
    u0_z_batch: Optional[TensorLike] = None,
    gp_params: Optional[Dict[str, float]] = None,
    u0_ranges: Optional[Sequence[Tuple[float, float]]] = None,
    seed: int = 42,
    t_eval: Optional[TensorLike] = None,
    *,
    dtype: torch.dtype = torch.float32,
    device: Union[str, torch.device] = "cpu",
    dt: Optional[float] = None,
    safety_factor: float = 0.9,
) -> Dict[str, torch.Tensor]:
    """
    Batched explicit finite-difference solver for Terzaghi's 3D consolidation PDE.
    """
    device = torch.device(device)
    t0, tf = map(float, t_span)
    if tf <= t0:
        raise ValueError("t_span must satisfy t_span[1] > t_span[0]")

    dx = (x_range[1] - x_range[0]) / (nx - 1)
    dy = (y_range[1] - y_range[0]) / (ny - 1)
    dz = (z_range[1] - z_range[0]) / (nz - 1)

    inv_dx2 = 1.0 / (dx * dx)
    inv_dy2 = 1.0 / (dy * dy)
    inv_dz2 = 1.0 / (dz * dz)

    if t_eval is None:
        times = torch.linspace(t0, tf, 50, dtype=dtype, device=device)
    else:
        times = torch.as_tensor(t_eval, dtype=dtype, device=device)
        if times.ndim != 1:
            raise ValueError("t_eval must be 1-D.")

    nt = int(times.shape[0])

    if u0_z_batch is not None:
        u0_tensor = torch.as_tensor(u0_z_batch, dtype=dtype, device=device)
        batch = int(u0_tensor.shape[0])
    else:
        if batch_size is None or gp_params is None or u0_ranges is None:
            raise ValueError("Provide batch_size, gp_params and u0_ranges to sample ICs.")
        u0_tensor = random_gaussian_pwp_batch(
            batch_size, nz, z_range, gp_params, u0_ranges, seed=seed, device=device, dtype=dtype
        )
        batch = batch_size

    U = build_initial_field_from_profiles(u0_tensor, nx, ny, nz)

    dt_limit = 1.0 / (2.0 * Cv * (inv_dx2 + inv_dy2 + inv_dz2))
    max_step = dt if dt is not None else safety_factor * dt_limit
    if max_step <= 0 or max_step > dt_limit:
        raise ValueError("Choose dt within (0, stability_limit].")

    sols = torch.empty((batch, nt, nx, ny, nz), dtype=dtype, device=device)
    sols[:, 0] = U

    current_time = float(times[0])

    with torch.no_grad():
        for idx in range(1, nt):
            target_time = float(times[idx])
            interval = target_time - current_time
            n_sub = max(1, int(math.ceil(interval / max_step)))
            sub_dt = interval / n_sub

            for _ in range(n_sub):
                core = U[:, 1:-1, 1:-1, 1:-1]
                lap = (
                    (U[:, 2:, 1:-1, 1:-1] - 2.0 * core + U[:, :-2, 1:-1, 1:-1]) * inv_dx2
                    + (U[:, 1:-1, 2:, 1:-1] - 2.0 * core + U[:, 1:-1, :-2, 1:-1]) * inv_dy2
                    + (U[:, 1:-1, 1:-1, 2:] - 2.0 * core + U[:, 1:-1, 1:-1, :-2]) * inv_dz2
                )
                core.add_(sub_dt * Cv * lap)
                _apply_drained_dirichlet_bc_(U)

            sols[:, idx] = U
            current_time = target_time

    _apply_drained_dirichlet_bc_(sols)

    return {
        "t": times,
        "u": sols,
        "dx": torch.tensor(dx, dtype=dtype, device=device),
        "dy": torch.tensor(dy, dtype=dtype, device=device),
        "dz": torch.tensor(dz, dtype=dtype, device=device),
    }