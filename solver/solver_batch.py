import math
import random as _random
from typing import Dict, List, Optional, Sequence, Tuple, Union

import gstools as gs
import numpy as np
import torch


TensorLike = Union[torch.Tensor, np.ndarray, Sequence[float]]


def random_gaussian_pwp_batch(
    n_samples: int,
    nx: int,
    ny: int,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    gp_params: Dict[str, float],
    u0_ranges: Sequence[Tuple[float, float]],
    seed: int = 42,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate random Gaussian batch surfaces u(x, y) on an (nx, ny) grid."""
    device = torch.device("cpu") if device is None else device

    np.random.seed(seed)
    _random.seed(seed)

    gp_params = gp_params or {}

    x = np.linspace(x_range[0], x_range[1], nx)
    y = np.linspace(y_range[0], y_range[1], ny)
    model = gs.Gaussian(
        dim=2,
        var=gp_params.get("output_scale", 1.0) ** 2,
        len_scale=gp_params.get("length_scales", 1.0),
    )

    surfaces: List[np.ndarray] = []
    for _ in range(n_samples):
        lo, hi = _random.choice(list(u0_ranges))
        mean = np.random.uniform(lo, hi)
        srf = gs.SRF(model, mean=mean)
        surfaces.append(np.asarray(srf.structured([x, y]), dtype=float))

    return torch.as_tensor(np.stack(surfaces, axis=0), dtype=dtype, device=device)


def _apply_drained_dirichlet_bc_(U: torch.Tensor) -> None:
    """In-place Dirichlet (u=0) boundary condition across the last three dims."""
    U[..., 0, :, :] = 0.0
    U[..., -1, :, :] = 0.0
    U[..., :, 0, :] = 0.0
    U[..., :, -1, :] = 0.0
    U[..., :, :, 0] = 0.0
    U[..., :, :, -1] = 0.0


def build_initial_field_from_surfaces(
    u0_batch: torch.Tensor, nx: int, ny: int, nz: int
) -> torch.Tensor:
    """Broadcast horizontal surfaces (batch, nx, ny) into 3D grids with drained BC."""
    if u0_batch.ndim != 3 or u0_batch.shape[1:] != (nx, ny):
        raise ValueError(f"u0_batch must have shape (batch, {nx}, {ny})")

    field = u0_batch[:, :, :, None].expand(-1, -1, -1, nz).contiguous().clone()
    _apply_drained_dirichlet_bc_(field)
    return field


def solve_terzaghi_3d_fdm_batch(
    Cv_batch: TensorLike,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    z_range: Tuple[float, float],
    nx: int,
    ny: int,
    nz: int,
    t_span: Tuple[float, float],
    u0_xy_batch: TensorLike,
    t_eval: Optional[TensorLike] = None,
    *,
    dtype: torch.dtype = torch.float32,
    device: Union[str, torch.device] = "cuda",
    dt: Optional[float] = None,
    safety_factor: float = 0.9,
) -> Dict[str, torch.Tensor]:
    """
    Batched explicit finite-difference solver for Terzaghi's 3D consolidation PDE.

    Parameters
    - Cv_batch: per-sample consolidation coefficients shaped (batch,)
    - x_range, y_range, z_range: (min, max) domain bounds
    - nx, ny, nz: grid resolution (includes boundaries)
    - t_span: (t0, tf)
    - u0_xy_batch: initial surfaces shaped (batch, nx, ny) broadcast across depth
    - t_eval: optional time samples (defaults to 50 linspace points)
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

    u0_tensor = torch.as_tensor(u0_xy_batch, dtype=dtype, device=device)
    if u0_tensor.ndim != 3 or u0_tensor.shape[1:] != (nx, ny):
        raise ValueError(f"u0_xy_batch must have shape (batch, {nx}, {ny})")
    batch = int(u0_tensor.shape[0])

    U = build_initial_field_from_surfaces(u0_tensor, nx, ny, nz)

    Cv_tensor = torch.as_tensor(Cv_batch, dtype=dtype, device=device).view(-1)
    if Cv_tensor.shape[0] != batch:
        raise ValueError(f"Cv_batch must have shape ({batch},)")
    if torch.any(Cv_tensor <= 0):
        raise ValueError("All Cv values must be positive.")

    diffusion_scale = inv_dx2 + inv_dy2 + inv_dz2
    dt_limits = 1.0 / (2.0 * Cv_tensor * diffusion_scale)
    dt_limit = torch.min(dt_limits).item()
    max_step = float(dt) if dt is not None else safety_factor * dt_limit
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
                core.add_(sub_dt * Cv_tensor.view(batch, 1, 1, 1) * lap)
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
