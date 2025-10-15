import numpy as np
from typing import Tuple, List, Optional, Sequence
from scipy.integrate import solve_ivp
import random as _random
import gstools as gs


def random_gaussian_pwp(
    n_samples: int,
    m: int,
    z_range: Tuple[float, float],
    gp_params: dict,
    u0_ranges: Sequence[Tuple[float, float]],
    seed: int = 42,
) -> List[np.ndarray]:
    """
    Generate random Gaussian input functions u(z) for an initial vertical
    pore-water-pressure profile u_0(z) using GSTools for Gaussian process generation.
    """

    np.random.seed(seed)
    _random.seed(seed)

    # Generate input sensor locations
    z = np.linspace(z_range[0], z_range[1], m)
    
    # Create Gaussian model using GSTools
    model = gs.Gaussian(
        dim=1,
        var=gp_params.get("output_scale", 1.0) ** 2,
        len_scale=gp_params.get("length_scales", 1.0)
    )

    outputs: List[np.ndarray] = []
    for _ in range(n_samples):
        # Randomly select base value (u0) from the provided ranges
        lo, hi = _random.choice(list(u0_ranges))
        u0_mean = np.random.uniform(lo, hi)
        
        # Generate spatial random field with mean u0_mean
        srf = gs.SRF(model, mean=u0_mean)
        u = srf(z)
        outputs.append(u)
    return outputs


def _apply_drained_dirichlet_bc(U: np.ndarray) -> None:
    """In-place set drained Dirichlet BC (u=0) on all six domain faces."""
    U[0, :, :] = 0.0
    U[-1, :, :] = 0.0
    U[:, 0, :] = 0.0
    U[:, -1, :] = 0.0
    U[:, :, 0] = 0.0
    U[:, :, -1] = 0.0


def build_initial_field_from_profile(
    u0_z: np.ndarray, nx: int, ny: int, nz: int
) -> np.ndarray:
    """
    Broadcast a vertical profile u0(z) over x and y, then apply drained BC.

    Parameters
    - u0_z: array-like shape (nz,) representing u(x,y,z,0) varying only with z
    - nx, ny, nz: grid sizes

    Returns
    - U0: array shape (nx, ny, nz)
    """
    u0_z = np.asarray(u0_z, dtype=float)
    # Broadcast along x,y
    U0 = np.broadcast_to(u0_z.reshape(1, 1, nz), (nx, ny, nz)).astype(float).copy()
    # Enforce drained boundaries
    _apply_drained_dirichlet_bc(U0)
    return U0


def solve_terzaghi_3d_fdm(
    Cv: float,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    z_range: Tuple[float, float],
    nx: int,
    ny: int,
    nz: int,
    t_span: Tuple[float, float],
    u0_z: Optional[np.ndarray] = None,
    gp_params: Optional[dict] = None,
    u0_ranges: Optional[Sequence[Tuple[float, float]]] = None,
    seed: int = 42,
    t_eval: Optional[np.ndarray] = None,
    rtol: float = 1e-6,
    atol: float = 1e-9,
):
    """
    Solve Terzaghi's 3D consolidation PDE using finite differences in space
    and SciPy's explicit RK45 time integrator.

    PDE: du/dt = Cv * (uxx + uyy + uzz)
    BC: drained (Dirichlet u=0) on all six faces
    IC: u(x,y,z,0) = u0(z), broadcast across x,y

    Parameters
    - Cv: consolidation coefficient
    - x_range, y_range, z_range: (min, max) domain bounds
    - nx, ny, nz: number of grid points in x, y, z (including boundaries)
    - t_span: (t0, tf)
    - u0_z: optional array of shape (nz,) for initial profile; if None, one
      sample is generated via random_gaussian_pwp using gp_params and u0_ranges
    - gp_params, u0_ranges, seed: parameters for random_gaussian_pwp when u0_z is None
    - t_eval: times at which to store the computed solution; defaults to 50 linspace samples
    - rtol, atol: tolerances for solve_ivp

    Returns
    - result: dict with keys
        - 't': times (array of shape (nt,))
        - 'u': solution array of shape (nt, nx, ny, nz)
        - 'dx', 'dy', 'dz': grid spacings
    """
    if t_eval is None:
        t_eval = np.linspace(t_span[0], t_span[1], 50)

    # Grid spacings (uniform grid)
    dx = (x_range[1] - x_range[0]) / float(nx - 1)
    dy = (y_range[1] - y_range[0]) / float(ny - 1)
    dz = (z_range[1] - z_range[0]) / float(nz - 1)

    if u0_z is None:
        u0_z = random_gaussian_pwp(1, nz, z_range, gp_params, u0_ranges, seed)[0]

    # Build full 3D initial field and enforce drained boundaries
    U0 = build_initial_field_from_profile(np.asarray(u0_z, dtype=float), nx, ny, nz)

    # Flatten helpers
    def _flatten(U: np.ndarray) -> np.ndarray:
        return U.ravel(order="C")
    def _unflatten(u_vec: np.ndarray) -> np.ndarray:
        return u_vec.reshape((nx, ny, nz), order="C")

    # ODE RHS: du/dt = Cv * Laplacian(u) in the interior; du/dt=0 on boundaries
    def rhs(t: float, u_vec: np.ndarray) -> np.ndarray:  # noqa: ARG001
        U = _unflatten(u_vec)
        dUdt = np.zeros_like(U)

        # Compute 3D Laplacian on the interior
        core = U[1:-1, 1:-1, 1:-1]
        lap = (
            (U[2:, 1:-1, 1:-1] - 2.0 * core + U[:-2, 1:-1, 1:-1]) / (dx * dx)
            + (U[1:-1, 2:, 1:-1] - 2.0 * core + U[1:-1, :-2, 1:-1]) / (dy * dy)
            + (U[1:-1, 1:-1, 2:] - 2.0 * core + U[1:-1, 1:-1, :-2]) / (dz * dz)
        )
        dUdt[1:-1, 1:-1, 1:-1] = Cv * lap

        # Enforce Dirichlet BCs: boundary values should remain zero -> du/dt = 0
        # (boundaries are already zero in U if U0 had them set; this keeps them fixed)
        return _flatten(dUdt)

    # Solve
    sol = solve_ivp(
        rhs,
        t_span=t_span,
        y0=_flatten(U0),
        method="RK45",
        t_eval=np.asarray(t_eval, dtype=float),
        rtol=rtol,
        atol=atol,
        vectorized=False,
    )

    nt = sol.t.shape[0]
    U_ts = np.empty((nt, nx, ny, nz), dtype=float)
    for i in range(nt):
        U_ts[i] = _unflatten(sol.y[:, i])

    # Re-apply BCs to be safe against numerical drift
    for i in range(nt):
        _apply_drained_dirichlet_bc(U_ts[i])

    return {
        "t": sol.t,
        "u": U_ts,
        "dx": dx,
        "dy": dy,
        "dz": dz,
    }