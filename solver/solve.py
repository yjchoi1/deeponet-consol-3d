import numpy as np
from typing import Tuple, List, Optional, Sequence
from scipy.integrate import solve_ivp
import random as _random
import gstools as gs
import time


def random_gaussian_pwp(
    n_samples: int,
    nx: int,
    ny: int,
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    gp_params: dict,
    u0_ranges: Sequence[Tuple[float, float]],
    seed: int = 42,
) -> List[np.ndarray]:
    """
    Generate random Gaussian input surfaces u(x, y) for an initial pore-water-pressure
    field that is constant with depth and varies only in the horizontal plane.
    """

    gp_params = gp_params or {}

    np.random.seed(seed)
    _random.seed(seed)

    x = np.linspace(x_range[0], x_range[1], nx)
    y = np.linspace(y_range[0], y_range[1], ny)

    model = gs.Gaussian(
        dim=2,
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
        field = np.asarray(srf.structured([x, y]), dtype=float)
        outputs.append(field)
    return outputs


def _apply_drained_dirichlet_bc(U: np.ndarray) -> None:
    """In-place set drained Dirichlet BC (u=0) on all six domain faces."""
    U[0, :, :] = 0.0
    U[-1, :, :] = 0.0
    U[:, 0, :] = 0.0
    U[:, -1, :] = 0.0
    U[:, :, 0] = 0.0
    U[:, :, -1] = 0.0


def build_initial_field_from_surface(
    u0_xy: np.ndarray, nx: int, ny: int, nz: int
) -> np.ndarray:
    """
    Broadcast a horizontal surface u0(x, y) uniformly across depth, then apply drained BC.

    Parameters
    - u0_xy: array-like shape (nx, ny) representing u(x,y,z,0) varying with x,y only
    - nx, ny, nz: grid sizes

    Returns
    - U0: array shape (nx, ny, nz)
    """
    u0_xy = np.asarray(u0_xy, dtype=float)
    if u0_xy.shape != (nx, ny):
        raise ValueError(f"u0_xy must have shape ({nx}, {ny}), got {u0_xy.shape}")

    # Broadcast along z
    U0 = np.repeat(u0_xy[:, :, None], nz, axis=2).astype(float).copy()
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
    u0_xy: np.ndarray,
    t_eval: Optional[np.ndarray] = None,
    rtol: float = 1e-6,
    atol: float = 1e-9,
    method: str = "RK45",
):
    """
    Solve Terzaghi's 3D consolidation PDE using finite differences in space
    and SciPy's explicit RK45 time integrator.

    PDE: du/dt = Cv * (uxx + uyy + uzz)
    BC: drained (Dirichlet u=0) on all six faces
    IC: u(x,y,z,0) = u0(x, y), constant across depth

    Parameters
    - Cv: consolidation coefficient
    - x_range, y_range, z_range: (min, max) domain bounds
    - nx, ny, nz: number of grid points in x, y, z (including boundaries)
    - t_span: (t0, tf)
    - u0_xy: array of shape (nx, ny) for initial surface replicated across depth
    - t_eval: times at which to store the computed solution; defaults to 50 linspace samples
    - rtol, atol: tolerances for solve_ivp
    - method: integration method for solve_ivp (e.g., 'RK45', 'BDF')

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

    # Build full 3D initial field and enforce drained boundaries
    U0 = build_initial_field_from_surface(np.asarray(u0_xy, dtype=float), nx, ny, nz)

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
    integration_method = method.upper()

    sol = solve_ivp(
        rhs,
        t_span=t_span,
        y0=_flatten(U0),
        method=integration_method,
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


if __name__ == "__main__":
    # Hard-coded test inputs
    test_params = {
        "Cv": 1.0,
        "x_range": (0.0, 1.0),
        "y_range": (0.0, 1.0),
        "z_range": (0.0, 1.0),
        "nx": 51,
        "ny": 51,
        "nz": 51,
        "t_span": (0.0, 0.20),
        "rtol": 1e-6,
        "atol": 1e-9,
        "method": "RK45",
    }

    # Generate a simple test initial condition (Gaussian surface)
    nx = test_params["nx"]
    ny = test_params["ny"]
    x = np.linspace(test_params["x_range"][0], test_params["x_range"][1], nx)
    y = np.linspace(test_params["y_range"][0], test_params["y_range"][1], ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    # Simple Gaussian initial condition
    u0_xy = 10.0 * np.exp(-((X - 5.0) ** 2 + (Y - 5.0) ** 2) / 4.0)

    print("Testing Terzaghi 3D FDM solver...")
    print(f"Grid size: {nx} x {ny} x {test_params['nz']}")
    print(f"Time span: {test_params['t_span']}")
    print(f"Consolidation coefficient Cv: {test_params['Cv']}")

    # Measure execution time
    start_time = time.time()
    result = solve_terzaghi_3d_fdm(
        Cv=test_params["Cv"],
        x_range=test_params["x_range"],
        y_range=test_params["y_range"],
        z_range=test_params["z_range"],
        nx=test_params["nx"],
        ny=test_params["ny"],
        nz=test_params["nz"],
        t_span=test_params["t_span"],
        u0_xy=u0_xy,
        method=test_params["method"],
    )
    elapsed_time = time.time() - start_time

    print(f"\nSolver completed successfully!")
    print(f"Execution time: {elapsed_time:.4f} seconds")
    print(f"Solution shape: {result['u'].shape}")
    print(f"Number of time steps: {len(result['t'])}")
    print(f"Grid spacings: dx={result['dx']:.4f}, dy={result['dy']:.4f}, dz={result['dz']:.4f}")
    print(f"Initial u range: [{result['u'][0].min():.4f}, {result['u'][0].max():.4f}]")
    print(f"Final u range: [{result['u'][-1].min():.4f}, {result['u'][-1].max():.4f}]")
