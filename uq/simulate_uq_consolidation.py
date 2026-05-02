from __future__ import annotations

import sys
from pathlib import Path

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import math
import h5py
import numpy as np
import torch
from omegaconf import OmegaConf

from solver.solver_batch import (
    random_gaussian_pwp_batch,
    solve_terzaghi_3d_fdm_batch,
)
from train.models import build_model


# ============================================================================
# CONFIGURATION (All inputs here)
# ============================================================================

# data_v2
CASE = "case3_data_v2_vanilla_ff_scaling"
MODE = "solver"
UQ_CONFIG = {
    # Evaluation mode: "deeponet" (fast, needs model) or "solver" (reference)
    "mode": MODE,

    # Model and normalization (when mode == "deeponet")
    "train_config_path": f"train/model/{CASE}/config.yaml",
    "checkpoint_path": f"train/model/{CASE}/latest.pt",
    "normalization_data_path": "data/train.h5",

    # Grid parameters (must match training/data gen)
    "nx": 51,
    "ny": 51,
    "nz": 51,
    "x_range": (0.0, 1.0),
    "y_range": (0.0, 1.0),
    "z_range": (0.0, 1.0),

    # Time domain
    "t_span": (0.0, 0.20),
    "nt": 51,

    # Monte Carlo settings
    "num_samples": 100,
    "seed": 2025,
    # Base seed for GRF u0 so samples match across modes (per-sample offset applied)
    "u0_seed_base": 12025,

    # GRF initial condition parameters
    "gp_params": {
        "output_scale": 1000.0,
        "length_scales": 0.30,
    },
    "u0_ranges": [(15000.0, 15000.0)],

    # Uncertain Cv ~ Normal(mean, std), truncated at cv_min
    "cv_mean": 0.5,
    "cv_std": 0.05,
    "cv_min": 0.3,

    # Drainage path H_dr (Tv = Cv * t / H_dr^2)
    "H_DR": 0.5,

    # Inference batch size (DeepONet)
    "batch_size": 51 * 51 * 51 * 3,

    # Outputs (NPZ only from this module; CSV/plots are in postprocess)
    "output_dir": f"uq/{CASE}/{MODE}/",
    "uv_timeseries_npz": f"uq/{CASE}/{MODE}/Uv_timeseries_constant_cv.npz",
}


# data_v1
# CASE = "case3_vanilla_ff"
# MODE = "solver"
# UQ_CONFIG = {
#     # Evaluation mode: "deeponet" (fast, needs model) or "solver" (reference)
#     "mode": MODE,

#     # Model and normalization (when mode == "deeponet")
#     "train_config_path": f"train/model/{CASE}/config.yaml",
#     "checkpoint_path": f"train/model/{CASE}/latest.pt",
#     "normalization_data_path": "data/train.h5",

#     # Grid parameters (must match training/data gen)
#     "nx": 51,
#     "ny": 51,
#     "nz": 51,
#     "x_range": (0.0, 1.0),
#     "y_range": (0.0, 1.0),
#     "z_range": (0.0, 1.0),

#     # Time domain
#     "t_span": (0.0, 1.0),
#     "nt": 51,

#     # Monte Carlo settings
#     "num_samples": 100,
#     "seed": 2025,
#     # Base seed for GRF u0 so samples match across modes (per-sample offset applied)
#     "u0_seed_base": 12025,

#     # GRF initial condition parameters
#     "gp_params": {
#         "output_scale": 1000.0,
#         "length_scales": 0.15,
#     },
#     "u0_ranges": [(15000.0, 20000.0)],

#     # Uncertain Cv ~ Normal(mean, std), truncated at cv_min
#     "cv_mean": 0.05,
#     "cv_std": 0.0,
#     "cv_min": 0.02,

#     # Drainage path H_dr (Tv = Cv * t / H_dr^2)
#     "H_DR": 0.5,

#     # Inference batch size (DeepONet)
#     "batch_size": 51 * 51 * 51 * 3,

#     # Outputs (NPZ only from this module; CSV/plots are in postprocess)
#     "output_dir": f"uq/{CASE}/{MODE}/",
#     "uv_timeseries_npz": f"uq/{CASE}/{MODE}/Uv_timeseries_constant_cv.npz",
# }


# ============================================================================
# HELPERS
# ============================================================================
def load_normalization_stats(data_path: Path) -> dict:
    with h5py.File(data_path, "r") as f:
        stats_group = f["stats"]
        stats = {
            "u_mean": float(np.asarray(stats_group["u_mean"], dtype=np.float32)),
            "u_std": float(np.asarray(stats_group["u_std"], dtype=np.float32)),
            "cv_mean": float(np.asarray(stats_group["cv_mean"], dtype=np.float32)),
            "cv_std": float(np.asarray(stats_group["cv_std"], dtype=np.float32)),
            "coord_mean": np.asarray(stats_group["coord_mean"], dtype=np.float32),
            "coord_std": np.asarray(stats_group["coord_std"], dtype=np.float32),
            "s_mean": float(np.asarray(stats_group["s_mean"], dtype=np.float32)),
            "s_std": float(np.asarray(stats_group["s_std"], dtype=np.float32)),
        }
    return stats


def enforce_drained_dirichlet_bc(u_xy: torch.Tensor) -> torch.Tensor:
    if u_xy.ndim != 2:
        raise ValueError("u_xy must have shape (nx, ny)")
    u_xy[0, :] = 0.0
    u_xy[-1, :] = 0.0
    u_xy[:, 0] = 0.0
    u_xy[:, -1] = 0.0
    return u_xy


def create_query_points(eval_times, x_range, y_range, z_range, nx, ny, nz):
    xs = np.linspace(x_range[0], x_range[1], nx, dtype=np.float32)
    ys = np.linspace(y_range[0], y_range[1], ny, dtype=np.float32)
    zs = np.linspace(z_range[0], z_range[1], nz, dtype=np.float32)

    all_coords = []
    for t in eval_times:
        T, X, Y, Z = np.meshgrid([t], xs, ys, zs, indexing="ij")
        coords = np.stack([T.ravel(), X.ravel(), Y.ravel(), Z.ravel()], axis=1)
        all_coords.append(coords)

    all_coords = np.concatenate(all_coords, axis=0)
    return all_coords


def evaluate_deeponet(model, u0, cv_value, coords, stats, device, batch_size, flatten_branch=True):
    model.eval()

    u_norm = (u0 - stats["u_mean"]) / stats["u_std"]
    if flatten_branch:
        u_norm = u_norm.reshape(-1)

    cv_norm = (cv_value - stats["cv_mean"]) / stats["cv_std"]
    coords_norm = (coords - stats["coord_mean"]) / stats["coord_std"]

    u_tensor = torch.as_tensor(u_norm, dtype=torch.float32, device=device)
    cv_scalar = torch.tensor([cv_norm], dtype=torch.float32, device=device)
    coords_tensor = torch.as_tensor(coords_norm, dtype=torch.float32, device=device)

    n_points = coords_tensor.shape[0]
    predictions = []
    with torch.no_grad():
        for i in range(0, n_points, batch_size):
            batch_coords = coords_tensor[i : i + batch_size]
            batch_cv = cv_scalar.expand(len(batch_coords), 1)
            batch_u = u_tensor.unsqueeze(0).expand(len(batch_coords), -1)
            output = model(batch_u, batch_cv, batch_coords)
            predictions.append(output.cpu())

    predictions_tensor = torch.cat(predictions, dim=0)
    predictions_normalized = predictions_tensor.numpy().ravel()
    predictions_physical = predictions_normalized * stats["s_std"] + stats["s_mean"]
    return predictions_physical


def compute_Uv_timeseries(u0_xy: np.ndarray, u_timeseries: np.ndarray) -> np.ndarray:
    u0_bar = float(np.mean(u0_xy))
    eps = 1e-12
    Uv = []
    for i in range(u_timeseries.shape[0]):
        u_bar = float(np.mean(u_timeseries[i]))
        Uv.append(1.0 - (u_bar / max(u0_bar, eps)))
    return np.asarray(Uv, dtype=np.float64)


def interp_t50(eval_times: np.ndarray, Uv: np.ndarray) -> float:
    Uv_mono = np.maximum.accumulate(Uv)
    if Uv_mono[-1] < 0.5:
        return float("nan")
    return float(np.interp(0.5, Uv_mono, eval_times))


# ============================================================================
# SIMULATION ENTRYPOINT
# ============================================================================
def run_uq(cfg: dict | None = None, save_npz_path: str | None = None) -> dict:
    cfg = UQ_CONFIG if cfg is None else cfg

    # Time grid
    t0, t1 = cfg["t_span"]
    nt = int(cfg.get("nt", 51))
    eval_times = np.linspace(t0, t1, nt, dtype=np.float32)

    # Model if needed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None
    stats = None
    flatten_branch = True
    if cfg["mode"].lower() == "deeponet":
        train_cfg = OmegaConf.load(Path(cfg["train_config_path"]))
        model_config = OmegaConf.to_container(train_cfg.model, resolve=True)
        model = build_model(model_config)
        model.to(device)
        checkpoint = torch.load(Path(cfg["checkpoint_path"]), map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        flatten_branch = bool(train_cfg.data.flatten_branch)
        stats = load_normalization_stats(Path(cfg["normalization_data_path"]))

    # Grid and sampling settings
    nx, ny, nz = cfg["nx"], cfg["ny"], cfg["nz"]
    x_range, y_range, z_range = cfg["x_range"], cfg["y_range"], cfg["z_range"]
    gp_params = cfg["gp_params"]
    u0_ranges = cfg["u0_ranges"]
    batch_size = int(cfg.get("batch_size", nx * ny * nz))
    H_DR = float(cfg["H_DR"])  # kept for completeness; used in postprocess for Tv

    # Monte Carlo
    num_samples = int(cfg["num_samples"])
    base_seed = int(cfg["seed"])
    rng_cv = np.random.default_rng(base_seed)
    u0_seed_base = int(cfg.get("u0_seed_base", base_seed + 10000))
    cv_mean = float(cfg["cv_mean"])
    cv_std = float(cfg["cv_std"])
    cv_min = float(cfg.get("cv_min", 0.0))

    # Storage
    Uv_all = np.zeros((num_samples, eval_times.size), dtype=np.float64)
    t50_all = np.full((num_samples,), np.nan, dtype=np.float64)
    Tv50_all = np.full((num_samples,), np.nan, dtype=np.float64)
    Cv_all = np.zeros((num_samples,), dtype=np.float64)

    # Run samples
    for s in range(num_samples):
        # Sample Cv (positive)
        cv_val = float(rng_cv.normal(cv_mean, cv_std))
        cv_val = max(cv_val, cv_min)
        Cv_all[s] = cv_val

        # Sample u0 via GRF and enforce drained BC
        u0_seed = u0_seed_base + s
        u0_batch = random_gaussian_pwp_batch(
            n_samples=1,
            nx=nx,
            ny=ny,
            x_range=x_range,
            y_range=y_range,
            gp_params=gp_params,
            u0_ranges=u0_ranges,
            seed=u0_seed,
            device=device,
            dtype=torch.float32,
        )
        u0_xy = enforce_drained_dirichlet_bc(u0_batch[0]).cpu().numpy()

        # Solve/evaluate
        if cfg["mode"].lower() == "deeponet":
            coords = create_query_points(eval_times, x_range, y_range, z_range, nx, ny, nz)
            pred_phys = evaluate_deeponet(
                model,
                u0_xy,
                cv_val,
                coords,
                stats,
                device,
                batch_size=batch_size,
                flatten_branch=flatten_branch,
            )
            per_t = nx * ny * nz
            u_timeseries = np.zeros((eval_times.size, nx, ny, nz), dtype=np.float32)
            for i in range(eval_times.size):
                seg = pred_phys[i * per_t : (i + 1) * per_t]
                u_timeseries[i] = seg.reshape(nx, ny, nz)
        else:
            sol = solve_terzaghi_3d_fdm_batch(
                Cv_batch=[cv_val],
                x_range=x_range,
                y_range=y_range,
                z_range=z_range,
                nx=nx,
                ny=ny,
                nz=nz,
                t_span=(float(eval_times[0]), float(eval_times[-1])),
                u0_xy_batch=u0_batch,
                t_eval=[float(t) for t in eval_times],
                dtype=torch.float32,
                device=device,
            )
            u_timeseries = sol["u"].squeeze(0).cpu().numpy()

        # Metrics
        Uv = compute_Uv_timeseries(u0_xy, u_timeseries)
        t50 = interp_t50(eval_times.astype(np.float64), Uv)
        Tv50 = (cv_val * t50 / (H_DR ** 2)) if not math.isnan(t50) else float("nan")

        Uv_all[s] = Uv
        t50_all[s] = t50
        Tv50_all[s] = Tv50

        # Simple progress
        print(f"Progress: {s + 1}/{num_samples}")

    results = {
        "Uv_all": Uv_all,
        "eval_times": eval_times,
        "t50_all": t50_all,
        "Tv50_all": Tv50_all,
        "Cv_all": Cv_all,
        "H_DR": H_DR,
        "CASE": CASE,
    }

    # Save NPZ if requested or configured
    out_npz = (
        Path(save_npz_path)
        if save_npz_path is not None
        else Path(UQ_CONFIG["uv_timeseries_npz"])
    )
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_npz, **results)

    return results


if __name__ == "__main__":
    print("Running UQ simulation and saving NPZ...")
    run_uq()
    print("Done.")


