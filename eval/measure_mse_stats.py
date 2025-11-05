from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf

from solver.solver_batch import (
    random_gaussian_pwp_batch,
    solve_terzaghi_3d_fdm_batch,
)
from train.models import build_model


# =====================================================================================
# Configuration (edit as needed)
# =====================================================================================
CASE = "case3_vanilla_ff"

CONFIG = {
    # Model and checkpoint paths
    "train_config_path": f"train/model/{CASE}/config.yaml",
    "checkpoint_path": f"train/model/{CASE}/latest.pt",
    "normalization_data_path": "train/data/deeponet_terzaghi_train.h5",

    # Device control: None -> use training config (if CUDA available), else CPU
    "device": None,

    # Train/Val sampling configuration
    "trainval": {
        "num_steps": 100,  # number of batches to evaluate for each of train and val
    },

    # Test (inference) evaluation configuration
    "test": {
        "num_samples": 5,            # number of random input functions to test
        "nx": 51,
        "ny": 51,
        "nz": 51,
        "x_range": (0.0, 1.0),
        "y_range": (0.0, 1.0),
        "z_range": (0.0, 1.0),
        "eval_times": [0.0, 0.2, 1.0],
        "t_span": (0.0, 1.0),
        # Spacetime MSE configuration (dense temporal grid)
        "compute_spacetime_mse": True,
        "nt_mse": 51,  # number of time points between t_span[0] and t_span[1]
        # Cv sampling: prefer "cv_ranges" (list of (min,max)) or "cv_range" (single (min,max)).
        # If neither is set or empty, falls back to constant "cv_value".
        # "cv_ranges": [0.03, 0.1],
        # "cv_range": (0.03, 0.1),
        "cv_value": 0.10,
        "gp_params": {
            "output_scale": 1000.0,
            "length_scales": 0.15,
        },
        "u0_ranges": [(10000.0, 20000.0)],
        "seed": 999,
        # batched inference: number of points per forward pass
        "batch_size": 51 * 51 * 51 * 3,
    },
}


# =====================================================================================
# Utilities
# =====================================================================================
H_DR = 0.5


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


def enforce_drained_dirichlet_bc(u_batch: torch.Tensor) -> torch.Tensor:
    if u_batch.ndim != 3:
        raise ValueError("u_batch must have shape (batch, nx, ny)")
    u_batch[:, 0, :] = 0.0
    u_batch[:, -1, :] = 0.0
    u_batch[:, :, 0] = 0.0
    u_batch[:, :, -1] = 0.0
    return u_batch


def create_query_points(
    eval_times: List[float],
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    z_range: Tuple[float, float],
    nx: int,
    ny: int,
    nz: int,
):
    xs = np.linspace(x_range[0], x_range[1], nx, dtype=np.float32)
    ys = np.linspace(y_range[0], y_range[1], ny, dtype=np.float32)
    zs = np.linspace(z_range[0], z_range[1], nz, dtype=np.float32)

    all_coords = []
    for t in eval_times:
        T, X, Y, Z = np.meshgrid([t], xs, ys, zs, indexing="ij")
        coords = np.stack([T.ravel(), X.ravel(), Y.ravel(), Z.ravel()], axis=1)
        all_coords.append(coords)

    all_coords = np.concatenate(all_coords, axis=0)
    return all_coords, xs, ys, zs


@torch.no_grad()
def evaluate_deeponet(
    model: torch.nn.Module,
    u0: np.ndarray,
    cv: float,
    coords: np.ndarray,
    stats: dict,
    device: torch.device,
    batch_size: int,
    flatten_branch: bool = True,
):
    model.eval()

    u_norm = (u0 - stats["u_mean"]) / stats["u_std"]
    if flatten_branch:
        u_norm = u_norm.reshape(-1)

    cv_norm = (cv - stats["cv_mean"]) / stats["cv_std"]
    coords_norm = (coords - stats["coord_mean"]) / stats["coord_std"]

    u_tensor = torch.as_tensor(u_norm, dtype=torch.float32, device=device)
    cv_scalar = torch.tensor([cv_norm], dtype=torch.float32, device=device)
    coords_tensor = torch.as_tensor(coords_norm, dtype=torch.float32, device=device)

    n_points = coords_tensor.shape[0]
    predictions: List[torch.Tensor] = []

    for i in range(0, n_points, batch_size):
        batch_coords = coords_tensor[i : i + batch_size]
        batch_cv = cv_scalar.expand(len(batch_coords), 1)
        batch_u = u_tensor.unsqueeze(0).expand(len(batch_coords), -1)
        output = model(batch_u, batch_cv, batch_coords)
        predictions.append(output.detach().cpu())

    predictions_np = torch.cat(predictions, dim=0).numpy().ravel()
    predictions_np = predictions_np * stats["s_std"] + stats["s_mean"]
    return predictions_np


def get_device(train_cfg, override: str | None) -> torch.device:
    if override is not None:
        return torch.device(override)
    # follow training device if CUDA is available, else CPU
    desired = str(train_cfg.training.device)
    if torch.cuda.is_available() and "cuda" in desired:
        return torch.device(desired)
    return torch.device("cpu")


def build_data_loaders(train_cfg):
    # Reuse the training dataloader builder
    from train.train import build_dataloaders

    return build_dataloaders(train_cfg)


def sample_cv_value(rng: np.random.Generator, test_cfg: dict) -> float:
    cv_ranges = test_cfg.get("cv_ranges")
    if cv_ranges:
        idx = int(rng.integers(0, len(cv_ranges)))
        low, high = cv_ranges[idx]
        return float(rng.uniform(low, high))
    cv_range = test_cfg.get("cv_range")
    if cv_range:
        low, high = cv_range
        return float(rng.uniform(low, high))
    return float(test_cfg["cv_value"])  # fallback constant


@torch.no_grad()
def compute_batch_mse_stats(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_steps: int,
    progress_label: str | None = None,
):
    model.eval()
    criterion = torch.nn.MSELoss()
    mses: List[float] = []
    for step_idx, batch in enumerate(loader):
        if step_idx >= num_steps:
            break
        if progress_label:
            print(f"    [{progress_label} {step_idx + 1}/{num_steps}]")
        u = batch["u"].to(device)
        cv = batch["cv"].to(device)
        coord = batch["coord"].to(device)
        target = batch["s"].to(device)
        output = model(u, cv, coord)
        loss = criterion(output, target)
        mses.append(float(loss.item()))

    mean = float(np.mean(mses)) if mses else float("nan")
    std = float(np.std(mses)) if mses else float("nan")
    return mean, std, mses


def main():
    print("=" * 80)
    print("MSE Statistics: Train / Val (limited steps) and Test (inference)")
    print("=" * 80)

    cfg = CONFIG

    # Load training configuration and build model
    print("\n[1] Loading training configuration and building model...")
    train_config_path = Path(cfg["train_config_path"])
    train_cfg = OmegaConf.load(train_config_path)
    model_cfg_dict = OmegaConf.to_container(train_cfg.model, resolve=True)
    assert isinstance(model_cfg_dict, dict)

    device = get_device(train_cfg, cfg.get("device"))
    print(f"    Using device: {device}")

    model = build_model(model_cfg_dict)
    model.to(device)

    # Load checkpoint
    checkpoint_path = Path(cfg["checkpoint_path"])
    print("\n[2] Loading checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"    Loaded: {checkpoint_path}")

    # Train/Val evaluation (limited steps)
    # Load normalization stats to convert MSE to physical scale
    stats_path = Path(cfg["normalization_data_path"]) 
    stats = load_normalization_stats(stats_path)

    print("\n[3] Building data loaders and evaluating train/val (limited steps)...")
    train_loader, val_loader = build_data_loaders(train_cfg)
    num_steps = int(cfg["trainval"]["num_steps"])
    print(f"    Evaluating first {num_steps} batches")

    train_mean, train_std, _ = compute_batch_mse_stats(
        model, train_loader, device, num_steps, progress_label="Train"
    )
    val_mean, val_std, _ = compute_batch_mse_stats(
        model, val_loader, device, num_steps, progress_label="Val"
    )

    # Convert normalized MSE statistics to physical scale: MSE_phys = (s_std^2) * MSE_norm
    scale_factor = stats["s_std"] ** 2
    train_mean *= scale_factor
    train_std *= scale_factor
    val_mean *= scale_factor
    val_std *= scale_factor

    print("\nTrain MSE over batches:")
    print(f"  mean = {train_mean:.6e}")
    print(f"  std  = {train_std:.6e}")
    print("Val MSE over batches:")
    print(f"  mean = {val_mean:.6e}")
    print(f"  std  = {val_std:.6e}")

    # Test-time evaluation (multiple random input functions)
    print("\n[4] Evaluating test MSE over random input functions (inference)...")
    test_cfg = cfg["test"]
    nx, ny, nz = int(test_cfg["nx"]), int(test_cfg["ny"]), int(test_cfg["nz"])
    x_range = tuple(test_cfg["x_range"])  # type: ignore[assignment]
    y_range = tuple(test_cfg["y_range"])  # type: ignore[assignment]
    z_range = tuple(test_cfg["z_range"])  # type: ignore[assignment]
    eval_times = list(test_cfg["eval_times"])  # type: ignore[assignment]
    num_samples = int(test_cfg["num_samples"])  # how many random u0
    t_span = tuple(test_cfg["t_span"])  # type: ignore[assignment]
    gp_params = dict(test_cfg["gp_params"])  # type: ignore[assignment]
    u0_ranges = list(test_cfg["u0_ranges"])  # type: ignore[assignment]
    seed = int(test_cfg["seed"]) if test_cfg.get("seed") is not None else None
    infer_batch_size = int(test_cfg["batch_size"])  # for coordinate batching

    # Points to evaluate
    all_coords, _, _, _ = create_query_points(
        eval_times, x_range, y_range, z_range, nx, ny, nz
    )

    sample_mses: List[float] = []
    sample_mses_spacetime: List[float] = []
    points_per_time = nx * ny * nz

    rng = np.random.default_rng(seed)

    for i in range(num_samples):
        print(f"    [Test {i + 1}/{num_samples}]")
        cv_value = sample_cv_value(rng, test_cfg)
        u0_batch = random_gaussian_pwp_batch(
            n_samples=1,
            nx=nx,
            ny=ny,
            x_range=x_range,
            y_range=y_range,
            gp_params=gp_params,
            u0_ranges=u0_ranges,
            seed=(seed + i) if seed is not None else None,
            device=device,
            dtype=torch.float32,
        )
        u0_batch = enforce_drained_dirichlet_bc(u0_batch)
        u0 = u0_batch[0].detach().cpu().numpy()

        # Solve true solution for this sample across eval_times
        solver_result = solve_terzaghi_3d_fdm_batch(
            Cv_batch=[cv_value],
            x_range=x_range,
            y_range=y_range,
            z_range=z_range,
            nx=nx,
            ny=ny,
            nz=nz,
            t_span=t_span,
            u0_xy_batch=u0_batch,
            t_eval=eval_times,
            dtype=torch.float32,
            device=device,
        )
        true_u = solver_result["u"].squeeze(0).detach().cpu().numpy()  # (T, nx, ny, nz)

        # Predict with DeepONet at all coordinates
        preds = evaluate_deeponet(
            model,
            u0,
            cv_value,
            all_coords,
            stats,
            device,
            batch_size=infer_batch_size,
            flatten_branch=bool(train_cfg.data.flatten_branch),
        )

        # Reshape predictions to (T, nx, ny, nz)
        T = len(eval_times)
        preds_stack = np.empty((T, nx, ny, nz), dtype=np.float32)
        for t_idx in range(T):
            s = t_idx * points_per_time
            e = s + points_per_time
            preds_stack[t_idx] = preds[s:e].reshape(nx, ny, nz)

        # MSE for this sample across all times and points
        mse = float(np.mean((preds_stack - true_u) ** 2))
        sample_mses.append(mse)

        # Optional: compute spacetime MSE on a dense temporal grid
        if bool(test_cfg.get("compute_spacetime_mse", False)):
            nt_mse = int(test_cfg.get("nt_mse", 51))
            t0, t1 = t_span
            eval_times_dense = np.linspace(t0, t1, nt_mse, dtype=np.float32)

            solver_result_dense = solve_terzaghi_3d_fdm_batch(
                Cv_batch=[cv_value],
                x_range=x_range,
                y_range=y_range,
                z_range=z_range,
                nx=nx,
                ny=ny,
                nz=nz,
                t_span=t_span,
                u0_xy_batch=u0_batch,
                t_eval=eval_times_dense.tolist(),
                dtype=torch.float32,
                device=device,
            )
            true_u_dense = solver_result_dense["u"].squeeze(0).detach().cpu().numpy()  # (T_dense, nx, ny, nz)

            se_total = 0.0
            count_total = 0

            for t_i, t in enumerate(eval_times_dense):
                coords_t, _, _, _ = create_query_points([float(t)], x_range, y_range, z_range, nx, ny, nz)
                preds_t = evaluate_deeponet(
                    model,
                    u0,
                    cv_value,
                    coords_t,
                    stats,
                    device,
                    batch_size=infer_batch_size,
                    flatten_branch=bool(train_cfg.data.flatten_branch),
                )
                true_t = true_u_dense[t_i]
                diff = preds_t - true_t.ravel()
                se_total += float(np.dot(diff, diff))
                count_total += diff.size

            mse_spacetime = se_total / max(count_total, 1)
            sample_mses_spacetime.append(float(mse_spacetime))

    test_mean = float(np.mean(sample_mses)) if sample_mses else float("nan")
    test_std = float(np.std(sample_mses)) if sample_mses else float("nan")

    print("\nTest MSE over samples:")
    print(f"  samples = {num_samples}")
    print(f"  mean    = {test_mean:.6e}")
    print(f"  std     = {test_std:.6e}")

    if sample_mses_spacetime:
        test_mean_st = float(np.mean(sample_mses_spacetime))
        test_std_st = float(np.std(sample_mses_spacetime))
        print("\nTest Spacetime MSE over samples (dense time grid):")
        print(f"  samples = {num_samples}")
        print(f"  mean    = {test_mean_st:.6e}")
        print(f"  std     = {test_std_st:.6e}")

    print("\n" + "=" * 80)
    print("Done.")
    print("=" * 80)


if __name__ == "__main__":
    main()


