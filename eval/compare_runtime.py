"""Benchmark DeepONet inference time against the numpy finite-difference solver."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Dict, List

import numpy as np
import torch
from omegaconf import OmegaConf


# Add project root to path so we can import project modules
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from solver.solve import random_gaussian_pwp, solve_terzaghi_3d_fdm
from train.models import build_model

from eval.evaluate_full_field import (  # noqa: E402  (import after path setup)
    create_query_points,
    load_normalization_stats,
)


BENCHMARK_CONFIG: Dict[str, object] = {
    "train_config_path": "train/model/cose3_vanilla_ff/config.yaml",
    "checkpoint_path": "train/model/cose3_vanilla_ff/latest.pt",
    "normalization_data_path": "train/data/deeponet_terzaghi_val.h5",
    "nx": 51,
    "ny": 51,
    "nz": 51,
    "x_range": (0.0, 1.0),
    "y_range": (0.0, 1.0),
    "z_range": (0.0, 1.0),
    "eval_times": [0.0, 0.2, 1.0],
    "t_span": (0.0, 1.0),
    "gp_params": {
        "output_scale": 1000.0,
        "length_scales": 0.15,
    },
    "u0_ranges": [(10000.0, 20000.0)],
    "num_samples": 1,
    "base_seed": 999,
    "deeponet_batch_size": 51*51*51*20,
    "cases": [
        {"name": "cv_0.05_rk45", "cv_value": 0.05, "solver_method": "RK45", "seed_offset": 0},
        {"name": "cv_0.05_bdf", "cv_value": 0.05, "solver_method": "RK45", "seed_offset": 0},
        {"name": "cv_0.10_bdf", "cv_value": 0.10, "solver_method": "RK45", "seed_offset": 10},
        {"name": "cv_0.20_bdf", "cv_value": 0.20, "solver_method": "RK45", "seed_offset": 20},
    ],
}


@dataclass
class SampleRuntime:
    sample_index: int
    deeponet_time: float
    solver_time: float
    speedup: float
    mse: float
    rel_l2: float


def apply_dirichlet_bc(surface: np.ndarray) -> np.ndarray:
    """Return a copy of ``surface`` with drained Dirichlet BC enforced."""
    u = np.asarray(surface, dtype=np.float32).copy()
    u[0, :] = 0.0
    u[-1, :] = 0.0
    u[:, 0] = 0.0
    u[:, -1] = 0.0
    return u


def synchronize_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def build_deeponet_model(train_config_path: Path, checkpoint_path: Path, device: torch.device):
    train_cfg = OmegaConf.load(train_config_path)
    model_cfg = OmegaConf.to_container(train_cfg.model, resolve=True)
    model = build_model(model_cfg)
    model.to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    flatten_branch = bool(getattr(train_cfg.data, "flatten_branch", True))

    return model, flatten_branch


def run_deeponet_inference_only(
    model: torch.nn.Module,
    u_branch_flat: torch.Tensor,
    cv_scalar: torch.Tensor,
    coords_tensor: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Run ONLY the model forward passes in batches and return normalized predictions.

    Inputs (u_branch_flat, cv_scalar, coords_tensor) must already be normalized and
    on the proper device. This function measures pure inference when timed outside.
    """
    model.eval()
    n_points = int(coords_tensor.shape[0])
    preds: List[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, n_points, batch_size):
            batch_coords = coords_tensor[i:i + batch_size]
            batch_cv = cv_scalar.expand(len(batch_coords), 1)
            batch_u = u_branch_flat.unsqueeze(0).expand(len(batch_coords), -1)
            out = model(batch_u, batch_cv, batch_coords)
            preds.append(out)
    return torch.cat(preds, dim=0)


def benchmark_case(
    case_cfg: Dict[str, object],
    base_config: Dict[str, object],
    model: torch.nn.Module,
    stats: Dict[str, object],
    device: torch.device,
    flatten_branch: bool,
    coords: np.ndarray,
) -> List[SampleRuntime]:
    nx = int(base_config["nx"])
    ny = int(base_config["ny"])
    nz = int(base_config["nz"])
    t_span = tuple(base_config["t_span"])  # type: ignore[arg-type]
    eval_times = list(base_config["eval_times"])  # type: ignore[arg-type]
    gp_params = dict(base_config["gp_params"])  # type: ignore[arg-type]
    u0_ranges = list(base_config["u0_ranges"])  # type: ignore[arg-type]
    x_range = tuple(base_config["x_range"])  # type: ignore[arg-type]
    y_range = tuple(base_config["y_range"])  # type: ignore[arg-type]
    z_range = tuple(base_config["z_range"])  # type: ignore[arg-type]

    num_samples = int(base_config["num_samples"])
    base_seed = int(base_config["base_seed"])
    deeponet_batch_size = int(base_config["deeponet_batch_size"])

    case_seed = base_seed + int(case_cfg.get("seed_offset", 0))
    cv_value = float(case_cfg["cv_value"])
    solver_method = str(case_cfg.get("solver_method", "BDF"))

    surfaces = random_gaussian_pwp(
        n_samples=num_samples,
        nx=nx,
        ny=ny,
        x_range=x_range,
        y_range=y_range,
        gp_params=gp_params,
        u0_ranges=u0_ranges,
        seed=case_seed,
    )

    # Precompute normalized coords tensor once per case (outside timing)
    coords_norm = (coords - stats["coord_mean"]) / stats["coord_std"]
    coords_tensor = torch.as_tensor(coords_norm, dtype=torch.float32, device=device)

    runtimes: List[SampleRuntime] = []
    points_per_time = nx * ny * nz

    for sample_idx, surface in enumerate(surfaces):
        u0_surface = apply_dirichlet_bc(surface)

        # Prepare normalized inputs (outside timing)
        u_norm = (u0_surface - stats["u_mean"]) / stats["u_std"]
        if flatten_branch:
            u_norm = u_norm.reshape(-1)
        u_branch_flat = torch.as_tensor(u_norm, dtype=torch.float32, device=device)

        cv_norm = (cv_value - stats["cv_mean"]) / stats["cv_std"]
        cv_scalar = torch.tensor([cv_norm], dtype=torch.float32, device=device)

        # Pure inference timing only
        synchronize_if_cuda(device)
        t0 = perf_counter()
        pred_norm_tensor = run_deeponet_inference_only(
            model,
            u_branch_flat,
            cv_scalar,
            coords_tensor,
            batch_size=deeponet_batch_size,
            device=device,
        )
        synchronize_if_cuda(device)
        deeponet_elapsed = perf_counter() - t0

        # Post-process (outside timing): move to CPU, ravel, and denormalize
        predictions = pred_norm_tensor.detach().cpu().numpy().ravel()
        predictions = predictions * stats["s_std"] + stats["s_mean"]

        pred_fields = []
        for time_index in range(len(eval_times)):
            start = time_index * points_per_time
            end = start + points_per_time
            pred_fields.append(predictions[start:end].reshape(nx, ny, nz))

        t1 = perf_counter()
        solver_result = solve_terzaghi_3d_fdm(
            Cv=cv_value,
            x_range=x_range,
            y_range=y_range,
            z_range=z_range,
            nx=nx,
            ny=ny,
            nz=nz,
            t_span=t_span,
            u0_xy=u0_surface,
            t_eval=np.asarray(eval_times, dtype=float),
            method=solver_method,
        )
        solver_elapsed = perf_counter() - t1

        true_fields = solver_result["u"]

        pred_stack = np.stack(pred_fields, axis=0)
        true_stack = np.asarray(true_fields, dtype=np.float32)

        diff = pred_stack - true_stack
        mse = float(np.mean(diff ** 2))
        rel_l2 = float(np.linalg.norm(diff) / (np.linalg.norm(true_stack) + 1e-12))

        runtimes.append(
            SampleRuntime(
                sample_index=sample_idx,
                deeponet_time=deeponet_elapsed,
                solver_time=solver_elapsed,
                speedup=solver_elapsed / deeponet_elapsed if deeponet_elapsed > 0 else float("inf"),
                mse=mse,
                rel_l2=rel_l2,
            )
        )

    return runtimes


def summarize_case(name: str, runtimes: List[SampleRuntime]) -> None:
    deeponet_times = np.array([r.deeponet_time for r in runtimes], dtype=float)
    solver_times = np.array([r.solver_time for r in runtimes], dtype=float)
    speedups = np.array([r.speedup for r in runtimes], dtype=float)
    mses = np.array([r.mse for r in runtimes], dtype=float)
    rel_l2s = np.array([r.rel_l2 for r in runtimes], dtype=float)

    print("\n" + "=" * 80)
    print(f"Case: {name}")
    print("=" * 80)

    for r in runtimes:
        print(
            f"Sample {r.sample_index:02d} | DeepONet: {r.deeponet_time:6.3f} s | "
            f"Solver: {r.solver_time:6.3f} s | Speedup: {r.speedup:6.2f}x | "
            f"MSE: {r.mse:.3e} | Rel L2: {r.rel_l2:.3e}"
        )

    print("-" * 80)
    print(
        "DeepONet avg: {mean:.3f} s (± {std:.3f})".format(
            mean=float(deeponet_times.mean()), std=float(deeponet_times.std())
        )
    )
    print(
        "Solver avg:   {mean:.3f} s (± {std:.3f})".format(
            mean=float(solver_times.mean()), std=float(solver_times.std())
        )
    )
    print(
        "Speedup avg:  {mean:.2f}x (min {min:.2f}x, max {max:.2f}x)".format(
            mean=float(speedups.mean()),
            min=float(speedups.min()),
            max=float(speedups.max()),
        )
    )
    print(
        "MSE avg:      {mean:.3e} (± {std:.3e})".format(
            mean=float(mses.mean()), std=float(mses.std())
        )
    )
    print(
        "Rel L2 avg:   {mean:.3e} (± {std:.3e})".format(
            mean=float(rel_l2s.mean()), std=float(rel_l2s.std())
        )
    )


def main() -> None:
    cfg = BENCHMARK_CONFIG

    train_config_path = PROJECT_ROOT / str(cfg["train_config_path"])
    checkpoint_path = PROJECT_ROOT / str(cfg["checkpoint_path"])
    normalization_path = PROJECT_ROOT / str(cfg["normalization_data_path"])

    train_cfg = OmegaConf.load(train_config_path)
    requested_device = getattr(train_cfg.training, "device", "cuda")
    device = torch.device(requested_device if torch.cuda.is_available() else "cpu")

    print("=" * 80)
    print("DeepONet vs finite-difference solver runtime benchmark")
    print("=" * 80)
    print(f"Using device: {device}")

    model, flatten_branch = build_deeponet_model(train_config_path, checkpoint_path, device)
    stats = load_normalization_stats(normalization_path)

    coords, _, _, _ = create_query_points(
        cfg["eval_times"],
        cfg["x_range"],
        cfg["y_range"],
        cfg["z_range"],
        cfg["nx"],
        cfg["ny"],
        cfg["nz"],
    )
    coords = np.asarray(coords, dtype=np.float32)

    for case in cfg["cases"]:
        runtimes = benchmark_case(
            case,
            cfg,
            model,
            stats,
            device,
            flatten_branch,
            coords,
        )
        summarize_case(case["name"], runtimes)

    print("\nBenchmark complete.")


if __name__ == "__main__":
    main()


