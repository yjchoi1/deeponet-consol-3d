"""Evaluate a trained ROM model by reconstructing full fields and comparing
to solver ground truth."""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Dict, List

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf

from eval.evaluate_full_field import plot_comparison
from rom.model_rom import build_rom_model
from solver.solver_batch import (
    random_gaussian_pwp_batch,
    solve_terzaghi_3d_fdm_batch,
)


# =====================================================================
# Configuration
# =====================================================================
CONFIG: Dict[str, object] = {
    "train_config_path": "rom/rom/checkpoint/config.yaml",
    "checkpoint_path": "rom/rom/checkpoint/latest.pt",
    "basis_path": "data/basis.npz",
    "fields_path": "data/rom_fields.h5",
    "device": None,  # None -> auto
    # Visualization (saved per sample when output_figure is set)
    "output_figure": "eval_rom/comparison.png",
    "y_threshold": 0.5,
    "u0_colorbar_limits": [10_000.0, 20_000.0],
    "test": {
        "num_samples": 5,
        "bc": "drained",
        "nx": 51,
        "ny": 51,
        "nz": 51,
        "x_range": (0.0, 1.0),
        "y_range": (0.0, 1.0),
        "z_range": (0.0, 1.0),
        "t_span": (0.0, 0.2),
        "eval_times": [0.0, 0.05, 0.20],
        "cv_value": 0.35,
        "gp_params": {
            "output_scale": 1000.0,
            "length_scales": 0.30,
        },
        "u0_ranges": [(10_000.0, 20_000.0)],
        "seed": 999,
    },
}


# =====================================================================
# Helpers
# =====================================================================


def enforce_drained_dirichlet_bc(u_batch: torch.Tensor) -> torch.Tensor:
    u_batch[:, 0, :] = 0.0
    u_batch[:, -1, :] = 0.0
    u_batch[:, :, 0] = 0.0
    u_batch[:, :, -1] = 0.0
    return u_batch


def load_normalization_stats(fields_path: Path) -> dict:
    with h5py.File(fields_path, "r") as f:
        sg = f["stats"]
        return {
            "u_mean": float(np.asarray(sg["u_mean"])),
            "u_std": float(np.asarray(sg["u_std"])),
            "cv_mean": float(np.asarray(sg["cv_mean"])),
            "cv_std": float(np.asarray(sg["cv_std"])),
        }


@torch.no_grad()
def predict_field(
    model: torch.nn.Module,
    u0: np.ndarray,
    cv: float,
    t_value: float,
    stats: dict,
    t_mean: float,
    t_std: float,
    Phi: np.ndarray,
    mean_field: np.ndarray,
    coeffs_mean: np.ndarray,
    coeffs_std: np.ndarray,
    nx: int,
    ny: int,
    nz: int,
    device: torch.device,
) -> np.ndarray:
    """Predict a full (nx, ny, nz) field for a single (u0, cv, t) input."""
    model.eval()

    u0_norm = ((u0.reshape(-1) - stats["u_mean"]) / stats["u_std"]).astype(np.float32)
    cv_norm = np.float32((cv - stats["cv_mean"]) / stats["cv_std"])
    t_norm = np.float32((t_value - t_mean) / t_std)

    u0_t = torch.as_tensor(u0_norm, dtype=torch.float32, device=device).unsqueeze(0)
    cv_t = torch.tensor([[cv_norm]], dtype=torch.float32, device=device)
    t_t = torch.tensor([[t_norm]], dtype=torch.float32, device=device)

    coeffs_norm = model(u0_t, cv_t, t_t).cpu().numpy().ravel()  # (n_modes,)

    # Denormalize coefficients back to physical scale.
    coeffs = coeffs_norm * coeffs_std + coeffs_mean

    field_flat = mean_field + Phi @ coeffs  # (N,)
    return field_flat.reshape(nx, ny, nz)


# =====================================================================
# Main
# =====================================================================


def main() -> None:
    cfg = CONFIG
    test_cfg: dict = cfg["test"]  # type: ignore[assignment]

    # --- Load training config and build model ---
    print("=" * 80)
    print("ROM Evaluation")
    print("=" * 80)

    train_config_path = Path(str(cfg["train_config_path"]))
    train_cfg = OmegaConf.load(train_config_path)
    model_cfg_dict = OmegaConf.to_container(train_cfg.model, resolve=True)
    assert isinstance(model_cfg_dict, dict)

    device_override = cfg.get("device")
    if device_override is not None:
        device = torch.device(str(device_override))
    else:
        desired = str(train_cfg.training.device)
        device = (
            torch.device(desired)
            if torch.cuda.is_available() and "cuda" in desired
            else torch.device("cpu")
        )
    print(f"Using device: {device}")

    # Infer u0_dim from grid config.
    nx = int(test_cfg["nx"])
    ny = int(test_cfg["ny"])
    nz = int(test_cfg["nz"])
    model_cfg_dict["u0_dim"] = nx * ny

    model = build_rom_model(model_cfg_dict)
    model.to(device)

    checkpoint_path = Path(str(cfg["checkpoint_path"]))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint: {checkpoint_path}")

    # --- Load POD basis ---
    basis_path = Path(str(cfg["basis_path"]))
    basis = np.load(basis_path)
    Phi = basis["Phi"]  # (N, n_modes)
    mean_field = basis["mean_field"]  # (N,)
    coeffs_mean = basis["coeffs_mean"]  # (n_modes,)
    coeffs_std = basis["coeffs_std"]  # (n_modes,)
    basis_times = basis["times"]  # (nt,)

    # Time normalization (must match data_loader_rom).
    t_mean = float(np.mean(basis_times))
    t_std = float(np.std(basis_times))
    if t_std == 0.0:
        t_std = 1.0

    # --- Load normalization stats ---
    fields_path = Path(str(cfg["fields_path"]))
    stats = load_normalization_stats(fields_path)

    # --- Generate test samples and evaluate ---
    x_range = tuple(test_cfg["x_range"])
    y_range = tuple(test_cfg["y_range"])
    z_range = tuple(test_cfg["z_range"])
    t_span = tuple(test_cfg["t_span"])
    eval_times = list(test_cfg["eval_times"])
    num_samples = int(test_cfg["num_samples"])
    bc = str(test_cfg.get("bc", "drained"))
    gp_params = dict(test_cfg["gp_params"])
    u0_ranges = list(test_cfg["u0_ranges"])
    seed = int(test_cfg["seed"])
    cv_value = float(test_cfg["cv_value"])

    print(f"\nEvaluating {num_samples} test sample(s) at times {eval_times}")

    all_sample_mses: List[float] = []
    all_time_results: List[List[dict]] = []
    all_rom_times: List[float] = []

    for i in range(num_samples):
        print(f"\n  [Sample {i + 1}/{num_samples}]")

        u0_batch = random_gaussian_pwp_batch(
            n_samples=1,
            nx=nx,
            ny=ny,
            x_range=x_range,
            y_range=y_range,
            gp_params=gp_params,
            u0_ranges=u0_ranges,
            seed=seed + i,
            device=device,
            dtype=torch.float32,
        )
        u0_batch = enforce_drained_dirichlet_bc(u0_batch)
        u0_np = u0_batch[0].detach().cpu().numpy()

        # Solve ground truth.
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
            bc=bc,
            dtype=torch.float32,
            device=device,
        )
        true_fields = solver_result["u"].squeeze(0).detach().cpu().numpy()  # (T, nx, ny, nz)

        se_total = 0.0
        count_total = 0
        time_results: List[dict] = []
        pred_fields_list: List[np.ndarray] = []
        mse_values: List[float] = []
        rom_time_total = 0.0

        for t_idx, t_val in enumerate(eval_times):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            pred = predict_field(
                model, u0_np, cv_value, t_val, stats, t_mean, t_std,
                Phi, mean_field, coeffs_mean, coeffs_std, nx, ny, nz, device,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - t0
            rom_time_total += elapsed

            true = true_fields[t_idx]
            diff = pred - true
            mse = float(np.mean(diff ** 2))
            max_abs = float(np.max(np.abs(diff)))
            true_norm = float(np.sqrt(np.sum(true ** 2)))
            rel_l2 = float(np.sqrt(np.sum(diff ** 2)) / max(true_norm, 1e-12))

            se_total += float(np.sum(diff ** 2))
            count_total += diff.size

            pred_fields_list.append(pred)
            mse_values.append(mse)
            time_results.append(
                {"t": t_val, "mse": mse, "max_abs": max_abs, "rel_l2": rel_l2}
            )
            print(
                f"    t={t_val:.3f}  MSE={mse:.6e}  "
                f"MaxAbs={max_abs:.6e}  RelL2={rel_l2:.6e}  "
                f"ROM time={elapsed*1000:.2f}ms"
            )

        sample_mse = se_total / max(count_total, 1)
        all_sample_mses.append(sample_mse)
        all_time_results.append(time_results)
        all_rom_times.append(rom_time_total)
        print(f"    Overall MSE = {sample_mse:.6e}  ROM total = {rom_time_total*1000:.2f}ms")

        # Save visualization
        output_figure = cfg.get("output_figure")
        if output_figure:
            output_path = Path(str(output_figure))
            if num_samples > 1:
                output_path = output_path.parent / f"sample_{i}_{output_path.name}"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            xs = np.linspace(x_range[0], x_range[1], nx, dtype=np.float32)
            ys = np.linspace(y_range[0], y_range[1], ny, dtype=np.float32)
            zs = np.linspace(z_range[0], z_range[1], nz, dtype=np.float32)
            true_fields_list = [true_fields[t_idx] for t_idx in range(len(eval_times))]
            plot_comparison(
                u0_np,
                pred_fields_list,
                true_fields_list,
                eval_times,
                xs,
                ys,
                zs,
                float(cfg.get("y_threshold", 0.5)),
                output_path,
                u0_color_limits=cfg.get("u0_colorbar_limits"),
                cv_value=cv_value,
                mse_norm_values=mse_values,
            )

    # --- Summary ---
    print("\n" + "=" * 80)
    print("Summary over test samples")
    print("=" * 80)
    mean_mse = float(np.mean(all_sample_mses))
    std_mse = float(np.std(all_sample_mses))
    total_rom_time = sum(all_rom_times)
    mean_rom_time_ms = 1000.0 * float(np.mean(all_rom_times))
    print(f"  Samples: {num_samples}")
    print(f"  Mean MSE: {mean_mse:.6e}")
    print(f"  Std MSE:  {std_mse:.6e}")
    print(f"  Total ROM prediction time: {total_rom_time*1000:.2f}ms")
    print(f"  Mean ROM time per sample: {mean_rom_time_ms:.2f}ms")
    print("Done.")


if __name__ == "__main__":
    main()
