from __future__ import annotations

import sys
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from solver.solver_batch import random_gaussian_pwp_batch, solve_terzaghi_3d_fdm_batch
from train.models import build_model


# ============================================================================
# HARDCODED EVALUATION CONFIGURATION
# ============================================================================
EVAL_CONFIG = {
    "train_config_path": "train/model/case3_vanilla_ff/config.yaml",
    "checkpoint_path": "train/model/case3_vanilla_ff/latest.pt",
    "normalization_data_path": "train/data/deeponet_terzaghi_train.h5",
    "nx": 51,
    "ny": 51,
    "nz": 51,
    "x_range": (0.0, 1.0),
    "y_range": (0.0, 1.0),
    "z_range": (0.0, 1.0),
    "eval_times": [0.0, 0.1, 0.3, 0.7, 1.5],
    "t_span": (0.0, 1.0),
    "cv_value": 0.1,
    "gp_params": {
        "output_scale": 1000.0,
        "length_scales": 0.15,
    },
    "u0_ranges": [(10000.0, 20000.0)],
    "seed": 999,
    "batch_size": 10000,
    "line_plot": {
        "output_figure": "eval/case3_vanilla_ff/line_comparison.png",
        "axes": {
            "x": {"fixed_y": 0.5, "fixed_z": 0.5},
            "y": {"fixed_x": 0.5, "fixed_z": 0.5},
            "z": {"fixed_x": 0.5, "fixed_y": 0.5},
        },
    },
}


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
    return all_coords, xs, ys, zs


def evaluate_deeponet(model, u0, cv, coords, stats, device, batch_size, flatten_branch=True):
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
    predictions = []

    with torch.no_grad():
        for i in range(0, n_points, batch_size):
            batch_coords = coords_tensor[i : i + batch_size]
            batch_cv = cv_scalar.expand(len(batch_coords), 1)
            batch_u = u_tensor.unsqueeze(0).expand(len(batch_coords), -1)
            output = model(batch_u, batch_cv, batch_coords)
            predictions.append(output.cpu())

    predictions = torch.cat(predictions, dim=0).numpy().ravel()
    predictions = predictions * stats["s_std"] + stats["s_mean"]
    return predictions


def select_index(value: float, grid: np.ndarray) -> int:
    return int(np.argmin(np.abs(grid - value)))


def plot_line_comparison(
    pred_fields,
    true_fields,
    eval_times,
    xs,
    ys,
    zs,
    line_cfg,
    cv_value: float,
):
    axes_cfg = line_cfg["axes"]
    axis_order = [axis for axis in ["x", "y", "z"] if axis in axes_cfg]
    if not axis_order:
        raise ValueError("line_plot.axes must contain at least one axis definition.")

    coords_norm = {"x": xs / H_DR, "y": ys / H_DR, "z": zs / H_DR}

    fig, axes = plt.subplots(
        1, len(axis_order), figsize=(5 * len(axis_order), 4), sharey=True
    )
    if len(axis_order) == 1:
        axes = [axes]

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(eval_times)))

    for ax_idx, axis_name in enumerate(axis_order):
        axis_cfg = axes_cfg[axis_name]
        axis = axes[ax_idx]

        if axis_name == "x":
            iy = select_index(axis_cfg["fixed_y"], ys)
            iz = select_index(axis_cfg["fixed_z"], zs)
            coord = coords_norm["x"]
            coord_label = r"$x / H_{dr}$"
        elif axis_name == "y":
            ix = select_index(axis_cfg["fixed_x"], xs)
            iz = select_index(axis_cfg["fixed_z"], zs)
            coord = coords_norm["y"]
            coord_label = r"$y / H_{dr}$"
        elif axis_name == "z":
            ix = select_index(axis_cfg["fixed_x"], xs)
            iy = select_index(axis_cfg["fixed_y"], ys)
            coord = coords_norm["z"]
            coord_label = r"$z / H_{dr}$"
        else:
            raise ValueError(f"Unsupported axis '{axis_name}'.")

        for time_idx, t in enumerate(eval_times):
            color = colors[time_idx]
            tv = (cv_value * t) / H_DR

            if axis_name == "x":
                pred_line = pred_fields[time_idx][:, iy, iz]
                true_line = true_fields[time_idx][:, iy, iz]
            elif axis_name == "y":
                pred_line = pred_fields[time_idx][ix, :, iz]
                true_line = true_fields[time_idx][ix, :, iz]
            else:  # axis_name == "z"
                pred_line = pred_fields[time_idx][ix, iy, :]
                true_line = true_fields[time_idx][ix, iy, :]

            axis.plot(
                coord,
                pred_line,
                color=color,
                linestyle="-",
                label=f"DeepONet (Tv={tv:.2f})",
            )
            axis.plot(
                coord,
                true_line,
                color=color,
                linestyle="--",
                label=f"Solver (Tv={tv:.2f})",
            )

        axis.set_xlabel(coord_label)
        axis.grid(True, linestyle="--", alpha=0.3)

        if axis_name == "x":
            y_fix = axis_cfg["fixed_y"] / H_DR
            z_fix = axis_cfg["fixed_z"] / H_DR
            axis.set_title(rf"Along $x$, $y/H_{{dr}}$={y_fix:.2f}, $z/H_{{dr}}$={z_fix:.2f}")
        elif axis_name == "y":
            x_fix = axis_cfg["fixed_x"] / H_DR
            z_fix = axis_cfg["fixed_z"] / H_DR
            axis.set_title(rf"Along $y$, $x/H_{{dr}}$={x_fix:.2f}, $z/H_{{dr}}$={z_fix:.2f}")
        else:
            x_fix = axis_cfg["fixed_x"] / H_DR
            y_fix = axis_cfg["fixed_y"] / H_DR
            axis.set_title(rf"Along $z$, $x/H_{{dr}}$={x_fix:.2f}, $y/H_{{dr}}$={y_fix:.2f}")

    axes[0].set_ylabel("Excess PWP (Pa)")

    # Consolidate legends into the last axis to avoid duplicates across subplots
    handles, labels = axes[-1].get_legend_handles_labels()
    dedup = dict(zip(labels, handles))
    axes[-1].legend(dedup.values(), dedup.keys(), fontsize=9, frameon=False)

    fig.tight_layout()

    output_path = Path(line_cfg["output_figure"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Line comparison figure saved to {output_path}")


def main():
    print("=" * 80)
    print("DeepONet Time-Specific Line Evaluation")
    print("=" * 80)

    cfg = EVAL_CONFIG

    train_config_path = Path(cfg["train_config_path"])
    train_cfg = OmegaConf.load(train_config_path)
    print(f"Loaded training config: {train_config_path}")

    device = torch.device(train_cfg.training.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    stats = load_normalization_stats(Path(cfg["normalization_data_path"]))
    print("Loaded normalization statistics")

    model_config = OmegaConf.to_container(train_cfg.model, resolve=True)
    model = build_model(model_config)
    model.to(device)
    checkpoint = torch.load(Path(cfg["checkpoint_path"]), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from {cfg['checkpoint_path']}")

    nx, ny, nz = cfg["nx"], cfg["ny"], cfg["nz"]
    x_range, y_range, z_range = cfg["x_range"], cfg["y_range"], cfg["z_range"]
    cv_value = cfg["cv_value"]

    u0_batch = random_gaussian_pwp_batch(
        n_samples=1,
        nx=nx,
        ny=ny,
        x_range=x_range,
        y_range=y_range,
        gp_params=cfg["gp_params"],
        u0_ranges=cfg["u0_ranges"],
        seed=cfg["seed"],
        device=device,
        dtype=torch.float32,
    )
    u0_batch = enforce_drained_dirichlet_bc(u0_batch)
    u0 = u0_batch[0].cpu().numpy()
    print("Generated initial condition")

    eval_times = cfg["eval_times"]
    solver_result = solve_terzaghi_3d_fdm_batch(
        Cv_batch=[cv_value],
        x_range=x_range,
        y_range=y_range,
        z_range=z_range,
        nx=nx,
        ny=ny,
        nz=nz,
        t_span=cfg["t_span"],
        u0_xy_batch=u0_batch,
        t_eval=eval_times,
        dtype=torch.float32,
        device=device,
    )
    solver_u = solver_result["u"].squeeze(0).cpu().numpy()
    print("Solver evaluation complete")

    coords, xs, ys, zs = create_query_points(
        eval_times, x_range, y_range, z_range, nx, ny, nz
    )

    predictions = evaluate_deeponet(
        model,
        u0,
        cv_value,
        coords,
        stats,
        device,
        batch_size=cfg["batch_size"],
        flatten_branch=bool(train_cfg.data.flatten_branch),
    )
    print("DeepONet evaluation complete")

    points_per_time = nx * ny * nz
    pred_fields = []
    true_fields = []
    for i in range(len(eval_times)):
        start = i * points_per_time
        end = start + points_per_time
        pred_fields.append(predictions[start:end].reshape(nx, ny, nz))
        true_fields.append(solver_u[i])

    line_cfg = cfg["line_plot"]
    plot_line_comparison(
        pred_fields,
        true_fields,
        eval_times,
        xs,
        ys,
        zs,
        line_cfg,
        cv_value=cv_value,
    )

    print("=" * 80)
    print("Time-specific line evaluation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()


