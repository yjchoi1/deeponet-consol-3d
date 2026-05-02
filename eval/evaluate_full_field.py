from __future__ import annotations

import sys
from pathlib import Path

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import h5py
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
import torch
from omegaconf import OmegaConf

from solver.solver_batch import random_gaussian_pwp_batch, solve_terzaghi_3d_fdm_batch
from train.models import build_model

# ============================================================================
# HARDCODED EVALUATION CONFIGURATION
# ============================================================================

# For data_v2
case = "case3_data_v2_vanilla_ff_scaling"
EVAL_CONFIG = {
    # Model and checkpoint paths
    "train_config_path": f"train/model/{case}/config.yaml",
    "checkpoint_path": f"train/model/{case}/latest.pt",
    "normalization_data_path": "data/train.h5",
    # Boundary condition for u used by the reference solver (ground truth generation).
    # Options:
    #   - "drained": Dirichlet u=0 on all six faces (fully drained)
    #   - "drained_xy_top_nodrain_bottom": Drained on x/y faces and top z face,
    #     no-drain (Neumann du/dz=0) on bottom z face
    "bc": "drained",
    
    # Grid parameters (should match training data generation)
    "nx": 51,
    "ny": 51,
    "nz": 51,
    "x_range": (0.0, 1.0),
    "y_range": (0.0, 1.0),
    "z_range": (0.0, 1.0),
    
    # Time points to evaluate
    "eval_times": [0.0, 0.05, 0.20],
    "t_span": (0.0, 0.20),  
    # Spacetime MSE evaluation configuration
    "compute_spacetime_mse": True,
    "nt_mse": 51,  # number of time points between t_span[0] and t_span[1]
    
    # Test sample parameters
    "cv_value": 0.5,
    "gp_params": {
        "output_scale": 1000.0,
        "length_scales": 0.30,
    },
    "u0_ranges": [(10000.0, 20000.0)],
    "seed": 999,
    
    # Visualization parameters
    "y_threshold": 0.5,  # Show points where y < y_threshold
    "output_figure": f"eval/{case}/comparison.png",
    "u0_colorbar_limits": [15000.0, 20000.0],
    
    # Inference parameters
    "batch_size": 51*51*51*3,  # For batched inference
}


# For data_v1
# case = "case3_vanilla_ff"
# EVAL_CONFIG = {
#     # Model and checkpoint paths
#     "train_config_path": f"train/model/{case}/config.yaml",
#     "checkpoint_path": f"train/model/{case}/latest.pt",
#     "normalization_data_path": "train/data/deeponet_terzaghi_train.h5",
    
#     # Grid parameters (should match training data generation)
#     "nx": 51,
#     "ny": 51,
#     "nz": 51,
#     "x_range": (0.0, 1.0),
#     "y_range": (0.0, 1.0),
#     "z_range": (0.0, 1.0),
    
#     # Time points to evaluate
#     "eval_times": [0.0, 0.2, 1.0],
#     "t_span": (0.0, 1.0),
#     # Spacetime MSE evaluation configuration
#     "compute_spacetime_mse": True,
#     "nt_mse": 51,  # number of time points between t_span[0] and t_span[1]
    
#     # Test sample parameters
#     "cv_value": 0.10,
#     "gp_params": {
#         "output_scale": 1000.0,
#         "length_scales": 0.15,
#     },
#     "u0_ranges": [(10000.0, 20000.0)],
#     "seed": 999,
    
#     # Visualization parameters
#     "y_threshold": 0.5,  # Show points where y < y_threshold
#     "output_figure": f"eval/{case}/comparison_cv_0.10.png",
#     "u0_colorbar_limits": [15000.0, 20000.0],
    
#     # Inference parameters
#     "batch_size": 51*51*51*3,  # For batched inference
# }

H_DR = 0.5


def load_normalization_stats(data_path: Path) -> dict:
    """Load normalization statistics from HDF5 dataset."""
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
    """Clamp horizontal surfaces to zero along the drained boundaries."""
    if u_batch.ndim != 3:
        raise ValueError("u_batch must have shape (batch, nx, ny)")
    u_batch[:, 0, :] = 0.0
    u_batch[:, -1, :] = 0.0
    u_batch[:, :, 0] = 0.0
    u_batch[:, :, -1] = 0.0
    return u_batch


def create_query_points(eval_times, x_range, y_range, z_range, nx, ny, nz):
    """Create meshgrid of query points for evaluation."""
    xs = np.linspace(x_range[0], x_range[1], nx, dtype=np.float32)
    ys = np.linspace(y_range[0], y_range[1], ny, dtype=np.float32)
    zs = np.linspace(z_range[0], z_range[1], nz, dtype=np.float32)
    
    all_coords = []
    all_times_list = []
    
    for t in eval_times:
        T, X, Y, Z = np.meshgrid(
            [t], xs, ys, zs, indexing='ij'
        )
        coords = np.stack([T.ravel(), X.ravel(), Y.ravel(), Z.ravel()], axis=1)
        all_coords.append(coords)
        all_times_list.append(np.full(len(coords), t, dtype=np.float32))
    
    all_coords = np.concatenate(all_coords, axis=0)
    all_times = np.concatenate(all_times_list, axis=0)
    
    return all_coords, xs, ys, zs


def evaluate_deeponet(model, u0, cv, coords, stats, device, batch_size, flatten_branch=True):
    """Evaluate DeepONet on query points with batching.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Tuple of (predictions_physical, predictions_normalized).
    """
    model.eval()
    
    # Normalize inputs
    u_norm = (u0 - stats["u_mean"]) / stats["u_std"]
    if flatten_branch:
        u_norm = u_norm.reshape(-1)
    
    cv_norm = (cv - stats["cv_mean"]) / stats["cv_std"]
    coords_norm = (coords - stats["coord_mean"]) / stats["coord_std"]
    
    # Convert to tensors
    u_tensor = torch.as_tensor(u_norm, dtype=torch.float32, device=device)
    cv_scalar = torch.tensor([cv_norm], dtype=torch.float32, device=device)
    coords_tensor = torch.as_tensor(coords_norm, dtype=torch.float32, device=device)

    # Batched inference
    n_points = coords_tensor.shape[0]
    predictions = []
    
    with torch.no_grad():
        for i in range(0, n_points, batch_size):
            batch_coords = coords_tensor[i:i + batch_size]
            batch_cv = cv_scalar.expand(len(batch_coords), 1)
            batch_u = u_tensor.unsqueeze(0).expand(len(batch_coords), -1)
            
            output = model(batch_u, batch_cv, batch_coords)
            predictions.append(output.cpu())

    predictions_tensor = torch.cat(predictions, dim=0)
    predictions_normalized = predictions_tensor.numpy().ravel()

    # Denormalize for physical-space outputs
    predictions_physical = predictions_normalized * stats["s_std"] + stats["s_mean"]

    return predictions_physical, predictions_normalized


def plot_comparison(
    u0,
    pred_fields,
    true_fields,
    eval_times,
    xs,
    ys,
    zs,
    y_threshold,
    output_path,
    u0_color_limits=None,
    cv_value=None,
    mse_norm_values=None,
):
    """Create comparison figure with initial field and 3D scatter plots."""
    if cv_value is None:
        raise ValueError("cv_value must be provided for computing Tv.")
    if mse_norm_values is None or len(mse_norm_values) != len(eval_times):
        raise ValueError("mse_norm_values must be provided with one entry per evaluation time.")

    n_times = len(eval_times)
    total_rows = n_times + 1
    total_cols = 3
    fig = plt.figure(figsize=(12, 3 * total_rows))

    pred_stack = np.stack(pred_fields, axis=0)
    true_stack = np.stack(true_fields, axis=0)
    error_stack = np.abs(pred_stack - true_stack)

    sol_vmin = min(pred_stack.min(), true_stack.min())
    sol_vmax = max(pred_stack.max(), true_stack.max())
    err_vmin = error_stack.min()
    err_vmax = error_stack.max()

    if u0_color_limits is not None:
        u0_vmin, u0_vmax = u0_color_limits
    else:
        u0_vmin = u0.min()
        u0_vmax = u0.max()

    xs_norm = xs / H_DR
    ys_norm = ys / H_DR
    zs_norm = zs / H_DR

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
    mask = Y >= y_threshold
    if not np.any(mask):
        raise ValueError("Mask for y >= y_threshold produced no points to plot.")

    X_norm = X / H_DR
    Y_norm = Y / H_DR
    Z_norm = Z / H_DR

    x_plot = X_norm[mask]
    y_plot = Y_norm[mask]
    z_plot = Z_norm[mask]

    # Centered initial condition on first row
    left_ax = fig.add_subplot(total_rows, total_cols, 1)
    left_ax.axis('off')
    ax_u0 = fig.add_subplot(total_rows, total_cols, 2)
    right_ax = fig.add_subplot(total_rows, total_cols, 3)
    right_ax.axis('off')

    im1 = ax_u0.imshow(
        u0.T,
        origin='lower',
        extent=[xs_norm[0], xs_norm[-1], ys_norm[0], ys_norm[-1]],
        cmap='viridis',
        aspect='equal',
        vmin=u0_vmin,
        vmax=u0_vmax,
    )
    ax_u0.set_xlabel(r'$x / H_{dr}$')
    ax_u0.set_ylabel(r'$y / H_{dr}$')
    ax_u0.set_title('Initial Excess PWP (Pa)')
    plt.colorbar(im1, ax=ax_u0, extend='both')

    for time_idx, t in enumerate(eval_times):
        pred = pred_fields[time_idx]
        true = true_fields[time_idx]
        error = np.abs(pred - true)
        mse_norm_val = float(mse_norm_values[time_idx])

        pred_plot = pred[mask]
        true_plot = true[mask]
        error_plot = error[mask]

        pred_min = float(pred_plot.min())
        pred_max = float(pred_plot.max())
        true_min = float(true_plot.min())
        true_max = float(true_plot.max())
        err_min = float(error_plot.min())
        err_max = float(error_plot.max())

        if pred_min == pred_max:
            pred_max = pred_min + 1e-6
        if true_min == true_max:
            true_max = true_min + 1e-6
        if err_min == err_max:
            err_max = err_min + 1e-6

        Tv = (cv_value * t) / H_DR**2

        base_idx = 3 * (time_idx + 1) + 1

        ax2 = fig.add_subplot(total_rows, total_cols, base_idx, projection='3d')
        sc2 = ax2.scatter(
            x_plot,
            y_plot,
            z_plot,
            c=true_plot,
            cmap='viridis',
            s=1,
            vmin=sol_vmin,
            vmax=sol_vmax,
        )
        ax2.set_xlabel(r'$x / H_{dr}$')
        ax2.set_ylabel(r'$y / H_{dr}$')
        ax2.set_zlabel(r'$z / H_{dr}$')
        ax2.set_title(rf'True Solution (Pa), Tv = {Tv:.2f}')
        ax2.set_xlim(xs_norm[0], xs_norm[-1])
        ax2.set_ylim(ys_norm[0], ys_norm[-1])
        ax2.set_zlim(zs_norm[0], zs_norm[-1])
        cbar2 = plt.colorbar(
            sc2,
            ax=ax2,
            shrink=0.5,
            pad=0.2,
            extend='both',
            boundaries=np.linspace(true_min, true_max, 256),
        )
        cbar2.locator = ticker.MaxNLocator(4)
        cbar2.update_ticks()

        ax3 = fig.add_subplot(total_rows, total_cols, base_idx + 1, projection='3d')
        sc3 = ax3.scatter(
            x_plot,
            y_plot,
            z_plot,
            c=pred_plot,
            cmap='viridis',
            s=1,
            vmin=sol_vmin,
            vmax=sol_vmax,
        )
        ax3.set_xlabel(r'$x / H_{dr}$')
        ax3.set_ylabel(r'$y / H_{dr}$')
        ax3.set_zlabel(r'$z / H_{dr}$')
        ax3.set_title(f'Prediction (Pa)\nMSE: {mse_norm_val:.2e}')
        ax3.set_xlim(xs_norm[0], xs_norm[-1])
        ax3.set_ylim(ys_norm[0], ys_norm[-1])
        ax3.set_zlim(zs_norm[0], zs_norm[-1])
        cbar3 = plt.colorbar(
            sc3,
            ax=ax3,
            shrink=0.5,
            pad=0.2,
            extend='both',
            boundaries=np.linspace(pred_min, pred_max, 256),
        )
        cbar3.locator = ticker.MaxNLocator(4)
        cbar3.update_ticks()

        ax4 = fig.add_subplot(total_rows, total_cols, base_idx + 2, projection='3d')
        sc4 = ax4.scatter(
            x_plot,
            y_plot,
            z_plot,
            c=error_plot,
            cmap='Reds',
            s=1,
            vmin=err_vmin,
            vmax=err_vmax,
        )
        ax4.set_xlabel(r'$x / H_{dr}$')
        ax4.set_ylabel(r'$y / H_{dr}$')
        ax4.set_zlabel(r'$z / H_{dr}$')
        ax4.set_title(f'Absolute error (Pa)\nMax: ${err_max:.2e} \\mathrm{{Pa}}$')
        ax4.set_xlim(xs_norm[0], xs_norm[-1])
        ax4.set_ylim(ys_norm[0], ys_norm[-1])
        ax4.set_zlim(zs_norm[0], zs_norm[-1])
        cbar4 = plt.colorbar(
            sc4,
            ax=ax4,
            shrink=0.5,
            pad=0.2,
            extend='both',
            boundaries=np.linspace(err_min, err_max, 256),
        )
        cbar4.locator = ticker.MaxNLocator(4)
        cbar4.update_ticks()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Figure saved to {output_path}")


def main():
    print("=" * 80)
    print("DeepONet Evaluation")
    print("=" * 80)
    
    cfg = EVAL_CONFIG
    
    # Load training config
    print("\n[1] Loading training configuration...")
    train_config_path = Path(cfg["train_config_path"])
    train_cfg = OmegaConf.load(train_config_path)
    print(f"    Loaded from: {train_config_path}")
    
    # Determine device
    device = torch.device(train_cfg.training.device if torch.cuda.is_available() else "cpu")
    print(f"    Using device: {device}")
    
    # Load normalization statistics
    print("\n[2] Loading normalization statistics...")
    norm_data_path = Path(cfg["normalization_data_path"])
    stats = load_normalization_stats(norm_data_path)
    print(f"    Loaded from: {norm_data_path}")
    
    # Build model
    print("\n[3] Building model...")
    model_config = OmegaConf.to_container(train_cfg.model, resolve=True)
    model = build_model(model_config)
    model.to(device)
    print(f"    Model built with architecture from config")
    
    # Load checkpoint
    print("\n[4] Loading checkpoint...")
    checkpoint_path = Path(cfg["checkpoint_path"])
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    epoch = checkpoint.get("epoch", "unknown")
    print(f"    Loaded from: {checkpoint_path}")
    print(f"    Checkpoint epoch: {epoch}")
    
    # Generate test sample
    print("\n[5] Generating test sample...")
    nx, ny, nz = cfg["nx"], cfg["ny"], cfg["nz"]
    x_range, y_range, z_range = cfg["x_range"], cfg["y_range"], cfg["z_range"]
    cv_value = cfg["cv_value"]
    bc = str(cfg.get("bc", "drained"))
    
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
    print(f"    Generated initial condition: shape={u0.shape}, mean={u0.mean():.2f}, std={u0.std():.2f}")
    print(f"    Cv value: {cv_value}")
    
    # Run solver
    print("\n[6] Running solver...")
    print(f"    Using device: {device}")
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
        bc=bc,
        dtype=torch.float32,
        device=device,
    )
    solver_u = solver_result["u"].squeeze(0).cpu().numpy()  # Remove batch dimension and move to CPU
    print(f"    Solver complete: solution shape={solver_u.shape}")
    
    # Create query points
    print("\n[7] Creating query points...")
    all_coords, xs, ys, zs = create_query_points(
        eval_times, x_range, y_range, z_range, nx, ny, nz
    )
    print(f"    Total query points: {len(all_coords)}")
    
    # Evaluate DeepONet
    print("\n[8] Evaluating DeepONet...")
    print(f"    Using device: {device}")
    predictions, predictions_normalized = evaluate_deeponet(
        model, u0, cv_value, all_coords, stats, device, 
        batch_size=cfg["batch_size"],
        flatten_branch=bool(train_cfg.data.flatten_branch)
    )
    print(f"    DeepONet evaluation complete")
    
    # Reshape predictions to match solver output
    points_per_time = nx * ny * nz
    pred_fields = []
    true_fields = []
    mse_norm_values = []
    
    for i, t in enumerate(eval_times):
        start_idx = i * points_per_time
        end_idx = start_idx + points_per_time
        pred_field = predictions[start_idx:end_idx].reshape(nx, ny, nz)
        pred_field_norm = predictions_normalized[start_idx:end_idx].reshape(nx, ny, nz)
        true_field = solver_u[i]
        true_field_norm = (true_field - stats["s_mean"]) / stats["s_std"]
        
        pred_fields.append(pred_field)
        true_fields.append(true_field)
        
        # Compute errors
        max_err = np.abs(pred_field - true_field).max()
        rel_err = np.linalg.norm(pred_field - true_field) / np.linalg.norm(true_field)
        mse_norm = np.mean((pred_field_norm - true_field_norm) ** 2)
        mse_norm_values.append(float(mse_norm))
        
        print(f"\n    Time t={t:.2f}:")
        print(f"      MSE: {mse_norm:.6e}")
        print(f"      Max absolute error: {max_err:.6e}")
        print(f"      Relative error (L2): {rel_err:.6e}")

    # Optional: compute spacetime MSE over dense temporal grid (nt_mse points)
    if cfg.get("compute_spacetime_mse", False):
        print("\n[8b] Computing spacetime MSE on dense temporal grid...")
        t0, t1 = cfg["t_span"]
        nt_mse = int(cfg.get("nt_mse", 51))
        eval_times_dense = np.linspace(t0, t1, nt_mse, dtype=np.float32)

        # Run solver once for all dense time points
        solver_result_dense = solve_terzaghi_3d_fdm_batch(
            Cv_batch=[cv_value],
            x_range=x_range,
            y_range=y_range,
            z_range=z_range,
            nx=nx,
            ny=ny,
            nz=nz,
            t_span=cfg["t_span"],
            u0_xy_batch=u0_batch,
            t_eval=eval_times_dense.tolist(),
            bc=bc,
            dtype=torch.float32,
            device=device,
        )
        solver_u_dense = solver_result_dense["u"].squeeze(0).cpu().numpy()  # (nt, nx, ny, nz)

        # Accumulate errors without storing all predictions at once
        se_norm_total = 0.0
        se_phys_total = 0.0
        true_phys_sq_total = 0.0
        count_total = 0

        for i, t in enumerate(eval_times_dense):
            coords_t, _, _, _ = create_query_points([float(t)], x_range, y_range, z_range, nx, ny, nz)
            pred_phys_t, pred_norm_t = evaluate_deeponet(
                model,
                u0,
                cv_value,
                coords_t,
                stats,
                device,
                batch_size=cfg["batch_size"],
                flatten_branch=bool(train_cfg.data.flatten_branch),
            )

            true_phys_t = solver_u_dense[i]
            true_norm_t = (true_phys_t - stats["s_mean"]) / stats["s_std"]

            diff_norm = pred_norm_t - true_norm_t.ravel()
            diff_phys = pred_phys_t - true_phys_t.ravel()

            se_norm_total += float(np.dot(diff_norm, diff_norm))
            se_phys_total += float(np.dot(diff_phys, diff_phys))
            true_phys_sq_total += float(np.dot(true_phys_t.ravel(), true_phys_t.ravel()))
            count_total += diff_norm.size

        mse_norm_spacetime = se_norm_total / max(count_total, 1)
        mse_phys_spacetime = se_phys_total / max(count_total, 1)
        rel_l2_spacetime = (se_phys_total ** 0.5) / (true_phys_sq_total ** 0.5) if true_phys_sq_total > 0 else float("nan")

        print(f"    Spacetime MSE (normalized): {mse_norm_spacetime:.6e}")
        print(f"    Spacetime MSE (physical units Pa^2): {mse_phys_spacetime:.6e}")
        print(f"    Spacetime Relative L2 (physical): {rel_l2_spacetime:.6e}")
    
    # Create visualization
    print("\n[9] Creating visualization...")
    output_path = Path(cfg["output_figure"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    plot_comparison(
        u0, pred_fields, true_fields, eval_times, 
        xs, ys, zs, cfg["y_threshold"], output_path,
        u0_color_limits=cfg.get("u0_colorbar_limits"),
        cv_value=cv_value,
        mse_norm_values=mse_norm_values,
    )
    
    print("\n" + "=" * 80)
    print("Evaluation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

