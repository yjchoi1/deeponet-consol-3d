from __future__ import annotations

import sys
from pathlib import Path

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf

from solver.solver_batch import random_gaussian_pwp_batch, solve_terzaghi_3d_fdm_batch
from train.deeponet import build_model

# ============================================================================
# HARDCODED EVALUATION CONFIGURATION
# ============================================================================
EVAL_CONFIG = {
    # Model and checkpoint paths
    "train_config_path": "train/conf/config.yaml",
    "checkpoint_path": "train/checkpoints/latest.pt",
    "normalization_data_path": "train/data/deeponet_terzaghi_val.h5",
    
    # Grid parameters (should match training data generation)
    "nx": 41,
    "ny": 41,
    "nz": 41,
    "x_range": (0.0, 1.0),
    "y_range": (0.0, 1.0),
    "z_range": (0.0, 1.0),
    
    # Time points to evaluate
    "eval_times": [0.2, 0.5, 1.0],
    "t_span": (0.0, 1.0),
    
    # Test sample parameters
    "cv_value": 0.05,
    "gp_params": {
        "output_scale": 1000.0,
        "length_scales": 0.15,
    },
    "u0_ranges": [(10000.0, 20000.0)],
    "seed": 999,
    
    # Visualization parameters
    "y_threshold": 0.5,  # Show points where y < y_threshold
    "output_figure": "eval/comparison.png",
    
    # Inference parameters
    "batch_size": 10000,  # For batched inference
}


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
    """Evaluate DeepONet on query points with batching."""
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
    
    predictions = torch.cat(predictions, dim=0).numpy().ravel()
    
    # Denormalize
    predictions = predictions * stats["s_std"] + stats["s_mean"]
    
    return predictions


def plot_comparison(u0, pred_fields, true_fields, eval_times, xs, ys, zs, y_threshold, output_path):
    """Create 4-subplot comparison figure with 2D initial field and 3D scatter plots."""
    n_times = len(eval_times)
    fig = plt.figure(figsize=(16, 4 * n_times))
    
    for time_idx, t in enumerate(eval_times):
        # Extract field at this time
        pred = pred_fields[time_idx]
        true = true_fields[time_idx]
        error = np.abs(pred - true)
        
        # Create mask for y < y_threshold
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing='ij')
        mask = Y < y_threshold
        
        # Flatten and filter
        x_plot = X[mask]
        y_plot = Y[mask]
        z_plot = Z[mask]
        pred_plot = pred[mask]
        true_plot = true[mask]
        error_plot = error[mask]
        
        # Determine color limits
        vmin = min(pred_plot.min(), true_plot.min())
        vmax = max(pred_plot.max(), true_plot.max())
        
        base_idx = time_idx * 4
        
        # Subplot 1: Initial condition (2D heatmap)
        ax1 = fig.add_subplot(n_times, 4, base_idx + 1)
        im1 = ax1.imshow(u0.T, origin='lower', extent=[xs[0], xs[-1], ys[0], ys[-1]], 
                         cmap='viridis', aspect='auto')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_title(f't={t:.2f}: Initial u0 (x-y surface)')
        plt.colorbar(im1, ax=ax1, label='PWP')
        
        # Subplot 2: DeepONet prediction (3D scatter)
        ax2 = fig.add_subplot(n_times, 4, base_idx + 2, projection='3d')
        sc2 = ax2.scatter(x_plot, z_plot, y_plot, c=pred_plot, cmap='viridis', 
                         s=1, vmin=vmin, vmax=vmax)
        ax2.set_xlabel('X')
        ax2.set_ylabel('Z')
        ax2.set_zlabel('Y')
        ax2.set_title(f't={t:.2f}: DeepONet Prediction')
        plt.colorbar(sc2, ax=ax2, label='PWP', shrink=0.5)
        
        # Subplot 3: Solver solution (3D scatter)
        ax3 = fig.add_subplot(n_times, 4, base_idx + 3, projection='3d')
        sc3 = ax3.scatter(x_plot, z_plot, y_plot, c=true_plot, cmap='viridis', 
                         s=1, vmin=vmin, vmax=vmax)
        ax3.set_xlabel('X')
        ax3.set_ylabel('Z')
        ax3.set_zlabel('Y')
        ax3.set_title(f't={t:.2f}: Solver Solution')
        plt.colorbar(sc3, ax=ax3, label='PWP', shrink=0.5)
        
        # Subplot 4: Error (3D scatter)
        ax4 = fig.add_subplot(n_times, 4, base_idx + 4, projection='3d')
        sc4 = ax4.scatter(x_plot, z_plot, y_plot, c=error_plot, cmap='Reds', s=1)
        ax4.set_xlabel('X')
        ax4.set_ylabel('Z')
        ax4.set_zlabel('Y')
        ax4.set_title(f't={t:.2f}: Absolute Error')
        plt.colorbar(sc4, ax=ax4, label='|Error|', shrink=0.5)
    
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
    
    u0_batch = random_gaussian_pwp_batch(
        n_samples=1,
        nx=nx,
        ny=ny,
        x_range=x_range,
        y_range=y_range,
        gp_params=cfg["gp_params"],
        u0_ranges=cfg["u0_ranges"],
        seed=cfg["seed"],
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    u0_batch = enforce_drained_dirichlet_bc(u0_batch)
    u0 = u0_batch[0].numpy()
    print(f"    Generated initial condition: shape={u0.shape}, mean={u0.mean():.2f}, std={u0.std():.2f}")
    print(f"    Cv value: {cv_value}")
    
    # Run solver
    print("\n[6] Running solver...")
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
        device="cpu",
    )
    solver_u = solver_result["u"].squeeze(0).numpy()  # Remove batch dimension
    print(f"    Solver complete: solution shape={solver_u.shape}")
    
    # Create query points
    print("\n[7] Creating query points...")
    all_coords, xs, ys, zs = create_query_points(
        eval_times, x_range, y_range, z_range, nx, ny, nz
    )
    print(f"    Total query points: {len(all_coords)}")
    
    # Evaluate DeepONet
    print("\n[8] Evaluating DeepONet...")
    predictions = evaluate_deeponet(
        model, u0, cv_value, all_coords, stats, device, 
        batch_size=cfg["batch_size"],
        flatten_branch=bool(train_cfg.data.flatten_branch)
    )
    print(f"    DeepONet evaluation complete")
    
    # Reshape predictions to match solver output
    points_per_time = nx * ny * nz
    pred_fields = []
    true_fields = []
    
    for i, t in enumerate(eval_times):
        start_idx = i * points_per_time
        end_idx = start_idx + points_per_time
        pred_field = predictions[start_idx:end_idx].reshape(nx, ny, nz)
        true_field = solver_u[i]
        
        pred_fields.append(pred_field)
        true_fields.append(true_field)
        
        # Compute errors
        mse = np.mean((pred_field - true_field) ** 2)
        max_err = np.abs(pred_field - true_field).max()
        rel_err = np.linalg.norm(pred_field - true_field) / np.linalg.norm(true_field)
        
        print(f"\n    Time t={t:.2f}:")
        print(f"      MSE: {mse:.6e}")
        print(f"      Max absolute error: {max_err:.6e}")
        print(f"      Relative error (L2): {rel_err:.6e}")
    
    # Create visualization
    print("\n[9] Creating visualization...")
    output_path = Path(cfg["output_figure"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    plot_comparison(
        u0, pred_fields, true_fields, eval_times, 
        xs, ys, zs, cfg["y_threshold"], output_path
    )
    
    print("\n" + "=" * 80)
    print("Evaluation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

