from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import h5py
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch

# Add project root to Python path for imports
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from solver.solver_batch import solve_terzaghi_3d_fdm_batch


CONFIG: Dict[str, object] = {
    "dataset_path": Path("train/data/deeponet_terzaghi_train.h5"),
    "sample_index": 0,
    "time_index": 40,
    "volume_opacity": 0.12,
    "volume_surface_count": 18,
    "marker_size": 6,
    "colorscale": "Turbo",
    # Fixed color range for all colorbars (set None to auto)
    "color_min": 0,
    "color_max": 20000,
}


def load_sample(
    dataset_path: Path,
    sample_index: int,
) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray, Dict[str, object]]:
    """Load stored DeepONet sample components from the HDF5 dataset."""
    with h5py.File(dataset_path, "r") as h5_file:
        total_samples = h5_file["u"].shape[0]
        idx = sample_index % total_samples

        u0 = np.array(h5_file["u"][idx])
        cv_value = float(h5_file["Cv"][idx])
        sample_points = np.array(h5_file["y"][idx])
        sample_values = np.array(h5_file["s"][idx])

        config = json.loads(h5_file.attrs["config"])
        config["x_coords"] = np.array(h5_file.attrs["x_coords"])
        config["y_coords"] = np.array(h5_file.attrs["y_coords"])
        config["z_coords"] = np.array(h5_file.attrs["z_coords"])
        config["time_samples"] = np.array(h5_file.attrs["time_samples"])

    return u0, cv_value, sample_points, sample_values, config


def compute_full_field(
    u0: np.ndarray,
    cv_value: float,
    config: Dict[str, object],
) -> np.ndarray:
    """Re-simulate the 3D consolidation field for the stored initial surface."""
    torch_dtype = getattr(torch, config.get("torch_dtype", "float32"))

    time_samples = np.asarray(config["time_samples"], dtype=np.float32)
    t_span = (float(time_samples[0]), float(time_samples[-1]))

    u0_tensor = torch.as_tensor(u0, dtype=torch_dtype).unsqueeze(0)
    Cv_batch = torch.as_tensor([cv_value], dtype=torch_dtype)

    result = solve_terzaghi_3d_fdm_batch(
        Cv_batch=Cv_batch,
        x_range=tuple(config["x_range"]),
        y_range=tuple(config["y_range"]),
        z_range=tuple(config["z_range"]),
        nx=int(config["nx"]),
        ny=int(config["ny"]),
        nz=int(config["nz"]),
        t_span=t_span,
        u0_xy_batch=u0_tensor,
        t_eval=time_samples,
        dtype=torch_dtype,
    )

    return result["u"][0].cpu().numpy()


def build_figure(
    u0: np.ndarray,
    full_field: np.ndarray,
    sample_points: np.ndarray,
    sample_values: np.ndarray,
    config: Dict[str, object],
    viz_cfg: Dict[str, object],
    sample_index: int,
    time_index: int,
    cv_value: float,
) -> go.Figure:
    """Construct the Plotly figure with 2D surface, 3D volume, and sampled points."""
    x_coords = np.asarray(config["x_coords"], dtype=np.float32)
    y_coords = np.asarray(config["y_coords"], dtype=np.float32)
    z_coords = np.asarray(config["z_coords"], dtype=np.float32)
    time_samples = np.asarray(config["time_samples"], dtype=np.float32)

    nt = time_samples.shape[0]
    if time_index < 0:
        t_idx = max(0, nt + time_index)
    else:
        t_idx = min(time_index, nt - 1)

    time_value = float(time_samples[t_idx])
    field_at_time = full_field[t_idx]

    mask = np.isclose(sample_points[:, 0], time_value, rtol=1e-5, atol=1e-6)
    if not np.any(mask):
        nearest = int(np.argmin(np.abs(sample_points[:, 0] - time_value)))
        mask = np.zeros(sample_points.shape[0], dtype=bool)
        mask[nearest] = True

    points_at_time = sample_points[mask]
    values_at_time = sample_values[mask]

    grid_x, grid_y, grid_z = np.meshgrid(
        x_coords, y_coords, z_coords, indexing="ij"
    )

    colorscale = str(viz_cfg.get("colorscale", "Turbo"))
    volume_opacity = float(viz_cfg.get("volume_opacity", 0.12))
    surface_count = int(viz_cfg.get("volume_surface_count", 18))
    marker_size = int(viz_cfg.get("marker_size", 6))

    color_min_raw = viz_cfg.get("color_min")
    color_max_raw = viz_cfg.get("color_max")
    color_min = float(color_min_raw) if color_min_raw is not None else None
    color_max = float(color_max_raw) if color_max_raw is not None else None

    fig = make_subplots(
        rows=1,
        cols=3,
        specs=[[{"type": "heatmap"}, {"type": "scene"}, {"type": "scene"}]],
        column_titles=[
            "Initial Excess PWP u₀(x, y)",
            "PWP Field Volume",
            "Sampled Points",
        ],
    )

    heatmap_kwargs = dict(
        x=x_coords,
        y=y_coords,
        z=u0.T,
        colorscale=colorscale,
        colorbar=dict(title="u₀"),
    )
    if color_min is not None:
        heatmap_kwargs["zmin"] = color_min
    if color_max is not None:
        heatmap_kwargs["zmax"] = color_max

    fig.add_trace(
        go.Heatmap(**heatmap_kwargs),
        row=1,
        col=1,
    )

    volume_kwargs = dict(
        x=grid_x.ravel(),
        y=grid_y.ravel(),
        z=grid_z.ravel(),
        value=field_at_time.ravel(),
        opacity=volume_opacity,
        surface_count=surface_count,
        colorscale=colorscale,
        colorbar=dict(title="PWP"),
        caps=dict(x_show=False, y_show=False, z_show=False),
    )
    if color_min is not None:
        volume_kwargs["cmin"] = color_min
    if color_max is not None:
        volume_kwargs["cmax"] = color_max

    fig.add_trace(
        go.Volume(**volume_kwargs),
        row=1,
        col=2,
    )

    marker_kwargs = dict(
        color=values_at_time,
        colorscale=colorscale,
        size=marker_size,
        colorbar=dict(title="Sampled PWP"),
    )
    if color_min is not None:
        marker_kwargs["cmin"] = color_min
    if color_max is not None:
        marker_kwargs["cmax"] = color_max

    fig.add_trace(
        go.Scatter3d(
            x=points_at_time[:, 1],
            y=points_at_time[:, 2],
            z=points_at_time[:, 3],
            mode="markers",
            marker=marker_kwargs,
        ),
        row=1,
        col=3,
    )

    fig.update_xaxes(title="x", row=1, col=1)
    fig.update_yaxes(title="y", row=1, col=1)

    fig.update_layout(
        title=f"Sample {sample_index} at t = {time_value:.4f}, Cv = {cv_value:.4f}",
        scene=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="z",
            aspectmode="cube",
        ),
        scene2=dict(
            xaxis_title="x",
            yaxis_title="y",
            zaxis_title="z",
            aspectmode="cube",
        ),
    )

    return fig


def visualize_sample(cfg: Dict[str, object]) -> None:
    """Load a dataset sample, re-simulate the field, and render Plotly figure."""
    dataset_path = Path(cfg["dataset_path"])  # type: ignore[arg-type]
    sample_index = int(cfg["sample_index"])
    time_index = int(cfg["time_index"])

    u0, cv_value, points, values, data_cfg = load_sample(dataset_path, sample_index)

    data_cfg.setdefault("x_range", [float(data_cfg["x_coords"][0]), float(data_cfg["x_coords"][-1])])
    data_cfg.setdefault("y_range", [float(data_cfg["y_coords"][0]), float(data_cfg["y_coords"][-1])])
    data_cfg.setdefault("z_range", [float(data_cfg["z_coords"][0]), float(data_cfg["z_coords"][-1])])
    data_cfg.setdefault("nx", int(len(data_cfg["x_coords"])))
    data_cfg.setdefault("ny", int(len(data_cfg["y_coords"])))
    data_cfg.setdefault("nz", int(len(data_cfg["z_coords"])))

    full_field = compute_full_field(u0, cv_value, data_cfg)
    fig = build_figure(u0, full_field, points, values, data_cfg, cfg, sample_index, time_index, cv_value)
    fig.write_html(f"sample_{sample_index}_time_{time_index}.html")


def main() -> None:
    visualize_sample(CONFIG)


if __name__ == "__main__":
    main()
