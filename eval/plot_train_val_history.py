#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib

# Set Times Roman font for journal-ready plots
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
matplotlib.rcParams["mathtext.fontset"] = "stix"

# Get workspace root (parent of eval directory)
WORKSPACE_ROOT = Path(__file__).parent.parent

# Configuration
CONFIG = {
    "logdir": WORKSPACE_ROOT / "train/model/case3_data_v2_vanilla_ff_scaling",
    "output": None,  # None means <logdir>/train_val_history.png
    "dpi": 300,
    "title": None,
}

try:
    # TensorBoard event reader
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except Exception as exc:
    raise SystemExit(
        "TensorBoard is required to parse event files.\n"
        "Install with: pip install tensorboard\n"
        f"Import error: {exc}"
    )


def load_scalar_series(log_dir: Path, tags: List[str]) -> Dict[str, List[Tuple[int, float]]]:
    """Load scalar series for given tags from a TensorBoard log directory."""
    if not log_dir.exists():
        raise FileNotFoundError(f"Log directory not found: {log_dir}")

    # Size guidance to load all scalars
    size_guidance = {
        "scalars": 0,
        "histograms": 0,
        "images": 0,
        "audio": 0,
        "tensors": 0,
    }
    accumulator = EventAccumulator(str(log_dir), size_guidance=size_guidance)
    accumulator.Reload()

    available = set(accumulator.Tags().get("scalars", []))
    series: Dict[str, List[Tuple[int, float]]] = {}
    for tag in tags:
        if tag not in available:
            series[tag] = []
            continue
        events = accumulator.Scalars(tag)
        points = [(int(evt.step), float(evt.value)) for evt in events]
        points.sort(key=lambda p: p[0])
        series[tag] = points
    return series


def plot_history(
    series: Dict[str, List[Tuple[int, float]]],
    output_path: Path,
    *,
    title: str | None = "Training and validation loss",
    dpi: int = 150,
) -> None:
    """Plot loss curves and save to file."""
    train_points = series.get("loss/train", [])
    val_points = series.get("loss/val", [])

    if not train_points and not val_points:
        raise RuntimeError(
            "No scalar data found for tags 'loss/train' or 'loss/val' in the provided log directory."
        )

    fig, ax = plt.subplots(figsize=(4.7, 3.5))

    if train_points:
        steps_t, values_t = zip(*train_points)
        ax.plot(steps_t, values_t, label="Train loss", color="C0", linewidth=2.0)
    if val_points:
        steps_v, values_v = zip(*val_points)
        ax.plot(steps_v, values_v, label="Validation loss", color="C1", linewidth=2.0)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    if title:
        ax.set_title(title)
    ax.grid(True, which="both", linestyle="-", alpha=0.3)
    ax.legend()
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def main() -> None:
    log_dir: Path = CONFIG["logdir"]
    output: Path | None = CONFIG["output"]
    if output is None:
        output_path = log_dir / "train_val_history.png"
    elif output.is_dir():
        output_path = output / "train_val_history.png"
    else:
        output_path = output

    tags = ["loss/train", "loss/val"]
    series = load_scalar_series(log_dir, tags)
    plot_history(series, output_path, title=CONFIG["title"], dpi=CONFIG["dpi"])


if __name__ == "__main__":
    main()


