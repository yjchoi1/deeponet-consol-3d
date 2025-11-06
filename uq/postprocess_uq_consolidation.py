from __future__ import annotations

import sys
from pathlib import Path

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import pandas as pd

from uq.simulate_uq_consolidation import UQ_CONFIG, CASE


def summarize_and_visualize(
    npz_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    *,
    save_csv: bool = True,
    make_plots: bool = True,
) -> dict:
    cfg = UQ_CONFIG
    npz_path = Path(npz_path) if npz_path is not None else Path(cfg["uv_timeseries_npz"])
    data = np.load(npz_path)

    Uv_all = data["Uv_all"]  # (num_samples, nt)
    eval_times = data["eval_times"]
    t50_all = data["t50_all"]
    Tv50_all = data["Tv50_all"]
    Cv_all = data["Cv_all"]
    H_DR = float(data["H_DR"]) if "H_DR" in data else float(cfg["H_DR"])  # fallback

    out_dir = Path(output_dir) if output_dir is not None else Path(cfg["output_dir"]) / "post"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Summary statistics for U_v(t)
    Uv_mean = np.nanmean(Uv_all, axis=0)
    Uv_std = np.nanstd(Uv_all, axis=0)
    Uv_p05 = np.nanpercentile(Uv_all, 5, axis=0)
    Uv_p50 = np.nanpercentile(Uv_all, 50, axis=0)
    Uv_p95 = np.nanpercentile(Uv_all, 95, axis=0)

    # Simple text summary for t50 and Tv50
    t50_mean = float(np.nanmean(t50_all))
    t50_std = float(np.nanstd(t50_all))
    t50_p05 = float(np.nanpercentile(t50_all, 5))
    t50_p50 = float(np.nanpercentile(t50_all, 50))
    t50_p95 = float(np.nanpercentile(t50_all, 95))

    Tv50_mean = float(np.nanmean(Tv50_all))
    Tv50_std = float(np.nanstd(Tv50_all))
    Tv50_p05 = float(np.nanpercentile(Tv50_all, 5))
    Tv50_p50 = float(np.nanpercentile(Tv50_all, 50))
    Tv50_p95 = float(np.nanpercentile(Tv50_all, 95))

    if save_csv:
        # Per-sample metrics
        sample_csv_path = out_dir / "sample_results.csv"
        df_samples = pd.DataFrame({
            "sample": np.arange(Cv_all.shape[0]),
            "Cv": Cv_all,
            "t50": t50_all,
            "Tv50": Tv50_all,
        })
        df_samples.to_csv(sample_csv_path, index=False)

        # Time-series summary
        summary_csv_path = out_dir / "summary_stats.csv"
        df_summary = pd.DataFrame({
            "t": eval_times.astype(float),
            "Uv_mean": Uv_mean,
            "Uv_std": Uv_std,
            "Uv_p05": Uv_p05,
            "Uv_p50": Uv_p50,
            "Uv_p95": Uv_p95,
        })
        df_summary.to_csv(summary_csv_path, index=False)

    if make_plots:
        # Plot U_v mean and 2-sigma band
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.plot(eval_times, Uv_mean, label="Mean", color="black", lw=2)
        ax.plot(eval_times, Uv_p50, label="Median", color="C0", lw=1.5, linestyle="--")
        lower = Uv_mean - 2.0 * Uv_std
        upper = Uv_mean + 2.0 * Uv_std
        ax.fill_between(eval_times, lower, upper, color="C0", alpha=0.15, label="mean ± 2σ")
        ax.set_xlabel("t")
        ax.set_ylabel("U_v(t)")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, ls=":", alpha=0.5)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / "Uv_percentiles.png", dpi=300)
        plt.close(fig)

        # Histogram of Tv50
        fig, ax = plt.subplots(figsize=(5, 3.5))
        valid = np.isfinite(Tv50_all)
        ax.hist(Tv50_all[valid], bins=30, color="C2", alpha=0.8)
        ax.set_xlabel("Tv_50")
        ax.set_ylabel("Count")
        ax.set_title("Distribution of Tv_50")
        ax.grid(True, ls=":", alpha=0.5)
        fig.tight_layout()
        fig.savefig(out_dir / "Tv50_hist.png", dpi=300)
        plt.close(fig)

    results = {
        "Uv_mean": Uv_mean,
        "Uv_std": Uv_std,
        "Uv_p05": Uv_p05,
        "Uv_p50": Uv_p50,
        "Uv_p95": Uv_p95,
        "t50_mean": t50_mean,
        "t50_std": t50_std,
        "t50_p05": t50_p05,
        "t50_p50": t50_p50,
        "t50_p95": t50_p95,
        "Tv50_mean": Tv50_mean,
        "Tv50_std": Tv50_std,
        "Tv50_p05": Tv50_p05,
        "Tv50_p50": Tv50_p50,
        "Tv50_p95": Tv50_p95,
        "output_dir": str(out_dir),
    }
    return results


if __name__ == "__main__":
    print("Summarizing and visualizing UQ results...")
    out = summarize_and_visualize()
    print(
        "t50 mean/std/p05/p50/p95 = ",
        f"{out['t50_mean']:.4g}",
        f"{out['t50_std']:.4g}",
        f"{out['t50_p05']:.4g}",
        f"{out['t50_p50']:.4g}",
        f"{out['t50_p95']:.4g}",
    )
    print("Outputs in:", out["output_dir"])


