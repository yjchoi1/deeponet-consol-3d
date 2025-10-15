from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from solver_batch import solve_terzaghi_3d_fdm_batch


CONFIG = {
    "Cv": 1.0,
    "x_range": (0.0, 1.0),
    "y_range": (0.0, 1.0),
    "z_range": (0.0, 1.0),
    "nx": 41,
    "ny": 41,
    "nz": 41,
    "t_span": (0.0, 1.0),
    "n_times": 60,
    "batch_size": 4,
    "gp_output_scale": 500.0,
    "gp_length_scale": 0.2,
    "u0_range": (10000, 20000),
    "seed": 42,
    "sample_indices": [0, 1, 2, 3],
    "time_index": -1,
    "y_section": 0.25,
    "keep_positive_side": True,
    "scatter_stride": 2,
    "scatter_size": 20.0,
    "scatter_alpha": 1.0,
    "cmap": "turbo",
    "elev": 20.0,
    "azim": -60.0,
    "output_path": Path("solver/test/xz_cut_visualization_batch.png"),
    "save_solution": None,
}


def main() -> None:
    cfg = CONFIG
    t_eval = np.linspace(cfg["t_span"][0], cfg["t_span"][1], cfg["n_times"])

    result = solve_terzaghi_3d_fdm_batch(
        Cv=cfg["Cv"],
        x_range=cfg["x_range"],
        y_range=cfg["y_range"],
        z_range=cfg["z_range"],
        nx=cfg["nx"],
        ny=cfg["ny"],
        nz=cfg["nz"],
        t_span=cfg["t_span"],
        batch_size=cfg["batch_size"],
        gp_params={
            "output_scale": cfg["gp_output_scale"],
            "length_scales": cfg["gp_length_scale"],
        },
        u0_ranges=[cfg["u0_range"]],
        seed=cfg["seed"],
        t_eval=t_eval,
    )

    times = result["t"].cpu().numpy()

    if cfg["save_solution"]:
        payload = {
            "u": result["u"].cpu().numpy(),
            "t": times,
            "x": np.linspace(cfg["x_range"][0], cfg["x_range"][1], cfg["nx"]),
            "y": np.linspace(cfg["y_range"][0], cfg["y_range"][1], cfg["ny"]),
            "z": np.linspace(cfg["z_range"][0], cfg["z_range"][1], cfg["nz"]),
        }
        np.savez(cfg["save_solution"], **payload)

    xs = np.linspace(cfg["x_range"][0], cfg["x_range"][1], cfg["nx"])
    ys = np.linspace(cfg["y_range"][0], cfg["y_range"][1], cfg["ny"])
    zs = np.linspace(cfg["z_range"][0], cfg["z_range"][1], cfg["nz"])

    tidx = cfg["time_index"]
    if tidx < 0:
        tidx = len(times) + tidx

    y_section = cfg["y_section"]
    y_idx = np.argmin(np.abs(ys - y_section))
    y_value = ys[y_idx]

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    Xf = X.reshape(-1)
    Yf = Y.reshape(-1)
    Zf = Z.reshape(-1)

    if cfg["keep_positive_side"]:
        slice_mask = Yf >= y_value
    else:
        slice_mask = Yf <= y_value

    step = cfg["scatter_stride"]
    Xf = Xf[slice_mask][::step]
    Yf = Yf[slice_mask][::step]
    Zf = Zf[slice_mask][::step]

    sample_indices = [int(i) for i in cfg["sample_indices"]]

    for sample_idx in sample_indices:
        sample_idx = int(np.clip(sample_idx, 0, cfg["batch_size"] - 1))
        field = result["u"][sample_idx, tidx].cpu().numpy()
        Vf = field.reshape(-1)[slice_mask][::step]

        norm = plt.Normalize(vmin=field.min(), vmax=field.max())
        cmap = plt.get_cmap(cfg["cmap"])

        fig = plt.figure(figsize=(8, 6), constrained_layout=True)
        ax = fig.add_subplot(111, projection="3d")

        ax.scatter(
            Xf,
            Yf,
            Zf,
            c=Vf,
            cmap=cmap,
            norm=norm,
            s=cfg["scatter_size"],
            alpha=cfg["scatter_alpha"],
            depthshade=False,
        )

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_xlim(xs[0], xs[-1])
        ax.set_ylim(ys[0], ys[-1])
        ax.set_zlim(zs[0], zs[-1])
        ax.view_init(elev=cfg["elev"], azim=cfg["azim"])
        ax.set_title(
            f"Batch sample {sample_idx} at t={times[tidx]:.4f} (y = {y_value:.3f})"
        )

        fig.colorbar(
            plt.cm.ScalarMappable(norm=norm, cmap=cmap),
            ax=ax,
            shrink=0.75,
            aspect=30,
            pad=0.1,
            label="Excess pore-water pressure",
        )

        if cfg["output_path"]:
            output_path = cfg["output_path"]
            if output_path.suffix:
                out_file = output_path.with_name(
                    f"{output_path.stem}_sample{sample_idx}{output_path.suffix}"
                )
            else:
                out_file = output_path / f"sample_{sample_idx}.png"
            out_file.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_file, dpi=300)
            plt.close(fig)
        else:
            plt.show()


if __name__ == "__main__":
    main()
