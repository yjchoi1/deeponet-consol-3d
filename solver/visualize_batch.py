from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from solver_batch import random_gaussian_pwp_batch, solve_terzaghi_3d_fdm_batch


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
    "gp_length_scale_xy": 0.2,
    "u0_range": (10000, 20000),
    "seed": 42,
    "sample_indices": [0, 1, 2, 3],
    "time_index": 0,
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

    u0_xy_batch = random_gaussian_pwp_batch(
        cfg["batch_size"],
        cfg["nx"],
        cfg["ny"],
        cfg["x_range"],
        cfg["y_range"],
        {
            "output_scale": cfg["gp_output_scale"],
            "length_scales": cfg["gp_length_scale_xy"],
        },
        [cfg["u0_range"]],
        seed=cfg["seed"],
    )

    result = solve_terzaghi_3d_fdm_batch(
        Cv=cfg["Cv"],
        x_range=cfg["x_range"],
        y_range=cfg["y_range"],
        z_range=cfg["z_range"],
        nx=cfg["nx"],
        ny=cfg["ny"],
        nz=cfg["nz"],
        t_span=cfg["t_span"],
        u0_xy_batch=u0_xy_batch,
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

    idx = np.flatnonzero(slice_mask)[:: cfg["scatter_stride"]]
    Xs = Xf[idx]
    Ys = Yf[idx]
    Zs = Zf[idx]

    cmap = plt.get_cmap(cfg["cmap"])

    sample_indices = [int(i) for i in cfg["sample_indices"]]

    for sample_idx in sample_indices:
        sample_idx = int(np.clip(sample_idx, 0, cfg["batch_size"] - 1))
        plane = u0_xy_batch[sample_idx].cpu().numpy()
        init_field = result["u"][sample_idx, 0].cpu().numpy()
        field = result["u"][sample_idx, tidx].cpu().numpy()

        fig = plt.figure(figsize=(18, 6), constrained_layout=True)
        ax_plane = fig.add_subplot(131)
        ax_init = fig.add_subplot(132, projection="3d")
        ax_final = fig.add_subplot(133, projection="3d")

        plane_norm = plt.Normalize(vmin=plane.min(), vmax=plane.max())
        ax_plane.imshow(
            plane.T,
            origin="lower",
            extent=[xs[0], xs[-1], ys[0], ys[-1]],
            aspect="auto",
            cmap=cmap,
            norm=plane_norm,
        )
        ax_plane.set_title(f"Initial excess PWP (sample {sample_idx})")
        ax_plane.set_xlabel("x")
        ax_plane.set_ylabel("y")
        fig.colorbar(
            plt.cm.ScalarMappable(norm=plane_norm, cmap=cmap),
            ax=ax_plane,
            shrink=0.8,
            pad=0.08,
            label="Excess pore-water pressure",
        )

        def scatter_field(ax, values: np.ndarray, title: str) -> None:
            data = values.reshape(-1)[idx]
            norm = plt.Normalize(vmin=values.min(), vmax=values.max())
            ax.scatter(
                Xs,
                Ys,
                Zs,
                c=data,
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
            ax.set_title(title)
            fig.colorbar(
                plt.cm.ScalarMappable(norm=norm, cmap=cmap),
                ax=ax,
                shrink=0.75,
                aspect=30,
                pad=0.1,
                label="Excess pore-water pressure",
            )

        scatter_field(
            ax_init,
            init_field,
            f"Initial slice (y = {y_value:.3f})",
        )
        scatter_field(
            ax_final,
            field,
            f"t={times[tidx]:.4f} (sample {sample_idx})",
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
