from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from solve import solve_terzaghi_3d_fdm

# All knobs live here for easy tweaking.
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
    "gp_output_scale": 500.0,
    "gp_length_scale": 0.2,
    "u0_range": (10000, 20000),
    "seed": 42,
    "time_index": -1,
    "y_section": 0.25,
    "keep_positive_side": True,
    "scatter_stride": 2,
    "scatter_size": 20.0,
    "scatter_alpha": 1.0,
    "cmap": "turbo",
    "elev": 20.0,
    "azim": -60.0,
    "output_path": Path("solver/test/xz_cut_visualization.png"),
    "save_solution": None,  # set to Path("...npz") if you want to cache the run
}


def main() -> None:
    cfg = CONFIG
    t_eval = np.linspace(cfg["t_span"][0], cfg["t_span"][1], cfg["n_times"])

    result = solve_terzaghi_3d_fdm(
        Cv=cfg["Cv"],
        x_range=cfg["x_range"],
        y_range=cfg["y_range"],
        z_range=cfg["z_range"],
        nx=cfg["nx"],
        ny=cfg["ny"],
        nz=cfg["nz"],
        t_span=cfg["t_span"],
        gp_params={
            "output_scale": cfg["gp_output_scale"],
            "length_scales": cfg["gp_length_scale"],
        },
        u0_ranges=[cfg["u0_range"]],
        seed=cfg["seed"],
        t_eval=t_eval,
    )
    a=1

    if cfg["save_solution"]:
        payload = {
            "u": result["u"],
            "t": result["t"],
            "x": np.linspace(cfg["x_range"][0], cfg["x_range"][1], cfg["nx"]),
            "y": np.linspace(cfg["y_range"][0], cfg["y_range"][1], cfg["ny"]),
            "z": np.linspace(cfg["z_range"][0], cfg["z_range"][1], cfg["nz"]),
        }
        np.savez(cfg["save_solution"], **payload)

    xs = np.linspace(cfg["x_range"][0], cfg["x_range"][1], cfg["nx"])
    ys = np.linspace(cfg["y_range"][0], cfg["y_range"][1], cfg["ny"])
    zs = np.linspace(cfg["z_range"][0], cfg["z_range"][1], cfg["nz"])
    tidx = cfg["time_index"] if cfg["time_index"] >= 0 else len(result["t"]) + cfg["time_index"]

    field = result["u"][tidx]
    y_section = cfg["y_section"]
    y_idx = np.argmin(np.abs(ys - y_section))
    y_value = ys[y_idx]

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    Xf = X.reshape(-1)
    Yf = Y.reshape(-1)
    Zf = Z.reshape(-1)
    Vf = field.reshape(-1)

    if cfg["keep_positive_side"]:
        mask = Yf >= y_value
    else:
        mask = Yf <= y_value

    Xf = Xf[mask][:: cfg["scatter_stride"]]
    Yf = Yf[mask][:: cfg["scatter_stride"]]
    Zf = Zf[mask][:: cfg["scatter_stride"]]
    Vf = Vf[mask][:: cfg["scatter_stride"]]

    norm = plt.Normalize(vmin=field.min(), vmax=field.max())
    cmap = plt.get_cmap(cfg["cmap"])

    fig = plt.figure(figsize=(8, 6), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")

    sc = ax.scatter(
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
    ax.set_title(f"Excess PWP at t={result['t'][tidx]:.4f} (y = {y_value:.3f})")

    fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        shrink=0.75,
        aspect=30,
        pad=0.1,
        label="Excess pore-water pressure",
    )

    if cfg["output_path"]:
        cfg["output_path"].parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(cfg["output_path"], dpi=300)
    else:
        plt.show()


if __name__ == "__main__":
    main()
