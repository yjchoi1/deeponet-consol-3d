"""Offline POD basis computation for ROM.

Reads the full-field HDF5 file produced by data_generator_rom.py, builds a
snapshot matrix, computes a truncated POD basis via randomized SVD, and saves
the basis, mean field, singular values, and pre-projected coefficients.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import h5py
import numpy as np
from sklearn.utils.extmath import randomized_svd


CONFIG: Dict[str, object] = {
    "fields_path": Path("data/rom_fields.h5"),
    "output_path": Path("data/basis.npz"),
    "n_modes": 50,
    # Number of extra random vectors for the randomized SVD (higher = more accurate).
    "n_oversamples": 10,
    "seed": 42,
}


def compute_pod_basis(cfg: Dict[str, object]) -> None:
    """Compute POD basis from full-field ROM data."""
    fields_path = Path(cfg["fields_path"])  # type: ignore[arg-type]
    output_path = Path(cfg["output_path"])  # type: ignore[arg-type]
    n_modes = int(cfg["n_modes"])
    n_oversamples = int(cfg.get("n_oversamples", 10))  # type: ignore[arg-type]
    seed = int(cfg["seed"])

    print(f"Loading fields from {fields_path} ...")
    with h5py.File(fields_path, "r") as f:
        fields = np.asarray(f["fields"], dtype=np.float32)  # (n_samples, nt, nx, ny, nz)
        times = np.asarray(f["times"], dtype=np.float32)

    n_samples, nt, nx, ny, nz = fields.shape
    N = nx * ny * nz  # spatial DOFs per snapshot
    M = n_samples * nt  # total number of snapshots

    print(f"  n_samples={n_samples}, nt={nt}, nx={nx}, ny={ny}, nz={nz}")
    print(f"  Spatial DOFs N={N}, total snapshots M={M}")
    print(f"  Requested modes: {n_modes}")

    # Reshape into snapshot matrix (N, M).
    snapshots = fields.reshape(n_samples * nt, N).T  # (N, M)

    # Mean-center the snapshots.
    mean_field = snapshots.mean(axis=1)  # (N,)
    snapshots_centered = snapshots - mean_field[:, None]

    # Free original array to save memory.
    del fields, snapshots

    # Randomized SVD — only computes the requested number of modes.
    print("Computing randomized SVD ...")
    U, sigma, _ = randomized_svd(
        snapshots_centered,
        n_components=n_modes,
        n_oversamples=n_oversamples,
        random_state=seed,
    )
    # U: (N, n_modes), sigma: (n_modes,)

    # Energy report.
    # For the full energy we need the Frobenius norm of the centered matrix.
    total_energy = float(np.sum(snapshots_centered ** 2))
    captured = np.cumsum(sigma ** 2)
    print("\nSingular-value energy report:")
    for k in [1, 2, 5, 10, 20, min(n_modes, 50), n_modes]:
        if k > n_modes:
            break
        pct = captured[k - 1] / total_energy * 100.0
        print(f"  modes={k:4d}  energy={pct:.4f}%")

    # Project all snapshots onto the basis to get coefficients.
    # coeffs_flat: (n_modes, M)
    coeffs_flat = U.T @ snapshots_centered  # (n_modes, M)
    coeffs = coeffs_flat.T.reshape(n_samples, nt, n_modes)  # (n_samples, nt, n_modes)

    del snapshots_centered

    # Per-mode normalization statistics for training.
    coeffs_flat_all = coeffs.reshape(-1, n_modes)  # (n_samples * nt, n_modes)
    coeffs_mean = coeffs_flat_all.mean(axis=0)  # (n_modes,)
    coeffs_std = coeffs_flat_all.std(axis=0)  # (n_modes,)
    coeffs_std[coeffs_std == 0.0] = 1.0

    print("\nCoefficient normalization stats (first 10 modes):")
    for k in range(min(10, n_modes)):
        print(f"  mode {k:3d}  mean={coeffs_mean[k]:.4e}  std={coeffs_std[k]:.4e}")

    # Save.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        Phi=U.astype(np.float32),
        mean_field=mean_field.astype(np.float32),
        singular_values=sigma.astype(np.float32),
        coeffs=coeffs.astype(np.float32),
        coeffs_mean=coeffs_mean.astype(np.float32),
        coeffs_std=coeffs_std.astype(np.float32),
        times=times,
        nx=nx,
        ny=ny,
        nz=nz,
        nt=nt,
        n_modes=n_modes,
        n_samples=n_samples,
    )
    print(f"\nBasis saved to {output_path}")
    print(f"  Phi shape: ({N}, {n_modes})")
    print(f"  coeffs shape: ({n_samples}, {nt}, {n_modes})")


if __name__ == "__main__":
    compute_pod_basis(CONFIG)
