"""Generate 2D heat-equation datasets for operator learning.

Builds initial conditions from the two families in spec.md, evolves each to the
requested times with the spectral solver, and writes the results in two formats:

  * a compressed ``.npz`` bundle (u0, grid, per-time solutions, metadata), and
  * plain-text ``_P`` / ``_R`` / ``_U_t{t}`` files matching the repo's DeepONet
    dataset convention (P = input functions, R = spatial points, U = outputs).

Initial-condition families
---------------------------
(a) "sine"  : u0 = (sum_i a_i sin(i*pi*x)) (sum_j b_j sin(j*pi*y)),
              a_i, b_j ~ U[-1, 1], i, j = 1..K  (K = --n-modes).
(b) "grf"   : boundary-compatible Gaussian random field synthesised in the
              Dirichlet eigenbasis (generate_grfs.sample_grf_dirichlet), with
              length scale l cycled over the values passed via --l.

Example (smoke test, 10 samples each family):

    python generate_dataset.py --family all --num-samples 10 --outdir data2d
"""

import argparse
import os

import numpy as np

from diffusion_spectral import diffusion_spectral, interior_grid
from generate_grfs import sample_grf_dirichlet

# With D=1 on the unit square the characteristic diffusion time is
# tau = 1/(D*2*pi^2) ~ 0.05, so these times span the transient decay (the
# fundamental mode drops from ~0.82 at t=0.01 to ~0.14 at t=0.1). Larger times
# such as {0.1,0.2,0.5,1.0} collapse the field to ~machine zero for D=1.
DEFAULT_TIMES = (0.01, 0.02, 0.05, 0.1)


def sample_sine_series(n, n_modes, rng, side=1.0):
    """One initial condition of family (a) on the (n, n) interior grid."""
    X, Y = interior_grid(n, n, side=side)
    a = rng.uniform(-1.0, 1.0, size=n_modes)
    b = rng.uniform(-1.0, 1.0, size=n_modes)
    modes = np.arange(1, n_modes + 1)
    fx = np.sin(np.pi * modes[:, None] * (X[0][None, :] / side)).T @ a  # (n,)
    fy = np.sin(np.pi * modes[:, None] * (Y[:, 0][None, :] / side)).T @ b  # (n,)
    return fy[:, None] * fx[None, :]  # (ny, nx)


def build_u0(family, n, num_samples, n_modes, length_scales, seed):
    """Return (U0, meta) where U0 has shape (num_samples, n, n)."""
    rng = np.random.default_rng(seed)
    U0 = np.empty((num_samples, n, n))
    meta = {}
    if family == "sine":
        for s in range(num_samples):
            U0[s] = sample_sine_series(n, n_modes, rng)
        meta["n_modes"] = n_modes
    elif family == "grf":
        ls = np.asarray(length_scales, dtype=float)
        # Cycle the length scales across samples and record which was used.
        per_sample_l = ls[np.arange(num_samples) % len(ls)]
        for s in range(num_samples):
            # Draw an independent integer seed per sample for reproducibility.
            child = int(rng.integers(0, 2**31 - 1))
            U0[s], _, _ = sample_grf_dirichlet(n, per_sample_l[s], seed=child)
        meta["length_scales"] = ls
        meta["l_per_sample"] = per_sample_l
    else:
        raise ValueError(f"unknown family {family!r}")
    return U0, meta


def evolve(U0, D, times, side=1.0):
    """Return U with shape (num_samples, n_times, n, n)."""
    ns, n, _ = U0.shape
    U = np.empty((ns, len(times), n, n))
    for s in range(ns):
        for k, t in enumerate(times):
            U[s, k] = diffusion_spectral(U0[s], D=D, side=side, t=t)
    return U


def default_name(family, n, D, n_modes, length_scales):
    if family == "sine":
        return f"heat2d_sine_nx{n}_K{n_modes}_D{D:g}"
    ltag = "-".join(f"{l:g}" for l in length_scales)
    return f"heat2d_grf_nx{n}_l{ltag}_D{D:g}"


def save_dataset(outdir, name, U0, U, X, Y, times, D, side, meta):
    """Write the .npz bundle and the _P / _R / _U_t{t} text files."""
    os.makedirs(outdir, exist_ok=True)
    ns, nt, n, _ = U.shape

    # --- .npz bundle (full-resolution, structured) ---
    npz_path = os.path.join(outdir, name + ".npz")
    np.savez_compressed(
        npz_path,
        u0=U0, u=U, X=X, Y=Y, times=np.asarray(times, float),
        D=D, side=side, **meta,
    )

    # --- DeepONet-style text files (flattened, row-major over the grid) ---
    R = np.column_stack([X.ravel(), Y.ravel()])          # (n*n, 2)
    np.savetxt(os.path.join(outdir, name + "_R.txt"), R)
    np.savetxt(os.path.join(outdir, name + "_P.txt"),
               U0.reshape(ns, n * n))                     # inputs
    for k, t in enumerate(times):
        np.savetxt(os.path.join(outdir, f"{name}_U_t{t:g}.txt"),
                   U[:, k].reshape(ns, n * n))            # outputs at time t

    return npz_path


def generate(family, n, num_samples, n_modes, length_scales, D, times, side,
             seed, outdir, name=None):
    name = name or default_name(family, n, D, n_modes, length_scales)
    U0, meta = build_u0(family, n, num_samples, n_modes, length_scales, seed)
    X, Y = interior_grid(n, n, side=side)
    U = evolve(U0, D, times, side=side)
    npz_path = save_dataset(outdir, name, U0, U, X, Y, times, D, side, meta)
    print(f"[{family}] {num_samples} samples, grid {n}x{n}, times {list(times)}"
          f" -> {npz_path}")
    return npz_path


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--family", choices=["sine", "grf", "all"], default="all")
    p.add_argument("--n", type=int, default=64, help="interior grid points per axis")
    p.add_argument("--num-samples", type=int, default=10,
                   help="samples per family (default 10, a smoke-test size)")
    p.add_argument("--n-modes", type=int, default=8,
                   help="K, sine modes per axis for family (a)")
    p.add_argument("--l", type=float, nargs="+", default=[0.1, 0.2],
                   help="GRF length scale(s) for family (b); cycled across samples")
    p.add_argument("--D", type=float, default=1.0, help="diffusion coefficient")
    p.add_argument("--times", type=float, nargs="+", default=list(DEFAULT_TIMES))
    p.add_argument("--side", type=float, default=1.0, help="domain side length")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", default="data2d")
    p.add_argument("--name", default=None, help="override output basename")
    args = p.parse_args()

    families = ["sine", "grf"] if args.family == "all" else [args.family]
    for i, fam in enumerate(families):
        generate(fam, args.n, args.num_samples, args.n_modes, args.l, args.D,
                 tuple(args.times), args.side, args.seed + i, args.outdir,
                 args.name)


if __name__ == "__main__":
    main()
