#!/bin/bash
# Wrapper executed by HTCondor on the compute node: compute the singular-value
# spectrum of the three 2D heat _U matrices in data/datasets that are still
# missing a <stem>_singvals.txt (l0.05, l0.1-0.2, sine K8, all at t=0.004) and
# write one singular value per line next to each. Pure NumPy; CPU-only.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ -f "${REPO_ROOT}/.venv-linux/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.venv-linux/bin/activate"
elif [[ -f "${REPO_ROOT}/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.venv/bin/activate"
fi

# Let BLAS use the cores we requested for the SVD.
NCPUS="${OMP_NUM_THREADS:-8}"
export OMP_NUM_THREADS="${NCPUS}"
export OPENBLAS_NUM_THREADS="${NCPUS}"
export MKL_NUM_THREADS="${NCPUS}"

echo "host=$(hostname) repo=${REPO_ROOT} python=$(command -v python) cpus=${NCPUS}"

python - "${REPO_ROOT}/data/datasets" <<'PY'
import os, sys
import numpy as np

datadir = sys.argv[1]
stems = [
    "heat2d_grf_nx64_l0.05_D1_t0.004",
    "heat2d_grf_nx64_l0.1-0.2_D1_t0.004",
    "heat2d_sine_nx64_K8_D1_t0.004",
]
for stem in stems:
    path = os.path.join(datadir, f"{stem}_U.txt")
    out = os.path.join(datadir, f"{stem}_U_singvals.txt")
    if os.path.exists(out):
        print(f"{stem}: {os.path.basename(out)} already exists, skipping", flush=True)
        continue
    U = np.loadtxt(path)                # (spatial, samples) as stored
    if U.ndim == 1:
        U = U[:, None]
    s = np.linalg.svd(U, compute_uv=False)   # min(spatial, samples) values
    np.savetxt(out, s)
    print(f"{stem}: U{U.shape} -> {len(s)} singvals, "
          f"s1={s[0]:.4e}, s_last={s[-1]:.4e}  -> {os.path.basename(out)}",
          flush=True)
print("done")
PY
