#!/bin/bash
# Wrapper executed by HTCondor on the compute node: compute the singular-value
# spectrum of every 1D _U matrix in data/datasets and write <stem>_singvals.txt
# next to each (one singular value per line). Pure NumPy; CPU-only.
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
import glob, os, sys
import numpy as np

datadir = sys.argv[1]
files = sorted(glob.glob(os.path.join(datadir, "*_U.txt")))
print(f"found {len(files)} _U files in {datadir}")
for path in files:
    stem = os.path.basename(path)[:-len("_U.txt")]  # drop trailing "_U.txt"
    U = np.loadtxt(path)                # (spatial, samples) as stored
    if U.ndim == 1:
        U = U[:, None]
    s = np.linalg.svd(U, compute_uv=False)   # min(spatial, samples) values
    out = os.path.join(datadir, f"{stem}_U_singvals.txt")
    np.savetxt(out, s)
    print(f"{stem}: U{U.shape} -> {len(s)} singvals, "
          f"s1={s[0]:.4e}, s_last={s[-1]:.4e}  -> {os.path.basename(out)}",
          flush=True)
print("done")
PY
