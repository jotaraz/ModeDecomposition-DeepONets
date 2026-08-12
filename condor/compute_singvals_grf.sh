#!/bin/bash
# Wrapper executed by HTCondor on the compute node: compute the singular-value
# spectrum of the 5 single-length-scale 2D GRF U matrices in 2d-pdes/data2d and
# write <stem>_singvals.txt next to each (one singular value per line).
# Pure NumPy; CPU-only.
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

NCPUS="${OMP_NUM_THREADS:-8}"
export OMP_NUM_THREADS="${NCPUS}"
export OPENBLAS_NUM_THREADS="${NCPUS}"
export MKL_NUM_THREADS="${NCPUS}"

echo "host=$(hostname) repo=${REPO_ROOT} python=$(command -v python) cpus=${NCPUS}"

python - "${REPO_ROOT}/2d-pdes/data2d" <<'PY'
import glob, os, sys
import numpy as np

datadir = sys.argv[1]
# The 5 single-l GRF datasets at t=0.004 (excludes the mixed l0.1-0.2 set).
files = sorted(glob.glob(os.path.join(
    datadir, "heat2d_grf_nx64_l[0-9]*_D1_U_t0.004.txt")))
files = [f for f in files if "0.1-0.2" not in f]
print(f"found {len(files)} GRF _U files in {datadir}")
for path in files:
    stem = os.path.basename(path)[:-len(".txt")]     # keep the "_U_t0.004"
    U = np.loadtxt(path)                              # (samples, spatial) as stored
    if U.ndim == 1:
        U = U[:, None]
    s = np.linalg.svd(U, compute_uv=False)
    out = os.path.join(datadir, f"{stem}_singvals.txt")
    np.savetxt(out, s)
    print(f"{stem}: U{U.shape} -> {len(s)} singvals, "
          f"s1={s[0]:.4e}, s_last={s[-1]:.4e}  -> {os.path.basename(out)}",
          flush=True)
print("done")
PY
