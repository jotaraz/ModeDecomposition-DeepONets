#!/bin/bash
# Singular values of the 5000-sample 2D heat U matrices, same convention as
# condor/compute_singvals.sh: svd(U, compute_uv=False) on the stored
# (spatial, samples) matrix, written to <stem>_U_singvals.txt.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${REPO_ROOT}/.venv-linux/bin/activate"
echo "host=$(hostname)"
python - "${REPO_ROOT}/data/datasets" <<'PY'
import glob, os, sys, numpy as np
datadir = sys.argv[1]
files = sorted(glob.glob(os.path.join(datadir, "heat2d_*_m5000_t0.004_U.txt")))
print(f"found {len(files)} m5000 2D _U files", flush=True)
for path in files:
    stem = os.path.basename(path)[:-len("_U.txt")]
    U = np.loadtxt(path)
    if U.ndim == 1: U = U[:, None]
    s = np.linalg.svd(U, compute_uv=False)
    out = os.path.join(datadir, f"{stem}_U_singvals.txt")
    np.savetxt(out, s)
    print(f"{stem}: U{U.shape} -> {len(s)} singvals, s1={s[0]:.4e}, "
          f"s_last/s1={s[-1]/s[0]:.3e} -> {os.path.basename(out)}", flush=True)
print("done", flush=True)
PY
