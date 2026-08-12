#!/bin/bash
# Move the generated 2D heat datasets from 2d-pdes/data2d into data/datasets,
# converting them to the DeepONet training convention expected by
# src/don_code.py:load_dataset:
#   * U files: transpose (samples, spatial) -> (spatial, samples) AND rename
#     <name>_U_t0.004.txt  ->  <name>_t0.004_U.txt   (so uendtag = "t0.004")
#   * R / P / npz: moved unchanged (already in the right orientation).
# Singular-value files are left behind in data2d (analysis artifacts).
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

echo "host=$(hostname) repo=${REPO_ROOT} python=$(command -v python)"

python - "${REPO_ROOT}/2d-pdes/data2d" "${REPO_ROOT}/data/datasets" <<'PY'
import glob, os, shutil, sys
import numpy as np

src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)
UEND = "t0.004"

ufiles = sorted(glob.glob(os.path.join(src, "heat2d_*_U_t0.004.txt")))
ufiles = [u for u in ufiles if not u.endswith("_singvals.txt")]  # belt & braces
print(f"{len(ufiles)} datasets to move", flush=True)

for uf in ufiles:
    base = os.path.basename(uf)
    name = base[:-len("_U_t0.004.txt")]          # e.g. heat2d_sine_nx64_K8_D1
    out_u = os.path.join(dst, f"{name}_{UEND}_U.txt")
    if os.path.exists(out_u):
        raise SystemExit(f"REFUSING to overwrite existing {out_u}")

    U = np.loadtxt(uf)                            # (samples, spatial)
    assert U.ndim == 2, f"unexpected U shape {U.shape}"
    Ut = np.ascontiguousarray(U.T)               # (spatial, samples)
    np.savetxt(out_u, Ut)

    moved = []
    for suff in ("_R.txt", "_P.txt", ".npz"):
        s = os.path.join(src, name + suff)
        if os.path.exists(s):
            shutil.move(s, os.path.join(dst, name + suff))
            moved.append(suff)

    os.remove(uf)                                 # original (untransposed) U
    print(f"{name}: U {U.shape} -> {Ut.shape}; "
          f"wrote {os.path.basename(out_u)}; moved {moved}", flush=True)

print("done", flush=True)
PY
