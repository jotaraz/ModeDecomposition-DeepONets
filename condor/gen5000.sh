#!/bin/bash
# Generate 5000-sample versions of the three 2D heat datasets used by the sweep,
# then convert them to the training convention. NEW NAMES (_m5000 suffix) so the
# existing 1000-sample datasets and the 72 nets trained on them stay valid.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${REPO_ROOT}/.venv-linux/bin/activate"
export MPLBACKEND=Agg
echo "host=$(hostname)"
cd "${REPO_ROOT}/2d-pdes"
mkdir -p data2d

START=$(date +%s)
python generate_dataset.py --family sine --num-samples 5000 --n-modes 8 \
       --times 0.004 --outdir data2d --name heat2d_sine_nx64_K8_D1_m5000
python generate_dataset.py --family grf  --num-samples 5000 --l 0.1 0.2 \
       --times 0.004 --outdir data2d --name heat2d_grf_nx64_l0.1-0.2_D1_m5000
python generate_dataset.py --family grf  --num-samples 5000 --l 0.05 \
       --times 0.004 --outdir data2d --name heat2d_grf_nx64_l0.05_D1_m5000
echo "GENERATION_SECONDS $(( $(date +%s) - START ))"

# --- convert to the training convention (same logic as move_2d_to_datasets.sh):
#     transpose (samples, spatial) -> (spatial, samples) and rename
#     <name>_U_t0.004.txt -> <name>_t0.004_U.txt
CONV=$(date +%s)
python - <<'PY'
import glob, os, shutil, numpy as np
REPO = os.path.abspath(os.path.join(os.path.dirname(os.getcwd()), "."))
src = os.path.join(os.getcwd(), "data2d")
dst = os.path.join(os.path.dirname(os.getcwd()), "data", "datasets")
UEND = "t0.004"
ufiles = sorted(f for f in glob.glob(os.path.join(src, "heat2d_*_m5000_U_t0.004.txt"))
                if not f.endswith("_singvals.txt"))
print(f"{len(ufiles)} datasets to convert", flush=True)
for uf in ufiles:
    name = os.path.basename(uf)[:-len("_U_t0.004.txt")]
    out_u = os.path.join(dst, f"{name}_{UEND}_U.txt")
    if os.path.exists(out_u):
        raise SystemExit(f"REFUSING to overwrite existing {out_u}")
    U = np.loadtxt(uf)
    assert U.ndim == 2, U.shape
    Ut = np.ascontiguousarray(U.T)
    np.savetxt(out_u, Ut)
    moved = []
    for suff in ("_R.txt", "_P.txt", ".npz"):
        s = os.path.join(src, name + suff)
        if os.path.exists(s):
            shutil.move(s, os.path.join(dst, name + suff)); moved.append(suff)
    os.remove(uf)
    print(f"{name}: {U.shape} -> {Ut.shape}; wrote {os.path.basename(out_u)}; moved {moved}", flush=True)
print("done", flush=True)
PY
echo "CONVERSION_SECONDS $(( $(date +%s) - CONV ))"
ls -la "${REPO_ROOT}/data/datasets/"*_m5000_* | awk '{print "PRODUCED", $5, $NF}'
