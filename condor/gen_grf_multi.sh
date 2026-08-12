#!/bin/bash
# Wrapper executed by HTCondor on the compute node: generate 2D heat-equation
# GRF datasets, ONE fixed length scale per file (no mixing), 1000 samples each,
# at t=0.004. Produces 5 separate datasets:
#   l = 0.05, 0.1, 0.3, 0.5, 0.7  ->  heat2d_grf_nx64_l<l>_D1.*
#
# Pure NumPy + scipy.fft, CPU-only, I/O-bound (np.savetxt). No GPU.
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

NCPUS="${OMP_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${NCPUS}"
export OPENBLAS_NUM_THREADS="${NCPUS}"
export MKL_NUM_THREADS="${NCPUS}"
export NUMEXPR_NUM_THREADS="${NCPUS}"

echo "host=$(hostname) repo=${REPO_ROOT} python=$(command -v python) cpus=${NCPUS}"

cd "${REPO_ROOT}/2d-pdes"
for L in 0.05 0.1 0.3 0.5 0.7; do
    echo "=== generating grf l=${L} (1000 samples, t=0.004) ==="
    # Single --l value -> every sample uses that l (no cycling/mixing).
    python generate_dataset.py \
        --family grf \
        --num-samples 1000 \
        --l "${L}" \
        --times 0.004 \
        --outdir data2d
done
echo "all done"
