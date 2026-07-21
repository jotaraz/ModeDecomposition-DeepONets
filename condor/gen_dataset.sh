#!/bin/bash
# Wrapper executed by HTCondor on the compute node for 2d-pdes/generate_dataset.py.
#
# Generates the two 2D heat-equation initial-condition families from spec.md and
# evolves them to the requested time(s):
#   (a) sine series, K=8 modes per axis
#   (b) boundary-compatible GRFs, length scales cycled/alternated over {0.1, 0.2}
# 1000 samples each, saved at t=0.004.
#
# This is a PURE NumPy + scipy.fft job -- there is no JAX/CUDA code path, so it
# runs CPU-only, single-threaded, and is dominated by np.savetxt of the output
# text files (I/O-bound, not compute-bound). Do NOT request a GPU.
#
# generate_dataset.py imports diffusion_spectral and generate_grfs relatively,
# so we must run from the 2d-pdes/ directory.
set -euo pipefail

# --- Locate the repo root from this script's location ---------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# --- Python environment ---------------------------------------------------
# The committed .venv is a macOS venv and will NOT work on the Linux cluster.
# Provide a Linux env one of two ways (edit to match your setup):
#   (a) point CONDA/venv activation here, or
#   (b) `module load` + a prebuilt env.
# By default we activate $REPO_ROOT/.venv if present, else fall back to PATH.
# Only numpy + scipy are needed for generation.
if [[ -f "${REPO_ROOT}/.venv/bin/activate" ]]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.venv/bin/activate"
fi

# --- Cap thread pools (generation is essentially single-threaded anyway) ---
NCPUS="${OMP_NUM_THREADS:-1}"
export OMP_NUM_THREADS="${NCPUS}"
export OPENBLAS_NUM_THREADS="${NCPUS}"
export MKL_NUM_THREADS="${NCPUS}"
export NUMEXPR_NUM_THREADS="${NCPUS}"

echo "host=$(hostname) repo=${REPO_ROOT} python=$(command -v python) cpus=${NCPUS}"
python -c "import numpy, scipy; print('numpy', numpy.__version__, 'scipy', scipy.__version__)"

# --- Run generation -------------------------------------------------------
#   --family all      -> both (a) sine and (b) grf
#   --num-samples 1000
#   --n-modes 8       -> K=8 for the sine family
#   --l 0.1 0.2       -> GRF length scales, alternated across samples
#   --times 0.004     -> single output time
# Output text files total ~0.3-0.5 GB; write them under the repo (home/fast),
# NOT /tmp (login /tmp is only 1 GB).
cd "${REPO_ROOT}/2d-pdes"
exec python generate_dataset.py \
    --family all \
    --num-samples 1000 \
    --n-modes 8 \
    --l 0.1 0.2 \
    --times 0.004 \
    --outdir data2d
